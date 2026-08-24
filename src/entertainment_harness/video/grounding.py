"""Segment->region grounding with a verification judge (anchored-scroll,
Phase 3).

For each narration segment, the vision model sees its assigned page with the
panel regions (Phase 2's panels.json -> regions.regions_for_panels) outlined
and numbered in reading order, plus the segment text, and returns region
numbers — batched per page (all of a page's segments in one call), so the
model never emits coordinates itself (no coordinate-space ambiguity; only
indices into the numbered overlay). A grounding judge then verifies each
choice: (page+boxes with the chosen regions highlighted, beat text, chosen
regions) -> "does this region illustrate this story beat" — explicitly NOT
"is the beat's text visible in the region" (the narration is a retelling).
Rejections re-ground with the judge's issues fed back (the recap judge-loop
idiom, bounded by MAX_GROUND_ATTEMPTS); a page that still fails (rejected,
unassignable, or an unparseable judge verdict) is PRUNED from the segment —
the page never renders for that beat (safe over smooth, scoped to the page
level: an unverifiable choice never ships). Only a segment whose every
assigned page was pruned degrades to a whole-page pan over its first
original page (a "segment fallback").

Two quality metrics come out of the stage (see the initiative README for the
precise definitions): assignment quality — ok / (ok + pruned) over judged
segment-pages, measuring assign_pages' picks; and the rendered-slot hit rate
— verified hold anchors / (verified holds + segment-fallback pans), measuring
what actually ships (vacuous pages, where only a whole-page region exists,
are neither judged nor counted in either). The rendered-slot rate is the
Phase-3 gate.

The resolved anchors (region boxes, so assembly never re-reads regions) are
stamped onto Segment.regions and the surviving pages onto Segment.pages;
grounding.json next to chapter.json makes the stage re-runnable without
model calls: versioned, keyed by a content hash of the segment text + its
ORIGINAL (pre-prune) assigned pages, corrupt/foreign/malformed -> recompute.
A re-extracted panels.json does NOT invalidate entries (stamped boxes stay
self-consistent); delete grounding.json to force re-grounding.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from entertainment_harness.library import works
from entertainment_harness.models.base import ModelAdapter
from entertainment_harness.pipelines.judge import Verdict
from entertainment_harness.video.panels import extract_chapter_panels
from entertainment_harness.video.regions import Region, regions_for_panels
from entertainment_harness.video.script import Segment

GROUNDING_FILENAME = "grounding.json"
# 3: judge-rejected pages are now PRUNED from the segment (previously they
# kept a whole-page pan anchor, so rejected art still shipped); entries gain
# post-prune "pages" and "segment_fallback", and the hit-rate metric moved
# from judged pairs to rendered slots. v2 entries predate pruning and must
# be re-judged; the Phase-3 assign_pages tightening re-keys segments anyway.
CACHE_VERSION = 3

MAX_GROUND_ATTEMPTS = 3  # proposals per (segment, page): batch + focused retries

GROUND_PROMPT = """This is page {page} of {total} of chapter {chapter} of the manga "{title}", with its panel regions outlined and numbered in reading order.

These narration beats from a recap video are illustrated by something on this page:
{segments}

For EACH beat, give the region number(s) whose art best illustrates what the beat narrates, in region order. The narration retells the story in new words — match beats to regions by story content (who is shown, what happens, where), NOT by matching the beat's words to text printed in a region. Most beats match one region; a beat whose moment spans several panels may match several.

Output ONLY a JSON object mapping beat index to a list of region numbers:
{{"12": [2], "13": [1, 2]}}"""

JUDGE_GROUND_PROMPT = """This is page {page} of chapter {chapter} of the manga "{title}" with its panel regions outlined and numbered. The region(s) chosen for the narration beat below are highlighted in green.

Narration beat {index}: [{moment}] "{text}"
Chosen region(s): {regions}

Verdict task: decide whether the highlighted region(s) are the right ILLUSTRATION for this story beat — would a viewer hearing the beat while looking at the highlighted art see the scene being narrated?

The narration retells the story in new words. Judge at the level of scene and moment (who is present, what is broadly happening, where), never at the level of wording or fine detail:
- The beat's wording does NOT need to match text printed in the region, and paraphrased dialogue is expected.
- PASS when the highlighted art shows the beat's scene or moment — even when small narrated details (an exact expression, a prop, a glow, a named character being on-panel) are missing, simplified, or framed differently.
- FAIL only when the highlighted art clearly shows a DIFFERENT event, moment, or place than the beat — or when another numbered region on the page clearly shows the beat's moment instead.
- FAIL when none of the highlighted art relates to the beat at all.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["one sentence naming the moment the highlighted region actually shows instead", ...]}}"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class GroundingError(Exception):
    pass


@dataclass
class Anchor:
    """One viewport anchor for a segment: a region to hold on, or a
    whole-page pan (the safe fallback). `box` is the normalized region box
    for kind "hold"; None for kind "pan"."""

    page: int  # 1-based page number
    kind: str  # "hold" | "pan"
    box: list[float] | None = None

    def to_dict(self) -> dict:
        d = {"page": self.page, "kind": self.kind}
        if self.box is not None:
            d["box"] = self.box
        return d

    @staticmethod
    def from_dict(data) -> Anchor | None:
        if not isinstance(data, dict):
            return None
        page, kind = data.get("page"), data.get("kind")
        if not isinstance(page, (int, float)) or kind not in ("hold", "pan"):
            return None
        if kind == "hold":
            box = data.get("box")
            if not (isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)):
                return None
            return Anchor(int(page), "hold", [float(v) for v in box])
        return Anchor(int(page), "pan")


def draw_region_overlay(
    page: Path,
    regions: list[Region],
    dest: Path,
    *,
    highlight: tuple[int, ...] = (),
) -> Path:
    """Draw the page with numbered region boxes (reading order, 1-based
    labels); highlighted indices (0-based) get a thick green box, the rest a
    thin red one. Saved as JPEG (small model payload)."""
    with Image.open(page) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    font = ImageFont.load_default(size=max(24, w // 28))
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region.box
        rect = [x0 * w, y0 * h, x1 * w, y1 * h]
        chosen = i in highlight
        color = (0, 200, 0) if chosen else (230, 40, 40)
        width = max(6, w // 150) if chosen else max(3, w // 300)
        draw.rectangle(rect, outline=color, width=width)
        label = str(i + 1)
        tx, ty = rect[0] + 6, rect[1] + 6
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(
            [bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4],
            fill=color,
        )
        draw.text((tx, ty), label, font=font, fill="white")
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, format="JPEG", quality=88)
    return dest


def parse_grounding(
    raw: str, valid_segments: set[int], region_count: int
) -> dict[int, list[int]]:
    """Extract {segment_index: [region indices]} from model output.

    Model-facing region numbers are 1-based overlay labels; the returned
    indices are 0-based, deduplicated and sorted (the regions list is in
    reading order, so sorted indices are reading order). Out-of-range
    numbers, unknown segments, and non-list values are dropped; a segment
    left with no valid numbers is omitted. GroundingError when no JSON
    object is found at all.
    """
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        raise GroundingError("Grounding model did not return a JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GroundingError(
            f"Grounding model returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise GroundingError("Grounding model did not return a JSON object")
    result: dict[int, list[int]] = {}
    for key, value in data.items():
        try:
            seg_idx = int(key)
        except (TypeError, ValueError):
            continue
        if seg_idx not in valid_segments or not isinstance(value, list):
            continue
        indices = sorted({
            int(v) - 1 for v in value
            if isinstance(v, (int, float)) and 1 <= int(v) <= region_count
        })
        if indices:
            result[seg_idx] = indices
    return result


def parse_ground_verdict(raw: str) -> Verdict:
    """Strict verdict parse — the INVERSION of judge.parse_verdict's policy:
    unparseable output is a REJECTION, not a pass. A broken grounding judge
    must not wave unverified choices through (safe over smooth); the repair
    loop treats it like any rejection and the page is pruned from the segment
    after the bounded attempts."""
    match = _JSON_OBJ_RE.search(raw)
    data = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        return Verdict(passed=False, issues=["judge output was not valid JSON"])
    passed = bool(data.get("pass", data.get("passed", True)))
    issues = [str(i) for i in data.get("issues", []) if str(i).strip()]
    if passed:
        issues = []
    elif not issues:
        issues = ["judge gave no specific issues"]
    return Verdict(passed=passed, issues=issues)


def _segments_block(segments: list[Segment]) -> str:
    return "\n".join(f"{s.index}. [{s.moment}] {s.text}" for s in segments)


def ground_page(
    adapter: ModelAdapter,
    model: str,
    overlay: Path,
    page_num: int,
    total_pages: int,
    region_count: int,
    segments: list[Segment],
    title: str,
    chapter_num: float,
    *,
    log: Callable[[str], None] = lambda m: None,
) -> dict[int, list[int]]:
    """The batched per-page grounding call: all of the page's segments in one
    prompt. Unparseable output retries once; persistent failure yields {}
    (each segment then goes through the focused repair loop)."""
    prompt = GROUND_PROMPT.format(
        page=page_num, total=total_pages, chapter=f"{chapter_num:g}",
        title=title, segments=_segments_block(segments),
    )
    valid = {s.index for s in segments}
    for attempt in (1, 2):
        raw = adapter.generate(model, prompt, images=[overlay])
        try:
            return parse_grounding(raw, valid, region_count)
        except GroundingError:
            if attempt == 2:
                log(f"  warning: page {page_num}: grounding output unparseable"
                    " twice; segments will re-ground individually.")
    return {}


def _ground_one(
    adapter: ModelAdapter,
    model: str,
    overlay: Path,
    page_num: int,
    total_pages: int,
    region_count: int,
    segment: Segment,
    title: str,
    chapter_num: float,
    feedback: str,
) -> list[int] | None:
    """Focused (re-)grounding of a single segment, optionally with judge
    feedback appended (the recap judge-loop idiom). None = no assignment."""
    prompt = GROUND_PROMPT.format(
        page=page_num, total=total_pages, chapter=f"{chapter_num:g}",
        title=title, segments=_segments_block([segment]),
    ) + feedback
    raw = adapter.generate(model, prompt, images=[overlay])
    try:
        return parse_grounding(raw, {segment.index}, region_count).get(
            segment.index
        )
    except GroundingError:
        return None


def judge_grounding(
    adapter: ModelAdapter,
    model: str,
    overlay: Path,
    segment: Segment,
    indices: list[int],
    page_num: int,
    title: str,
    chapter_num: float,
) -> Verdict:
    """Judge one segment's chosen regions on a page (the overlay already has
    them highlighted). See parse_ground_verdict for the failure policy."""
    prompt = JUDGE_GROUND_PROMPT.format(
        page=page_num, chapter=f"{chapter_num:g}", title=title,
        index=segment.index, moment=segment.moment, text=segment.text,
        regions=", ".join(str(i + 1) for i in indices),
    )
    return parse_ground_verdict(adapter.generate(model, prompt, images=[overlay]))


def _judge_repair(
    adapter: ModelAdapter,
    model: str,
    page: Path,
    overlay: Path,
    regions: list[Region],
    segment: Segment,
    page_num: int,
    total_pages: int,
    initial: list[int] | None,
    title: str,
    chapter_num: float,
    tmpdir: Path,
    log: Callable[[str], None],
) -> tuple[list[int] | None, dict]:
    """Verify a segment's page assignment through the grounding judge,
    re-grounding with feedback on rejection (bounded). Returns the final
    0-based indices, or None plus a "pruned" record (the caller removes the
    page from the segment)."""
    indices = initial
    attempts = 1 if indices else 0  # the batch proposal counts as one
    feedback = ""
    reason = "unassigned"
    while True:
        if indices is None:
            if attempts >= MAX_GROUND_ATTEMPTS:
                return None, {"status": "pruned", "reason": reason,
                              "attempts": attempts}
            attempts += 1
            indices = _ground_one(
                adapter, model, overlay, page_num, total_pages, len(regions),
                segment, title, chapter_num, feedback,
            )
            if indices is None:
                feedback = (
                    "\n\nYou must assign at least one region number —"
                    " pick the closest match even when unsure."
                )
                continue
        highlighted = draw_region_overlay(
            page, regions,
            tmpdir / f"judge-{segment.index:03d}-p{page_num:03d}-{attempts}.jpg",
            highlight=tuple(indices),
        )
        verdict = judge_grounding(
            adapter, model, highlighted, segment, indices,
            page_num, title, chapter_num,
        )
        if verdict.passed:
            return indices, {"status": "ok", "attempts": attempts}
        reason = "rejected"
        log(f"  grounding judge rejected segment {segment.index} on page"
            f" {page_num} (attempt {attempts}/{MAX_GROUND_ATTEMPTS}):"
            f" {'; '.join(verdict.issues)}")
        feedback = (
            "\n\nA previous assignment for this beat was rejected by the"
            " evaluator for these reasons:\n"
            + "\n".join(f"- {i}" for i in verdict.issues)
            + "\nAssign region number(s) that fix these problems."
        )
        indices = None


def _segment_key(text: str, pages: list[int]) -> str:
    """Content key for a cached grounding: the segment text and its original
    (pre-prune) assigned pages. A regenerated script re-keys (and recomputes)
    only the segments that actually changed."""
    blob = text + "\x00" + ",".join(str(p) for p in sorted(pages))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def ground_segments(
    adapter: ModelAdapter,
    model: str,
    segments: list[Segment],
    page_paths: list[Path],
    series_id: str,
    chapter_id: str,
    title: str,
    chapter_num: float,
    *,
    log: Callable[[str], None] = lambda m: None,
) -> dict:
    """Ground every segment to region anchors (stamps Segment.regions) and
    cache the result per chapter. Judge-rejected pages are pruned from
    Segment.pages (a fully pruned segment keeps one whole-page pan anchor
    over its first original page). Returns the quality stats:
    {judged, ok, pruned, vacuous, segment_fallbacks, assignment_rate,
    verified, unverified, slots, rate} — `rate` is the rendered-slot hit
    rate (the Phase-3 gate), `assignment_rate` measures assign_pages.
    """
    cached = load_grounding(series_id, chapter_id) or {}
    # Keys are content-hashed from the ORIGINAL (pre-prune) pages; pruning
    # mutates Segment.pages only in the apply pass below, after every lookup.
    keys = {s.index: _segment_key(s.text, s.pages) for s in segments}
    todo = [s for s in segments if keys[s.index] not in cached]
    entries: dict[str, dict] = {}
    if todo:
        assigned = sorted({
            p for s in todo for p in s.pages if 1 <= p <= len(page_paths)
        })
        panels = extract_chapter_panels(
            adapter, model, title, chapter_num, series_id, chapter_id,
            [page_paths[p - 1] for p in assigned], log=log,
        )
        regions_by_page = {
            p: regions_for_panels(panels.get(page_paths[p - 1].name, []))
            for p in assigned
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for p in assigned:
                page_segments = [s for s in todo if p in s.pages]
                regions = regions_by_page[p]
                if len(regions) < 2:
                    continue  # whole-page region only: grounding is vacuous
                overlay = draw_region_overlay(
                    page_paths[p - 1], regions,
                    tmpdir / f"ground-p{p:03d}.jpg",
                )
                log(f"  grounding: page {p} ({len(page_segments)} beat(s),"
                    f" {len(regions)} regions)...")
                proposals = ground_page(
                    adapter, model, overlay, p, len(page_paths),
                    len(regions), page_segments, title, chapter_num, log=log,
                )
                for s in page_segments:
                    indices, record = _judge_repair(
                        adapter, model, page_paths[p - 1], overlay, regions,
                        s, p, len(page_paths), proposals.get(s.index),
                        title, chapter_num, tmpdir, log,
                    )
                    entry = entries.setdefault(keys[s.index],
                                               {"anchors": [], "judge": {}})
                    entry["judge"][str(p)] = record
                    if indices is not None:  # pruned pages get no anchor
                        entry["anchors"].extend(
                            Anchor(p, "hold", list(regions[i].box)).to_dict()
                            for i in indices
                        )
        # Pages with only the whole-page region never reached the loop above:
        # stamp their pan anchors + vacuous records here.
        for s in todo:
            entry = entries.setdefault(keys[s.index],
                                       {"anchors": [], "judge": {}})
            for p in sorted(
                p for p in s.pages
                if 1 <= p <= len(page_paths) and str(p) not in entry["judge"]
            ):
                entry["anchors"].append(Anchor(p, "pan").to_dict())
                entry["judge"][str(p)] = {"status": "vacuous"}
        # Prune judge-rejected pages (safe over smooth, scoped to the page
        # level): a pruned page never renders for that beat. A segment that
        # loses every page degrades to a whole-page pan over its first
        # original page — the one unverified slot kind the gate counts.
        for s in todo:
            entry = entries[keys[s.index]]
            surviving = [
                p for p in s.pages
                if entry["judge"].get(str(p), {}).get("status") != "pruned"
            ]
            if not surviving:
                first = min(max(s.pages[0], 1), len(page_paths)) if s.pages else 1
                surviving = [first]
                entry["anchors"] = [Anchor(first, "pan").to_dict()]
                entry["segment_fallback"] = True
            entry["anchors"].sort(key=lambda a: a["page"])
            entry["pages"] = surviving
        cached.update(entries)
        # Prune entries for segments no longer in the script (regenerations
        # re-key), then persist. Keys come from the pre-prune pages, so fresh
        # entries survive the GC.
        current = set(keys.values())
        cached = {k: v for k, v in cached.items() if k in current}
        store_grounding(series_id, chapter_id, cached)

    stats = {"judged": 0, "ok": 0, "pruned": 0, "vacuous": 0,
             "segment_fallbacks": 0, "verified": 0, "unverified": 0}
    for s in segments:
        entry = cached.get(keys[s.index], {"anchors": [], "judge": {}})
        s.regions = [a.to_dict() for a in
                     (Anchor.from_dict(a) for a in entry["anchors"]) if a]
        stored_pages = entry.get("pages")
        if isinstance(stored_pages, list) and stored_pages:
            s.pages = [int(p) for p in stored_pages]
        for record in entry["judge"].values():
            status = record.get("status")
            if status == "ok":
                stats["ok"] += 1
                stats["judged"] += 1
            elif status == "pruned":
                stats["pruned"] += 1
                stats["judged"] += 1
            elif status == "vacuous":
                stats["vacuous"] += 1
        if entry.get("segment_fallback"):
            stats["segment_fallbacks"] += 1
        for a in s.regions:
            if a["kind"] == "hold":
                stats["verified"] += 1
            elif entry.get("segment_fallback"):
                stats["unverified"] += 1  # vacuous pans are not counted
    stats["assignment_rate"] = (
        round(stats["ok"] / stats["judged"], 3) if stats["judged"] else None
    )
    stats["slots"] = stats["verified"] + stats["unverified"]
    stats["rate"] = (
        round(stats["verified"] / stats["slots"], 3) if stats["slots"] else None
    )
    if stats["slots"] or stats["judged"]:
        rate = f"{stats['rate']:.1%}" if stats["rate"] is not None else "n/a"
        a_rate = (f"{stats['assignment_rate']:.1%}"
                  if stats["assignment_rate"] is not None else "n/a")
        log(f"  grounding hit rate: {stats['verified']}/{stats['slots']}"
            f" rendered slots verified ({rate}); assignment quality"
            f" {stats['ok']}/{stats['judged']} pages kept ({a_rate}),"
            f" {stats['pruned']} page(s) pruned,"
            f" {stats['vacuous']} vacuous (no regions),"
            f" {stats['segment_fallbacks']} whole-page segment fallback(s).")
    return stats


# --- Cache (grounding.json next to chapter.json, content-keyed by segment) ---


def grounding_path(series_id: str, chapter_id: str) -> Path:
    return works.chapter_dir(series_id, chapter_id) / GROUNDING_FILENAME


def store_grounding(series_id: str, chapter_id: str, entries: dict) -> Path:
    payload = {"version": CACHE_VERSION, "segments": entries}
    path = grounding_path(series_id, chapter_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_grounding(series_id: str, chapter_id: str) -> dict | None:
    """Read the cached grounding entries (keyed by segment content hash over
    the ORIGINAL assigned pages), or None when absent/unreadable/foreign
    version. Entries whose anchors are all malformed are dropped so callers
    recompute them."""
    path = grounding_path(series_id, chapter_id)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or d.get("version") != CACHE_VERSION:
        return None
    segments = d.get("segments")
    if not isinstance(segments, dict):
        return None
    out: dict[str, dict] = {}
    for key, entry in segments.items():
        if not isinstance(entry, dict):
            continue
        anchors = [
            a.to_dict()
            for a in (Anchor.from_dict(a) for a in entry.get("anchors", []))
            if a is not None
        ]
        if not anchors:
            continue
        judge = entry.get("judge")
        loaded = {
            "anchors": anchors,
            "judge": judge if isinstance(judge, dict) else {},
        }
        pages = entry.get("pages")
        if (isinstance(pages, list)
                and all(isinstance(p, (int, float)) for p in pages)):
            loaded["pages"] = [int(p) for p in pages]
        if entry.get("segment_fallback"):
            loaded["segment_fallback"] = True
        out[key] = loaded
    return out
