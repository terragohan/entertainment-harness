"""Panel-first narration (anchored-scroll initiative, Phase 4).

The alternative stage-1 path for scroll-mode videos: instead of retelling
the chapter first and hunting for each beat's page afterwards (assign_pages
+ the grounding judge), the chapter's cached panel beats (panels.json,
Phase 2) ARE the script's skeleton. A text-role model groups ADJACENT
panels into narration beats and writes one spoken line per group, so every
segment is born with its panel span: Segment.pages and Segment.regions
(padded hold anchors, in reading order) are stamped at grouping time and
stages 3/3.5 are skipped entirely — no page assignment, no grounding
judge, and no whole-page fallbacks on paneled pages.

Grouping output is validated against a strict partition policy (see
parse_groups): every panel in exactly one group, in order; overlaps are
clamped, gaps close by extending the previous group, and unusable items
are dropped — a response with nothing usable retries (SCRIPT_ATTEMPTS,
the script.py idiom). The grouped retelling then goes through the
narration judge (completeness axis) exactly like a full-detail artifact:
per-page panel descriptions are the evidence batches, the steering
instruction is shown to both writer and judge, thinking "high" verifies
against the page images, and persistent failure keeps the FIRST attempt
(the recap judge_loop feedback-poisoning guard).

Cached per chapter as panelfirst.json next to chapter.json, keyed by a
hash of the beat sequence (page + description per beat) plus the steering
instruction: re-extracted panels or a new instruction regroup, everything
else is served from cache. Groups are stored as {"from","to"} indices
into the beat stream, so fresher panel boxes apply at segment build time
without invalidating the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.models.registry import Selection
from entertainment_harness.pipelines.judge import MAX_ATTEMPTS, judge_narration
from entertainment_harness.pipelines.recap import user_direction_block
from entertainment_harness.video.grounding import Anchor
from entertainment_harness.video.panels import extract_chapter_panels, load_panels
from entertainment_harness.video.regions import DEFAULT_PADDING, _expand
from entertainment_harness.video.script import (
    SCRIPT_ATTEMPTS,
    Segment,
    VideoError,
    split_long_segments,
)

PANELFIRST_FILENAME = "panelfirst.json"
CACHE_VERSION = 1


@dataclass
class Beat:
    """One panel in the chapter's reading order: the atomic narration unit."""

    index: int  # 1-based, global chapter order
    page: int  # 1-based page number
    box: tuple[float, float, float, float]  # padded, normalized
    description: str


def chapter_beats(
    page_paths: list[Path],
    series_id: str,
    chapter_id: str,
    *,
    log: Callable[[str], None] = lambda m: None,
) -> list[Beat]:
    """Flatten panels.json into the chapter's ordered beat stream: page
    order, then extraction (reading) order within the page. Pages with no
    extracted panels are skipped (logged) — with the panels.py whole-page
    describe fallback, a skip means extraction genuinely failed, not that
    the page was empty."""
    panels = load_panels(series_id, chapter_id) or {}
    beats: list[Beat] = []
    skipped: list[str] = []
    for page_num, path in enumerate(page_paths, start=1):
        page_panels = panels.get(path.name, [])
        if not page_panels:
            skipped.append(path.name)
            continue
        for panel in page_panels:
            beats.append(
                Beat(
                    index=len(beats) + 1,
                    page=page_num,
                    box=_expand(tuple(panel.box), DEFAULT_PADDING),
                    description=panel.description,
                )
            )
    if skipped:
        log(f"  panel-first: no panels on {len(skipped)} page(s), skipped:"
            f" {', '.join(skipped)}")
    return beats


GROUP_PROMPT = """You are scripting a narrated video of chapter {chapter} of the manga "{title}".

Below are the chapter's panels in reading order, numbered, each with its page and a description of what it shows.

Panels:
---
{beats}
---

Group ADJACENT panels into narration beats for the video: each group becomes one camera hold over those panels while a spoken line retells that moment.

Rules:
- Every panel appears in exactly one group, in order — the groups partition the list. Never skip, reorder, or repeat a panel.
- Most groups hold 1-3 panels; a panel carrying a big moment may stand alone.
- "text" is ONE spoken sentence (two at most), present tense, retelling what the group's panels show — actions, dialogue, reactions. Plain spoken English; no panel numbers, no "the panel shows", no headers.
- Only events and names visible in the descriptions above; no outside knowledge of any manga or anime.
- "moment" is a 1-3 word label for the story beat; reuse the same label for consecutive groups in the same scene.
{direction}{feedback}
Output ONLY a JSON array, no other text:
[{{"from": 1, "to": 2, "moment": "...", "text": "..."}}]"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_GROUP_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_items(raw: str) -> list:
    """Pull candidate group objects out of raw model output: the full JSON
    array when it parses, else individually salvaged {...} objects (the
    script.py truncated-array idiom)."""
    match = _JSON_ARRAY_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    items = []
    for candidate in _GROUP_OBJ_RE.findall(raw):
        try:
            item = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def _normalize_groups(items: list, beat_count: int) -> list[dict]:
    """Validate candidate items into a strict partition of beats 1..N.

    Policy: items without usable from/to/text are dropped; spans clamp into
    [1, N]; a group overlapping the span already covered is clamped to
    start after it (dropped when that empties it); a gap before a group's
    start closes by extending the previous group (a leading gap clamps the
    first group to beat 1); beats left over after the last group extend
    it. VideoError when nothing usable remains.
    """
    groups: list[dict] = []
    covered = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start, end = int(item.get("from")), int(item.get("to"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        moment = str(item.get("moment") or "").strip()
        start = min(max(start, 1), beat_count)
        end = min(max(end, 1), beat_count)
        start = max(start, covered + 1)
        if end < start:
            continue
        if start > covered + 1 and groups:
            groups[-1]["to"] = start - 1
        groups.append({"from": start, "to": end, "moment": moment, "text": text})
        covered = end
    if not groups:
        raise VideoError("Panel grouping produced no usable groups")
    groups[0]["from"] = 1
    if groups[-1]["to"] < beat_count:
        groups[-1]["to"] = beat_count
    return groups


def parse_groups(raw: str, beat_count: int) -> list[dict]:
    """Extract a validated, gapless, ordered group partition from raw model
    output. See _normalize_groups for the gap/overlap policy."""
    if beat_count < 1:
        raise VideoError("Panel grouping needs at least one beat")
    return _normalize_groups(_extract_items(raw), beat_count)


def group_beats(
    adapter,
    model: str,
    beats: list[Beat],
    *,
    title: str,
    chapter_num: float,
    instruction: str = "",
    feedback: str = "",
    max_attempts: int = SCRIPT_ATTEMPTS,
    log: Callable[[str], None] = lambda m: None,
) -> list[dict]:
    """One grouping call with script.py's retry policy: a response with no
    usable groups retries up to max_attempts, then VideoError."""
    prompt = GROUP_PROMPT.format(
        chapter=f"{chapter_num:g}",
        title=title,
        beats="\n".join(
            f"{b.index}. [page {b.page}] {b.description}" for b in beats
        ),
        direction=user_direction_block(instruction),
        feedback=feedback,
    )
    last_error: VideoError | None = None
    for attempt in range(1, max_attempts + 1):
        raw = adapter.generate(model, prompt)
        try:
            return parse_groups(raw, len(beats))
        except VideoError as exc:
            last_error = exc
            log(f"  warning: panel grouping attempt {attempt}/{max_attempts}"
                f" failed: {exc}")
    raise VideoError(
        f"Panel grouping failed after {max_attempts} attempts: {last_error}"
    )


def _narration_text(groups: list[dict]) -> str:
    return "\n\n".join(g["text"] for g in groups)


def _evidence_batches(beats: list[Beat]) -> list[str]:
    """The judge's source material: per-page blocks of panel descriptions,
    in reading order (the narration judge's 'sequential narrations of
    page/section batches')."""
    pages: dict[int, list[str]] = {}
    for beat in beats:
        pages.setdefault(beat.page, []).append(beat.description)
    return [
        f"Page {page}:\n" + "\n".join(f"- {d}" for d in descriptions)
        for page, descriptions in pages.items()
    ]


def _judged_groups(
    generate: Callable[[str], list[dict]],
    judge_fn,
    *,
    max_attempts: int,
    log: Callable[[str], None],
) -> tuple[list[dict], str]:
    """The recap.judge_loop policy carried on groups directly, so the kept
    version's panel spans always match its wording: regenerate with the
    judge's issues fed back until a pass (bounded by max_attempts); on
    persistent failure keep the FIRST attempt (feedback-poisoning guard).
    Returns (groups, status)."""
    feedback = ""
    first: list[dict] | None = None
    for attempt in range(1, max_attempts + 1):
        groups = generate(feedback)
        if first is None:
            first = groups
        verdict = judge_fn(_narration_text(groups))
        if verdict.passed:
            return groups, (
                "passed" if attempt == 1 else f"passed after {attempt} attempts"
            )
        log(f"  judge rejected panel-first narration"
            f" (attempt {attempt}/{max_attempts}): {'; '.join(verdict.issues)}")
        feedback = (
            "\n\nA previous attempt was rejected by the evaluator for these"
            " reasons:\n"
            + "\n".join(f"- {i}" for i in verdict.issues)
            + "\nFix these problems in the new version."
        )
    log(f"  warning: panel-first narration failed the judge {max_attempts}"
        " times; keeping the first version.")
    return first, "kept first"


def segments_from_groups(groups: list[dict], beats: list[Beat]) -> list[Segment]:
    """Stamp each group's panel span onto its segment(s): pages in reading
    order, one padded hold anchor per panel. An over-long group text splits
    at sentence boundaries (split_long_segments) and every piece re-stamps
    the same span — the split helper rebuilds plain Segments, so the
    re-stamp happens here, after it."""
    segments: list[Segment] = []
    for group in groups:
        span = beats[group["from"] - 1 : group["to"]]
        pages = list(dict.fromkeys(b.page for b in span))
        regions = [
            Anchor(page=b.page, kind="hold", box=list(b.box)).to_dict()
            for b in span
        ]
        pieces = split_long_segments(
            [Segment(index=0, text=group["text"], moment=group["moment"])]
        )
        for piece in pieces:
            piece.pages = list(pages)
            piece.regions = [dict(r) for r in regions]
            segments.append(piece)
    for i, segment in enumerate(segments):
        segment.index = i
    return segments


# --- Cache (panelfirst.json next to chapter.json, keyed by beat sequence) ---


def _cache_key(beats: list[Beat], instruction: str) -> str:
    """Content key for a cached grouping: the beat sequence (page +
    description per beat) and the steering instruction. Panel BOXES are
    deliberately excluded — groups are {"from","to"} indices, so fresher
    boxes apply at segment build time without regrouping."""
    h = hashlib.sha1()
    for beat in beats:
        h.update(f"{beat.page}\x00{beat.description}\n".encode("utf-8"))
    h.update(b"\x00instruction\x00")
    h.update(instruction.encode("utf-8"))
    return h.hexdigest()


def _panelfirst_path(series_id: str, chapter_id: str) -> Path:
    return works.chapter_dir(series_id, chapter_id) / PANELFIRST_FILENAME


def _load_groups(
    series_id: str, chapter_id: str, key: str, beat_count: int
) -> list[dict] | None:
    """Read the cached grouping, or None when absent/unreadable/foreign
    version/stale key/no longer normalizable (callers regroup)."""
    try:
        d = json.loads(_panelfirst_path(series_id, chapter_id).read_text(
            encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or d.get("version") != CACHE_VERSION:
        return None
    if d.get("key") != key:
        return None
    groups = d.get("groups")
    if not isinstance(groups, list):
        return None
    try:
        return _normalize_groups(groups, beat_count)
    except VideoError:
        return None


def _store_groups(
    series_id: str, chapter_id: str, key: str, groups: list[dict], status: str
) -> Path:
    payload = {
        "version": CACHE_VERSION,
        "key": key,
        "status": status,
        "groups": groups,
    }
    path = _panelfirst_path(series_id, chapter_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_panel_first_script(
    vision: Selection,
    text: Selection,
    judge: Selection | None,
    page_paths: list[Path],
    series,
    chapter,
    *,
    instruction: str = "",
    max_attempts: int = MAX_ATTEMPTS,
    vision_verify: bool = False,
    log: Callable[[str], None] = lambda m: None,
) -> tuple[list[Segment], str]:
    """Stage 1 for panel-first videos: group the chapter's cached panel
    beats into narration segments born with their spans.

    Missing pages are panel-extracted first (per-page cached, restart-safe)
    using the vision role. With a judge (thinking medium/high) the grouped
    retelling goes through the narration judge — evidence batches are the
    per-page panel descriptions; vision_verify judges against the page
    images instead (thinking high). Returns (segments, status).
    """
    title = series["title"]
    chapter_num = chapter["chapter_num"]
    cached_panels = load_panels(series["id"], chapter["id"]) or {}
    missing = [p for p in page_paths if p.name not in cached_panels]
    if missing:
        vision.adapter.ensure(vision.info.name)
        log(f"  panel-first: extracting panels on {len(missing)} page(s)...")
        extract_chapter_panels(
            vision.adapter, vision.info.name, title, chapter_num,
            series["id"], chapter["id"], missing, log=log,
        )
    beats = chapter_beats(page_paths, series["id"], chapter["id"], log=log)
    if not beats:
        raise VideoError(
            f"Chapter {chapter_num:g}: panel-first found no panels to narrate"
            " — run 'eh recap' first so the pages are cached."
        )
    key = _cache_key(beats, instruction)
    groups = _load_groups(series["id"], chapter["id"], key, len(beats))
    if groups is not None:
        status = "cached"
    else:
        def generate(feedback: str) -> list[dict]:
            return group_beats(
                text.adapter, text.info.name, beats,
                title=title, chapter_num=chapter_num,
                instruction=instruction, feedback=feedback, log=log,
            )

        if judge is None:
            groups = generate("")
            status = "unjudged"
        else:
            batches = _evidence_batches(beats)
            if vision_verify:
                judge_fn = lambda t: judge_narration(  # noqa: E731
                    vision.adapter, vision.info.name, t, batches, None,
                    title, chapter_num, pages=page_paths,
                    instruction=instruction,
                )
            else:
                judge_fn = lambda t: judge_narration(  # noqa: E731
                    judge.adapter, judge.info.name, t, batches, None,
                    title, chapter_num, instruction=instruction,
                )
            groups, status = _judged_groups(
                generate, judge_fn, max_attempts=max_attempts, log=log
            )
        _store_groups(series["id"], chapter["id"], key, groups, status)
    return segments_from_groups(groups, beats), status
