"""Per-page panel extraction (anchored-scroll initiative, Phase 2).

The vision model reads one manga page and returns its panels in reading
order as {"box", "description"} entries — the beat source for both the
grounded path (regions come from these boxes via
regions.regions_for_panels) and the panel-first narration path.

Parse/validation follows the translate.py bubble idiom (parse_bubbles):
tolerates code fences/preamble, repairs common JSON mistakes, detects the
model's coordinate system response-wide — normalized floats, true pixels
(the translate.py fallback), or qwen's relative-1000, which models flip
between unpredictably (observed live with qwen3-vl-32b) — clamps to
[0, 1], drops degenerate, inverted, speck-sized, and description-less
entries plus anything beyond MAX_PANELS, and salvages individual objects
when the full array won't parse. Extraction retries once on unparseable output; when no
panels survive (repeated parse failure or a valid empty result), one
plain-text describe-the-page call salvages a single whole-page panel, so
story content is never silently dropped from the beat stream; only when
that also fails does the page yield [] (whole-page regions downstream)
rather than killing the chapter — translate.py propagates instead, but
this stage feeds fallback-composing assembly, so degrading is correct here.

Cached per chapter as panels.json next to chapter.json, keyed by page
filename (page content never changes), mirroring regions.py's cache:
versioned, corrupt/foreign-version/malformed -> recompute.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from entertainment_harness.library import works
from entertainment_harness.models.base import ModelAdapter
from entertainment_harness.pipelines.translate import (
    _extract_json_objects,
    _JSON_ARRAY_RE,
    _repair_json,
)

PANELS_FILENAME = "panels.json"
CACHE_VERSION = 1

MIN_PANEL_AREA = 0.005  # normalized; a real panel is far bigger, this drops noise
MAX_PANELS = 15         # sanity cap; manga pages run ~2-8 panels

EXTRACT_PANELS_PROMPT = """This is page {page} of {total} of chapter {chapter} of the manga "{title}".

Divide the page into its panels and describe each one. List the panels in manga reading order (top to bottom, right to left).

For EACH panel output one entry:
- "box": bounding box [x1, y1, x2, y2] as normalized floats 0.0-1.0 (origin top-left), tight around that single panel including its art and any text — never spanning multiple panels
- "description": 1-2 sentences on what happens in the panel (characters, action, setting, dialogue content) — a reader who cannot see the page should be able to follow the story from the descriptions alone

Rules:
- Every panel exactly once, in reading order. Ignore scanlation credits and watermarks outside the panels.
- Only describe what is actually visible on the page; no outside knowledge of any manga or anime.
- A full-page spread with no panel divisions is a single panel.

Output ONLY a JSON array:
[{{"box": [0.05, 0.04, 0.95, 0.30], "description": "..."}}]"""

DESCRIBE_PAGE_PROMPT = """This is page {page} of {total} of chapter {chapter} of the manga "{title}".

Describe what happens on this page in 2-3 sentences: characters, action, setting, dialogue content — a reader who cannot see the page should be able to follow the story from the description alone.

Rules:
- Only describe what is actually visible on the page; no outside knowledge of any manga or anime.

Output ONLY the description as plain prose — no JSON, no lists, no preamble."""


class PanelError(Exception):
    pass


@dataclass
class Panel:
    box: list[float]  # normalized [x1, y1, x2, y2]
    description: str


def _parse_panel_object(text: str) -> Panel | None:
    """Parse one panel object after best-effort repair. Coordinates arrive
    pre-normalized by parse_panels (which detects the model's coordinate
    system response-wide), so this only validates shape and description."""
    for candidate in (text, _repair_json(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            return None
        box = data.get("box")
        if not (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)):
            return None
        coords = [float(v) for v in box]
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in coords)
        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) * (y2 - y1) < MIN_PANEL_AREA:
            return None
        description = str(data.get("description") or "").strip()
        if not description:
            return None
        return Panel(box=[x1, y1, x2, y2], description=description)
    return None


def _normalize_coord_system(
    entries: list, image_size: tuple[int, int] | None
) -> None:
    """Rewrite every entry's box to normalized floats, detecting the
    model's coordinate system once per response.

    Models flip between three formats (observed live with qwen3-vl-32b):
    proper normalized floats (left alone), true pixels (divided by the
    image dims, the translate.py idiom), and qwen's relative-1000 system
    (divided by 1000). Pixels and relative-1000 are told apart
    response-wide: a coordinate beyond the image dims can only be
    relative-1000, and a response that never exceeds 1000 on a page whose
    dims both exceed 1000 is relative-1000 too — true-pixel output on a
    fully-paneled page would reach the bottom panels' ~1500+ y-coords.
    """
    boxes = [
        e["box"] for e in entries
        if isinstance(e, dict)
        and isinstance(e.get("box"), list) and len(e["box"]) == 4
        and all(isinstance(v, (int, float)) for v in e["box"])
    ]
    if not boxes or image_size is None:
        return
    w, h = image_size
    coords = [[float(v) for v in box] for box in boxes]
    if max(max(c) for c in coords) <= 1.0:
        return  # proper normalized floats
    exceeds = any(c[0] > w or c[2] > w or c[1] > h or c[3] > h for c in coords)
    relative_1000 = exceeds or (
        w > 1000 and h > 1000 and max(max(c) for c in coords) <= 1000
    )
    for box, c in zip(boxes, coords):
        if relative_1000:
            box[:] = [v / 1000.0 for v in c]
        else:
            box[:] = [c[0] / w, c[1] / h, c[2] / w, c[3] / h]


def parse_panels(
    raw: str, *, page_label: str = "", image_size: tuple[int, int] | None = None
) -> list[Panel]:
    """Extract a validated panel list from model output.

    Same contract as translate.parse_bubbles: code fences/preamble
    tolerated, boxes clamped to [0, 1], malformed entries dropped
    individually, individual objects salvaged when the full array won't
    parse, PanelError when nothing usable remains. At most MAX_PANELS
    entries are kept.
    """
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        raise PanelError("Model did not return a JSON array")
    data = None
    last_exc: Exception | None = None
    for candidate in (match.group(0), _repair_json(match.group(0))):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue

    if isinstance(data, list):
        _normalize_coord_system(data, image_size)
        panels = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            panel = _parse_panel_object(json.dumps(entry))
            if panel is not None:
                panels.append(panel)
        return panels[:MAX_PANELS]

    # Full-array parse failed. Salvage individual objects so one malformed
    # panel doesn't kill the page.
    entries = []
    for obj_text in _extract_json_objects(raw):
        try:
            entry = json.loads(_repair_json(obj_text))
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    _normalize_coord_system(entries, image_size)
    salvaged: list[Panel] = []
    for entry in entries:
        panel = _parse_panel_object(json.dumps(entry))
        if panel is not None:
            salvaged.append(panel)
    if salvaged:
        return salvaged[:MAX_PANELS]

    label = f" ({page_label})" if page_label else ""
    snippet = raw[:800].replace("\n", "\\n")
    raise PanelError(
        f"Model returned invalid JSON{label}: {last_exc}\n"
        f"Raw snippet: {snippet}"
    ) from last_exc


def extract_panels(
    adapter: ModelAdapter,
    model: str,
    page: Path,
    prompt_args: dict,
    *,
    log: Callable[[str], None] = lambda m: None,
) -> list[Panel]:
    """Run the vision model on one page and return its panels in reading
    order. Unparseable output retries once; when no panels survive, a
    plain-text describe-the-page call salvages one whole-page panel
    (observed needed with qwen3-vl-32b, which intermittently fails to
    panel dense pages that clearly have story content). Only when the
    fallback also yields nothing does the page return [] (whole-page
    regions downstream)."""
    prompt = EXTRACT_PANELS_PROMPT.format(**prompt_args)
    label = f"page {prompt_args['page']} of chapter {prompt_args['chapter']}"
    with Image.open(page) as img:
        image_size = img.size

    for attempt in (1, 2):
        raw = adapter.generate(model, prompt, images=[page])
        try:
            panels = parse_panels(raw, page_label=label, image_size=image_size)
        except PanelError:
            continue
        if panels:
            return panels
        break  # valid empty result — retrying the same prompt won't help

    log(f"  warning: {label}: panel extraction yielded nothing;"
        " describing the whole page instead.")
    description = adapter.generate(
        model, DESCRIBE_PAGE_PROMPT.format(**prompt_args), images=[page]
    ).strip()
    if not description or description.startswith(("[", "{")):
        log(f"  warning: {label}: whole-page describe also failed;"
            " using the whole-page fallback downstream.")
        return []
    return [Panel(box=[0.0, 0.0, 1.0, 1.0], description=description)]


def extract_chapter_panels(
    adapter: ModelAdapter,
    model: str,
    series_title: str,
    chapter_num: float,
    series_id: str,
    chapter_id: str,
    pages: list[Path],
    *,
    log: Callable[[str], None] = lambda m: None,
) -> dict[str, list[Panel]]:
    """Panel extraction for one chapter with per-page caching: pages already
    in panels.json are served from cache (page content never changes), the
    rest are extracted and the cache is rewritten after each page."""
    cached = load_panels(series_id, chapter_id) or {}
    # Seed with everything cached, not just this call's hits: callers pass
    # subsets (assigned/missing pages only), and store_panels rewrites the
    # whole file — without the seed a subset call would clobber every other
    # page's cache entry.
    out: dict[str, list[Panel]] = dict(cached)
    for i, page in enumerate(pages, start=1):
        if page.name in cached:
            log(f"  page {i}/{len(pages)}: panels cached"
                f" ({len(cached[page.name])}).")
            out[page.name] = cached[page.name]
            continue
        panels = extract_panels(
            adapter, model, page,
            {"page": i, "total": len(pages),
             "chapter": f"{chapter_num:g}", "title": series_title},
            log=log,
        )
        log(f"  page {i}/{len(pages)}: {len(panels)} panel(s).")
        out[page.name] = panels
        store_panels(series_id, chapter_id, out)
    return out


# --- Cache (panels.json next to chapter.json, keyed by page filename) ---


def panels_path(series_id: str, chapter_id: str) -> Path:
    return works.chapter_dir(series_id, chapter_id) / PANELS_FILENAME


def store_panels(
    series_id: str, chapter_id: str, pages: dict[str, list[Panel]]
) -> Path:
    payload = {
        "version": CACHE_VERSION,
        "pages": {
            name: [{"box": p.box, "description": p.description} for p in panels]
            for name, panels in pages.items()
        },
    }
    path = panels_path(series_id, chapter_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_panels(series_id: str, chapter_id: str) -> dict[str, list[Panel]] | None:
    """Read the cached panels, or None when absent/unreadable/foreign
    version (callers recompute). Pages whose entries are all malformed are
    dropped so callers recompute them."""
    path = panels_path(series_id, chapter_id)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or d.get("version") != CACHE_VERSION:
        return None
    pages = d.get("pages")
    if not isinstance(pages, dict):
        return None
    out: dict[str, list[Panel]] = {}
    for name, entries in pages.items():
        if not isinstance(entries, list):
            continue
        panels = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            panel = _parse_panel_object(json.dumps(entry))
            if panel is not None:
                panels.append(panel)
        if panels:
            out[name] = panels
    return out
