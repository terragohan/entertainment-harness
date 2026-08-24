"""Ordered panel regions per page (anchored-scroll initiative, Phase 1).

Regions come from text locations on the page: boxes from any extraction
source — the translation stage's bubble parse, or the Phase-3 panel-beat
extraction; only the normalized "box" [x0, y0, x1, y1] key is read — are
clustered into panel bounding boxes, padded, and sorted into manga reading
order: top bands first, right-to-left within a band (RTL). A page with no
usable boxes falls back to a single whole-page region.

Availability finding (Phase-1 gate): the real library has NO bubble metadata
for non-translated chapters — bubbles are a by-product of the translation
stage (`eh recap --translated`), which chapters already in a preferred
language (all of kenja-no-mago) never run. Regions therefore become
available on real pages only once the Phase-3 extraction stage exists; this
module is deliberately extraction-format-agnostic.

Cached per chapter as regions.json next to chapter.json, keyed by page
filename. When the panel extraction (video/panels.py) is the source, use
regions_for_panels: panel-level boxes are not clustered, only padded and
ordering-validated.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from entertainment_harness.library import works

REGIONS_FILENAME = "regions.json"
CACHE_VERSION = 1

DEFAULT_PADDING = 0.02      # normalized page fraction; merge margin and the
                            # margin emitted around each region box
DEFAULT_BAND_OVERLAP = 0.3  # vertical overlap (fraction of the shorter span)
                            # for two regions to share a reading band
DEFAULT_TALL_FACTOR = 1.8   # a region this much taller than the page's
                            # median region spans bands (full-height column);
                            # it joins the topmost band it overlaps WITHOUT
                            # stretching that band's span page-wide

Box = tuple[float, float, float, float]
WHOLE_PAGE: Box = (0.0, 0.0, 1.0, 1.0)


@dataclass
class Region:
    """One panel region on a page, normalized (x0, y0, x1, y1) within [0, 1]."""

    box: Box
    bubbles: int = 1  # boxes merged into this region; 0 = whole-page fallback


def sanitize_box(raw: Any) -> Box | None:
    """Coerce a raw [x0, y0, x1, y1] into a normalized box, or None.

    Clamps to [0, 1]; drops anything that is not four finite numbers, and
    boxes that are inverted or zero-area after clamping.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in vals)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _expand(box: Box, padding: float) -> Box:
    x0, y0, x1, y1 = box
    return (
        max(x0 - padding, 0.0),
        max(y0 - padding, 0.0),
        min(x1 + padding, 1.0),
        min(y1 + padding, 1.0),
    )


def _overlaps(a: Box, b: Box) -> bool:
    return (
        min(a[2], b[2]) > max(a[0], b[0])
        and min(a[3], b[3]) > max(a[1], b[1])
    )


def cluster_bubbles(
    bubbles: Iterable[Mapping[str, Any]], *, padding: float = DEFAULT_PADDING
) -> list[Region]:
    """Merge bubble boxes into panel regions.

    Two bubbles belong to the same panel when their padding-expanded boxes
    overlap (transitively); each cluster becomes the padding-expanded
    bounding box of its members. Malformed boxes are dropped; no usable
    boxes yields an empty list (callers apply the whole-page fallback).
    Order is deterministic (first-member order) — use reading_order() for
    the presentation order.
    """
    boxes = [
        box
        for raw in bubbles
        if (box := sanitize_box(raw.get("box") if isinstance(raw, Mapping) else None))
        is not None
    ]
    if not boxes:
        return []
    expanded = [_expand(b, padding) for b in boxes]
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlaps(expanded[i], expanded[j]):
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi

    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)

    regions = []
    for members in groups.values():
        bbox: Box = (
            min(boxes[i][0] for i in members),
            min(boxes[i][1] for i in members),
            max(boxes[i][2] for i in members),
            max(boxes[i][3] for i in members),
        )
        regions.append(Region(box=_expand(bbox, padding), bubbles=len(members)))
    return regions


def reading_order(
    regions: Iterable[Region],
    *,
    band_overlap: float = DEFAULT_BAND_OVERLAP,
    tall_factor: float = DEFAULT_TALL_FACTOR,
) -> list[Region]:
    """Sort regions into manga reading order.

    Two passes. First, regions whose vertical spans overlap (by
    band_overlap of the shorter span) are swept into horizontal bands.
    Then "tall" regions — taller than tall_factor × the median region
    height, i.e. column-spanners like a full-height right panel — are
    placed into the topmost band they overlap, WITHOUT extending that
    band's span (letting a page-tall box stretch its band page-wide
    collapses every row into one band and scrambles the order, seen live
    on kenja ch-001 p1). Bands run top to bottom; within a band regions
    run right to left.
    """
    regions = list(regions)
    if not regions:
        return []
    median_h = statistics.median(r.box[3] - r.box[1] for r in regions)
    tall, short = [], []
    for r in regions:
        (tall if (r.box[3] - r.box[1]) > tall_factor * median_h else short
         ).append(r)

    bands: list[list[Region]] = []
    spans: list[list[float]] = []  # [y0, y1] per band, parallel to bands
    key = lambda r: (r.box[1], -(r.box[0] + r.box[2]))

    def place(r: Region, *, extend_span: bool) -> None:
        for i, (y0, y1) in enumerate(spans):
            overlap = min(r.box[3], y1) - max(r.box[1], y0)
            shorter = min(r.box[3] - r.box[1], y1 - y0)
            if shorter > 0 and overlap >= band_overlap * shorter:
                bands[i].append(r)
                if extend_span:
                    spans[i] = [min(y0, r.box[1]), max(y1, r.box[3])]
                return
        bands.append([r])
        spans.append([r.box[1], r.box[3]])

    for r in sorted(short, key=key):
        place(r, extend_span=True)
    for r in sorted(tall, key=key):
        place(r, extend_span=False)

    ordered: list[Region] = []
    for i in sorted(range(len(bands)), key=lambda i: spans[i][0]):
        ordered.extend(
            sorted(bands[i], key=lambda r: -(r.box[0] + r.box[2]))
        )
    return ordered


def regions_for_page(
    bubbles: Iterable[Mapping[str, Any]],
    *,
    padding: float = DEFAULT_PADDING,
    band_overlap: float = DEFAULT_BAND_OVERLAP,
) -> list[Region]:
    """Ordered panel regions for one page; a single whole-page region when
    the page has no usable boxes."""
    regions = cluster_bubbles(bubbles, padding=padding)
    if not regions:
        return [Region(box=WHOLE_PAGE, bubbles=0)]
    return reading_order(regions, band_overlap=band_overlap)


def regions_for_panels(
    panels: Iterable[Any],
    *,
    padding: float = DEFAULT_PADDING,
    band_overlap: float = DEFAULT_BAND_OVERLAP,
) -> list[Region]:
    """Ordered regions from panel-level boxes (the panels.json extraction).

    No clustering here — the extraction is already panel-level, so merging
    distinct adjacent panels would be wrong; each valid panel box becomes
    one padded region, and the set passes through reading_order as an
    ordering validation (the model returns reading order; geometry
    double-checks it). Entries may be mappings with a "box" key or objects
    with a .box attribute (panels.Panel). No usable boxes yields the
    whole-page fallback.
    """
    regions = []
    for raw in panels:
        box_raw = (
            raw.get("box") if isinstance(raw, Mapping) else getattr(raw, "box", None)
        )
        box = sanitize_box(box_raw)
        if box is not None:
            regions.append(Region(box=_expand(box, padding), bubbles=1))
    if not regions:
        return [Region(box=WHOLE_PAGE, bubbles=0)]
    return reading_order(regions, band_overlap=band_overlap)


# --- Cache (regions.json next to chapter.json, keyed by page filename) ---


def regions_path(series_id: str, chapter_id: str) -> Path:
    return works.chapter_dir(series_id, chapter_id) / REGIONS_FILENAME


def store_regions(
    series_id: str, chapter_id: str, pages: Mapping[str, list[Region]]
) -> Path:
    payload = {
        "version": CACHE_VERSION,
        "pages": {
            name: [{"box": list(r.box), "bubbles": r.bubbles} for r in regs]
            for name, regs in pages.items()
        },
    }
    path = regions_path(series_id, chapter_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_regions(series_id: str, chapter_id: str) -> dict[str, list[Region]] | None:
    """Read the cached regions, or None when absent/unreadable/foreign
    version (callers recompute). Pages whose entries are all malformed are
    dropped so callers recompute them."""
    path = regions_path(series_id, chapter_id)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict) or d.get("version") != CACHE_VERSION:
        return None
    pages = d.get("pages")
    if not isinstance(pages, dict):
        return None
    out: dict[str, list[Region]] = {}
    for name, regs in pages.items():
        if not isinstance(regs, list):
            continue
        regions = []
        for entry in regs:
            if not isinstance(entry, dict):
                continue
            box = sanitize_box(entry.get("box"))
            if box is None:
                continue
            try:
                count = int(entry.get("bubbles", 1))
            except (TypeError, ValueError):
                count = 1
            regions.append(Region(box=box, bubbles=count))
        if regions:
            out[name] = regions
    return out
