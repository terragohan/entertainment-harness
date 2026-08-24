"""Stage 3: pick which chapter pages illustrate each narration segment.

v1 uses full pages with Ken Burns motion (panel crops deferred per docs/design.md).
The vision model reads labeled contact sheets (thumbnails + page numbers) a
chunk at a time and returns, per segment, the pages in that chunk that best
illustrate it. Segments the model leaves unassigned fall back to their
neighbors' pages, so every segment always has at least one page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

from entertainment_harness.models.base import ModelAdapter
from entertainment_harness.video.script import Segment, VideoError

CHUNK_SIZE = 6  # pages per contact sheet; small sheets keep the model local —
# the Phase-3 gate showed 12-page sheets inviting thematic grabs from distant
# pages (one exposition page assigned to 26 fight-arc beats)
SHEET_COLS = 4
THUMB_W = 420  # thumbnail width in px; height follows aspect

MOTIONS = ("pan_down", "zoom_in")

ASSIGN_PROMPT = """The attached image is a contact sheet of pages {start}-{end} of {where}. Each panel is labeled with its page number.

Below is the narration script for a recap video. For each segment, list the page numbers FROM PAGES {start}-{end} that show the EXACT moment the segment narrates — the same characters, the same action, the same place. A page matches only when a viewer hearing the segment while looking at the page would see the very scene being narrated.

Never assign a page because it is thematically related, shows the same characters at a different moment, or depicts a flashback or flash-forward of the event. Most segments will match ZERO pages in this range — when in doubt, omit the segment. A page may serve several segments, and a segment may match several pages when its moment genuinely spans them.

Output ONLY a JSON object mapping segment index to a list of page numbers, omitting segments with no match in this range:
{{"2": [7], "5": [10, 11]}}

Segments:
{segments}"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_contact_sheet(pages: list[Path], first_page_num: int, dest: Path) -> Path:
    """Tile page thumbnails into a labeled grid image."""
    thumbs = []
    for offset, page in enumerate(pages):
        img = Image.open(page)
        img.thumbnail((THUMB_W, THUMB_W * 2))
        canvas = Image.new("RGB", (img.width, img.height + 28), "white")
        canvas.paste(img, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, img.width, 28], fill="black")
        draw.text((8, 8), f"PAGE {first_page_num + offset}", fill="white")
        thumbs.append(canvas)

    cols = min(SHEET_COLS, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "gray")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * cell_w, (i // cols) * cell_h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest


def parse_assignments(raw: str, valid_pages: set[int]) -> dict[int, list[int]]:
    """Extract {segment_index: [pages]} from model output, dropping page
    numbers outside the chunk that was shown."""
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        raise VideoError("Page-assignment model did not return a JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise VideoError(f"Page-assignment model returned invalid JSON: {exc}") from exc
    result: dict[int, list[int]] = {}
    for key, value in data.items():
        try:
            seg_idx = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, list):
            continue
        pages = sorted({int(p) for p in value
                        if isinstance(p, (int, float)) and int(p) in valid_pages})
        if pages:
            result[seg_idx] = pages
    return result


def _fill_gaps(segments: list[Segment], page_count: int) -> None:
    """Every segment needs a page: inherit from a neighbor, else page 1."""
    for i, seg in enumerate(segments):
        if seg.pages:
            continue
        neighbor = next(
            (segments[j].pages[0]
             for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(segments)))
             if segments[j].pages),
            1,
        )
        seg.pages = [min(max(neighbor, 1), page_count)]


def assign_pages(
    adapter: ModelAdapter,
    model: str,
    segments: list[Segment],
    page_paths: list[Path],
    workdir: Path,
    title: str,
    chapter_num: float,
    log=lambda m: None,
    where: str | None = None,
) -> None:
    """Fill segment.pages (vision model) and segment.motion (alternating).

    `where` overrides the prompt's description of what the pages belong to
    (default: "chapter N of the manga '<title>'") — whole-work videos pass a
    work-level label since the page list spans chapters.
    """
    where = where or f'chapter {chapter_num:g} of the manga "{title}"'
    seg_block = "\n".join(
        f"{s.index}. [{s.moment}] {s.text}" for s in segments
    )
    for chunk_start in range(0, len(page_paths), CHUNK_SIZE):
        chunk = page_paths[chunk_start : chunk_start + CHUNK_SIZE]
        first = chunk_start + 1  # page numbers are 1-based
        last = chunk_start + len(chunk)
        sheet = build_contact_sheet(
            chunk, first, workdir / f"sheet-{first:03d}-{last:03d}.png"
        )
        prompt = ASSIGN_PROMPT.format(
            start=first, end=last, where=where, segments=seg_block,
        )
        log(f"  page assignment: pages {first}-{last}...")
        raw = adapter.generate(model, prompt, images=[sheet])
        for seg_idx, pages in parse_assignments(raw, set(range(first, last + 1))).items():
            if 0 <= seg_idx < len(segments):
                merged = sorted({*segments[seg_idx].pages, *pages})
                segments[seg_idx].pages = merged

    _fill_gaps(segments, len(page_paths))
    for seg in segments:
        seg.motion = MOTIONS[seg.index % len(MOTIONS)]
