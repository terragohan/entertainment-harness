"""Credits end card appended to every rendered video.

Sharing content made with Entertainment Harness requires crediting both the
source material and the tool (see SHARING.md); the end card makes that the
default instead of something sharers have to remember. `[video] credits =
false` skips it for private viewing — the attribution requirement for shared
content stands either way.

The card is a PIL-rendered PNG (dark, purple accent — the terragohan.com
palette), cached by content hash so a title/author edit renders a fresh card
while an identical re-run reuses it. The muxer (video/assemble.py) turns it
into a static clip of CREDITS_SECONDS and appends matching silence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CREDITS_SECONDS = 4.0

BACKGROUND = (7, 4, 15)  # terragohan.com --bg
ACCENT = (168, 85, 247)  # terragohan.com purple
TEXT = (232, 227, 248)
MUTED = (167, 159, 208)

TOOL_LINE = "Made with Entertainment Harness"
TOOL_URL = "terragohan.com"

_SOURCE_LABELS = {"import": "local import", "search": "online search"}


@dataclass
class Credits:
    """The attribution stamped onto a video's end card."""

    title: str
    source: str
    chapter_label: str = ""  # "Chapter 47"; "" for whole-work videos
    author: str | None = None

    def meta_line(self) -> str:
        parts = []
        if self.author:
            parts.append(f"by {self.author}")
        parts.append(f"Source: {_SOURCE_LABELS.get(self.source, self.source)}")
        if self.chapter_label:
            parts.append(self.chapter_label)
        return "  ·  ".join(parts)


def _fit_font(draw: ImageDraw.ImageDraw, text: str, size: int,
              max_width: int, floor: int = 16) -> ImageFont.FreeTypeFont:
    """Largest load_default size (<= size) whose text fits max_width."""
    while size > floor:
        font = ImageFont.load_default(size=size)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size = int(size * 0.9)
    return ImageFont.load_default(size=floor)


def render_card(credits: Credits, size: tuple[int, int], dest: Path) -> Path:
    """Render the end card PNG at dest. One centered block: work title,
    byline/source/chapter meta, an accent rule, then the tool line."""
    width, height = size
    margin = int(width * 0.08)
    max_text_w = width - 2 * margin

    img = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(img)

    base = min(width, height)
    heading_font = _fit_font(draw, credits.title, int(base / 11), max_text_w)
    meta = credits.meta_line()
    meta_font = _fit_font(draw, meta, int(base / 22), max_text_w)
    tool_font = _fit_font(draw, TOOL_LINE, int(base / 20), max_text_w)
    url_font = ImageFont.load_default(size=max(16, int(base / 30)))

    def h(text: str, font) -> int:
        box = draw.textbbox((0, 0), text, font=font)
        return box[3] - box[1]

    gap = int(base / 40)
    rule_w = int(width * 0.22)
    block_h = (
        h(credits.title, heading_font) + gap
        + h(meta, meta_font) + gap * 2 + 3 + gap * 2
        + h(TOOL_LINE, tool_font) + gap
        + h(TOOL_URL, url_font)
    )
    y = (height - block_h) // 2

    def centered(text: str, font, fill: str | tuple[int, int, int]) -> None:
        nonlocal y
        box = draw.textbbox((0, 0), text, font=font)
        x = (width - (box[2] - box[0])) // 2 - box[0]
        draw.text((x, y - box[1]), text, font=font, fill=fill)
        y += (box[3] - box[1])

    centered(credits.title, heading_font, TEXT)
    y += gap
    centered(meta, meta_font, MUTED)
    y += gap * 2
    draw.rectangle(
        [(width - rule_w) // 2, y, (width + rule_w) // 2, y + 3], fill=ACCENT
    )
    y += 3 + gap * 2
    centered(TOOL_LINE, tool_font, ACCENT)
    y += gap
    centered(TOOL_URL, url_font, MUTED)

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return dest


def card_path(credits: Credits, size: tuple[int, int], workdir: Path) -> Path:
    """Content-hash-keyed card PNG: identical inputs reuse the cached render
    (and the identically named clip the muxer builds from it); any change to
    the attribution or frame size renders fresh instead of serving stale."""
    key = hashlib.sha256(
        f"{credits.title}|{credits.author}|{credits.source}"
        f"|{credits.chapter_label}|{size[0]}x{size[1]}".encode()
    ).hexdigest()[:12]
    dest = workdir / "clips" / f"credits-{key}.png"
    if not dest.exists():
        render_card(credits, size, dest)
    return dest
