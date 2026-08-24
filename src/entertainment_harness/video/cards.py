"""Caption-card images for book short-form videos (no page art to show).

One 1080x1920 card per narration segment: moment label up top, the segment's
spoken text in large word-wrapped type, over a blurred cover backdrop when the
import found a cover (dark gradient otherwise). The first card also shows the
work's title. Rendered at the output resolution, so assembly only adds a slow
zoom. No font files needed: Pillow's scalable load_default (Aileron) is used,
falling back to the bitmap default.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from entertainment_harness.video.script import Segment

ACCENT = (255, 209, 102)
FOREGROUND = (245, 245, 245)


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow: bitmap default only
        return ImageFont.load_default()


def _backdrop(size: tuple[int, int], cover: Path | None) -> Image.Image:
    width, height = size
    if cover is not None:
        with Image.open(cover) as img:
            img = img.convert("RGB")
            scale = max(width / img.width, height / img.height)
            img = img.resize((round(img.width * scale), round(img.height * scale)))
            left = (img.width - width) // 2
            top = (img.height - height) // 2
            img = img.crop((left, top, left + width, top + height))
        img = img.filter(ImageFilter.GaussianBlur(60))
        return Image.blend(img, Image.new("RGB", size, (10, 10, 14)), 0.55)
    # dark vertical gradient
    base = Image.new("RGB", (1, height))
    for y in range(height):
        v = round(14 + 22 * y / height)
        base.putpixel((0, y), (v, v, v + 8))
    return base.resize(size)


def _wrap_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int
) -> str:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return "\n".join(lines)


def _centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_y: float,
    size: tuple[int, int],
    fill: tuple[int, int, int],
) -> None:
    width, _ = size
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=round(font.size * 0.35))
    block_h = bbox[3] - bbox[1]
    draw.multiline_text(
        (width / 2, center_y - block_h / 2),
        text,
        font=font,
        fill=fill,
        anchor="ma",
        align="center",
        spacing=round(font.size * 0.35),
    )


def render_card(
    segment: Segment,
    dest: Path,
    size: tuple[int, int] = (1080, 1920),
    cover: Path | None = None,
    title: str | None = None,
) -> Path:
    width, height = size
    img = _backdrop(size, cover)
    draw = ImageDraw.Draw(img)

    font_scale = width / 1080  # fonts are tuned for a 1080-wide card
    label_font = _font(round(52 * font_scale))
    body_font = _font(round(62 * font_scale))
    title_font = _font(round(96 * font_scale))
    max_w = round(width * 0.84)

    if title:
        wrapped_title = _wrap_to_width(draw, title, title_font, max_w)
        _centered(draw, wrapped_title, title_font, height * 0.22, size, ACCENT)
    moment = segment.moment.upper() if segment.moment else ""
    if moment:
        wrapped_moment = _wrap_to_width(draw, moment, label_font, max_w)
        _centered(
            draw, wrapped_moment, label_font,
            height * (0.38 if title else 0.24), size, ACCENT,
        )
    body = _wrap_to_width(draw, segment.text, body_font, max_w)
    _centered(draw, body, body_font, height * (0.62 if title else 0.55), size, FOREGROUND)

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return dest


def render_cards(
    segments: list[Segment],
    title: str,
    workdir: Path,
    size: tuple[int, int] = (1080, 1920),
    cover: Path | None = None,
) -> list[Path]:
    """One card per segment; the first also carries the title. Returns the
    ordered card paths (1-based page numbering matches segment order)."""
    cards_dir = workdir / "cards"
    paths = []
    for i, seg in enumerate(segments):
        dest = cards_dir / f"page-{i + 1:03d}.png"
        if dest.exists():
            paths.append(dest)
            continue
        paths.append(
            render_card(
                seg, dest, size=size, cover=cover,
                title=title if i == 0 else None,
            )
        )
    return paths
