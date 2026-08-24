"""Render the Entertainment Harness logo into favicons, site assets, and the
macOS iconset.

Idempotent — safe to re-run. Raster output is drawn from scratch with PIL (no
SVG rasterizer required); assets/logo.svg is the hand-maintained vector twin of
the same mark.

Outputs:
  site/assets/{favicon.ico, favicon-32.png, apple-touch-icon.png,
               logo-512.png, logo.svg, og.png}
  ui/icon.iconset/icon_*.png        (Electrobun build.mac.icons default)

Usage: uv run python scripts/render_icons.py
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SITE_ASSETS = ROOT / "site" / "assets"
ICONSET = ROOT / "ui" / "icon.iconset"
SVG_SOURCE = ROOT / "assets" / "logo.svg"

MASTER = 1024
SUPER = 4
CORNER = 224
TOP = (109, 40, 217)  # violet-700
BOTTOM = (30, 27, 75)  # indigo-950
WHITE = (255, 255, 255)
PLANET = (251, 191, 36)  # amber-400

OG_SIZE = (1200, 630)
OG_TITLE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
]
OG_TAG_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def _gradient(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / (size - 1)
        draw.line(
            [(0, y), (size, y)],
            fill=tuple(_lerp(TOP[i], BOTTOM[i], t) for i in range(3)),
        )
    return img


def tile(size: int, *, rounded: bool = True) -> Image.Image:
    """The EH mark at `size` px: gradient squircle + orbital ring + play glyph."""
    s = MASTER * SUPER
    canvas = _gradient(s).convert("RGBA")
    if rounded:
        mask = Image.new("L", (s, s), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, s - 1, s - 1], radius=CORNER * SUPER, fill=255
        )
        tile_ = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        tile_.paste(canvas, (0, 0), mask)
    else:
        tile_ = canvas

    glyph = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    g = ImageDraw.Draw(glyph)
    c = s // 2

    ring_w, ring_h = s * 0.335, s * 0.245
    g.ellipse(
        [c - ring_w, c - ring_h, c + ring_w, c + ring_h],
        outline=(*WHITE, 255),
        width=round(s * 0.028),
    )

    ang = math.radians(38)
    px, py = c + ring_w * math.cos(ang), c + ring_h * math.sin(ang)
    pr = s * 0.036
    g.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(*PLANET, 255))

    half_w, h = s * 0.11, s * 0.24
    tx = c + s * 0.012  # optical nudge right
    g.polygon(
        [(tx - half_w, c - h / 2), (tx - half_w, c + h / 2), (tx + half_w, c)],
        fill=(*WHITE, 255),
    )

    tile_.alpha_composite(glyph)
    return tile_.resize((size, size), Image.LANCZOS)


def _load_font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def og_card() -> Image.Image:
    w, h = OG_SIZE
    # Vertical gradient at the card's own aspect.
    bg = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(bg)
    for y in range(h):
        t = y / (h - 1)
        draw.line([(0, y), (w, y)], fill=tuple(_lerp(TOP[i], BOTTOM[i], t) for i in range(3)))
    card = bg.convert("RGBA")

    mark = tile(440)
    card.alpha_composite(mark, (70, (h - 440) // 2))

    d = ImageDraw.Draw(card)
    x = 570
    max_w = w - x - 50
    size = 72
    while size > 28:
        title_font = _load_font(OG_TITLE_FONT_CANDIDATES, size)
        if d.textlength("Entertainment Harness", font=title_font) <= max_w:
            break
        size -= 4
    d.text((x, 225), "Entertainment Harness", font=title_font, fill=(*WHITE, 255))
    tag_font = _load_font(OG_TAG_FONT_CANDIDATES, 30)
    d.text(
        (x, 330),
        "Local-first media library, recaps, and video.",
        font=tag_font,
        fill=(221, 214, 254, 255),
    )
    return card


def main() -> None:
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    ICONSET.mkdir(parents=True, exist_ok=True)

    tile(48).save(SITE_ASSETS / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])

    tile(32).save(SITE_ASSETS / "favicon-32.png")
    tile(512).save(SITE_ASSETS / "logo-512.png")

    touch = Image.new("RGB", (180, 180), BOTTOM)
    touch.paste(tile(180, rounded=False), (0, 0))
    touch.save(SITE_ASSETS / "apple-touch-icon.png")

    og_card().save(SITE_ASSETS / "og.png")

    shutil.copyfile(SVG_SOURCE, SITE_ASSETS / "logo.svg")

    for name, px in {
        "icon_16x16": 16,
        "icon_16x16@2x": 32,
        "icon_32x32": 32,
        "icon_32x32@2x": 64,
        "icon_128x128": 128,
        "icon_128x128@2x": 256,
        "icon_256x256": 256,
        "icon_256x256@2x": 512,
        "icon_512x512": 512,
        "icon_512x512@2x": 1024,
    }.items():
        tile(px).save(ICONSET / f"{name}.png")

    print(f"wrote site assets -> {SITE_ASSETS}")
    print(f"wrote iconset -> {ICONSET}")


if __name__ == "__main__":
    main()
