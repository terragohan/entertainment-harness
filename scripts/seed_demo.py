"""Seed a throwaway EH_DATA_DIR with fictional demo works for screenshots.

Generates synthetic comic pages (abstract shapes + fictional titles — no real
or copyrighted content), packs them into .cbz files, and imports them via the
real CLI (`eh import`) so the library DB is populated exactly as a user's would
be.

Usage:
    uv run python scripts/seed_demo.py [--data-dir /tmp/eh-demo-data]

Requires ffmpeg-free operation — import only needs the standard deps.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

TITLE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
BODY_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

# Fictional works — hue rotates per work so the library grid is colorful.
# `eh import` dedupes by title slug, so each work is a single import (one CBZ,
# several pages).
WORKS = [
    {"title": "Starfall Chronicles", "author": "M. Aoyama", "hue": 265},
    {"title": "The Clockwork Garden", "author": "P. Ferro", "hue": 150},
    {"title": "Aster City Blues", "author": "J. Callisto", "hue": 25},
    {"title": "Paper Moons", "author": "R. Okonkwo", "hue": 330},
    {"title": "Neon Koi", "author": "T. Vasquez", "hue": 200},
]

PAGE_W, PAGE_H = 1000, 1500
PAGES_PER_WORK = 10


def _font(candidates: list[str], size: int) -> ImageFont.ImageFont:
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _hsv(h: int, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return round(r * 255), round(g * 255), round(b * 255)


def draw_page(work: dict, chapter: int, page: int) -> Image.Image:
    hue = work["hue"]
    img = Image.new("RGB", (PAGE_W, PAGE_H), _hsv(hue, 0.28, 0.97))
    d = ImageDraw.Draw(img)

    # Header band with the fictional title.
    d.rectangle([0, 0, PAGE_W, 190], fill=_hsv(hue, 0.45, 0.32))
    title_font = _font(TITLE_FONT_CANDIDATES, 64)
    body_font = _font(BODY_FONT_CANDIDATES, 40)
    d.text((50, 45), work["title"], font=title_font, fill=(245, 243, 250))
    d.text(
        (50, 125),
        f"by {work['author']}  ·  Chapter {chapter}",
        font=body_font,
        fill=(214, 208, 228),
    )

    # Abstract "panels": rounded rects with varied tints and geometric fills.
    top = 230
    rng = [(page * 7 + chapter * 13 + i * 29) % 100 for i in range(6)]
    for col in range(2):
        for row in range(2):
            x0 = 60 + col * ((PAGE_W - 140) // 2 + 20)
            y0 = top + row * ((PAGE_H - top - 90) // 2 + 20)
            x1 = x0 + (PAGE_W - 140) // 2
            y1 = y0 + (PAGE_H - top - 90) // 2
            tint = _hsv((hue + 40 + rng[row * 2 + col]) % 360, 0.25, 0.92)
            d.rounded_rectangle([x0, y0, x1, y1], radius=28, fill=tint)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            span = min(x1 - x0, y1 - y0)
            shape = (row * 2 + col + page) % 3
            if shape == 0:
                r = span // 4
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_hsv(hue, 0.5, 0.45))
            elif shape == 1:
                w, h = span // 3, span // 5
                d.rectangle([cx - w, cy - h, cx + w, cy + h], fill=_hsv(hue, 0.55, 0.5))
            else:
                d.polygon(
                    [
                        (cx, cy - span // 4),
                        (cx + span // 4, cy + span // 4),
                        (cx - span // 4, cy + span // 4),
                    ],
                    fill=_hsv(hue, 0.5, 0.42),
                )

    # Footer with the page number.
    d.text(
        (PAGE_W - 160, PAGE_H - 80),
        f"p.{page}",
        font=body_font,
        fill=(120, 116, 132),
    )
    return img


def build_cbz(work: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for page in range(1, PAGES_PER_WORK + 1):
            img = draw_page(work, 1, page)
            zf.writestr(f"page-{page:02d}.jpg", _jpeg_bytes(img), compress_type=zipfile.ZIP_DEFLATED)
    return out


def _jpeg_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/eh-demo-data"))
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="eh-demo-") as tmp:
        tmpdir = Path(tmp)
        for work in WORKS:
            cbz = build_cbz(
                work,
                tmpdir / f"{work['title'].replace(' ', '_')}.cbz",
            )
            env = dict(os.environ, EH_DATA_DIR=str(data_dir))
            subprocess.run(
                    [
                        "uv",
                        "run",
                        "eh",
                        "import",
                        str(cbz),
                        "--title",
                        work["title"],
                        "--kind",
                        "comic",
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                )

    print(f"seeded demo library -> {data_dir}")
    print("launch with:  EH_DATA_DIR=%s bun run dev   (from ui/)" % data_dir)


if __name__ == "__main__":
    main()
