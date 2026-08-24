"""Shared helpers for content source clients."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


# Canonical extension for each PIL format we care about.
_FORMAT_TO_EXT: dict[str, str] = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
}


def canonical_image_ext(data: bytes, fallback: str | None = None) -> str | None:
    """Return a canonical file extension for the image format in *data*.

    Uses Pillow to inspect the actual bytes, so it works even when the
    source URL lies about the format (e.g. a JPEG served from a `.png`
    URL). Returns *fallback* if the bytes cannot be parsed as an image.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
    except Exception:
        return fallback
    return _FORMAT_TO_EXT.get(fmt, fallback)


def page_dest_from_url(url: str, index: int) -> str:
    """Default filename for a downloaded page based on the URL extension."""
    ext = Path(url.split("?", 1)[0]).suffix or ".jpg"
    return f"page-{index:03d}{ext}"
