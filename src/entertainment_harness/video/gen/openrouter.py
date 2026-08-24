"""OpenAI-compatible image generation client (OpenRouter).

Image generation over the chat-completions API: the request asks for
``modalities: ["image", "text"]``, reference images go in as data-URL
content parts (the prompt's @tag mentions stay as ordinary words — unlike
Runway's gen4_image there is no server-side tagging), and the first image
in the response is normalized — center-cropped and resized to the
requested ratio — so chained frames share exact dimensions for ffmpeg.

Credentials come from ``[models.openai_compat]`` with the same env
fallbacks as the text/vision backend (OPENAI_COMPAT_API_KEY /
OPENROUTER_API_KEY). The interface mirrors RunwayProvider.text_to_image
so the frames/sequence animators duck-type either client.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import httpx
from PIL import Image

from entertainment_harness.config import Config
from entertainment_harness.video.script import VideoConfigError, VideoError

#: Fallback image model when [frames]/[sequence] model is empty. Any
#: OpenRouter image-generation model id works (one whose architecture has
#: "image" in its output modalities, e.g. "openai/gpt-5-image").
DEFAULT_IMAGE_MODEL = "google/gemini-2.5-flash-image"

TIMEOUT_S = 300.0  # image generation is a single slow round-trip

# The endpoint intermittently answers 200 with no image payload (same
# request succeeds a minute later); retry a few times before failing the
# anchor. Only the no-image case retries — HTTP errors raise immediately.
NO_IMAGE_ATTEMPTS = 3


def _data_uri(image: Path) -> str:
    """The reference image as a data URL (JPEG for RGB — manga pages are
    big and base64 inflates by a third; PNG otherwise)."""
    with Image.open(image) as img:
        fmt = "JPEG" if img.mode in ("RGB", "RGBA") else "PNG"
        if img.mode == "RGBA" and fmt == "JPEG":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _ratio_pixels(ratio: str) -> tuple[int, int]:
    """"1920:1080" -> (1920, 1080); garbage -> 16:9."""
    width, _, height = ratio.partition(":")
    try:
        return int(width), int(height)
    except ValueError:
        return 1920, 1080


def _extract_image_url(data: dict) -> str | None:
    """The first image URL in a chat-completions response, or None."""
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    for image in message.get("images") or []:
        if not isinstance(image, dict):
            continue
        url = (image.get("image_url") or {}).get("url") or image.get("url")
        if url:
            return str(url)
    return None


class OpenRouterImageProvider:
    """Image generation via the OpenAI-compatible endpoint ([models.openai_compat])."""

    name = "openrouter"
    animated = True
    capabilities = frozenset({"image-gen"})

    def __init__(self, config: Config) -> None:
        remote = config.models.openai_compat
        self.base_url = (
            remote.base_url or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.api_key = (
            remote.api_key
            or os.environ.get("OPENAI_COMPAT_API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not self.api_key:
            raise VideoError(
                "OpenRouter image generation needs an API key: set api_key"
                " under [models.openai_compat] in config.toml, or the"
                " OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY env var."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, payload: dict) -> dict:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=TIMEOUT_S,
                )
        except httpx.HTTPError as exc:
            raise VideoError(f"Image generation request failed: {exc}") from exc
        if resp.status_code == 404 and "output modalities" in resp.text:
            model = payload.get("model", "?")
            raise VideoConfigError(
                f"{model!r} cannot generate images at {self.base_url} — the"
                " endpoint has no image-output route for it (a model can"
                " accept image input without being able to draw; check the"
                " model's output modalities). Set [frames]/[sequence] model"
                f" to an image model, e.g. {DEFAULT_IMAGE_MODEL!r} or"
                " \"openai/gpt-5-image\" — or blank for the default."
            )
        if resp.status_code != 200:
            raise VideoError(
                f"Image generation failed ({resp.status_code})"
                f" from {self.base_url}: {resp.text[:500]}"
            )
        return resp.json()

    def _fetch_bytes(self, url: str) -> bytes:
        if url.startswith("data:"):
            _, _, b64 = url.partition("base64,")
            try:
                return base64.b64decode(b64)
            except Exception as exc:
                raise VideoError(
                    f"Image response carried an undecodable data URL: {exc}"
                ) from exc
        try:
            with httpx.Client() as client:
                resp = client.get(url, timeout=TIMEOUT_S)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise VideoError(f"Image download failed: {exc}") from exc
        return resp.content

    def _write_normalized(self, raw: bytes, ratio: str, dest: Path) -> None:
        """Center-crop to the ratio's aspect and resize to its pixels, so
        every frame in a chain has identical dimensions for ffmpeg."""
        target_w, target_h = _ratio_pixels(ratio)
        with Image.open(io.BytesIO(raw)) as img:
            src_w, src_h = img.size
            target = target_w / target_h
            src = src_w / src_h
            if src > target:  # too wide: crop the sides
                new_w = int(src_h * target)
                left = (src_w - new_w) // 2
                img = img.crop((left, 0, left + new_w, src_h))
            elif src < target:  # too tall: crop top/bottom
                new_h = int(src_w / target)
                top = (src_h - new_h) // 2
                img = img.crop((0, top, src_w, top + new_h))
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, format="PNG")

    def text_to_image(
        self,
        prompt: str,
        references: list[tuple[str, Path]],
        *,
        model: str,
        ratio: str,
        ref_ratio: tuple[int, int] = (0, 0),  # parity with Runway; inputs go as-is
        dest: Path,
    ) -> Path:
        """One image generation round-trip. The prompt leads (its @tag
        mentions read as ordinary words), reference images follow in order;
        the result is normalized to `ratio` and written to dest."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for _tag, path in references:
            content.append(
                {"type": "image_url", "image_url": {"url": _data_uri(path)}}
            )
        payload = {
            "model": model,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": content}],
        }
        url = None
        for attempt in range(1, NO_IMAGE_ATTEMPTS + 1):
            data = self._request(payload)
            url = _extract_image_url(data)
            if url is not None:
                break
            if attempt < NO_IMAGE_ATTEMPTS:
                time.sleep(2.0 * attempt)
        if url is None:
            raise VideoError(
                f"The endpoint returned no image after {NO_IMAGE_ATTEMPTS}"
                f" attempts — is {model!r} an image-generation model?"
                " (image models answer with a message.images payload)"
            )
        self._write_normalized(self._fetch_bytes(url), ratio, dest)
        return dest
