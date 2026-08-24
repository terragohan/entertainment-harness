"""Runway ML generation client.

Runway Dev API: https://api.dev.runwayml.com, X-Runway-Version: 2024-11-06,
over plain httpx (no SDK). One client serves both callers:

- TikTok/short-form (`generate_segment`): gen3a_turbo image-to-video,
  hard-cropped 768:1280 vertical — legacy behavior, unchanged.
- Anime scenes (`text_to_image` / `image_to_video`): gen4_image keyframes with
  tagged reference images, landscape 1280:720 5-second shots.
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
from entertainment_harness.video.script import Segment, VideoError

RUNWAY_BASE = "https://api.dev.runwayml.com"
RUNWAY_VERSION = "2024-11-06"
POLL_INTERVAL = 5.0
MAX_POLL_SECONDS = 600.0

# Legacy vertical path (TikTok). Anime scenes pass their own ratio.
LEGACY_RATIO = (768, 1280)

# gen4_image rejects reference/prompt assets whose width/height ratio falls
# outside this window (a tall manga panel is ~0.27) with a 400.
MIN_REF_ASPECT = 0.5
MAX_REF_ASPECT = 2.0


def _pad_to_ref_window(img: Image.Image) -> Image.Image:
    """Pad an extreme-aspect image into Runway's reference window with white
    bars — never a crop, so no panel content is lost."""
    w, h = img.size
    if MIN_REF_ASPECT <= w / h <= MAX_REF_ASPECT:
        return img
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    size = (round(h * MIN_REF_ASPECT), h) if w / h < MIN_REF_ASPECT else (
        w, round(w / MAX_REF_ASPECT))
    color = (255, 255, 255, 255) if img.mode == "RGBA" else (
        255 if img.mode == "L" else (255, 255, 255))
    canvas = Image.new(img.mode, size, color)
    canvas.paste(img, ((size[0] - w) // 2, (size[1] - h) // 2))
    return canvas


class RunwayProvider:
    name = "runway"
    animated = True
    # gen3a_turbo image-to-video (generate_segment, image_to_video) and
    # gen4_image text-to-image (text_to_image) share this one client.
    capabilities = frozenset({"image-to-video", "image-gen"})

    def __init__(self, config: Config) -> None:
        self.config = config
        self.api_key = config.video_gen.runway.api_key or os.getenv("RUNWAY_API_KEY")
        self.model = config.video_gen.runway.model or "gen3a_turbo"
        if not self.api_key:
            raise VideoError(
                "Runway provider requires an API key. Set [video_gen.runway] api_key"
                " in config.toml or the RUNWAY_API_KEY environment variable."
            )

    # --- shared request plumbing -------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Runway-Version": RUNWAY_VERSION,
            "Content-Type": "application/json",
        }

    def _submit_task(self, endpoint: str, payload: dict) -> str:
        """POST a generation task; return its task id."""
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{RUNWAY_BASE}{endpoint}",
                    headers=self._headers(),
                    json=payload,
                    timeout=60.0,
                )
        except httpx.HTTPError as exc:
            raise VideoError(f"Runway request failed: {exc}") from exc
        if resp.status_code != 200:
            raise VideoError(
                f"Runway generation failed ({resp.status_code}): {resp.text}"
            )
        task_id = resp.json().get("id")
        if not task_id:
            raise VideoError("Runway response missing task id")
        return task_id

    def _poll(self, task_id: str) -> str:
        """Poll a task until it finishes; return its first output URL."""
        start = time.time()
        with httpx.Client() as client:
            while time.time() - start < MAX_POLL_SECONDS:
                try:
                    resp = client.get(
                        f"{RUNWAY_BASE}/v1/tasks/{task_id}",
                        headers=self._headers(),
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    raise VideoError(f"Runway task poll failed: {exc}") from exc
                data = resp.json()
                status = data.get("status", "").upper()
                if status in {"SUCCEEDED", "COMPLETED"}:
                    outputs = data.get("output", []) or []
                    if not outputs:
                        outputs = data.get("outputs", [])
                    if not outputs:
                        raise VideoError(
                            "Runway task completed but returned no output"
                        )
                    return outputs[0]
                if status in {"FAILED", "CANCELED", "CANCELLED"}:
                    raise VideoError(
                        f"Runway task {task_id} ended with status {status}"
                    )
                time.sleep(POLL_INTERVAL)
        raise VideoError(f"Runway task {task_id} timed out after {MAX_POLL_SECONDS}s")

    def _download(self, url: str, dest: Path) -> None:
        try:
            with httpx.Client() as client:
                resp = client.get(url, timeout=120.0)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise VideoError(f"Runway output download failed: {exc}") from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)

    # --- image encoding ------------------------------------------------------

    def _prepare_image(
        self, image: Path, ratio: tuple[int, int] = LEGACY_RATIO
    ) -> tuple[bytes, str]:
        """Center-crop to the target aspect ratio (so Runway's server-side
        crop can't cut content we care about), resize to fit it, and pad
        extreme aspects into Runway's reference window (white bars, never a
        crop — a tall panel would otherwise 400 the request)."""
        target_w, target_h = ratio
        with Image.open(image) as img:
            src_w, src_h = img.size
            target_ratio = target_w / target_h
            src_ratio = src_w / src_h
            if src_ratio > target_ratio:
                # Too wide: crop width
                new_w = int(src_h * target_ratio)
                left = (src_w - new_w) // 2
                img = img.crop((left, 0, left + new_w, src_h))
            elif src_ratio < target_ratio:
                # Too tall: crop height
                new_h = int(src_w / target_ratio)
                top = (src_h - new_h) // 2
                img = img.crop((0, top, src_w, top + new_h))
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img = _pad_to_ref_window(img)
            fmt = "JPEG" if img.mode in ("RGB", "RGBA") else "PNG"
            buf = io.BytesIO()
            if img.mode == "RGBA" and fmt == "JPEG":
                img = img.convert("RGB")
            img.save(buf, format=fmt)
            data = buf.getvalue()
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return data, mime

    def data_uri(
        self, image: Path, ratio: tuple[int, int] = LEGACY_RATIO
    ) -> str:
        data, mime = self._prepare_image(image, ratio)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # --- generation endpoints -------------------------------------------------

    def text_to_image(
        self,
        prompt: str,
        references: list[tuple[str, Path]],
        *,
        model: str,
        ratio: str,
        ref_ratio: tuple[int, int],
        dest: Path,
    ) -> Path:
        """gen4_image text-to-image with tagged reference images. The prompt
        references each image as @<tag>. Downloads the result to dest."""
        payload = {
            "model": model,
            "promptText": prompt[:1000],  # gen4_image caps promptText at 1000
            "ratio": ratio,
            "referenceImages": [
                {"uri": self.data_uri(path, ref_ratio), "tag": tag}
                for tag, path in references
            ],
        }
        task_id = self._submit_task("/v1/text_to_image", payload)
        self._download(self._poll(task_id), dest)
        return dest

    def image_to_video(
        self,
        image: Path,
        prompt: str,
        *,
        duration: float,
        ratio: str,
        model: str | None = None,
        ref_ratio: tuple[int, int] = LEGACY_RATIO,
        dest: Path,
    ) -> Path:
        """Image-to-video: the input image is the first frame."""
        payload = {
            "model": model or self.model,
            "promptImage": self.data_uri(image, ref_ratio),
            "promptText": prompt[:512] if prompt else "",
            "duration": self._nearest_duration(duration),
            "ratio": ratio,
            "watermark": False,
        }
        task_id = self._submit_task("/v1/image_to_video", payload)
        self._download(self._poll(task_id), dest)
        return dest

    def _nearest_duration(self, duration: float) -> int:
        return 10 if duration >= 7.5 else 5

    # --- legacy TikTok path (unchanged behavior) ------------------------------

    def generate_segment(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
    ) -> Path:
        dest = workdir / f"seg-runway-{segment.index:02d}.mp4"
        return self.image_to_video(
            image, segment.text, duration=duration, ratio="768:1280", dest=dest
        )
