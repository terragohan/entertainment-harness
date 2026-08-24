"""Frame animators for [video] mode = "animate".

A frame animator turns one grounded panel crop into the sequence of frames
the assembly stitches (xfade) into that beat's clip. Providers register in
the shared plugin registry under the "frames" category:

- `local`  — animated=False: no image generator configured; the assembly
  falls back to the panels-mode Ken Burns crop clip.
- `runway` — gen4_image text-to-image with the crop as a tagged @panel
  reference and a motion-phase prompt, so the generated frames keep the
  chapter's own art (composition, characters, style) and only the motion
  advances.

Generated frames are content-addressed (sha256 of the crop bytes + prompt +
model + ratio) under workdir/"frames": they survive clip invalidation and
can never be stale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from PIL import Image

from entertainment_harness.config import Config
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginRegistry
from entertainment_harness.video.script import Segment, VideoError

# Ratios gen4_image accepts (Runway docs, 2024-11-06 API version).
SUPPORTED_RATIOS = (
    "1024:1024", "1080:1920", "1920:1080", "1360:768", "1080:1080",
    "1168:880", "1440:1080", "1080:1440", "1808:768", "2112:912",
    "1280:720", "720:1280", "720:720", "960:720", "720:960", "1680:720",
)

FRAME_PROMPT = (
    "Redraw the manga panel @panel as one frame of a subtle animation: keep"
    " the exact same composition, characters, and art style, and advance"
    " natural motion a small step — hair, cloth, smoke and effects drifting,"
    " expressions and impact lines shifting slightly. This is frame {phase}"
    " of {count}. Scene context: {text}"
)


def _aspect(path: Path) -> float:
    with Image.open(path) as img:
        return img.width / img.height


def nearest_ratio(image: Path) -> str:
    """The supported gen4_image ratio closest to the image's aspect."""
    import math

    aspect = _aspect(image)
    return min(
        SUPPORTED_RATIOS,
        key=lambda r: abs(
            math.log(aspect) - math.log(int(r.split(":")[0]) / int(r.split(":")[1]))
        ),
    )


def _own_size(image: Path) -> tuple[int, int]:
    """Reference-prep size that preserves the whole image: the client's
    center-crop-to-ratio is a no-op when the ratio is the image's own."""
    with Image.open(image) as img:
        return img.width, img.height


def _frame_key(image: Path, prompt: str, model: str, ratio: str) -> str:
    h = hashlib.sha256(image.read_bytes())
    h.update(prompt.encode())
    h.update(model.encode())
    h.update(ratio.encode())
    return h.hexdigest()[:16]


class FrameAnimator(Protocol):
    name: str
    animated: bool  # True = generates frames; False = panels-mode fallback
    capabilities: frozenset[str]  # e.g. {"image-gen"}

    def generate_frames(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
        log=lambda m: None,
    ) -> list[Path]:
        """Return ordered frame images animating `image` across `duration`
        seconds of screen time (the animator picks the frame count — it owns
        the cost model). Only called when `animated` is True."""
        ...


class LocalFrameAnimator:
    name = "local"
    animated = False

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate_frames(self, image, segment, duration, workdir, log=lambda m: None):
        raise VideoError(
            "The local frame animator generates nothing — set [frames]"
            " provider = \"runway\" to animate panels with generated frames."
        )


class RunwayFrameAnimator:
    name = "runway"
    animated = True
    capabilities = frozenset({"image-gen"})

    def __init__(self, config: Config) -> None:
        from entertainment_harness.video.gen.runway import RunwayProvider

        self.config = config
        try:
            self.client = RunwayProvider(config)  # raises without an API key
        except VideoError as exc:
            raise VideoError(
                f"{exc} Alternatively, set [frames] provider = \"openrouter\""
                " to animate panels via the OpenAI-compatible endpoint"
                " ([models.openai_compat])."
            ) from exc
        self.model = config.frames.model or "gen4_image"

    def generate_frames(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
        log=lambda m: None,
    ) -> list[Path]:
        return generate_frames_with(
            self.client, self.model, self.config.frames,
            image, segment, duration, workdir, log,
        )


class OpenRouterFrameAnimator:
    name = "openrouter"
    animated = True
    capabilities = frozenset({"image-gen"})

    def __init__(self, config: Config) -> None:
        from entertainment_harness.video.gen.openrouter import (
            DEFAULT_IMAGE_MODEL,
            OpenRouterImageProvider,
        )

        self.config = config
        self.client = OpenRouterImageProvider(config)  # raises without a key
        self.model = config.frames.model or DEFAULT_IMAGE_MODEL

    def generate_frames(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
        log=lambda m: None,
    ) -> list[Path]:
        return generate_frames_with(
            self.client, self.model, self.config.frames,
            image, segment, duration, workdir, log,
        )


def generate_frames_with(
    client,
    model: str,
    cfg,
    image: Path,
    segment: Segment,
    duration: float,
    workdir: Path,
    log=lambda m: None,
) -> list[Path]:
    """The shared animate-mode loop: `frame_count` motion-phase variations
    of the panel crop, content-addressed so re-renders reuse them. `client`
    is any image provider with Runway's text_to_image interface."""
    count = frame_count(duration, cfg.seconds_per_frame, cfg.max_frames)
    ratio = nearest_ratio(image)
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    generated = 0
    for i in range(count):
        prompt = FRAME_PROMPT.format(
            phase=i + 1, count=count, text=segment.text[:300]
        )
        dest = frames_dir / f"f-{_frame_key(image, prompt, model, ratio)}-{i}.png"
        if not dest.exists():
            client.text_to_image(
                prompt,
                [("panel", image)],
                model=model,
                ratio=ratio,
                ref_ratio=_own_size(image),  # never crop the panel
                dest=dest,
            )
            generated += 1
        paths.append(dest)
    log(f"    segment {segment.index:02d}: {count} frame(s)"
        f" ({generated} generated, {count - generated} cached)")
    return paths


REGISTRY = PluginRegistry("frames", ENTRY_POINT_GROUPS["frames"])
REGISTRY.register("local", "entertainment_harness.video.frames:LocalFrameAnimator")
REGISTRY.register("runway", "entertainment_harness.video.frames:RunwayFrameAnimator")
REGISTRY.register(
    "openrouter", "entertainment_harness.video.frames:OpenRouterFrameAnimator"
)
# Frame-by-frame expansion + chained frames for [video] mode = "sequence".
REGISTRY.register(
    "sequence", "entertainment_harness.video.sequence:RunwaySequenceAnimator"
)
REGISTRY.register(
    "openrouter-sequence",
    "entertainment_harness.video.sequence:OpenRouterSequenceAnimator",
)


def get_animator(name: str, config: Config) -> FrameAnimator:
    return REGISTRY.create(name, config)


def resolve_image_model(provider: str, config_model: str) -> str:
    """The image model a built-in frames/sequence animator will actually use
    — render-state keying must see a changed default (a retired upstream id
    swapped out) just like an explicit `[frames]`/`[sequence]` model edit.
    Third-party providers key on the raw config value."""
    if config_model:
        return config_model
    if provider in ("runway", "sequence"):
        return "gen4_image"
    if provider in ("openrouter", "openrouter-sequence"):
        from entertainment_harness.video.gen.openrouter import (
            DEFAULT_IMAGE_MODEL,
        )
        return DEFAULT_IMAGE_MODEL
    return ""


def frame_count(share_s: float, seconds_per_frame: float = 2.0,
                max_frames: int = 6) -> int:
    """Frames per beat: one per seconds_per_frame of slot share, at least two
    (a single frame cannot animate), capped for cost."""
    return max(2, min(max_frames, round(share_s / seconds_per_frame)))
