"""Local ffmpeg/Ken-Burns video generation provider."""

from __future__ import annotations

from pathlib import Path

from entertainment_harness.config import Config
from entertainment_harness.video.assemble import build_clip
from entertainment_harness.video.script import Segment


class LocalVideoGenProvider:
    name = "local"
    animated = False  # stills with Ken Burns motion, not generated video
    capabilities = frozenset({"stills"})

    def __init__(self, config: Config) -> None:
        self.config = config

    def generate_segment(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
    ) -> Path:
        dest = workdir / f"seg-local-{segment.index:02d}.mp4"
        motion = segment.motion or "zoom_in"
        build_clip(image, duration, motion, (1080, 1920), dest)
        return dest
