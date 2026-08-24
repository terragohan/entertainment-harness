"""Video generation providers for TikTok-style clips.

Providers register in the shared plugin registry (plugins.py) and turn one
input image and a text hint into a short video clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from entertainment_harness.config import Config
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginRegistry
from entertainment_harness.video.script import Segment


class VideoGenProvider(Protocol):
    name: str
    animated: bool  # True = generates moving clips; False = stills (Ken Burns)
    capabilities: frozenset[str]  # e.g. {"stills"}, {"image-to-video"}

    def generate_segment(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
    ) -> Path:
        """Return an MP4 clip for one narration segment.

        The returned clip's actual duration may differ from the requested
        duration; callers must read it and update Segment.duration_s.
        """
        ...


REGISTRY = PluginRegistry("video_gen", ENTRY_POINT_GROUPS["video_gen"])
REGISTRY.register("local", "entertainment_harness.video.gen.local:LocalVideoGenProvider")
REGISTRY.register("runway", "entertainment_harness.video.gen.runway:RunwayProvider")


def get_provider(name: str, config: Config) -> VideoGenProvider:
    return REGISTRY.create(name, config)
