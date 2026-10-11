"""Timeline compositor (animation-conductor initiative).

The compositor's whole contract: one resolved FrameState plus the
timeline's visual assets -> one image. It understands no story intent and
makes no timing decisions — it draws exactly what the sequencer resolved.

Phase 1 rendering model, deliberately simple:

- Layers composite in timeline order onto an RGBA canvas over the
  timeline's background color. Each layer's asset is scaled so its height
  is `layer.scale` x the canvas height (aspect preserved) and centered at
  the layer's normalized (x, y). Alpha is honored, so PNG sprites composite
  over backgrounds.
- The camera is a crop window over the composed canvas: `scale` zooms
  (1.0 = whole canvas), normalized (x, y) centers the window, clamped to
  the canvas edges so the output size is always exact. With scale 1.0 the
  window IS the canvas — a pan only shows once zoomed in.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from entertainment_harness.anime.sequencer import FrameState
from entertainment_harness.anime.timeline import Timeline, TimelineError


def _hex_rgba(color: str) -> tuple[int, int, int, int]:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        255,
    )


class Compositor:
    """Draws FrameStates for one timeline. Assets load once (they are the
    expensive side); rendering a frame is pure deterministic Pillow work."""

    def __init__(self, timeline: Timeline, base_dir: Path) -> None:
        self.timeline = timeline
        self.assets: dict[str, Image.Image] = {}
        for name, rel in timeline.assets.items():
            path = Path(rel)
            if not path.is_absolute():
                path = base_dir / path
            try:
                with Image.open(path) as img:
                    self.assets[name] = img.convert("RGBA")
            except (FileNotFoundError, OSError) as exc:
                raise TimelineError(
                    f"Asset {name!r} cannot be loaded from {path}:"
                    f" {exc.strerror or exc}"
                ) from exc

    def render(self, state: FrameState) -> Image.Image:
        """Frame state -> one RGB image of exactly (width, height)."""
        width, height = self.timeline.width, self.timeline.height
        canvas = Image.new("RGBA", (width, height), _hex_rgba(self.timeline.background))
        for layer in state.layers:
            sprite = self.assets[layer.asset]
            draw_h = max(1, round(layer.scale * height))
            draw_w = max(1, round(sprite.width * draw_h / sprite.height))
            if (draw_w, draw_h) != sprite.size:
                sprite = sprite.resize((draw_w, draw_h), Image.LANCZOS)
            left = round(layer.x * width - draw_w / 2)
            top = round(layer.y * height - draw_h / 2)
            canvas.alpha_composite(sprite, (left, top))

        camera = state.camera
        crop_w = width / camera.scale
        crop_h = height / camera.scale
        left = min(max(camera.x * width - crop_w / 2, 0.0), width - crop_w)
        top = min(max(camera.y * height - crop_h / 2, 0.0), height - crop_h)
        if left == 0.0 and top == 0.0 and camera.scale == 1.0:
            return canvas.convert("RGB")
        # AFFINE maps output -> input: sample (left + ox/scale, top + oy/scale).
        framed = canvas.transform(
            (width, height),
            Image.AFFINE,
            (1 / camera.scale, 0.0, left, 0.0, 1 / camera.scale, top),
            resample=Image.BILINEAR,
        )
        return framed.convert("RGB")
