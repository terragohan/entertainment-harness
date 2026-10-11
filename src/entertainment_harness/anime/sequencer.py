"""Deterministic timeline sequencer (animation-conductor initiative).

Given a sparse Timeline, answer: "what should the scene look like at frame
N?" — the complete resolved state, ready for the compositor. No pixels are
drawn here, no models are involved, and the same (timeline, frame) pair
always resolves to the same state.

Resolution rules (Phase 1):

- **Camera** — linear interpolation between the surrounding sparse
  keyframes; before the first / after the last keyframe the nearest value
  holds. No keyframes at all means a static full-frame camera.
- **Layers** — a layer's asset is persistent: an asset event at frame F
  applies from F until the next asset event on that layer. An event at
  frame 36 means state_at(35) and state_at(36) differ — boundaries are
  exact.
- **Mouth** — while a mouth interval covers frame N it overrides the
  layer's persistent asset with the deterministic cycle
  `states[((N - start) // period) % len(states)]`; outside the interval the
  persistent asset applies again.

This is NOT video/sequence.py: that module asks an image model to invent
chained frames; this one resolves an explicit plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from entertainment_harness.anime.timeline import (
    AssetEvent,
    MouthEvent,
    Timeline,
    TimelineError,
)


@dataclass
class CameraState:
    x: float = 0.5
    y: float = 0.5
    scale: float = 1.0


@dataclass
class LayerState:
    id: str
    asset: str
    x: float
    y: float
    scale: float


@dataclass
class FrameState:
    """Everything the compositor needs for one frame, nothing more."""

    frame: int
    camera: CameraState
    layers: list[LayerState]


def camera_at(timeline: Timeline, frame: int) -> CameraState:
    """The camera at `frame`: piecewise-linear between keyframes."""
    keys = timeline.camera
    if not keys:
        return CameraState()
    if frame <= keys[0].frame:
        key = keys[0]
        return CameraState(key.x, key.y, key.scale)
    if frame >= keys[-1].frame:
        key = keys[-1]
        return CameraState(key.x, key.y, key.scale)
    for before, after in zip(keys, keys[1:]):
        if before.frame <= frame <= after.frame:
            span = after.frame - before.frame
            t = (frame - before.frame) / span
            return CameraState(
                before.x + (after.x - before.x) * t,
                before.y + (after.y - before.y) * t,
                before.scale + (after.scale - before.scale) * t,
            )
    raise AssertionError("unreachable: keyframes bracket every frame")


def _persistent_asset(timeline: Timeline, layer_id: str, default: str,
                      frame: int) -> str:
    """The asset the layer holds at `frame`: the latest asset event at or
    before it (validation guarantees at most one event per (layer, frame)),
    else the layer's initial asset."""
    asset = default
    best_frame = -1
    for event in timeline.events:
        if (
            isinstance(event, AssetEvent)
            and event.target == layer_id
            and best_frame < event.frame <= frame
        ):
            asset, best_frame = event.asset, event.frame
    return asset


def _mouth_asset(timeline: Timeline, layer_id: str, frame: int) -> str | None:
    """The mouth-cycle asset covering `frame`, or None when no mouth
    interval does (validation guarantees intervals on one layer never
    overlap, so at most one can apply)."""
    for event in timeline.events:
        if (
            isinstance(event, MouthEvent)
            and event.target == layer_id
            and event.start_frame <= frame <= event.end_frame
        ):
            index = ((frame - event.start_frame) // event.period_frames) % len(
                event.states
            )
            return event.states[index]
    return None


def state_at(timeline: Timeline, frame: int) -> FrameState:
    """Resolve the complete scene state at an exact frame."""
    if not 0 <= frame < timeline.duration_frames:
        raise TimelineError(
            f"Frame {frame} is outside the timeline's 0.."
            f"{timeline.duration_frames - 1} (duration_frames="
            f"{timeline.duration_frames})"
        )
    layers = [
        LayerState(
            id=layer.id,
            asset=_mouth_asset(timeline, layer.id, frame)
            or _persistent_asset(timeline, layer.id, layer.asset, frame),
            x=layer.x,
            y=layer.y,
            scale=layer.scale,
        )
        for layer in timeline.layers
    ]
    return FrameState(frame=frame, camera=camera_at(timeline, frame), layers=layers)
