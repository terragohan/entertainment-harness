"""Sparse animation timeline domain model (animation-conductor initiative).

A timeline.json is the EXECUTABLE counterpart to the semantic scene.json
(anime/scene.py): it says exactly when animation state changes, on an
integer-frame clock. It is deliberately sparse — camera keyframes and
discrete events, never per-frame repetition — human-editable, versioned,
and validated before any rendering happens.

Canonical clock: integer frames. A 5-second scene at fps 24 has
duration_frames = 120 and valid frame indices 0..119; seconds only ever
appear as frame / fps at the ffmpeg boundary.

The sequencer (anime/sequencer.py) resolves this sparse form into complete
per-frame state; nothing in this module draws pixels or runs ffmpeg.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


class TimelineError(Exception):
    pass


SCHEMA_VERSION = 1
DEFAULT_FPS = 24

ASSET_EVENT = "asset"  # set a layer's asset at an exact frame (persistent)
MOUTH_EVENT = "mouth"  # cycle mouth-state assets over an inclusive frame range


def _int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineError(f"{what} must be an integer, got {value!r}")
    if int(value) != value:
        raise TimelineError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _float(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimelineError(f"{what} must be a number, got {value!r}")
    return float(value)


@dataclass
class CameraKey:
    """One camera keyframe: normalized center (x, y in 0..1) and zoom scale
    (1.0 = the whole canvas visible) at an exact frame."""

    frame: int
    x: float = 0.5
    y: float = 0.5
    scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> CameraKey:
        return CameraKey(
            frame=_int(data.get("frame"), "camera keyframe frame"),
            x=_float(data.get("x", 0.5), "camera keyframe x"),
            y=_float(data.get("y", 0.5), "camera keyframe y"),
            scale=_float(data.get("scale", 1.0), "camera keyframe scale"),
        )


@dataclass
class Layer:
    """A drawable subject: which asset it starts with, where its center sits
    (normalized 0..1), and its size as a fraction of the canvas height."""

    id: str
    asset: str
    x: float = 0.5
    y: float = 0.5
    scale: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Layer:
        return Layer(
            id=str(data.get("id", "")).strip(),
            asset=str(data.get("asset", "")).strip(),
            x=_float(data.get("x", 0.5), f"layer {data.get('id')!r} x"),
            y=_float(data.get("y", 0.5), f"layer {data.get('id')!r} y"),
            scale=_float(data.get("scale", 1.0), f"layer {data.get('id')!r} scale"),
        )


@dataclass
class AssetEvent:
    """At `frame`, `target` layer switches to `asset` — persistent until the
    next event for that layer. Cuts, pose/expression swaps, and blinks are
    all this one mechanism."""

    frame: int
    target: str
    asset: str
    type: str = field(default=ASSET_EVENT, init=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> AssetEvent:
        return AssetEvent(
            frame=_int(data.get("frame"), "asset event frame"),
            target=str(data.get("target", "")).strip(),
            asset=str(data.get("asset", "")).strip(),
        )


@dataclass
class MouthEvent:
    """Crude dialogue animation: over the INCLUSIVE range
    [start_frame, end_frame] the target layer's asset cycles through
    `states`, each held for `period_frames` frames. Outside the range the
    layer's persistent asset applies again."""

    start_frame: int
    end_frame: int
    target: str
    states: list[str]
    period_frames: int = 1
    type: str = field(default=MOUTH_EVENT, init=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> MouthEvent:
        return MouthEvent(
            start_frame=_int(data.get("start_frame"), "mouth event start_frame"),
            end_frame=_int(data.get("end_frame"), "mouth event end_frame"),
            target=str(data.get("target", "")).strip(),
            states=[str(s).strip() for s in data.get("states", [])],
            period_frames=_int(
                data.get("period_frames", 1), "mouth event period_frames"
            ),
        )


Event = AssetEvent | MouthEvent


def _event_from_dict(data: dict) -> Event:
    kind = str(data.get("type", "")).strip()
    if kind == ASSET_EVENT:
        return AssetEvent.from_dict(data)
    if kind == MOUTH_EVENT:
        return MouthEvent.from_dict(data)
    raise TimelineError(
        f"Unknown event type {kind!r}; expected one of: {ASSET_EVENT}, {MOUTH_EVENT}"
    )


@dataclass
class AudioClip:
    """One audio file starting at an exact frame: it begins playing at
    start_frame / fps seconds into the video. The clip never stretches the
    scene — the timeline's duration_frames stays authoritative."""

    start_frame: int
    path: str
    volume: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> AudioClip:
        return AudioClip(
            start_frame=_int(data.get("start_frame"), "audio start_frame"),
            path=str(data.get("path", "")).strip(),
            volume=_float(data.get("volume", 1.0), "audio volume"),
        )


_BACKGROUND_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass
class Timeline:
    fps: int
    width: int
    height: int
    duration_frames: int
    assets: dict[str, str] = field(default_factory=dict)  # id -> image path
    layers: list[Layer] = field(default_factory=list)
    camera: list[CameraKey] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    audio: list[AudioClip] = field(default_factory=list)
    background: str = "#000000"  # visible where no opaque layer covers
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_frames": self.duration_frames,
            "background": self.background,
            "assets": dict(self.assets),
            "layers": [layer.to_dict() for layer in self.layers],
            "camera": [key.to_dict() for key in self.camera],
            "events": [event.to_dict() for event in self.events],
            "audio": [clip.to_dict() for clip in self.audio],
        }

    @staticmethod
    def from_dict(data: dict) -> Timeline:
        if not isinstance(data, dict):
            raise TimelineError("Timeline is not a JSON object")
        return Timeline(
            version=_int(data.get("version", SCHEMA_VERSION), "version"),
            fps=_int(data.get("fps", DEFAULT_FPS), "fps"),
            width=_int(data.get("width"), "width"),
            height=_int(data.get("height"), "height"),
            duration_frames=_int(data.get("duration_frames"), "duration_frames"),
            background=str(data.get("background", "#000000")).strip(),
            assets={
                str(k).strip(): str(v).strip()
                for k, v in dict(data.get("assets", {})).items()
            },
            layers=[Layer.from_dict(layer) for layer in data.get("layers", [])],
            camera=[CameraKey.from_dict(key) for key in data.get("camera", [])],
            events=[_event_from_dict(e) for e in data.get("events", [])],
            audio=[AudioClip.from_dict(clip) for clip in data.get("audio", [])],
        )

    @property
    def duration_seconds(self) -> float:
        """Nominal visual duration; derived, never stored, so frames stay
        the single authoritative clock."""
        return self.duration_frames / self.fps

    def frame_seconds(self, frame: int) -> float:
        """When frame `frame` begins, in seconds (audio placement uses this
        exact conversion)."""
        return frame / self.fps


def validate_timeline(timeline: Timeline) -> None:
    """Enforce the timeline contract. Human-edited timeline.json files go
    through the same validation, so a bad hand edit fails before any
    rendering."""
    if timeline.version != SCHEMA_VERSION:
        raise TimelineError(
            f"Unsupported timeline version {timeline.version};"
            f" this build reads version {SCHEMA_VERSION}"
        )
    for field_name in ("fps", "width", "height", "duration_frames"):
        value = getattr(timeline, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TimelineError(
                f"{field_name} must be an integer, got {value!r}"
            )
    if timeline.fps <= 0:
        raise TimelineError(f"fps must be > 0, got {timeline.fps}")
    if timeline.width <= 0 or timeline.height <= 0:
        raise TimelineError(
            f"width/height must be > 0, got {timeline.width}x{timeline.height}"
        )
    if timeline.duration_frames <= 0:
        raise TimelineError(
            f"duration_frames must be > 0, got {timeline.duration_frames}"
        )
    if not _BACKGROUND_RE.match(timeline.background):
        raise TimelineError(
            f"background must be #rgb or #rrggbb, got {timeline.background!r}"
        )
    last = timeline.duration_frames - 1  # valid frame indices: 0..last

    def check_frame(frame: int, what: str) -> None:
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise TimelineError(f"{what} must be an integer, got {frame!r}")
        if not 0 <= frame <= last:
            raise TimelineError(
                f"{what} is frame {frame}, outside the valid range 0..{last}"
            )

    for name, path in timeline.assets.items():
        if not name:
            raise TimelineError("Asset ids must be non-empty")
        if not path:
            raise TimelineError(f"Asset {name!r} has an empty path")

    layer_ids: list[str] = []
    for layer in timeline.layers:
        if not layer.id:
            raise TimelineError("Layers must have a non-empty id")
        if layer.id in layer_ids:
            raise TimelineError(f"Duplicate layer id {layer.id!r}")
        layer_ids.append(layer.id)
        if layer.asset not in timeline.assets:
            raise TimelineError(
                f"Layer {layer.id!r} references unknown asset {layer.asset!r}"
            )
        if not 0.0 <= layer.x <= 1.0 or not 0.0 <= layer.y <= 1.0:
            raise TimelineError(
                f"Layer {layer.id!r} position must be normalized (0.0-1.0),"
                f" got ({layer.x}, {layer.y})"
            )
        if layer.scale <= 0:
            raise TimelineError(f"Layer {layer.id!r} scale must be > 0")

    previous_frame = -1
    for key in timeline.camera:
        check_frame(key.frame, "Camera keyframe")
        if key.frame <= previous_frame:
            raise TimelineError(
                "Camera keyframes must be ordered by frame with no duplicates"
                f" (frame {key.frame} follows {previous_frame})"
            )
        previous_frame = key.frame
        if not 0.0 <= key.x <= 1.0 or not 0.0 <= key.y <= 1.0:
            raise TimelineError(
                f"Camera keyframe at frame {key.frame} must use normalized"
                f" x/y (0.0-1.0), got ({key.x}, {key.y})"
            )
        if key.scale <= 0:
            raise TimelineError(
                f"Camera keyframe at frame {key.frame} scale must be > 0"
            )

    seen_swaps: set[tuple[str, int]] = set()
    mouth_ranges: dict[str, list[tuple[int, int]]] = {}
    for event in timeline.events:
        if event.target not in layer_ids:
            raise TimelineError(
                f"{event.type} event targets unknown layer {event.target!r}"
            )
        if isinstance(event, AssetEvent):
            check_frame(event.frame, f"asset event on {event.target!r}")
            if event.asset not in timeline.assets:
                raise TimelineError(
                    f"asset event on {event.target!r} references unknown"
                    f" asset {event.asset!r}"
                )
            key = (event.target, event.frame)
            if key in seen_swaps:
                raise TimelineError(
                    f"Two asset events set layer {event.target!r} at frame"
                    f" {event.frame} — ambiguous"
                )
            seen_swaps.add(key)
        else:
            if not event.states:
                raise TimelineError(
                    f"mouth event on {event.target!r} has no states"
                )
            for state in event.states:
                if state not in timeline.assets:
                    raise TimelineError(
                        f"mouth event on {event.target!r} references unknown"
                        f" asset {state!r}"
                    )
            check_frame(
                event.start_frame, f"mouth event start on {event.target!r}"
            )
            check_frame(event.end_frame, f"mouth event end on {event.target!r}")
            if event.end_frame < event.start_frame:
                raise TimelineError(
                    f"mouth event on {event.target!r} ends at frame"
                    f" {event.end_frame}, before it starts ({event.start_frame})"
                )
            if (
                isinstance(event.period_frames, bool)
                or not isinstance(event.period_frames, int)
                or event.period_frames < 1
            ):
                raise TimelineError(
                    f"mouth event on {event.target!r} period_frames must be"
                    f" an integer >= 1"
                )
            for lo, hi in mouth_ranges.get(event.target, []):
                if event.start_frame <= hi and lo <= event.end_frame:
                    raise TimelineError(
                        f"Overlapping mouth events on layer {event.target!r}"
                        f" ({lo}..{hi} and"
                        f" {event.start_frame}..{event.end_frame}) — ambiguous"
                    )
            mouth_ranges.setdefault(event.target, []).append(
                (event.start_frame, event.end_frame)
            )

    for clip in timeline.audio:
        check_frame(clip.start_frame, f"audio clip {clip.path!r} start")
        if not clip.path:
            raise TimelineError("Audio clips must have a non-empty path")
        if clip.volume < 0:
            raise TimelineError(
                f"Audio clip {clip.path!r} volume must be >= 0"
            )


def save_timeline(timeline: Timeline, dest: Path) -> None:
    """Write timeline.json (validated first — never persist a broken one)."""
    validate_timeline(timeline)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(timeline.to_dict(), indent=2) + "\n")


def load_timeline(path: Path) -> Timeline:
    """Load a (possibly human-edited) timeline.json and re-validate it, so a
    malformed file fails here — before any frame is rendered."""
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise TimelineError(f"Cannot read {path}: {exc}") from exc
    try:
        timeline = Timeline.from_dict(payload)
    except TimelineError:
        raise
    except (TypeError, AttributeError, KeyError, ValueError) as exc:
        raise TimelineError(f"Malformed timeline {path}: {exc}") from exc
    validate_timeline(timeline)
    return timeline
