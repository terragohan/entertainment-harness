"""Experiment files: TOML-declared sweeps over the frame-generation seams.

An experiment is data: a name, a mode ("video-animation" | "frame-animation"), panels, a
slot duration (seconds of screen time each panel gets), optional scene
context (becomes the Segment text embedded in the frame prompts), a judge
flag, and arms. Each arm is one config variation — provider + model +
cadence knobs — that the runner applies to a deep copy of the loaded
Config. Adding an arm never requires code.

Validation is strict: unknown keys, a bad mode, duplicate or path-unsafe
names, non-positive durations, out-of-mode knobs, and missing panels all
fail the load. Panel paths resolve from the current working directory
first, then relative to the experiment file's directory — the seed
experiments use repo-root-relative paths and are meant to run from the
repo root.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("video-animation", "frame-animation")
PROVIDERS = ("openrouter", "runway", "diffusers", "fal", "replicate")

# Cadence knobs per mode. `resolution` is frame-animation-only: the outpainting
# ratio derives from config.video.resolution, so it changes the generated
# frames there. `interp_fps` is assembly-only (preview-clip playback) and
# never changes generated frames — the manifest/gallery label it.
ANIMATE_KNOBS = frozenset({"seconds_per_frame", "max_frames"})
SEQUENCE_KNOBS = frozenset(
    {"fps", "max_frames", "interp_fps", "critic", "resolution", "director"}
)

_TOP_KEYS = frozenset(
    {"name", "mode", "duration", "panels", "context", "judge", "arms"}
)
_ARM_KEYS = (
    frozenset({"name", "provider", "model"}) | ANIMATE_KNOBS | SEQUENCE_KNOBS
)

_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")  # fullmatch; dir-safe
_RESOLUTION_RE = re.compile(r"[0-9]+x[0-9]+")

_FLOAT_KNOBS = {"seconds_per_frame", "fps"}
_INT_KNOBS = {"max_frames", "interp_fps"}


class ExperimentError(ValueError):
    """An invalid experiment file (schema or values)."""


@dataclass
class Arm:
    name: str
    provider: str = "openrouter"
    model: str = ""  # empty = the provider's default image model
    knobs: dict = field(default_factory=dict)  # mode-specific overrides


@dataclass
class Experiment:
    name: str
    mode: str
    duration: float
    panels: list[Path]
    context: str = ""
    judge: bool = True
    arms: list[Arm] = field(default_factory=list)
    path: Path | None = None  # the file it loaded from


def _slug(value: object, what: str, path: Path) -> str:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ExperimentError(
            f"{path}: {what} must be a lowercase slug ([a-z0-9._-], no"
            f" slashes), got {value!r}"
        )
    return value


def _knob_value(key: str, value: object, arm_name: str, path: Path):
    where = f"{path}: arm {arm_name!r}"
    if key in _FLOAT_KNOBS:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ExperimentError(f"{where}: {key} must be a positive number")
        return float(value)
    if key in _INT_KNOBS:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ExperimentError(f"{where}: {key} must be an int >= 1")
        return value
    if key in ("critic", "director"):
        if not isinstance(value, bool):
            raise ExperimentError(f"{where}: {key} must be a bool")
        return value
    if key == "resolution":
        if not isinstance(value, str) or not _RESOLUTION_RE.fullmatch(value):
            raise ExperimentError(
                f"{where}: resolution must look like \"1920x1080\""
            )
        return value
    raise ExperimentError(f"{where}: unknown knob {key!r}")  # unreachable


def _panels(raw: object, path: Path) -> list[Path]:
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(p, str) and p for p in raw)
    ):
        raise ExperimentError(
            f"{path}: panels must be a non-empty list of image paths"
        )
    panels = []
    for entry in raw:
        candidate = Path(entry)  # from the cwd first
        if not candidate.is_file():
            fallback = path.parent / entry  # then the experiment file's dir
            if fallback.is_file():
                candidate = fallback
            else:
                raise ExperimentError(
                    f"{path}: panel {entry!r} not found (tried the cwd and"
                    f" {path.parent})"
                )
        panels.append(candidate)
    return panels


def _arms(raw: object, mode: str, path: Path) -> list[Arm]:
    if not isinstance(raw, list) or not raw:
        raise ExperimentError(
            f"{path}: [[arms]] must be a non-empty array of tables"
        )
    knob_set = ANIMATE_KNOBS if mode == "video-animation" else SEQUENCE_KNOBS
    seen: set[str] = set()
    arms = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ExperimentError(f"{path}: arms[{i}] must be a table")
        unknown = sorted(set(entry) - _ARM_KEYS)
        if unknown:
            raise ExperimentError(
                f"{path}: unknown arm key(s): {', '.join(unknown)} (known:"
                f" {', '.join(sorted(_ARM_KEYS))})"
            )
        name = _slug(entry.get("name"), f"arms[{i}].name", path)
        if name in seen:
            raise ExperimentError(f"{path}: duplicate arm name {name!r}")
        seen.add(name)
        provider = entry.get("provider", "openrouter")
        if provider not in PROVIDERS:
            raise ExperimentError(
                f"{path}: arm {name!r}: provider must be one of"
                f" {', '.join(PROVIDERS)}, got {provider!r}"
            )
        model = entry.get("model", "")
        if not isinstance(model, str):
            raise ExperimentError(f"{path}: arm {name!r}: model must be a string")
        knobs = {}
        for key in sorted(set(entry) & (ANIMATE_KNOBS | SEQUENCE_KNOBS)):
            if key not in knob_set:
                other = "frame-animation" if key in SEQUENCE_KNOBS else "video-animation"
                raise ExperimentError(
                    f"{path}: arm {name!r}: {key!r} is a {other}-mode knob"
                    f" — the experiment mode is {mode!r}"
                )
            knobs[key] = _knob_value(key, entry[key], name, path)
        if knobs.get("director") and knobs.get("critic"):
            raise ExperimentError(
                f"{path}: arm {name!r}: director and critic conflict — the"
                " director subsumes the drift critic"
            )
        arms.append(Arm(name=name, provider=provider, model=model, knobs=knobs))
    return arms


def load_experiment(path: Path) -> Experiment:
    """Load and strictly validate one experiment TOML."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentError(f"cannot read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ExperimentError(f"{path}: invalid TOML: {exc}") from exc

    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise ExperimentError(
            f"{path}: unknown top-level key(s): {', '.join(unknown)}"
            f" (known: {', '.join(sorted(_TOP_KEYS))})"
        )
    name = _slug(data.get("name"), "name", path)
    mode = data.get("mode")
    if mode not in MODES:
        raise ExperimentError(
            f"{path}: mode must be one of {', '.join(MODES)}, got {mode!r}"
        )
    duration = data.get("duration")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or duration <= 0
    ):
        raise ExperimentError(f"{path}: duration must be a positive number")
    context = data.get("context", "")
    if not isinstance(context, str):
        raise ExperimentError(f"{path}: context must be a string")
    judge = data.get("judge", True)
    if not isinstance(judge, bool):
        raise ExperimentError(f"{path}: judge must be a bool")
    return Experiment(
        name=name,
        mode=mode,
        duration=float(duration),
        panels=_panels(data.get("panels"), path),
        context=context,
        judge=judge,
        arms=_arms(data.get("arms"), mode, path),
        path=path,
    )


def load_experiments(directory: Path) -> list[Experiment]:
    """Every *.toml in a directory, filename-sorted."""
    return [
        load_experiment(toml) for toml in sorted(Path(directory).glob("*.toml"))
    ]


def arm_frame_count(experiment: Experiment, arm: Arm) -> int:
    """Frames one panel requests from the loop under this arm — the list
    command's cost preview (frame-animation critic retries can double an arm
    and are not counted). Knobs default to the Config dataclass defaults."""
    if experiment.mode == "video-animation":
        from entertainment_harness.config import FramesConfig
        from entertainment_harness.video.frames import frame_count

        base = FramesConfig()
        return frame_count(
            experiment.duration,
            arm.knobs.get("seconds_per_frame", base.seconds_per_frame),
            arm.knobs.get("max_frames", base.max_frames),
        )
    from entertainment_harness.config import FrameAnimationConfig
    from entertainment_harness.video.sequence import frame_count

    base = FrameAnimationConfig()
    return frame_count(
        experiment.duration,
        arm.knobs.get("fps", base.fps),
        arm.knobs.get("max_frames", base.max_frames),
    )


def estimated_generations(experiment: Experiment) -> int:
    """Total image generations a full run will request (cost preview)."""
    return (
        sum(arm_frame_count(experiment, arm) for arm in experiment.arms)
        * len(experiment.panels)
    )
