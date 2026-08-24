"""Anime-scene domain model: SceneSpec / Shot, validation, manual crops.

A Shot is NOT a recap Segment: it carries no narration or audio timing — it
describes one 5-second anime shot (source manga page, optional crop,
keyframe/animation prompts) inside one continuous scene. scene.json is
deliberately human-editable: plan first, edit by hand, resume generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image


class SceneError(Exception):
    pass


MIN_SHOTS = 3
MAX_SHOTS = 5
SHOT_SECONDS = 5  # every shot is exactly this long
MAX_SCENE_SECONDS = 30


@dataclass
class Shot:
    index: int
    source_page: int  # 1-based page number within the chapter
    action: str
    keyframe_prompt: str
    animation_prompt: str
    shot_type: str = "medium"
    composition: str = ""
    camera: str = ""
    continuity: str = ""
    source_crop: list[float] | None = None  # normalized [left, top, right, bottom]
    duration_s: float = SHOT_SECONDS

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Shot:
        crop = data.get("source_crop")
        return Shot(
            index=int(data["index"]),
            source_page=int(data["source_page"]),
            action=str(data.get("action", "")).strip(),
            keyframe_prompt=str(data.get("keyframe_prompt", "")).strip(),
            animation_prompt=str(data.get("animation_prompt", "")).strip(),
            shot_type=str(data.get("shot_type", "medium")).strip() or "medium",
            composition=str(data.get("composition", "")).strip(),
            camera=str(data.get("camera", "")).strip(),
            continuity=str(data.get("continuity", "")).strip(),
            source_crop=[float(v) for v in crop] if crop is not None else None,
            duration_s=float(data.get("duration_s", SHOT_SECONDS)),
        )


@dataclass
class SceneSpec:
    title: str
    chapter: float
    source_pages: list[int]  # selected 1-based page numbers, ascending
    instruction: str
    scene_summary: str
    continuity: str
    shots: list[Shot] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> SceneSpec:
        return SceneSpec(
            title=str(data.get("title", "")),
            chapter=float(data.get("chapter", 0)),
            source_pages=[int(p) for p in data.get("source_pages", [])],
            instruction=str(data.get("instruction", "")),
            scene_summary=str(data.get("scene_summary", "")),
            continuity=str(data.get("continuity", "")),
            shots=[Shot.from_dict(s) for s in data.get("shots", [])],
        )

    @property
    def total_seconds(self) -> float:
        return sum(s.duration_s for s in self.shots)


def validate_scene(spec: SceneSpec) -> None:
    """Enforce the v0 contract. Human-edited scene.json files go through the
    same validation, so a bad hand edit fails before any paid generation."""
    if not (MIN_SHOTS <= len(spec.shots) <= MAX_SHOTS):
        raise SceneError(
            f"Scene needs {MIN_SHOTS}-{MAX_SHOTS} shots, got {len(spec.shots)}"
        )
    if [s.index for s in spec.shots] != list(range(len(spec.shots))):
        raise SceneError("Shot indices must be sequential starting at 0")
    pages = set(spec.source_pages)
    for shot in spec.shots:
        if shot.duration_s != SHOT_SECONDS:
            raise SceneError(
                f"Shot {shot.index} is {shot.duration_s:g}s; shots must be"
                f" exactly {SHOT_SECONDS}s"
            )
        if shot.source_page not in pages:
            raise SceneError(
                f"Shot {shot.index} references page {shot.source_page},"
                f" outside the selected pages {sorted(pages)}"
            )
        for field_name in ("action", "keyframe_prompt", "animation_prompt"):
            if not getattr(shot, field_name):
                raise SceneError(
                    f"Shot {shot.index} has an empty {field_name}"
                )
        _validate_crop(shot.source_crop, shot.index)
    if spec.total_seconds > MAX_SCENE_SECONDS:
        raise SceneError(
            f"Scene runs {spec.total_seconds:g}s; max is {MAX_SCENE_SECONDS}s"
        )


def _validate_crop(crop: list[float] | None, index: int) -> None:
    if crop is None:
        return
    if len(crop) != 4:
        raise SceneError(
            f"Shot {index} crop must be [left, top, right, bottom]"
        )
    left, top, right, bottom = crop
    if not all(0.0 <= v <= 1.0 for v in crop):
        raise SceneError(f"Shot {index} crop values must be 0.0-1.0")
    if right <= left or bottom <= top:
        raise SceneError(f"Shot {index} crop has zero or negative area")


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_scene(raw: str, source_pages: list[int] | None = None) -> SceneSpec:
    """Extract the scene JSON object from raw planner output (tolerates
    fences/preamble) and validate it. When source_pages is given it overrides
    whatever the model claimed — the caller knows which pages were selected."""
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        raise SceneError("Planner did not return a JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise SceneError(f"Planner returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SceneError("Planner output is not a JSON object")
    spec = SceneSpec.from_dict(data)
    if source_pages is not None:
        spec.source_pages = list(source_pages)
    validate_scene(spec)
    return spec


def crop_page(page: Path, crop: list[float] | None, dest: Path) -> Path:
    """Write the shot's source image: the page, or its normalized
    [left, top, right, bottom] crop. Returns dest."""
    _validate_crop(crop, -1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page) as img:
        img = img.convert("RGB")
        if crop is not None:
            left, top, right, bottom = crop
            box = (
                round(left * img.width),
                round(top * img.height),
                round(right * img.width),
                round(bottom * img.height),
            )
            img = img.crop(box)
        img.save(dest)
    return dest


def save_scene(spec: SceneSpec, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")


def load_scene(path: Path) -> SceneSpec:
    """Load a (possibly human-edited) scene.json and re-validate it."""
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SceneError(f"Cannot read {path}: {exc}") from exc
    spec = SceneSpec.from_dict(payload)
    validate_scene(spec)
    return spec
