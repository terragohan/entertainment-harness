"""Anime-scene pipeline: plan -> keyframes -> shots -> assembly.

Staged and cached under works/<series>/chapters/<ch>/anime-scene/; the
filesystem is the source of truth (no DB state). scene.json is
human-editable: once it exists it is always loaded, never silently
regenerated — delete it (or change --pages/--instruction) to re-plan.
Fingerprints in state.json are sha256 content hashes, so editing one shot
regenerates only that shot and whatever references its keyframe.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from entertainment_harness.anime.planner import PLANNER_PROMPT_VERSION, plan_scene
from entertainment_harness.anime.scene import (
    MAX_SCENE_SECONDS,
    SHOT_SECONDS,
    SceneError,
    SceneSpec,
    Shot,
    crop_page,
    load_scene,
    save_scene,
)
from entertainment_harness.config import Config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.library import works
from entertainment_harness.models.registry import get_vision_model
from entertainment_harness.video.assemble import video_duration

# Capabilities the anime pipeline needs from its [anime] provider.
ANIME_PROVIDER_CAPS = frozenset({"image-gen", "image-to-video"})


def get_scene_provider(config: Config):
    """The [anime] provider: a video_gen plugin declaring image-gen AND
    image-to-video. Configurable so third-party generators slot in without
    touching this pipeline."""
    from entertainment_harness.video.gen import REGISTRY

    name = config.anime.provider
    cls = REGISTRY.load(name)  # unknown name -> PluginError listing options
    missing = ANIME_PROVIDER_CAPS - getattr(cls, "capabilities", frozenset())
    if missing:
        raise SceneError(
            f"[anime] provider {name!r} lacks capabilities {sorted(missing)}."
            f" Providers that can image-gen: {REGISTRY.names_with('image-gen')};"
            f" image-to-video: {REGISTRY.names_with('image-to-video')}"
        )
    return REGISTRY.create(name, config)

KEYFRAME_SIZE = (1280, 720)  # 16:9 720p
KEYFRAME_RATIO = "1280:720"
VIDEO_FPS = 30

KEYFRAME_PROMPT = """Create one polished 2D anime production frame.

@Source is the authoritative manga reference for the action/composition.
{refs}
SOURCE FIDELITY RULES:
- Preserve recognizable character identity.
- Preserve hair, face, clothing, props and creature design.
- Preserve the setting and story action from @Source.
- Remove speech bubbles, captions, page borders, screentone artifacts and printed text.
- Do not add new characters or objects.
- Do not invent story information.
- Do not change outfits unless visibly required by @Source.

SHOT:
{shot_type}

FRAME DESCRIPTION:
{frame}

ACTION AT THIS FRAME:
{action}

COMPOSITION:
{composition}

CAMERA:
{camera}

CONTINUITY REQUIREMENTS:
{continuity}

USER DIRECTION:
{instruction}

Output should look like a finished 16:9 anime frame rather than a colored manga page."""

_REFS_WITH_ANCHOR = """@Anchor is the canonical anime appearance for recurring characters, clothing,
palette and rendering style.

@Previous shows continuity immediately before this shot.
"""

ANIMATION_PROMPT = """Animate this exact anime frame for {seconds} seconds.

Preserve character identity, clothing, colors, background, composition and
drawing style from the input frame.

ACTION:
{action}

CHARACTER MOTION:
{motion}

CAMERA:
{camera}

CONTINUITY:
{continuity}

Do not introduce new characters, objects, text or locations.
Do not change clothing, hairstyle or character design.
Do not morph faces or bodies.
Do not cut to another scene.
Do not add subtitles or captions.
Keep the action readable and physically coherent."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(*parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _state_path(workdir: Path) -> Path:
    return workdir / "state.json"


def _load_state(workdir: Path) -> dict:
    try:
        return json.loads(_state_path(workdir).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(workdir: Path, state: dict) -> None:
    _state_path(workdir).write_text(json.dumps(state, indent=2) + "\n")


def _keyframe_prompt(shot: Shot, instruction: str) -> str:
    refs = _REFS_WITH_ANCHOR if shot.index > 0 else ""
    return KEYFRAME_PROMPT.format(
        refs=refs,
        shot_type=shot.shot_type,
        frame=shot.keyframe_prompt,
        action=shot.action,
        composition=shot.composition or "as described",
        camera=shot.camera or "static",
        continuity=shot.continuity or "maintain scene continuity",
        instruction=instruction or "(none)",
    )


def _animation_prompt(shot: Shot) -> str:
    return ANIMATION_PROMPT.format(
        seconds=SHOT_SECONDS,
        action=shot.action,
        motion=shot.animation_prompt,
        camera=shot.camera or "static",
        continuity=shot.continuity or "maintain scene continuity",
    )


def _keyframe_fp(
    shot: Shot, source_png: Path, refs: list[tuple[str, Path]],
    instruction: str, model: str,
) -> str:
    return _fingerprint(
        "keyframe", model, instruction,
        shot.to_dict(), _sha256_file(source_png),
        [(tag, _sha256_file(p)) for tag, p in refs],
    )


def _clip_fp(shot: Shot, keyframe: Path, model: str) -> str:
    return _fingerprint(
        "clip", model, SHOT_SECONDS, KEYFRAME_RATIO,
        _sha256_file(keyframe), _animation_prompt(shot),
    )


def _source_image(shot: Shot, page_paths: dict[int, Path],
                  workdir: Path) -> Path:
    """The manga page/crop supplied to the keyframe model for this shot."""
    dest = workdir / "sources" / f"shot-{shot.index:02d}.png"
    crop_page(page_paths[shot.source_page], shot.source_crop, dest)
    return dest


def _keyframe_refs(shot: Shot, workdir: Path) -> list[tuple[str, Path]]:
    """Continuity references for shot N: @Anchor is shot 0's keyframe,
    @Previous the immediately prior one when distinct."""
    refs: list[tuple[str, Path]] = []
    if shot.index == 0:
        return refs
    anchor = workdir / "keyframes" / "shot-00.png"
    refs.append(("Anchor", anchor))
    previous = workdir / "keyframes" / f"shot-{shot.index - 1:02d}.png"
    if previous != anchor:
        refs.append(("Previous", previous))
    return refs


def _normalize_clip(clip: Path, dest: Path) -> Path:
    """Uniform geometry/codec so the concat demuxer can hard-cut between
    clips: 1280x720 (scale+pad), 30 fps, h264 yuv420p, no audio."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    w, h = KEYFRAME_SIZE
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(clip),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={VIDEO_FPS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dest),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise SceneError(f"ffmpeg failed normalizing {clip}:\n{tail}")
    return dest


def assemble_scene(clips: list[Path], workdir: Path) -> Path:
    """Hard-cut the shot clips in storyboard order into out.mp4 (no audio,
    no recap muxer). Verifies the result is <= MAX_SCENE_SECONDS."""
    normalized = [
        _normalize_clip(clip, workdir / "clips" / f"norm-{clip.stem}.mp4")
        for clip in clips
    ]
    lst = workdir / "clips.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in normalized))
    out = workdir / "out.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise SceneError(f"ffmpeg failed concatenating clips:\n{tail}")
    duration = video_duration(out)
    if duration > MAX_SCENE_SECONDS:
        raise SceneError(
            f"Assembled scene runs {duration:.1f}s; max is {MAX_SCENE_SECONDS}s"
        )
    return out


def parse_page_spec(spec: str, page_count: int) -> list[int]:
    """Parse a --pages selector like "8-10" or "8,9,12" into 1-based page
    numbers, bounds-checked against the chapter's cached page count."""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise SceneError(f"Empty entry in page spec {spec!r}")
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                raise SceneError(f"Bad page range {part!r} in {spec!r}") from None
            if hi < lo:
                raise SceneError(f"Bad page range {part!r} in {spec!r}")
            pages.extend(range(lo, hi + 1))
        else:
            try:
                pages.append(int(part))
            except ValueError:
                raise SceneError(f"Bad page {part!r} in {spec!r}") from None
    pages = sorted(set(pages))
    if not pages:
        raise SceneError(f"Empty page spec {spec!r}")
    if pages[0] < 1 or pages[-1] > page_count:
        raise SceneError(
            f"Pages {pages[0]}-{pages[-1]} out of range;"
            f" chapter has {page_count} pages"
        )
    return pages


def build_scene(
    series_id: str,
    chapter_id: str,
    chapter_num: float,
    title: str,
    page_paths: dict[int, Path],
    config: Config,
    profile: HardwareProfile,
    instruction: str,
    stop_after: str | None = None,
    regenerate_shots: set[int] | None = None,
    provider=None,
    log=lambda m: None,
) -> Path:
    """Build (or resume) the anime scene for one chapter's selected pages.

    Returns scene.json when stop_after is set, else out.mp4. A cached stage
    is redone only when its fingerprint changed (or the shot is listed in
    regenerate_shots). Missing any stage artifact rebuilds it and everything
    downstream of it.
    """
    if stop_after not in (None, "plan", "keyframes"):
        raise SceneError(f"Bad --stop-after {stop_after!r}; plan or keyframes")
    regenerate = regenerate_shots or set()
    workdir = works.anime_scene_dir(series_id, chapter_id)
    workdir.mkdir(parents=True, exist_ok=True)
    scene_path = workdir / "scene.json"
    state = _load_state(workdir)

    # --- stage 1: guided storyboard -----------------------------------------
    if scene_path.exists():
        spec = load_scene(scene_path)  # re-validates human edits
        plan_fp = _fingerprint(
            "plan", PLANNER_PROMPT_VERSION,
            [_sha256_file(page_paths[p]) for p in spec.source_pages],
            spec.instruction,
        )
        if state.get("plan") not in (None, plan_fp):
            log("  plan: scene.json inputs differ from the last generation"
                " — using the existing file (delete it to re-plan)")
        state["plan"] = plan_fp
        log(f"Stage 1/4 plan: cached ({len(spec.shots)} shots)")
    else:
        vision = get_vision_model(config, profile)
        log(f"Stage 1/4 plan: storyboarding with {vision.info.name}...")
        vision.adapter.ensure(vision.info.name)
        ordered = sorted(page_paths)
        spec = plan_scene(
            vision.adapter, vision.info.name,
            [page_paths[p] for p in ordered], ordered,
            title, chapter_num, instruction,
        )
        save_scene(spec, scene_path)
        state = {"plan": _fingerprint(
            "plan", PLANNER_PROMPT_VERSION, vision.info.name,
            [_sha256_file(page_paths[p]) for p in spec.source_pages],
            instruction,
        )}  # a new plan invalidates everything downstream
        _save_state(workdir, state)
        log(f"Stage 1/4 plan: wrote {scene_path} ({len(spec.shots)} shots,"
            f" {spec.total_seconds:g}s)")

    if stop_after == "plan":
        log("Stopped after plan — edit scene.json, then rerun to continue.")
        return scene_path

    # --- stage 2: anime keyframes ([anime] provider text_to_image) ------------
    if provider is None:
        provider = get_scene_provider(config)
    keyframe_model = config.anime.keyframe_model
    keyframes: dict[int, Path] = {}
    keyframe_fps = dict(state.get("keyframes", {}))
    for shot in spec.shots:
        dest = workdir / "keyframes" / f"shot-{shot.index:02d}.png"
        source_png = _source_image(shot, page_paths, workdir)
        refs = _keyframe_refs(shot, workdir)
        fp = _keyframe_fp(shot, source_png, refs, spec.instruction, keyframe_model)
        if (
            shot.index not in regenerate
            and dest.exists()
            and keyframe_fps.get(str(shot.index)) == fp
        ):
            log(f"  keyframe shot-{shot.index:02d}: cached")
        else:
            log(f"  keyframe shot-{shot.index:02d}: generating"
                f" ({keyframe_model}, refs: {', '.join('@' + t for t, _ in refs) or '@Source'})...")
            provider.text_to_image(
                _keyframe_prompt(shot, spec.instruction),
                [("Source", source_png), *refs],
                model=keyframe_model,
                ratio=KEYFRAME_RATIO,
                ref_ratio=KEYFRAME_SIZE,
                dest=dest,
            )
            keyframe_fps[str(shot.index)] = fp
        keyframes[shot.index] = dest
    state["keyframes"] = keyframe_fps
    _save_state(workdir, state)

    if stop_after == "keyframes":
        log("Stopped after keyframes — inspect keyframes/, then rerun.")
        return scene_path

    # --- stage 3: animate each keyframe ([anime] provider image_to_video) -----
    video_model = config.anime.video_model
    clips: list[Path] = []
    clip_fps = dict(state.get("clips", {}))
    for shot in spec.shots:
        dest = workdir / "clips" / f"shot-{shot.index:02d}.mp4"
        fp = _clip_fp(shot, keyframes[shot.index], video_model)
        if (
            shot.index not in regenerate
            and dest.exists()
            and clip_fps.get(str(shot.index)) == fp
        ):
            log(f"  clip shot-{shot.index:02d}: cached")
        else:
            log(f"  clip shot-{shot.index:02d}: generating"
                f" ({video_model}, {SHOT_SECONDS}s)...")
            provider.image_to_video(
                keyframes[shot.index],
                _animation_prompt(shot),
                duration=SHOT_SECONDS,
                ratio=KEYFRAME_RATIO,
                model=video_model,
                ref_ratio=KEYFRAME_SIZE,
                dest=dest,
            )
            clip_fps[str(shot.index)] = fp
        clips.append(dest)
    state["clips"] = clip_fps
    _save_state(workdir, state)

    # --- stage 4: assembly -----------------------------------------------------
    out_fp = _fingerprint("out", [_sha256_file(c) for c in clips])
    out = workdir / "out.mp4"
    if out.exists() and state.get("out") == out_fp and not regenerate:
        log("Stage 4/4 assembly: cached")
    else:
        log("Stage 4/4 assembly: cutting clips together...")
        out = assemble_scene(clips, workdir)
        state["out"] = out_fp
        _save_state(workdir, state)
    log(f"Done: {out} ({video_duration(out):.1f}s)")
    return out
