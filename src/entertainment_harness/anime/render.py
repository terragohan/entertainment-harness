"""Sequencing Renderer (animation-conductor initiative).

The deterministic runtime that executes a timeline.json end to end:

1. load + validate the timeline (malformed files fail before any frame),
2. load the referenced visual assets,
3. iterate exactly `range(duration_frames)` — the timeline is the clock;
   ffmpeg is never allowed to invent timing,
4. resolve each frame through the sequencer (`state_at`),
5. render each frame through the compositor -> `frames/frame-000000.png` …,
6. assemble the PNG sequence with ffmpeg at exactly the timeline's fps,
7. place audio at `start_frame / fps` seconds and mix it over a silent
   base, cut to exactly `duration_frames / fps` seconds (audio never
   stretches or shortens the scene),
8. produce `out.mp4`.

Frame compositing is cheap and AI generation is the expensive operation, so
Phase 1 re-renders every frame on each run and keys nothing off timestamps;
asset identity stays addressable (the timeline's `assets` map) so future
selective regeneration never has to rebuild unrelated assets.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from entertainment_harness.anime.compositor import Compositor
from entertainment_harness.anime.sequencer import state_at
from entertainment_harness.anime.timeline import (
    Timeline,
    TimelineError,
    load_timeline,
)
from entertainment_harness.video.assemble import AUDIO_BITRATE, video_duration
from entertainment_harness.video.script import VideoError

FRAME_NAME = "frame-%06d.png"


def _check_audio_files(timeline: Timeline, base_dir: Path) -> list[Path]:
    """Resolve audio clip paths (relative to the timeline file) and fail
    before any rendering when one is missing."""
    paths = []
    for clip in timeline.audio:
        path = Path(clip.path)
        if not path.is_absolute():
            path = base_dir / path
        if not path.is_file():
            raise TimelineError(
                f"Audio clip {clip.path!r} does not exist at {path}"
            )
        paths.append(path)
    return paths


def render_frames(timeline: Timeline, compositor: Compositor,
                  frames_dir: Path, log=lambda m: None) -> list[Path]:
    """Render exactly duration_frames PNGs, 0..duration_frames-1. Stale
    frames from a previous (longer) render are removed first, so the ffmpeg
    image2 demuxer can never pick up leftovers and the output count is
    exactly the timeline's."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame-*.png"):
        stale.unlink()
    frames = []
    for frame in range(timeline.duration_frames):
        state = state_at(timeline, frame)
        dest = frames_dir / (FRAME_NAME % frame)
        compositor.render(state).save(dest)
        frames.append(dest)
    log(f"  rendered {len(frames)} frame(s) at"
        f" {timeline.width}x{timeline.height} -> {frames_dir}")
    return frames


def assemble_video(timeline: Timeline, frames_dir: Path,
                   audio_paths: list[Path], out: Path,
                   log=lambda m: None) -> Path:
    """ffmpeg: PNG sequence -> mp4 at the timeline's fps, audio clips delayed
    to their exact frame offsets and mixed over a silent base. The output is
    pinned to duration_frames frames / duration_frames/fps seconds — the
    timeline, not the media, decides when the video ends."""
    duration_s = timeline.duration_seconds
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(timeline.fps),
        "-i", str(frames_dir / FRAME_NAME),
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
    ]
    for path in audio_paths:
        cmd += ["-i", str(path)]
    if audio_paths:
        filters = []
        labels = ["1:a"]
        for index, (clip, _) in enumerate(zip(timeline.audio, audio_paths)):
            delay_ms = round(timeline.frame_seconds(clip.start_frame) * 1000)
            filters.append(
                f"[{index + 2}:a]aresample=48000,"
                f"aformat=channel_layouts=stereo,volume={clip.volume:g},"
                f"adelay={delay_ms}|{delay_ms}[a{index}]"
            )
            labels.append(f"a{index}")
        inputs = "".join(f"[{label}]" for label in labels)
        filters.append(
            f"{inputs}amix=inputs={len(labels)}:normalize=0[aout]"
        )
        cmd += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += [
        "-frames:v", str(timeline.duration_frames),
        "-t", f"{duration_s:.6f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise VideoError(f"ffmpeg failed assembling {out}:\n{tail}")
    actual = video_duration(out)
    log(f"  assembled {out} ({actual:.3f}s nominal {duration_s:.3f}s,"
        f" {timeline.fps} fps)")
    return out


def render_timeline(timeline_path: Path, workdir: Path | None = None,
                    log=lambda m: None) -> Path:
    """Render one timeline.json to out.mp4. Returns the output path.

    Default workdir is `<timeline dir>/render/` holding the full frame
    sequence (`frames/frame-000000.png` …) plus `out.mp4`. Every run
    re-renders all frames — deterministically: the same timeline and assets
    produce the same frames.
    """
    timeline = load_timeline(timeline_path)  # validates before any rendering
    base_dir = timeline_path.resolve().parent
    workdir = workdir or timeline_path.parent / "render"
    workdir.mkdir(parents=True, exist_ok=True)

    compositor = Compositor(timeline, base_dir)
    audio_paths = _check_audio_files(timeline, base_dir)

    log(f"Timeline: {timeline.duration_frames} frames at {timeline.fps} fps"
        f" ({timeline.duration_seconds:g}s), {timeline.width}x{timeline.height},"
        f" {len(timeline.layers)} layer(s), {len(timeline.events)} event(s),"
        f" {len(timeline.audio)} audio clip(s)")
    frames_dir = workdir / "frames"
    render_frames(timeline, compositor, frames_dir, log)
    return assemble_video(timeline, frames_dir, audio_paths, workdir / "out.mp4", log)
