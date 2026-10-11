"""Standalone eval harness for the animation-conductor Sequencing Renderer.

This is deliberately NOT a unit test: it is the release gate a human runs to
prove the deterministic runtime works end to end, the same way `bun run
smoke` gates the desktop UI. It builds its own synthetic fixtures (no
network, no models, no paid APIs), renders real MP4s with the real ffmpeg,
and checks exact behavior — frame counts, filenames, fps, duration, audio
offsets, pixel changes at event boundaries, byte-determinism.

Usage:
    uv run python scripts/eval_animation_conductor.py [--out DIR]

Requires ffmpeg/ffprobe on PATH. Exits 0 when every check passes, 1 on the
first failed scenario (all failures listed). The fast unit gate stays
`uv run pytest`; this harness is the separate end-to-end one.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

os.environ.setdefault("COLUMNS", "200")
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")

from PIL import Image, ImageChops

from entertainment_harness.anime.render import FRAME_NAME, render_timeline
from entertainment_harness.anime.timeline import (
    AssetEvent,
    AudioClip,
    CameraKey,
    Layer,
    Timeline,
    save_timeline,
)
from entertainment_harness.video.assemble import probe_format, video_duration

RED = (200, 30, 30)
BLUE = (30, 30, 200)
GREEN = (30, 180, 60)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail and not ok else ""
    print(f"{mark}  {name}{suffix}")


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# --- synthetic fixtures --------------------------------------------------------


def _solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGBA", size, (*color, 255)).save(path)


def _half_bg(path: Path, width: int, height: int) -> None:
    img = Image.new("RGBA", (width, height), (*RED, 255))
    for x in range(width // 2, width):
        for y in range(height):
            img.putpixel((x, y), (*BLUE, 255))
    img.save(path)


def _tone_wav(path: Path, seconds: float = 1.0, rate: int = 22050) -> None:
    frames = bytearray()
    for i in range(int(rate * seconds)):
        sample = 0.5 * math.sin(2 * math.pi * 440.0 * i / rate)
        frames += int(sample * 32767).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))


def _rms(path: Path, start_s: float, end_s: float) -> float:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        wav.setpos(round(start_s * rate))
        raw = wav.readframes(round((end_s - start_s) * rate))
    samples = array.array("h", raw)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _audio_wav(mp4: Path, dest: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(mp4),
         "-ac", "1", "-ar", "8000", "-f", "wav", str(dest)],
        check=True, capture_output=True,
    )
    return dest


def _diff_pixels(a: Path, b: Path, threshold: int = 60) -> int:
    """Pixels that changed SUBSTANTIALLY between two frames (any channel
    delta >= threshold). Smooth camera drift moves pixels by small amounts;
    an asset swap (blink, pose) changes whole regions by large ones."""
    with Image.open(a) as ia, Image.open(b) as ib:
        diff = ImageChops.difference(ia.convert("RGB"), ib.convert("RGB"))
    data = diff.tobytes()
    return sum(
        1
        for i in range(0, len(data), 3)
        if data[i] >= threshold or data[i + 1] >= threshold
        or data[i + 2] >= threshold
    )


# --- scenario 1: exact behavior -------------------------------------------------


def scenario_exact(root: Path) -> None:
    print("== scenario: exact (160x90, 96 frames @ 24fps, tone at frame 48) ==")
    root.mkdir(parents=True, exist_ok=True)
    _half_bg(root / "bg.png", 160, 90)
    _solid(root / "red.png", (30, 30), RED)
    _solid(root / "green.png", (30, 30), GREEN)
    _tone_wav(root / "tone.wav")
    timeline = Timeline(
        fps=24, width=160, height=90, duration_frames=96,
        assets={"bg": "bg.png", "red": "red.png", "green": "green.png"},
        layers=[
            Layer(id="bg", asset="bg"),
            Layer(id="box", asset="red", x=0.5, y=0.6, scale=0.4),
        ],
        camera=[
            CameraKey(frame=0, x=0.25, y=0.5, scale=2.0),
            CameraKey(frame=95, x=0.75, y=0.5, scale=2.0),
        ],
        events=[AssetEvent(frame=48, target="box", asset="green")],
        audio=[AudioClip(start_frame=48, path="tone.wav")],
    )
    timeline_path = root / "timeline.json"
    save_timeline(timeline, timeline_path)

    # A broken timeline must be rejected before any frame is rendered.
    bad = root / "broken.json"
    bad.write_text(json.dumps({**timeline.to_dict(), "fps": 0}))
    try:
        render_timeline(bad, workdir=root / "broken-render", log=lambda m: None)
        check("invalid timeline rejected before rendering", False,
              "fps=0 rendered without error")
    except Exception as exc:
        check("invalid timeline rejected before rendering",
              "fps must be > 0" in str(exc), str(exc))
    check("no output produced for the invalid timeline",
          not (root / "broken-render").exists())

    out = render_timeline(timeline_path, log=lambda m: None)
    workdir = root / "render"
    frames_dir = workdir / "frames"

    names = sorted(p.name for p in frames_dir.iterdir())
    check("exactly duration_frames files with deterministic names",
          names == [FRAME_NAME % i for i in range(96)],
          f"got {len(names)} files")
    with Image.open(frames_dir / (FRAME_NAME % 0)) as img:
        check("frame dimensions are exact", img.size == (160, 90),
              f"got {img.size}")

    def pixel(frame: int, xy: tuple[int, int]) -> tuple:
        with Image.open(frames_dir / (FRAME_NAME % frame)) as img:
            return img.getpixel(xy)

    check("camera pan: first frame shows the red half",
          pixel(0, (80, 45))[0] > pixel(0, (80, 45))[2])
    check("camera pan: last frame shows the blue half",
          pixel(95, (80, 45))[2] > pixel(95, (80, 45))[0])
    check("asset swap holds before its frame", pixel(47, (80, 54))[:3] == RED,
          str(pixel(47, (80, 54))))
    check("asset swap applies at its exact frame",
          pixel(48, (80, 54))[:3] == GREEN, str(pixel(48, (80, 54))))

    again = render_timeline(timeline_path, workdir=root / "render-2",
                            log=lambda m: None)
    same = all(
        (frames_dir / (FRAME_NAME % i)).read_bytes()
        == (root / "render-2" / "frames" / (FRAME_NAME % i)).read_bytes()
        for i in range(96)
    )
    check("re-render is byte-identical (deterministic)", same)

    codec, width, height, fps = probe_format(out)
    check("mp4 is h264 at the timeline's exact geometry and fps",
          (codec, width, height, fps) == ("h264", 160, 90, "24/1"),
          f"{codec},{width},{height},{fps}")
    check("mp4 duration is duration_frames / fps (audio does not stretch it)",
          approx(video_duration(out), 4.0, 0.1),
          f"{video_duration(out):.3f}s")
    check("second render's mp4 matches too",
          approx(video_duration(again), 4.0, 0.1))

    wav = _audio_wav(out, workdir / "audio.wav")
    before = _rms(wav, 0.4, 1.6)
    during = _rms(wav, 2.2, 2.9)
    after = _rms(wav, 3.4, 3.9)
    check("audio starts at frame 48 / 24fps = 2.0s (silent before)",
          before < 5.0, f"RMS {before:.1f}")
    check("audio is audible after 2.0s", during > 1000.0,
          f"RMS {during:.1f}")
    check("1s tone ends but the video continues", after < 5.0,
          f"RMS {after:.1f}")


# --- scenario 2: the synthetic demo ----------------------------------------------


def scenario_demo(root: Path) -> None:
    print("== scenario: demo (640x360, 120 frames @ 24fps = 5s) ==")
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anime_conductor_demo import write_demo

    timeline_path = write_demo(root)
    out = render_timeline(timeline_path, log=lambda m: None)
    frames_dir = root / "render" / "frames"

    names = sorted(p.name for p in frames_dir.iterdir())
    check("5s at 24fps produces exactly 120 frames (0..119, no frame 120)",
          names == [FRAME_NAME % i for i in range(120)],
          f"got {len(names)} files")
    codec, width, height, fps = probe_format(out)
    check("mp4 geometry/fps match the timeline",
          (codec, width, height, fps) == ("h264", 640, 360, "24/1"),
          f"{codec},{width},{height},{fps}")
    check("mp4 duration is exactly 5.0s", approx(video_duration(out), 5.0, 0.1),
          f"{video_duration(out):.3f}s")

    # Event boundaries must stand out over the (smooth) camera drift: the
    # frame-to-frame pixel difference spikes at the blink and the pose swap.
    def diff(frame: int) -> int:
        return _diff_pixels(
            frames_dir / (FRAME_NAME % (frame - 1)),
            frames_dir / (FRAME_NAME % frame),
        )

    drift = max(diff(34), diff(35), 1)
    check("blink at frame 36 is visible over camera drift",
          diff(36) > 3 * drift, f"blink diff {diff(36)} vs drift {drift}")
    drift = max(diff(98), diff(99), 1)
    check("pose swap at frame 100 is visible over camera drift",
          diff(100) > 3 * drift, f"swap diff {diff(100)} vs drift {drift}")

    wav = _audio_wav(out, root / "render" / "audio.wav")
    check("dialogue tone starts at frame 60 / 24fps = 2.5s",
          _rms(wav, 1.0, 2.2) < 5.0 and _rms(wav, 2.7, 3.5) > 1000.0)
    check("tone ends at 4.0s while the video runs to 5.0s",
          _rms(wav, 4.3, 4.9) < 5.0)

    # The CLI path: same timeline through `eh anime-render`.
    from typer.testing import CliRunner

    from entertainment_harness.cli import app

    result = CliRunner().invoke(app, ["anime-render", str(timeline_path)])
    check("`eh anime-render` exits 0 and prints the output path",
          result.exit_code == 0 and "out.mp4" in result.output,
          result.output.strip().splitlines()[-1] if result.output else "no output")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="keep eval artifacts here (default: a temp dir, deleted after)",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("FATAL: this eval needs ffmpeg and ffprobe on PATH.")
        return 1

    if args.out is not None:
        root = args.out.expanduser().resolve()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        scenario_exact(root / "exact")
        scenario_demo(root / "demo")
        print(f"artifacts kept under {root}")
    else:
        with tempfile.TemporaryDirectory(prefix="eh-eval-") as tmp:
            root = Path(tmp)
            scenario_exact(root / "exact")
            scenario_demo(root / "demo")

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    print(f"== summary: {passed}/{total} checks passed ==")
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    if passed == total:
        print("EVAL PASS")
        return 0
    print("EVAL FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
