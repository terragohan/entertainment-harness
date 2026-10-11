"""Tests for the compositor and the Sequencing Renderer (animation-conductor
initiative, Phase 1).

Compositor and frame-loop tests are pure Pillow — tiny resolutions, exact
pixel assertions. End-to-end MP4/audio tests use the real ffmpeg/ffprobe and
are guarded by shutil.which, like tests/test_anime_scene.py. No test here
touches the network, a model, or a paid API.
"""

from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from entertainment_harness.anime.compositor import Compositor
from entertainment_harness.anime.render import (
    FRAME_NAME,
    render_frames,
    render_timeline,
)
from entertainment_harness.anime.sequencer import state_at
from entertainment_harness.anime.timeline import (
    AssetEvent,
    AudioClip,
    CameraKey,
    Layer,
    Timeline,
    TimelineError,
    save_timeline,
)
from entertainment_harness.cli import app
from entertainment_harness.video.assemble import probe_format, video_duration

RED = (200, 30, 30)
BLUE = (30, 30, 200)
GREEN = (30, 180, 60)

runner = CliRunner()


def _solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGBA", size, (*color, 255)).save(path)


def _half_bg(path: Path, width: int, height: int) -> None:
    """Background asset: red left half, blue right half — camera pans across
    the boundary are exact pixel checks."""
    img = Image.new("RGBA", (width, height), (*RED, 255))
    for x in range(width // 2, width):
        for y in range(height):
            img.putpixel((x, y), (*BLUE, 255))
    img.save(path)


def _tone_wav(path: Path, seconds: float = 1.0, freq: float = 440.0,
              rate: int = 22050) -> None:
    frames = bytearray()
    for i in range(int(rate * seconds)):
        sample = 0.5 * math.sin(2 * math.pi * freq * i / rate)
        frames += int(sample * 32767).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))


def _timeline(**overrides) -> Timeline:
    fields = dict(
        fps=24, width=64, height=36, duration_frames=24,
        assets={}, layers=[], camera=[], events=[], audio=[],
    )
    fields.update(overrides)
    return Timeline(**fields)


# --- Compositor ---------------------------------------------------------------


def _studio(tmp_path: Path) -> tuple[Timeline, Compositor]:
    """64x36 canvas: full-frame background plus a centered 50%-height box."""
    _solid(tmp_path / "bg.png", (64, 36), BLUE)
    _solid(tmp_path / "red.png", (20, 20), RED)
    _solid(tmp_path / "green.png", (20, 20), GREEN)
    timeline = _timeline(
        assets={
            "bg": "bg.png", "red": "red.png", "green": "green.png",
        },
        layers=[
            Layer(id="bg", asset="bg"),
            Layer(id="box", asset="red", x=0.5, y=0.5, scale=0.5),
        ],
    )
    return timeline, Compositor(timeline, tmp_path)


def test_compositor_draws_layers_in_order_at_exact_size(tmp_path):
    timeline, compositor = _studio(tmp_path)
    img = compositor.render(state_at(timeline, 0))
    assert img.size == (64, 36)
    # box: scale 0.5 -> 18px tall, 20 wide, centered at (32, 18)
    assert img.getpixel((32, 18)) == RED
    assert img.getpixel((2, 2)) == BLUE       # uncovered corner = bg
    assert img.getpixel((32, 2)) == BLUE      # above the box


def test_compositor_asset_swap_changes_exact_pixels(tmp_path):
    timeline, compositor = _studio(tmp_path)
    timeline.events = [AssetEvent(frame=12, target="box", asset="green")]
    before = compositor.render(state_at(timeline, 11))
    after = compositor.render(state_at(timeline, 12))
    assert before.getpixel((32, 18)) == RED
    assert after.getpixel((32, 18)) == GREEN
    assert after.getpixel((2, 2)) == BLUE     # background untouched


def test_compositor_camera_zoom_and_pan(tmp_path):
    _half_bg(tmp_path / "bg.png", 64, 36)
    timeline = _timeline(
        width=64, height=36, duration_frames=100,
        assets={"bg": "bg.png"},
        layers=[Layer(id="bg", asset="bg")],
        camera=[
            CameraKey(frame=0, x=0.25, y=0.5, scale=2.0),
            CameraKey(frame=99, x=0.75, y=0.5, scale=2.0),
        ],
    )
    compositor = Compositor(timeline, tmp_path)
    r, _, b = compositor.render(state_at(timeline, 0)).getpixel((32, 18))
    assert r > 150 and b < 100       # zoomed into the red half
    r, _, b = compositor.render(state_at(timeline, 99)).getpixel((32, 18))
    assert b > 150 and r < 100       # panned into the blue half
    # every frame is exactly the output size, however the camera moves
    for frame in (0, 33, 66, 99):
        assert compositor.render(state_at(timeline, frame)).size == (64, 36)


def test_compositor_missing_asset_fails_before_rendering(tmp_path):
    timeline = _timeline(
        assets={"bg": "no-such-file.png"}, layers=[Layer(id="bg", asset="bg")]
    )
    with pytest.raises(TimelineError, match="cannot be loaded"):
        Compositor(timeline, tmp_path)


def test_compositor_transparent_sprite_composites_over_background(tmp_path):
    _solid(tmp_path / "bg.png", (64, 36), BLUE)
    sprite = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    for x in range(5):
        for y in range(10):
            sprite.putpixel((x, y), (*RED, 255))  # left half opaque
    sprite.save(tmp_path / "sprite.png")
    timeline = _timeline(
        assets={"bg": "bg.png", "sprite": "sprite.png"},
        layers=[
            Layer(id="bg", asset="bg"),
            Layer(id="sprite", asset="sprite", x=0.5, y=0.5, scale=0.5),
        ],
    )
    img = Compositor(timeline, tmp_path).render(state_at(timeline, 0))
    # sprite is 18px tall -> 20 wide, centered: spans x 22..41, left half red
    assert img.getpixel((24, 18))[0] > 150          # opaque half
    assert img.getpixel((40, 18))[2] > 150          # transparent half shows bg


# --- Frame loop -----------------------------------------------------------------


def test_render_frames_produces_exactly_duration_frames(tmp_path):
    timeline, compositor = _studio(tmp_path)
    frames = render_frames(timeline, compositor, tmp_path / "frames")
    assert len(frames) == 24
    names = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert names == [FRAME_NAME % i for i in range(24)]
    assert not (tmp_path / "frames" / (FRAME_NAME % 24)).exists()
    with Image.open(frames[0]) as img:
        assert img.size == (64, 36)


def test_render_frames_removes_stale_frames_from_longer_renders(tmp_path):
    timeline, compositor = _studio(tmp_path)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    stale = frames_dir / (FRAME_NAME % 30)
    stale.write_bytes(b"leftover")
    render_frames(timeline, compositor, frames_dir)
    assert not stale.exists()


def test_rerender_is_byte_deterministic(tmp_path):
    timeline, compositor = _studio(tmp_path)
    first = render_frames(timeline, compositor, tmp_path / "a")
    second = render_frames(timeline, compositor, tmp_path / "b")
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes(), a.name


# --- End-to-end MP4 + audio (real ffmpeg) ---------------------------------------


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _rms(path: Path, start_s: float, end_s: float) -> float:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        wav.setpos(round(start_s * rate))
        raw = wav.readframes(round((end_s - start_s) * rate))
    samples = array.array("h", raw)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


@pytest.fixture
def render_dir(tmp_path):
    """A 4 s / 24 fps / 96-frame timeline: half-red/half-blue background,
    camera panning across it, a box swapping color at frame 48, and a 1 s
    tone starting at frame 48 (= 2.0 s)."""
    _half_bg(tmp_path / "bg.png", 160, 90)
    _solid(tmp_path / "red.png", (30, 30), RED)
    _solid(tmp_path / "green.png", (30, 30), GREEN)
    _tone_wav(tmp_path / "tone.wav")
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
    path = tmp_path / "timeline.json"
    save_timeline(timeline, path)
    return path


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_render_timeline_exact_frames_fps_and_duration(render_dir):
    out = render_timeline(render_dir, log=lambda m: None)
    workdir = render_dir.parent / "render"
    frames = sorted((workdir / "frames").glob("frame-*.png"))
    assert [p.name for p in frames] == [FRAME_NAME % i for i in range(96)]
    codec, width, height, fps = probe_format(out)
    assert (codec, width, height) == ("h264", 160, 90)
    assert fps == "24/1"
    # 96 frames at 24 fps = exactly 4 s; the 1 s tone must not stretch it.
    assert video_duration(out) == pytest.approx(4.0, abs=0.1)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_render_timeline_places_audio_at_the_exact_frame(render_dir):
    out = render_timeline(render_dir, log=lambda m: None)
    wav = render_dir.parent / "render" / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out), "-ac", "1", "-ar", "8000",
         "-f", "wav", str(wav)],
        capture_output=True, check=True,
    )
    # frame 48 at 24 fps -> the tone begins at 2.0 s.
    assert _rms(wav, 0.4, 1.6) < 5.0        # silent before
    assert _rms(wav, 2.2, 2.9) > 1000.0     # tone playing
    assert _rms(wav, 3.4, 3.9) < 5.0        # 1 s tone over, video continues


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_render_timeline_camera_moves_and_event_swaps_pixels(render_dir):
    render_timeline(render_dir, log=lambda m: None)
    frames = render_dir.parent / "render" / "frames"

    def center(frame: int) -> tuple[int, int, int, int]:
        with Image.open(frames / (FRAME_NAME % frame)) as img:
            return img.getpixel((80, 45))

    assert center(0)[0] > center(0)[2]      # red half of the background
    assert center(95)[2] > center(95)[0]    # camera panned to the blue half
    # the box sits at y 0.6 (row 54); above it only the background shows,
    # so the swap at frame 48 is visible there via the box, not the sky.
    def box_pixel(frame: int) -> tuple[int, int, int, int]:
        with Image.open(frames / (FRAME_NAME % frame)) as img:
            return img.getpixel((80, 54))

    assert box_pixel(47)[:3] == RED
    assert box_pixel(48)[:3] == GREEN


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_render_timeline_without_audio_still_produces_valid_mp4(tmp_path):
    _solid(tmp_path / "bg.png", (64, 36), BLUE)
    timeline = _timeline(
        assets={"bg": "bg.png"}, layers=[Layer(id="bg", asset="bg")],
    )
    path = tmp_path / "timeline.json"
    save_timeline(timeline, path)
    out = render_timeline(path, log=lambda m: None)
    codec, width, height, fps = probe_format(out)
    assert (codec, width, height, fps) == ("h264", 64, 36, "24/1")
    assert video_duration(out) == pytest.approx(1.0, abs=0.1)


# --- CLI ------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_anime_render_cli_renders_and_prints_output(tmp_path):
    _solid(tmp_path / "bg.png", (64, 36), BLUE)
    timeline = _timeline(
        assets={"bg": "bg.png"}, layers=[Layer(id="bg", asset="bg")],
    )
    path = tmp_path / "timeline.json"
    save_timeline(timeline, path)
    result = runner.invoke(app, ["anime-render", str(path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "render" / "out.mp4").is_file()
    assert "out.mp4" in result.output


def test_anime_render_cli_rejects_invalid_timeline_before_rendering(tmp_path):
    payload = _timeline(fps=0).to_dict()
    path = tmp_path / "timeline.json"
    path.write_text(json.dumps(payload))
    result = runner.invoke(app, ["anime-render", str(path)])
    assert result.exit_code == 1
    assert "fps must be > 0" in result.output
    assert not (tmp_path / "render").exists()


def test_anime_render_cli_reports_missing_file(tmp_path):
    result = runner.invoke(app, ["anime-render", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "Cannot read" in result.output
