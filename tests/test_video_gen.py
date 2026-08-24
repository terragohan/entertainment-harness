"""Tests for the video generation provider registry and providers.

Local provider exercises the real ffmpeg path only when ffmpeg is available;
Runway provider is fully mocked.
"""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest
import respx
from PIL import Image

from entertainment_harness.config import Config
from entertainment_harness.plugins import PluginError
from entertainment_harness.video.gen import REGISTRY, get_provider
from entertainment_harness.video.gen.local import LocalVideoGenProvider
from entertainment_harness.video.gen.runway import RunwayProvider
from entertainment_harness.video.script import Segment, VideoError


@pytest.fixture
def image(tmp_path: Path) -> Path:
    p = tmp_path / "input.png"
    Image.new("RGB", (1080, 1920), "white").save(p)
    return p


def test_registry_has_local_and_runway():
    assert REGISTRY.names() == ["local", "runway"]


def test_animated_capability_flag():
    assert LocalVideoGenProvider.animated is False
    assert RunwayProvider.animated is True


def test_get_provider_unknown_raises():
    with pytest.raises(PluginError, match="Unknown video_gen plugin"):
        get_provider("does-not-exist", Config())


def test_local_provider_uses_existing_ffmpeg_clip(image, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    provider = LocalVideoGenProvider(Config())
    seg = Segment(index=0, text="hello", motion="zoom_in", duration_s=1.0)
    clip = provider.generate_segment(image, seg, 1.0, tmp_path)
    assert clip.exists()


def test_runway_provider_requires_api_key(monkeypatch):
    monkeypatch.setenv("RUNWAY_API_KEY", "")
    config = Config()
    config.video_gen.runway.api_key = ""
    with pytest.raises(VideoError, match="API key"):
        RunwayProvider(config)


def test_runway_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("RUNWAY_API_KEY", "rk_test")
    config = Config()
    config.video_gen.runway.api_key = ""
    provider = RunwayProvider(config)
    assert provider.api_key == "rk_test"


def test_runway_prepare_image_crops_to_vertical(image, tmp_path):
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    provider = RunwayProvider(config)
    data, mime = provider._prepare_image(image)
    assert mime == "image/jpeg"
    out = tmp_path / "out.jpg"
    out.write_bytes(data)
    with Image.open(out) as img:
        assert img.size == (768, 1280)


def test_runway_generate_segment_polls_and_downloads(image, tmp_path, monkeypatch, respx_mock):
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    provider = RunwayProvider(config)

    submitted = {"id": "task-1"}
    completed = {
        "id": "task-1",
        "status": "SUCCEEDED",
        "output": ["https://cdn.runwayml.com/clip.mp4"],
    }

    # Capture submitted payload to verify data URI and aspect ratio
    captured: dict = {}

    def capture_request(request):
        captured["payload"] = request.content.decode()
        return respx.MockResponse(200, json=submitted)

    route_submit = respx_mock.post("https://api.dev.runwayml.com/v1/image_to_video").mock(
        side_effect=capture_request
    )
    route_poll = respx_mock.get("https://api.dev.runwayml.com/v1/tasks/task-1").mock(
        return_value=respx.MockResponse(200, json=completed)
    )
    route_video = respx_mock.get("https://cdn.runwayml.com/clip.mp4").mock(
        return_value=respx.MockResponse(200, content=b"fake-mp4")
    )

    seg = Segment(index=0, text="A mage casts fire", duration_s=5.0)
    clip = provider.generate_segment(image, seg, 5.0, tmp_path)

    assert clip.exists()
    assert clip.read_bytes() == b"fake-mp4"
    assert route_submit.called
    assert route_poll.called
    assert route_video.called
    assert "gen3a_turbo" in captured.get("payload", "")
    assert "768:1280" in captured.get("payload", "")


def test_runway_poll_failure_raises(tmp_path, monkeypatch, respx_mock):
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    provider = RunwayProvider(config)
    monkeypatch.setattr(
        provider, "_prepare_image", lambda image, ratio=(768, 1280): (b"data", "image/jpeg")
    )

    respx_mock.post("https://api.dev.runwayml.com/v1/image_to_video").mock(
        return_value=respx.MockResponse(200, json={"id": "task-2"})
    )
    respx_mock.get("https://api.dev.runwayml.com/v1/tasks/task-2").mock(
        return_value=respx.MockResponse(200, json={"id": "task-2", "status": "FAILED"})
    )

    seg = Segment(index=1, text="boom", duration_s=5.0)
    with pytest.raises(VideoError, match="FAILED"):
        provider.generate_segment(Path(tmp_path / "dummy.png"), seg, 5.0, tmp_path)


def _sized_image(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size, (20, 40, 60)).save(path)
    return path


def test_runway_prepare_image_pads_tall_reference_into_window(tmp_path):
    # A tall manga panel (~0.27 aspect) would 400 the request — Runway
    # requires reference width/height >= 0.5, so pad, never crop.
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    provider = RunwayProvider(config)
    tall = _sized_image(tmp_path / "tall.png", (274, 1000))
    data, _ = provider._prepare_image(tall, (274, 1000))  # own-size ref prep
    out = tmp_path / "out.jpg"
    out.write_bytes(data)
    with Image.open(out) as img:
        assert img.size == (500, 1000)  # padded up to the 0.5 floor


def test_runway_prepare_image_pads_wide_reference_into_window(tmp_path):
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    provider = RunwayProvider(config)
    wide = _sized_image(tmp_path / "wide.png", (3000, 1000))  # 3.0 aspect
    data, _ = provider._prepare_image(wide, (3000, 1000))
    out = tmp_path / "out.jpg"
    out.write_bytes(data)
    with Image.open(out) as img:
        assert img.size == (3000, 1500)  # padded down to the 2.0 ceiling


def test_runway_prepare_image_leaves_in_window_aspect_alone(tmp_path):
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    provider = RunwayProvider(config)
    ok = _sized_image(tmp_path / "ok.png", (800, 1000))  # 0.8 aspect
    data, _ = provider._prepare_image(ok, (800, 1000))
    out = tmp_path / "out.jpg"
    out.write_bytes(data)
    with Image.open(out) as img:
        assert img.size == (800, 1000)
