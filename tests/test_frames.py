"""Frame-animator tests: ratio picking, content-addressed caching, the runway
client contract (mocked — no live API), and the local fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from entertainment_harness.config import Config, load_config
from entertainment_harness.video.frames import (
    LocalFrameAnimator,
    RunwayFrameAnimator,
    frame_count,
    get_animator,
    nearest_ratio,
    resolve_image_model,
)
from entertainment_harness.video.script import Segment, VideoError


def _write_image(path: Path, size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def test_frame_count_scales_with_share_and_respects_bounds():
    assert frame_count(1.0) == 2  # a single frame cannot animate
    assert frame_count(4.0) == 2
    assert frame_count(8.0) == 4
    assert frame_count(60.0) == 6  # cost cap
    assert frame_count(60.0, max_frames=10) == 10
    assert frame_count(9.0, seconds_per_frame=3.0) == 3


def test_nearest_ratio_picks_closest_supported(tmp_path):
    wide = _write_image(tmp_path / "wide.png", (1600, 900))
    tall = _write_image(tmp_path / "tall.png", (900, 1600))
    square = _write_image(tmp_path / "sq.png", (1000, 1000))
    assert nearest_ratio(wide) == "1920:1080"
    assert nearest_ratio(tall) == "1080:1920"
    assert nearest_ratio(square) in {"1024:1024", "1080:1080", "720:720"}


def test_local_animator_is_not_animated_and_raises():
    animator = get_animator("local", Config())
    assert isinstance(animator, LocalFrameAnimator)
    assert animator.animated is False
    with pytest.raises(VideoError, match="local frame animator"):
        animator.generate_frames(
            Path("x.png"), Segment(index=0, text="t"), 4.0, Path("/tmp")
        )


class FakeRunwayClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def text_to_image(self, prompt, references, *, model, ratio, ref_ratio,
                      dest):
        self.calls.append({
            "prompt": prompt, "references": references, "model": model,
            "ratio": ratio, "ref_ratio": ref_ratio, "dest": dest,
        })
        dest.write_bytes(b"frame")
        return dest


def _runway_animator(monkeypatch, tmp_path) -> tuple[RunwayFrameAnimator, FakeRunwayClient]:
    config = Config()
    config.video_gen.runway.api_key = "test-key"
    animator = RunwayFrameAnimator(config)
    client = FakeRunwayClient()
    monkeypatch.setattr(animator, "client", client)
    return animator, client


def test_runway_animator_generates_motion_phase_frames(monkeypatch, tmp_path):
    animator, client = _runway_animator(monkeypatch, tmp_path)
    crop = _write_image(tmp_path / "panel.png", (800, 600))
    seg = Segment(index=3, text="Shin charges forward.")
    frames = animator.generate_frames(crop, seg, 6.0, tmp_path)  # -> 3 frames
    assert len(frames) == 3
    assert all(f.parent == tmp_path / "frames" for f in frames)
    assert len(client.calls) == 3
    for i, call in enumerate(client.calls):
        assert call["references"] == [("panel", crop)]  # the panel is the ref
        assert call["model"] == "gen4_image"
        assert f"frame {i + 1} of 3" in call["prompt"]  # motion phases
        assert "Shin charges forward." in call["prompt"]
        assert call["ratio"] == "1440:1080"  # 800/600 is exactly 4:3
        assert call["ref_ratio"] == (800, 600)  # the panel is never cropped


def test_runway_animator_caches_frames_by_content(monkeypatch, tmp_path):
    animator, client = _runway_animator(monkeypatch, tmp_path)
    crop = _write_image(tmp_path / "panel.png", (800, 600))
    seg = Segment(index=0, text="Same beat.")
    first = animator.generate_frames(crop, seg, 4.0, tmp_path)  # -> 2 frames
    assert len(client.calls) == 2
    again = animator.generate_frames(crop, seg, 4.0, tmp_path)
    assert again == first
    assert len(client.calls) == 2  # all cached
    # different segment text -> different prompt -> new frames
    other = animator.generate_frames(
        crop, Segment(index=1, text="Another beat."), 4.0, tmp_path
    )
    assert other != first
    assert len(client.calls) == 4


def test_load_config_parses_frames_section(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[frames]\nprovider = "runway"\nmodel = "gen4_image"\n'
        "seconds_per_frame = 1.5\nmax_frames = 4\n"
    )
    config = load_config()
    assert config.frames.provider == "runway"
    assert config.frames.model == "gen4_image"
    assert config.frames.seconds_per_frame == 1.5
    assert config.frames.max_frames == 4


def test_frames_plugin_category_is_registered():
    from entertainment_harness.plugins import ENTRY_POINT_GROUPS

    assert ENTRY_POINT_GROUPS["frames"] == "entertainment_harness.frames"


# --- openrouter animator --------------------------------------------------------


def _openrouter_config() -> Config:
    config = Config()
    config.models.openai_compat.api_key = "test-or-key"
    return config


def test_openrouter_animator_registers_and_resolves_model(monkeypatch, tmp_path):
    from entertainment_harness.video.frames import OpenRouterFrameAnimator
    from entertainment_harness.video.gen.openrouter import DEFAULT_IMAGE_MODEL

    animator = get_animator("openrouter", _openrouter_config())
    assert isinstance(animator, OpenRouterFrameAnimator)
    assert animator.animated and "image-gen" in animator.capabilities
    assert animator.model == DEFAULT_IMAGE_MODEL  # empty config = default

    config = _openrouter_config()
    config.frames.model = "gpt-6-luna"
    assert OpenRouterFrameAnimator(config).model == "gpt-6-luna"


def test_openrouter_animator_shares_the_generation_loop(monkeypatch, tmp_path):
    animator = get_animator("openrouter", _openrouter_config())
    client = FakeRunwayClient()  # same duck-typed client interface
    monkeypatch.setattr(animator, "client", client)
    crop = _write_image(tmp_path / "panel.png", (800, 600))
    frames = animator.generate_frames(
        crop, Segment(index=0, text="Shin charges forward."), 6.0, tmp_path
    )
    assert len(frames) == 3
    assert all(c["references"] == [("panel", crop)] for c in client.calls)


def test_openrouter_animator_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(VideoError, match="api_key"):
        get_animator("openrouter", Config())


def test_runway_animator_missing_key_points_at_openrouter(monkeypatch):
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    with pytest.raises(VideoError, match="openrouter"):
        RunwayFrameAnimator(Config())


def test_resolve_image_model_tracks_defaults_and_explicit_values():
    from entertainment_harness.video.gen.openrouter import DEFAULT_IMAGE_MODEL

    assert resolve_image_model("runway", "") == "gen4_image"
    assert resolve_image_model("sequence", "") == "gen4_image"
    assert resolve_image_model("openrouter", "") == DEFAULT_IMAGE_MODEL
    assert resolve_image_model("openrouter-sequence", "") == DEFAULT_IMAGE_MODEL
    # an explicit [frames]/[sequence] model always wins
    assert resolve_image_model("openrouter-sequence", "x/y") == "x/y"
    assert resolve_image_model("sequence", "gen4_image_turbo") == "gen4_image_turbo"
    # third-party providers key on the raw config value
    assert resolve_image_model("third-party", "") == ""
    assert resolve_image_model("local", "") == ""
