"""Sequence-animator tests: expansion + chaining, content-addressed caching,
the drift critic (retry/truncate), and config. No live API."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from entertainment_harness.config import Config, parse_config
from entertainment_harness.video.frames import get_animator
from entertainment_harness.video.script import Segment
from entertainment_harness.video.sequence import (
    RunwaySequenceAnimator,
    drift_critic,
    frame_count,
    video_ratio,
)


def _write_image(path: Path, size: tuple[int, int] = (800, 600)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


class FakeRunwayClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def text_to_image(self, prompt, references, *, model, ratio, ref_ratio, dest):
        self.calls.append({
            "prompt": prompt, "references": references, "model": model,
            "ratio": ratio, "ref_ratio": ref_ratio, "dest": dest,
        })
        _write_image(dest, (1920, 1080))
        return dest


def _animator(monkeypatch, tmp_path) -> tuple[RunwaySequenceAnimator, FakeRunwayClient]:
    config = Config()
    config.video_gen.runway.api_key = "test-key"
    animator = RunwaySequenceAnimator(config)
    client = FakeRunwayClient()
    monkeypatch.setattr(animator, "client", client)
    return animator, client


# --- count + ratio --------------------------------------------------------------


def test_frame_count_scales_with_fps_and_respects_bounds():
    assert frame_count(1.0) == 2  # expansion + at least one chained frame
    assert frame_count(4.0) == 6  # 4s * 1.5fps
    assert frame_count(60.0) == 12  # cost cap
    assert frame_count(60.0, max_frames=20) == 20
    assert frame_count(4.0, fps=1.0) == 4


def test_video_ratio_matches_resolution():
    assert video_ratio("1920x1080") == "1920:1080"
    assert video_ratio("1080x1920") == "1080:1920"
    assert video_ratio("garbage") == "1920:1080"  # 16:9 fallback


# --- generation semantics ---------------------------------------------------------


def test_sequence_expands_then_chains(monkeypatch, tmp_path):
    animator, client = _animator(monkeypatch, tmp_path)
    panel = _write_image(tmp_path / "panel.png")
    seg = Segment(index=2, text="Shin charges forward.")
    frames = animator.generate_frames(panel, seg, 4.0, tmp_path)  # -> 6 frames

    assert len(frames) == 6
    assert all(f.parent == tmp_path / "sequence" for f in frames)
    expand, *chained = client.calls
    # Frame 0: outpainting expansion with only the panel as reference.
    assert expand["references"] == [("panel", panel)]
    assert "Outpaint" in expand["prompt"]
    assert expand["ratio"] == "1920:1080"
    # Frames 1..N: chained off the previous frame, anchored to the panel.
    for i, call in enumerate(chained, start=1):
        assert call["references"][0] == ("prev", frames[i - 1])
        assert call["references"][1] == ("panel", panel)
        assert f"frame {i + 1} of 6" in call["prompt"]


def test_sequence_frames_are_cached(monkeypatch, tmp_path):
    animator, client = _animator(monkeypatch, tmp_path)
    panel = _write_image(tmp_path / "panel.png")
    seg = Segment(index=0, text="t")
    first = animator.generate_frames(panel, seg, 4.0, tmp_path)
    assert len(client.calls) == 6
    second = animator.generate_frames(panel, seg, 4.0, tmp_path)
    assert second == first
    assert len(client.calls) == 6  # all cache hits


# --- drift critic ------------------------------------------------------------------


def test_critic_feedback_triggers_one_retry(monkeypatch, tmp_path):
    animator, client = _animator(monkeypatch, tmp_path)
    panel = _write_image(tmp_path / "panel.png")
    calls = {"n": 0}

    def flaky_critic(prev: Path, new: Path) -> str | None:
        calls["n"] += 1
        if calls["n"] == 1:  # reject the first attempt, accept the retry
            return "the hair changed color"
        return None

    frames = animator.generate_frames(
        panel, Segment(index=1, text="t"), 2.0, tmp_path, critic=flaky_critic
    )
    assert len(frames) == 3  # 2s * 1.5fps
    retry_prompts = [c["prompt"] for c in client.calls if "drifted" in c["prompt"]]
    assert len(retry_prompts) == 1
    assert "the hair changed color" in retry_prompts[0]


def test_persistent_drift_truncates_the_chain(monkeypatch, tmp_path):
    animator, client = _animator(monkeypatch, tmp_path)
    panel = _write_image(tmp_path / "panel.png")
    logs: list[str] = []
    frames = animator.generate_frames(
        panel, Segment(index=1, text="t"), 6.0, tmp_path,
        log=logs.append, critic=lambda p, n: "always drifted",
    )
    assert frames == [frames[0]]  # only the expansion survives
    assert len(frames) == 1
    assert any("truncating the chain" in m for m in logs)


def test_cached_frames_are_recritiqued_when_critic_on(monkeypatch, tmp_path):
    animator, client = _animator(monkeypatch, tmp_path)
    panel = _write_image(tmp_path / "panel.png")
    seg = Segment(index=1, text="t")
    animator.generate_frames(panel, seg, 2.0, tmp_path)  # cached, no critic
    calls_before = len(client.calls)

    calls = {"n": 0}

    def reject_first(prev: Path, new: Path) -> str | None:
        calls["n"] += 1
        if calls["n"] == 1:  # cached first attempt rejected; retry accepted
            return "drift"
        return None

    frames = animator.generate_frames(
        panel, seg, 2.0, tmp_path, critic=reject_first
    )
    assert len(frames) == 3
    # The cached first attempt was re-critiqued and rejected; the retry was
    # a fresh generation, then accepted.
    assert len(client.calls) == calls_before + 1


def test_drift_critic_parses_verdicts():
    class FakeAdapter:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        def generate(self, model, prompt, images=None):
            return self.reply

    assert drift_critic(FakeAdapter("OK"), "m")(Path("a"), Path("b")) is None
    assert drift_critic(FakeAdapter("ok, looks fine"), "m")(Path("a"), Path("b")) is None
    assert (
        drift_critic(FakeAdapter("DRIFT: the face melted"), "m")(Path("a"), Path("b"))
        == "the face melted"
    )
    assert "composition" in drift_critic(FakeAdapter("garbage"), "m")(Path("a"), Path("b"))


# --- registry + config --------------------------------------------------------------


def test_sequence_animator_registered():
    animator_cls = type(get_animator("sequence", _config_with_key()))
    assert animator_cls is RunwaySequenceAnimator
    assert RunwaySequenceAnimator.capabilities == frozenset(
        {"frame-sequence", "image-gen"}
    )


def _config_with_key() -> Config:
    config = Config()
    config.video_gen.runway.api_key = "test-key"
    return config


def test_sequence_config_parsed():
    config = parse_config({})
    assert config.sequence.provider == "sequence"
    assert config.sequence.fps == 1.5
    assert config.sequence.max_frames == 12
    assert config.sequence.interp_fps == 30
    assert config.sequence.critic is True
    overridden = parse_config({
        "sequence": {"fps": 2.0, "max_frames": 8, "critic": False}
    })
    assert overridden.sequence.fps == 2.0
    assert overridden.sequence.max_frames == 8
    assert overridden.sequence.critic is False


# --- openrouter-sequence animator -------------------------------------------------


def _or_config() -> Config:
    config = Config()
    config.models.openai_compat.api_key = "test-or-key"
    return config


def test_openrouter_sequence_registered_and_resolves_model():
    from entertainment_harness.video.gen.openrouter import DEFAULT_IMAGE_MODEL
    from entertainment_harness.video.sequence import OpenRouterSequenceAnimator

    animator = get_animator("openrouter-sequence", _or_config())
    assert isinstance(animator, OpenRouterSequenceAnimator)
    assert animator.capabilities == frozenset({"frame-sequence", "image-gen"})
    assert animator.model == DEFAULT_IMAGE_MODEL  # empty config = default

    config = _or_config()
    config.sequence.model = "gpt-6-luna"
    assert OpenRouterSequenceAnimator(config).model == "gpt-6-luna"


def test_openrouter_sequence_shares_the_loop(monkeypatch, tmp_path):
    animator = get_animator("openrouter-sequence", _or_config())
    client = FakeRunwayClient()  # same duck-typed client interface
    monkeypatch.setattr(animator, "client", client)
    panel = _write_image(tmp_path / "panel.png")
    frames = animator.generate_frames(
        panel, Segment(index=0, text="Shin charges forward."), 4.0, tmp_path
    )
    assert len(frames) == 6
    expand, *chained = client.calls
    assert "Outpaint" in expand["prompt"]
    assert expand["references"] == [("panel", panel)]
    for i, call in enumerate(chained, start=1):
        assert call["references"][0] == ("prev", frames[i - 1])


def test_runway_sequence_missing_key_points_at_openrouter(monkeypatch):
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    from entertainment_harness.video.script import VideoError

    with pytest.raises(VideoError, match="openrouter-sequence"):
        RunwaySequenceAnimator(Config())
