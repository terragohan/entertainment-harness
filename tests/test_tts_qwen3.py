"""Qwen3-TTS clone engine tests: registry wiring, config overrides, and
synthesize via a fake mlx-style model. No mlx-audio install or downloads —
the lazy _load() is short-circuited with a fake."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from entertainment_harness.config import Config, load_config
from entertainment_harness.video import voices
from entertainment_harness.video.script import VideoError
from entertainment_harness.video.tts import REGISTRY, get_engine
from entertainment_harness.video.tts_qwen3 import Qwen3TTSCloneEngine


def _write_wav(path: Path, seconds: float = 4.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


class FakeResult:
    audio = np.array([0.0, 0.5, -0.5, 1.0] * 100, dtype=np.float32)
    sample_rate = 24000


class FakeModel:
    """Records generate() kwargs; returns one canned result."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return iter([FakeResult()])


@pytest.fixture
def engine(tmp_path):
    ref = tmp_path / "ref.wav"
    _write_wav(ref)
    voices.add_voice("narrator", ref, "Reference words.", base=tmp_path)
    eng = Qwen3TTSCloneEngine(voices_base=tmp_path)
    eng._model = FakeModel()
    return eng


def test_registry_lists_qwen3():
    assert "qwen3" in REGISTRY.names()


def test_get_engine_applies_tts_config_section():
    config = Config()
    config.tts = {"qwen3": {"model": "custom-model", "language": "English"}}
    eng = get_engine("qwen3", config)
    assert isinstance(eng, Qwen3TTSCloneEngine)
    assert eng._model_name == "custom-model"
    assert eng._language == "English"


def test_load_config_parses_tts_tables(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[tts.qwen3]\nmodel = "mlx-community/custom"\nlanguage = "Japanese"\n'
    )
    config = load_config(tmp_path / "config.toml")
    assert config.tts["qwen3"] == {"model": "mlx-community/custom",
                                   "language": "Japanese"}


def test_synthesize_writes_wav_in_icl_mode(engine, tmp_path):
    dest = tmp_path / "out.wav"
    engine.synthesize("New line.", "narrator", dest)
    with wave.open(str(dest), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 400
    call = engine._model.calls[0]
    assert call["text"] == "New line."
    assert call["ref_text"] == "Reference words."
    assert call["x_vector_only_mode"] is False
    assert call["ref_audio"].endswith("narrator.wav")
    assert call["max_new_tokens"] == 2048


def test_synthesize_uses_x_vector_without_transcript(engine, tmp_path):
    ref = tmp_path / "bare.wav"
    _write_wav(ref)
    voices.add_voice("bare", ref, x_vector=True, base=tmp_path)
    dest = tmp_path / "out.wav"
    engine.synthesize("Hi.", "bare", dest)
    call = engine._model.calls[0]
    assert call["ref_text"] is None
    assert call["x_vector_only_mode"] is True


def test_synthesize_rejects_empty_voice(engine, tmp_path):
    with pytest.raises(VideoError, match="--voice NAME"):
        engine.synthesize("Hi.", "", tmp_path / "out.wav")


def test_synthesize_unknown_voice_lists_registered(engine, tmp_path):
    with pytest.raises(VideoError, match="eh voices add ghost"):
        engine.synthesize("Hi.", "ghost", tmp_path / "out.wav")


def test_synthesize_wraps_model_failures(engine, tmp_path, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("mlx exploded")

    engine._model.generate = boom
    with pytest.raises(VideoError, match="qwen3 cloning failed"):
        engine.synthesize("Hi.", "narrator", tmp_path / "out.wav")


def test_engine_module_imports_without_mlx_audio():
    """The lazy dotted registration must not import mlx_audio at CLI startup."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import entertainment_harness.video.tts_qwen3;"
         "assert 'mlx_audio' not in __import__('sys').modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
