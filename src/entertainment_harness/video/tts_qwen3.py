"""Qwen3-TTS Base voice-cloning engine (via mlx-audio, Apple Silicon).

Clones a voice from the voice library (`video/voices.py`) using Alibaba's
Qwen3-TTS-12Hz Base models (Apache 2.0). Requires the `qwen3` extra:

    uv sync --extra qwen3
"""

from __future__ import annotations

import wave
from pathlib import Path

from entertainment_harness.config import Config, data_dir
from entertainment_harness.video.script import VideoError
from entertainment_harness.video.voices import resolve_voice

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
# Output sample rate of Qwen3-TTS; used only if the result omits it.
FALLBACK_SAMPLE_RATE = 24000


class Qwen3TTSCloneEngine:
    """Zero-shot voice cloning: `voice` is a name from `eh voices list`."""

    name = "qwen3"
    default_voice = ""
    capabilities = frozenset({"tts", "voice-clone"})

    def __init__(
        self,
        config: Config | None = None,
        model: str = DEFAULT_MODEL,
        language: str = "Auto",
        x_vector_only: bool = False,
        max_new_tokens: int = 2048,
        voices_base: Path | None = None,
    ) -> None:
        self._model_name = model
        self._language = language
        self._x_vector_only = x_vector_only
        self._max_new_tokens = max_new_tokens
        self._voices_base = voices_base  # None = data_dir(); tests override
        self._model = None  # lazy: loading takes seconds + GBs of RAM

    def _load(self):
        if self._model is None:
            try:
                from mlx_audio.tts.utils import load_model
            except ImportError as exc:
                raise VideoError(
                    "mlx-audio is not installed; enable qwen3 cloning with:"
                    " uv sync --extra qwen3 (or use --tts-engine kokoro)"
                ) from exc
            self._model = load_model(self._model_name)
        return self._model

    def synthesize(self, text: str, voice: str, dest: Path) -> None:
        import numpy as np

        if not voice:
            raise VideoError(
                "The qwen3 engine clones registered voices; pass --voice NAME"
                " and register one with: eh voices add NAME --audio sample.wav"
                " --text \"...\""
            )
        base = self._voices_base or data_dir()
        ref = resolve_voice(voice, base)
        x_vector_only = self._x_vector_only or ref.transcript is None
        try:
            results = list(
                self._load().generate(
                    text=text,
                    language=self._language,
                    ref_audio=str(ref.path),
                    ref_text=None if x_vector_only else ref.transcript,
                    x_vector_only_mode=x_vector_only,
                    max_new_tokens=self._max_new_tokens,
                )
            )
        except VideoError:
            raise
        except Exception as exc:
            raise VideoError(f"qwen3 cloning failed for voice {voice!r}: {exc}") from exc
        if not results:
            raise VideoError(f"qwen3 produced no audio for voice {voice!r}")
        result = results[0]
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        sample_rate = int(getattr(result, "sample_rate", FALLBACK_SAMPLE_RATE) or FALLBACK_SAMPLE_RATE)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
