"""Stage 2: narration. TTS engines register in the shared plugin registry
(plugins.py): Kokoro-82M (default, via kokoro-onnx — no torch/spacy needed),
macOS `say` (zero-dependency fallback), and Qwen3-TTS voice cloning via the
optional `qwen3` extra (mlx-audio; voices registered with `eh voices`).

Audio timing is authoritative: segment durations are read back from the
rendered WAV files, and the visuals/assembly stages conform to them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import httpx

from entertainment_harness.config import Config, cache_dir
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginRegistry
from entertainment_harness.video.script import Segment, VideoError

def _espeak_data_path() -> str | None:
    """espeak-ng stores the data path in a fixed ~160-byte buffer; a longer
    path is silently dropped in favor of the (bogus, build-machine) one
    compiled into the dylib, and espeak then exit(1)s. Deep install
    locations — like inside the packaged .app bundle — blow past that
    buffer, so hand espeak a short symlink instead."""
    try:
        import espeakng_loader
    except ImportError:
        return None
    path = espeakng_loader.get_data_path()
    if len(os.fsencode(path)) < 150:
        return path
    link = Path(tempfile.gettempdir()) / "eh-espeak-ng-data"
    try:
        if link.is_symlink() and os.path.realpath(link) != os.path.realpath(path):
            link.unlink()
        if not link.exists():
            os.symlink(path, link)
    except OSError:
        return path
    return str(link)


# The espeak-ng dylib bundled by espeakng-loader has a build-machine data path
# compiled in; left alone, espeak tries to read its phoneme tables from that
# nonexistent directory and the process dies. Pin the data dir that ships
# inside the package via espeak-ng's own env-var fallback so every espeak
# consumer in this process resolves it (the packaged backend included).
try:
    _data_path = _espeak_data_path()
    if _data_path is not None:
        os.environ.setdefault("ESPEAK_DATA_PATH", _data_path)
except Exception:
    pass

KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

SAY_SAMPLE_RATE = 44100


class TTSEngine(Protocol):
    name: str
    default_voice: str
    capabilities: frozenset[str]  # e.g. {"tts"}, {"tts", "voice-clone"}

    def synthesize(self, text: str, voice: str, dest: Path) -> None:
        """Render text to a WAV file at dest."""
        ...


class SayEngine:
    """macOS `say` — always available on a Mac, no downloads."""

    name = "say"
    default_voice = "Samantha"
    capabilities = frozenset({"tts"})

    def __init__(self, config: Config | None = None) -> None:
        pass  # contract: (config, **overrides); no [tts.say] section yet

    def synthesize(self, text: str, voice: str, dest: Path) -> None:
        if shutil.which("say") is None:
            raise VideoError("The 'say' TTS engine requires macOS")
        result = subprocess.run(
            [
                "say", "-v", voice, "-o", str(dest),
                f"--data-format=LEI16@{SAY_SAMPLE_RATE}", text,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not dest.exists():
            hint = result.stderr.strip() or "unknown error"
            raise VideoError(
                f"say failed for voice {voice!r}: {hint}"
                " (list voices with: say -v '?')"
            )


class KokoroEngine:
    """Kokoro-82M via kokoro-onnx. Model files download on first use."""

    name = "kokoro"
    default_voice = "af_heart"
    capabilities = frozenset({"tts"})

    def __init__(
        self, config: Config | None = None, models_dir: Path | None = None
    ) -> None:
        self._models_dir = models_dir or cache_dir() / "tts"
        self._kokoro = None  # lazy: loading the model takes seconds

    def _ensure_model_files(self) -> tuple[Path, Path]:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for filename, url in KOKORO_FILES.items():
            dest = self._models_dir / filename
            if not dest.exists():
                with httpx.stream("GET", url, follow_redirects=True,
                                  timeout=300.0) as resp:
                    resp.raise_for_status()
                    with dest.open("wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
            paths[filename] = dest
        return paths["kokoro-v1.0.onnx"], paths["voices-v1.0.bin"]

    def _load(self):
        if self._kokoro is None:
            try:
                import espeakng_loader
                from kokoro_onnx import Kokoro
                from kokoro_onnx.config import EspeakConfig
            except ImportError as exc:
                raise VideoError(
                    "kokoro-onnx is not installed; use --tts-engine say"
                ) from exc
            model_path, voices_path = self._ensure_model_files()
            # Pin the real data dir explicitly (the module-level
            # ESPEAK_DATA_PATH covers consumers that initialize espeak without
            # a path) so phonemizer can't fall back to the compiled-in one.
            data_path = _espeak_data_path()
            if data_path is None:
                raise VideoError(
                    "espeakng-loader is not installed; use --tts-engine say"
                )
            os.environ.setdefault("ESPEAK_DATA_PATH", data_path)
            self._kokoro = Kokoro(
                str(model_path),
                str(voices_path),
                espeak_config=EspeakConfig(
                    data_path=data_path,
                    lib_path=espeakng_loader.get_library_path(),
                ),
            )
        return self._kokoro

    def synthesize(self, text: str, voice: str, dest: Path) -> None:
        import numpy as np

        kokoro = self._load()
        try:
            samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0,
                                                 lang="en-us")
        except Exception as exc:
            raise VideoError(f"kokoro failed for voice {voice!r}: {exc}") from exc
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(dest), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())


REGISTRY = PluginRegistry("tts", ENTRY_POINT_GROUPS["tts"])
REGISTRY.register("kokoro", KokoroEngine)
REGISTRY.register("say", SayEngine)
# Lazy dotted string: keeps the mlx_audio import (qwen3 extra) out of CLI
# startup; Qwen3TTSCloneEngine clones voices from the `eh voices` library.
REGISTRY.register(
    "qwen3", "entertainment_harness.video.tts_qwen3:Qwen3TTSCloneEngine"
)


def get_engine(name: str, config: Config | None = None) -> TTSEngine:
    # Typed-dict settings still apply as explicit overrides; create()
    # additionally merges the raw [tts.<name>] table for entry-point engines.
    overrides = config.tts.get(name, {}) if config is not None else {}
    return REGISTRY.create(name, config, **overrides)


def wav_duration(path: Path) -> float:
    """Duration in seconds, read from the rendered file (authoritative)."""
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def render_narration(
    segments: list[Segment],
    engine: TTSEngine,
    voice: str,
    workdir: Path,
    log=lambda m: None,
) -> bool:
    """Render each segment to seg-NN.wav (skipping existing files) and stamp
    segment.duration_s from the rendered audio.

    Returns True if any segment was (re)rendered — callers must invalidate
    downstream artifacts (clips, out.mp4), since audio timing is authoritative.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    rendered_any = False
    for seg in segments:
        dest = workdir / f"seg-{seg.index:02d}.wav"
        if dest.exists():
            log(f"  narration seg-{seg.index:02d}: cached")
        else:
            engine.synthesize(seg.text, voice, dest)
            rendered_any = True
            log(f"  narration seg-{seg.index:02d}: rendered")
        seg.duration_s = wav_duration(dest)
    return rendered_any
