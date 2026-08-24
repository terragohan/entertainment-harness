"""Voice library for voice-cloning TTS engines.

A voice is a named reference sample: `<name>.wav` (clean, single-speaker,
3+ seconds recommended) plus an optional `<name>.txt` transcript. Voices
with a transcript clone in ICL mode (best quality); voices without one use
x-vector-only mode. Stored under `data_dir()/voices/` (user data, not
re-downloadable cache).
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from entertainment_harness.config import data_dir
from entertainment_harness.video.script import VideoError

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
MIN_REF_SECONDS = 3.0
REF_SAMPLE_RATE = 16000

# Sample lines synthesized during onboarding so the user can judge the clone
# before committing it to a narration. Varied on purpose: neutral narration,
# emotional dialogue with names, and a numbers run (digits stress intonation).
PREVIEW_LINES: tuple[str, ...] = (
    "In the year 2026, humanity finally reached the stars, and found them"
    " waiting.",
    "\"You're telling me the gate is already open?\" Mara said."
    " \"Then we go now, no matter the cost.\"",
    "Shin counted under his breath: one, two, three, four, five, six, seven,"
    " eight, nine, ten.",
)


@dataclass
class VoiceMeta:
    name: str
    path: Path
    duration_s: float
    transcript: str | None  # None = x-vector-only cloning (lower quality)


def voices_dir(base: Path | None = None) -> Path:
    return (base or data_dir()) / "voices"


def _validate_name(name: str) -> None:
    if not NAME_RE.match(name):
        raise VideoError(
            f"Invalid voice name {name!r}: use lowercase letters, digits,"
            " '-' or '_' (e.g. 'narrator-1')"
        )


def _convert_to_wav(src: Path, dest: Path) -> None:
    """Copy a WAV as-is; transcode anything else to mono 16 kHz WAV."""
    if src.suffix.lower() == ".wav":
        shutil.copyfile(src, dest)
        return
    if shutil.which("ffmpeg") is None:
        raise VideoError(
            f"Converting {src.suffix} reference audio requires ffmpeg"
            " (or pass a .wav file)"
        )
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar",
         str(REF_SAMPLE_RATE), str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not dest.exists():
        hint = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
        raise VideoError(f"ffmpeg could not convert {src}: {hint}")


def wav_seconds(path: Path) -> float:
    """Reference-sample duration; ffprobe fallback for non-PCM WAVs."""
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except wave.Error:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise VideoError(f"Could not read audio duration: {path}")
        return float(result.stdout.strip())


def record_reference(dest: Path, seconds: float = 15.0) -> Path:
    """Record microphone audio to `dest` — macOS `afrecord` when present,
    else ffmpeg's avfoundation backend. Used by `eh voices add` interactive
    onboarding."""
    if platform.system() != "Darwin":
        raise VideoError(
            "Interactive recording is macOS-only; pass --audio with a"
            " pre-recorded sample instead"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("afrecord"):
        cmd = ["afrecord", "-d", str(seconds), "-f", "m4af", str(dest)]
    elif shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0",
               "-t", str(seconds), str(dest)]
    else:
        raise VideoError(
            "No recorder found (tried afrecord, ffmpeg/avfoundation);"
            " pass --audio with a pre-recorded sample instead"
        )
    result = subprocess.run(cmd)
    if result.returncode != 0 or not dest.exists():
        raise VideoError(
            "Recording failed. If the mic never started: System Settings →"
            " Privacy & Security → Microphone, allow your terminal; then retry."
        )
    return dest


def add_voice(
    name: str,
    audio: Path,
    text: str | None = None,
    *,
    x_vector: bool = False,
    base: Path | None = None,
) -> VoiceMeta:
    """Register a voice from a reference sample. ICL mode (best quality)
    needs `text` (the exact transcript); pass `x_vector=True` to skip it."""
    _validate_name(name)
    if not audio.exists():
        raise VideoError(f"Reference audio not found: {audio}")
    if not x_vector and not text:
        raise VideoError(
            "A transcript is required for quality cloning: pass --text"
            " (the exact words in the sample) or --x-vector for lower"
            " quality without one"
        )
    dest_dir = voices_dir(base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.wav"
    _convert_to_wav(audio, dest)
    transcript = None
    transcript_path = dest_dir / f"{name}.txt"
    if text:
        transcript = text.strip()
        transcript_path.write_text(transcript)
    elif transcript_path.exists():
        transcript_path.unlink()
    return VoiceMeta(name, dest, wav_seconds(dest), transcript)


def list_voices(base: Path | None = None) -> list[VoiceMeta]:
    dest_dir = voices_dir(base)
    if not dest_dir.exists():
        return []
    voices = []
    for wav_path in sorted(dest_dir.glob("*.wav")):
        name = wav_path.stem
        transcript_path = dest_dir / f"{name}.txt"
        transcript = (
            transcript_path.read_text().strip() if transcript_path.exists() else None
        )
        voices.append(VoiceMeta(name, wav_path, wav_seconds(wav_path), transcript))
    return voices


def remove_voice(name: str, base: Path | None = None) -> None:
    _validate_name(name)
    dest_dir = voices_dir(base)
    wav_path = dest_dir / f"{name}.wav"
    if not wav_path.exists():
        raise VideoError(f"Unknown voice {name!r} (see: eh voices list)")
    wav_path.unlink()
    transcript_path = dest_dir / f"{name}.txt"
    if transcript_path.exists():
        transcript_path.unlink()
    for preview in preview_files(name, base):
        preview.unlink()


def previews_dir(base: Path | None = None) -> Path:
    """Onboarding previews live in a subdirectory so list_voices' *.wav glob
    never mistakes them for reference samples."""
    return voices_dir(base) / "previews"


def preview_files(name: str, base: Path | None = None) -> list[Path]:
    _validate_name(name)
    return sorted(previews_dir(base).glob(f"{name}-*.wav"))


def generate_previews(
    name: str,
    base: Path | None = None,
    log=lambda m: None,
) -> list[Path]:
    """Synthesize PREVIEW_LINES with the cloned voice for auditioning.

    Loads the qwen3 clone engine (seconds + GBs of RAM on first call).
    """
    from entertainment_harness.video.tts_qwen3 import Qwen3TTSCloneEngine

    resolve_voice(name, base)  # validates the voice exists first
    engine = Qwen3TTSCloneEngine(voices_base=base)
    dest_dir = previews_dir(base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, line in enumerate(PREVIEW_LINES):
        dest = dest_dir / f"{name}-{i}.wav"
        log(f"Preview {i + 1}/{len(PREVIEW_LINES)}: {line[:48]}…")
        engine.synthesize(line, name, dest)
        paths.append(dest)
    return paths


def resolve_voice(name: str, base: Path | None = None) -> VoiceMeta:
    """Look up a registered voice; error lists how to register one."""
    _validate_name(name)
    dest_dir = voices_dir(base)
    wav_path = dest_dir / f"{name}.wav"
    if not wav_path.exists():
        known = ", ".join(v.name for v in list_voices(base)) or "none"
        raise VideoError(
            f"Unknown voice {name!r} (registered: {known})."
            f" Add one with: eh voices add {name} --audio sample.wav"
            " --text \"...\""
        )
    transcript_path = dest_dir / f"{name}.txt"
    transcript = (
        transcript_path.read_text().strip() if transcript_path.exists() else None
    )
    return VoiceMeta(name, wav_path, wav_seconds(wav_path), transcript)
