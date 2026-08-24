"""Voice library tests: registration, listing, resolution, and the eh voices
CLI. No model downloads — reference samples are tiny generated WAVs."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from entertainment_harness.cli import app
from entertainment_harness.video import voices
from entertainment_harness.video.script import VideoError

runner = CliRunner()


def _write_wav(path: Path, seconds: float = 4.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


@pytest.fixture
def sample(tmp_path) -> Path:
    path = tmp_path / "ref.wav"
    _write_wav(path)
    return path


def test_add_and_resolve_with_transcript(tmp_path, sample):
    voice = voices.add_voice("narrator", sample, "Hello there.", base=tmp_path)
    assert voice.name == "narrator"
    assert voice.duration_s == pytest.approx(4.0)
    assert voice.transcript == "Hello there."
    resolved = voices.resolve_voice("narrator", tmp_path)
    assert resolved.path == tmp_path / "voices" / "narrator.wav"
    assert resolved.transcript == "Hello there."


def test_add_copies_audio_to_library(tmp_path, sample):
    voices.add_voice("narrator", sample, "Hello.", base=tmp_path)
    library_wav = tmp_path / "voices" / "narrator.wav"
    assert library_wav.exists()
    assert sample.read_bytes() == library_wav.read_bytes()


def test_add_requires_text_unless_x_vector(tmp_path, sample):
    with pytest.raises(VideoError, match="--text"):
        voices.add_voice("narrator", sample, base=tmp_path)
    voice = voices.add_voice("narrator", sample, x_vector=True, base=tmp_path)
    assert voice.transcript is None
    assert voices.resolve_voice("narrator", tmp_path).transcript is None


def test_add_rejects_bad_names(tmp_path, sample):
    for bad in ("Narrator", "my voice", "-lead", ""):
        with pytest.raises(VideoError, match="Invalid voice name"):
            voices.add_voice(bad, sample, "Hi.", base=tmp_path)


def test_add_rejects_missing_audio(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        voices.add_voice("narrator", tmp_path / "nope.wav", "Hi.", base=tmp_path)


def test_list_and_remove(tmp_path, sample):
    assert voices.list_voices(tmp_path) == []
    voices.add_voice("a", sample, "Hi.", base=tmp_path)
    voices.add_voice("b", sample, x_vector=True, base=tmp_path)
    metas = voices.list_voices(tmp_path)
    assert [m.name for m in metas] == ["a", "b"]
    assert metas[0].transcript == "Hi."
    assert metas[1].transcript is None
    voices.remove_voice("a", tmp_path)
    assert [m.name for m in voices.list_voices(tmp_path)] == ["b"]
    with pytest.raises(VideoError, match="Unknown voice"):
        voices.remove_voice("a", tmp_path)


def test_resolve_unknown_voice_lists_registered(tmp_path, sample):
    voices.add_voice("known", sample, "Hi.", base=tmp_path)
    with pytest.raises(VideoError, match="eh voices add ghost"):
        voices.resolve_voice("ghost", tmp_path)


# --- CLI ---------------------------------------------------------------------


def test_cli_add_list_remove(tmp_path, monkeypatch, sample):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    result = runner.invoke(
        app, ["voices", "add", "demo", "--audio", str(sample),
              "--text", "Hi there.", "--no-preview"]
    )
    assert result.exit_code == 0, result.output
    assert "--tts-engine qwen3" in result.output
    assert "--voice demo" in result.output

    result = runner.invoke(app, ["voices", "list"])
    assert result.exit_code == 0
    assert "demo" in result.output
    assert "ICL (transcript)" in result.output

    result = runner.invoke(app, ["voices", "remove", "demo"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["voices", "list"])
    assert "demo" not in result.output


def test_cli_add_without_text_fails(tmp_path, monkeypatch, sample):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["voices", "add", "demo", "--audio", str(sample)])
    assert result.exit_code == 1
    assert "--text" in result.output


def test_cli_add_warns_on_short_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    short = tmp_path / "short.wav"
    _write_wav(short, seconds=1.0)
    result = runner.invoke(
        app, ["voices", "add", "tiny", "--audio", str(short), "--text", "Hi.",
              "--no-preview"]
    )
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output


def test_cli_remove_unknown_voice_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["voices", "remove", "ghost"])
    assert result.exit_code == 1
    assert "Unknown voice" in result.output


# --- previews ------------------------------------------------------------------


def _fake_previews(tmp_path, monkeypatch):
    """Stub generation with tiny WAVs so no model loads."""

    def fake(name, base=None, log=lambda m: None):
        paths = []
        for i in range(len(voices.PREVIEW_LINES)):
            path = voices.previews_dir(base) / f"{name}-{i}.wav"
            _write_wav(path, seconds=0.5)
            paths.append(path)
        return paths

    monkeypatch.setattr(voices, "generate_previews", fake)
    return fake


def test_generate_previews_named_per_line(tmp_path, sample, monkeypatch):
    fake = _fake_previews(tmp_path, monkeypatch)
    voices.add_voice("narrator", sample, "Hi.", base=tmp_path)
    paths = fake("narrator", base=tmp_path)
    assert len(paths) == len(voices.PREVIEW_LINES)
    assert [p.name for p in paths] == ["narrator-0.wav", "narrator-1.wav",
                                       "narrator-2.wav"]
    assert all(p.exists() for p in paths)
    # previews must not leak into the voice list
    assert [v.name for v in voices.list_voices(tmp_path)] == ["narrator"]


def test_remove_voice_deletes_previews(tmp_path, sample):
    voices.add_voice("narrator", sample, "Hi.", base=tmp_path)
    (voices.previews_dir(tmp_path)).mkdir(parents=True, exist_ok=True)
    _write_wav(voices.previews_dir(tmp_path) / "narrator-0.wav", seconds=0.5)
    voices.remove_voice("narrator", tmp_path)
    assert voices.preview_files("narrator", tmp_path) == []


def test_generate_previews_unknown_voice(tmp_path):
    with pytest.raises(VideoError, match="Unknown voice"):
        voices.generate_previews("ghost", base=tmp_path)


def test_cli_add_runs_previews_by_default(tmp_path, monkeypatch, sample):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    fake = _fake_previews(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["voices", "add", "demo", "--audio", str(sample), "--text", "Hi."]
    )
    assert result.exit_code == 0, result.output
    assert "Previews for 'demo'" in result.output
    assert voices.preview_files("demo", tmp_path)


def test_cli_preview_command(tmp_path, monkeypatch, sample):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    _fake_previews(tmp_path, monkeypatch)
    voices.add_voice("demo", sample, "Hi.", base=tmp_path)
    result = runner.invoke(app, ["voices", "preview", "demo"])
    assert result.exit_code == 0, result.output
    assert "Previews for 'demo'" in result.output
    assert voices.preview_files("demo", tmp_path)


def test_preview_failure_warns_but_keeps_voice(tmp_path, monkeypatch, sample):
    """A broken preview run must not fail an otherwise-good `voices add`."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))

    def broken(name, base=None, log=lambda m: None):
        raise VideoError("mlx-audio is not installed")

    monkeypatch.setattr(voices, "generate_previews", broken)
    result = runner.invoke(
        app, ["voices", "add", "demo", "--audio", str(sample), "--text", "Hi."]
    )
    assert result.exit_code == 0, result.output
    assert "Previews failed" in result.output
    assert "eh voices preview demo" in result.output
    assert voices.resolve_voice("demo", tmp_path).name == "demo"


# --- interactive onboarding (no --audio) ---------------------------------------


def _fake_recorder(tmp_path, monkeypatch):
    """Stub afrecord with an instant 4 s 'recording'. Returns a .wav path so
    add_voice copies it without invoking ffmpeg."""

    def fake(dest: Path, seconds: float = 15.0) -> Path:
        wav_dest = dest.with_suffix(".wav")
        _write_wav(wav_dest, seconds=4.0)
        return wav_dest

    monkeypatch.setattr(voices, "record_reference", fake)
    return fake


def test_record_reference_requires_macos(tmp_path, monkeypatch):
    monkeypatch.setattr(voices.platform, "system", lambda: "Linux")
    with pytest.raises(VideoError, match="macOS-only"):
        voices.record_reference(tmp_path / "ref.m4a")


def test_record_reference_falls_back_to_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(voices.shutil, "which", lambda tool: None if tool == "afrecord" else "/opt/homebrew/bin/ffmpeg")
    calls = []

    class FakeResult:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake")
        return FakeResult()

    monkeypatch.setattr(voices.subprocess, "run", fake_run)
    dest = voices.record_reference(tmp_path / "ref.m4a", seconds=9)
    assert calls[0][:3] == ["ffmpeg", "-y", "-f"]
    assert "avfoundation" in calls[0]
    assert dest.exists()


def test_record_reference_no_recorder_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(voices.shutil, "which", lambda tool: None)
    with pytest.raises(VideoError, match="No recorder found"):
        voices.record_reference(tmp_path / "ref.m4a")


def test_cli_add_interactive_onboarding(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    _fake_previews(tmp_path, monkeypatch)
    _fake_recorder(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["voices", "add", "me"],
        input="\nn\nn\n",  # start recording, no playback, keep take
    )
    assert result.exit_code == 0, result.output
    assert "Read this line aloud" in result.output
    assert "Previews for 'me'" in result.output
    metas = voices.list_voices(tmp_path)
    assert [m.name for m in metas] == ["me"]
    assert metas[0].transcript == voices.PREVIEW_LINES[0]


def test_cli_add_interactive_re_records(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    _fake_previews(tmp_path, monkeypatch)
    recorder = _fake_recorder(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["voices", "add", "me"],
        input="\nn\ny\n\nn\nn\n",  # take 1, re-record, take 2, keep
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("Recording") >= 2


def test_cli_add_interactive_x_vector(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    _fake_previews(tmp_path, monkeypatch)
    _fake_recorder(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["voices", "add", "me", "--x-vector"],
        input="\nn\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "Speak naturally" in result.output
    assert voices.resolve_voice("me", tmp_path).transcript is None


def test_cli_add_interactive_custom_text(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    _fake_previews(tmp_path, monkeypatch)
    _fake_recorder(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["voices", "add", "me", "--text", "The quick brown fox."],
        input="\nn\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert '"The quick brown fox."' in result.output
    assert voices.resolve_voice("me", tmp_path).transcript == "The quick brown fox."


