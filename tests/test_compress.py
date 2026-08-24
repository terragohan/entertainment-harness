"""Compression preset tests: ffmpeg command construction and validation.
ffmpeg itself is faked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entertainment_harness.video import compress
from entertainment_harness.video.script import VideoError


def _fake_ffmpeg(monkeypatch, returncode=0):
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)

        class Result:
            stderr = ""

        Result.returncode = returncode
        return Result()

    monkeypatch.setattr(compress.subprocess, "run", fake_run)
    return calls


def test_compress_builds_crf_command(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch)
    src = tmp_path / "out.mp4"
    src.write_bytes(b"master")
    dest = compress.compress(src, "balanced", tmp_path)
    assert dest == tmp_path / "out-balanced.mp4"
    cmd = calls[0]
    assert cmd[0] == "ffmpeg"
    assert "scale=1280:720" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "25"
    assert "mov_text" in cmd  # subtitle track preserved


def test_compress_all_presets(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch)
    src = tmp_path / "out.mp4"
    src.write_bytes(b"master")
    for preset, (w, h, codec, crf) in compress.PRESETS.items():
        dest = compress.compress(src, preset, tmp_path)
        assert dest.name == f"out-{preset}.mp4"
        cmd = calls[-1]
        assert cmd[cmd.index("-c:v") + 1] == codec
        assert cmd[cmd.index("-crf") + 1] == str(crf)
    # h265 output gets the Apple-friendly hvc1 tag
    assert "hvc1" in calls[-1]


def test_compress_rejects_unknown_preset(tmp_path):
    with pytest.raises(VideoError, match="Unknown compression preset"):
        compress.compress(tmp_path / "out.mp4", "4k", tmp_path)


def test_compress_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, returncode=1)
    src = tmp_path / "out.mp4"
    src.write_bytes(b"master")
    with pytest.raises(VideoError, match="compression failed"):
        compress.compress(src, "small", tmp_path)
