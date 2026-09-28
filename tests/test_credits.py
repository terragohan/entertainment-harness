"""Credits end card: card rendering, content-keyed caching, mux integration,
the [video] credits toggle, and author plumbing (WorkMetadata + EPUB)."""

from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import entertainment_harness.video.assemble as assemble_mod
from entertainment_harness.config import Config, load_config
from entertainment_harness.db import utcnow
from entertainment_harness.library import works
from entertainment_harness.library.importer import read_epub_author
from entertainment_harness.video.credits import (
    CREDITS_SECONDS,
    Credits,
    card_path,
    render_card,
)
from entertainment_harness.video.script import Segment


def _credits(**kw) -> Credits:
    base = {"title": "Kenja no Mago", "source": "weebcentral",
            "chapter_label": "Chapter 47", "author": "Yoshioka Tsuyoshi"}
    base.update(kw)
    return Credits(**base)


# --- Credits lines ------------------------------------------------------------


def test_meta_line_includes_author_source_and_chapter():
    line = _credits().meta_line()
    assert "by Yoshioka Tsuyoshi" in line
    assert "Source: weebcentral" in line
    assert "Chapter 47" in line


def test_meta_line_omits_unknown_author():
    assert "by " not in _credits(author=None).meta_line()


def test_meta_line_omits_empty_chapter_label():
    assert "Chapter" not in _credits(chapter_label="").meta_line()


def test_meta_line_humanizes_internal_sources():
    assert "Source: local import" in _credits(source="import").meta_line()


# --- card rendering ------------------------------------------------------------


def test_render_card_produces_rgb_png_at_frame_size(tmp_path):
    dest = render_card(_credits(), (1280, 720), tmp_path / "card.png")
    with Image.open(dest) as img:
        assert img.size == (1280, 720)
        assert img.mode == "RGB"
        # Text was actually drawn: the frame is not a flat background fill.
        assert len(set(img.convert("L").getdata())) > 8


def test_render_card_handles_very_long_titles(tmp_path):
    dest = render_card(_credits(title="A" * 200), (320, 568), tmp_path / "c.png")
    with Image.open(dest) as img:
        assert img.size == (320, 568)


def test_card_path_is_content_keyed(tmp_path):
    first = card_path(_credits(), (1280, 720), tmp_path)
    mtime = first.stat().st_mtime_ns
    assert card_path(_credits(), (1280, 720), tmp_path) == first
    assert first.stat().st_mtime_ns == mtime  # cached, not re-rendered
    assert card_path(_credits(title="Other"), (1280, 720), tmp_path) != first
    assert card_path(_credits(), (640, 360), tmp_path) != first


# --- mux integration -------------------------------------------------------------


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


def _stub_ffmpeg(monkeypatch):
    """Record build_clip calls; fake ffprobe/ffmpeg so no real encode runs."""
    calls: list[dict] = []

    def fake_build_clip(page, duration, motion, resolution, dest):
        calls.append({"page": page, "duration": duration, "motion": motion,
                      "resolution": resolution, "dest": dest})
        dest.write_bytes(b"credits-clip")

    def fake_run(cmd):
        Path(cmd[-1]).write_bytes(b"out")  # each ffmpeg call's last arg is its output

    monkeypatch.setattr(assemble_mod, "build_clip", fake_build_clip)
    monkeypatch.setattr(assemble_mod, "_run", fake_run)
    monkeypatch.setattr(
        assemble_mod, "probe_format", lambda p: ("h264", 1280, 720, "30/1")
    )
    return calls


def _one_segment_workdir(tmp_path) -> tuple[list[Segment], list[Path], Path]:
    workdir = tmp_path / "work"
    (workdir / "clips").mkdir(parents=True)
    _write_wav(workdir / "seg-00.wav")
    clip = workdir / "clips" / "clip-00-0.mp4"
    clip.write_bytes(b"clip")
    seg = Segment(index=0, text="hello", pages=[1], duration_s=3.0,
                  pause_after_s=0.15)
    return [seg], [clip], workdir


def test_mux_clips_appends_credits_card(tmp_path, monkeypatch):
    calls = _stub_ffmpeg(monkeypatch)
    segments, clips, workdir = _one_segment_workdir(tmp_path)
    _, duration = assemble_mod.mux_clips(
        segments, clips, workdir, credits=_credits()
    )
    assert duration == pytest.approx(3.0 + 0.15 + CREDITS_SECONDS)

    assert len(calls) == 1  # one extra clip: the end card
    call = calls[0]
    assert call["duration"] == CREDITS_SECONDS
    assert call["motion"] == "static"  # no Ken Burns drift on the card
    assert call["resolution"] == (1280, 720)  # probed from the chapter clips
    assert call["dest"].name.startswith("credits-")

    concat = (workdir / "clips.txt").read_text().splitlines()
    assert len(concat) == 2
    assert concat[-1] == f"file '{call['dest'].as_posix()}'"

    wavs = (workdir / "wavs.txt").read_text().splitlines()
    assert len(wavs) == 3  # narration + beat pause + credits silence
    assert "gap-4000ms.wav" in wavs[-1]


def test_mux_clips_without_credits_is_unchanged(tmp_path, monkeypatch):
    calls = _stub_ffmpeg(monkeypatch)
    segments, clips, workdir = _one_segment_workdir(tmp_path)
    _, duration = assemble_mod.mux_clips(segments, clips, workdir)
    assert duration == pytest.approx(3.0 + 0.15)
    assert calls == []
    assert len((workdir / "clips.txt").read_text().splitlines()) == 1


def test_mux_clips_reuses_cached_card_clip(tmp_path, monkeypatch):
    calls = _stub_ffmpeg(monkeypatch)
    segments, clips, workdir = _one_segment_workdir(tmp_path)
    assemble_mod.mux_clips(segments, clips, workdir, credits=_credits())
    assert len(calls) == 1
    (workdir / "out.mp4").unlink()
    assemble_mod.mux_clips(segments, clips, workdir, credits=_credits())
    assert len(calls) == 1  # card + clip cached by content hash


# --- [video] credits toggle -------------------------------------------------------


def test_credits_toggle_defaults_on():
    assert Config().video.credits is True


def test_credits_toggle_parses_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[video]\ncredits = false\n")
    assert load_config(path).video.credits is False


# --- author plumbing ---------------------------------------------------------------


def test_work_metadata_author_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Book", source="import", source_id="test-book",
        added_at=utcnow(), kind="book", status="imported", author="Jane Author",
    ))
    loaded = works.read_work_metadata("s1")
    assert loaded is not None
    assert loaded.author == "Jane Author"


def test_work_metadata_author_defaults_none(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Old Book", source="import", source_id="old-book",
        added_at=utcnow(), kind="book",
    ))
    # A work.json written before the field existed still loads.
    raw = works.work_dir("s1") / "work.json"
    data = json.loads(raw.read_text())
    del data["author"]
    raw.write_text(json.dumps(data))
    assert works.read_work_metadata("s1").author is None


_CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""

_OPF_WITH_CREATOR = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">test</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:creator>Jane Author</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest><item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="ch1"/></spine>
</package>
"""


def _epub_with_creator(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("META-INF/container.xml", _CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", _OPF_WITH_CREATOR)
        zf.writestr("OEBPS/ch1.xhtml",
                    "<html><body><h1>Chapter 1</h1><p>words</p></body></html>")
    return path


def test_read_epub_author_extracts_dc_creator(tmp_path):
    assert read_epub_author(_epub_with_creator(tmp_path / "b.epub")) == "Jane Author"


def test_read_epub_author_tolerates_garbage(tmp_path):
    garbage = tmp_path / "not-an-epub.epub"
    garbage.write_bytes(b"nope")
    assert read_epub_author(garbage) is None
