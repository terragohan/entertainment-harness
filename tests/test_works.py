"""Tests for the resource-centric works layout module."""

from __future__ import annotations

import json

import pytest

from entertainment_harness.library import works
from entertainment_harness.db import utcnow


@pytest.fixture
def series_id():
    return "s1"


@pytest.fixture
def chapter_id():
    return "c1"


def test_work_metadata_roundtrip(tmp_path, monkeypatch, series_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    meta = works.WorkMetadata(
        id=series_id,
        title="Test Series",
        source="mangadex",
        source_id=series_id,
        added_at=utcnow(),
        kind="manga",
        alt_titles=["Alt"],
        status="ongoing",
        progress=works.WorkProgress(last_read_chapter=1.0, updated_at=utcnow()),
        context=works.WorkContext(rolling_summary="Summary", through_chapter=1.0),
    )
    works.write_work_metadata(meta)
    loaded = works.read_work_metadata(series_id)
    assert loaded is not None
    assert loaded.title == "Test Series"
    assert loaded.kind == "manga"
    assert loaded.progress.last_read_chapter == 1.0
    assert loaded.context.rolling_summary == "Summary"


def test_chapter_metadata_roundtrip(tmp_path, monkeypatch, series_id, chapter_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    meta = works.ChapterMetadata(
        id=chapter_id,
        chapter_num=1.0,
        title="Chapter 1",
        lang="en",
        pages=20,
        published_at=utcnow(),
        fetched_at=utcnow(),
    )
    works.write_chapter_metadata(series_id, meta)
    loaded = works.read_chapter_metadata(series_id, chapter_id)
    assert loaded is not None
    assert loaded.chapter_num == 1.0
    assert loaded.lang == "en"


def test_recap_roundtrip(tmp_path, monkeypatch, series_id, chapter_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    meta = works.RecapMetadata(summary="A recap", model="m", created_at=utcnow())
    works.write_recap(series_id, chapter_id, meta)
    loaded = works.read_recap(series_id, chapter_id)
    assert loaded is not None
    assert loaded.summary == "A recap"


def test_recap_metadata_reads_pre_instruction_files(
    tmp_path, monkeypatch, series_id, chapter_id
):
    """recap.json written before detail/instruction existed still parses,
    with the defaults applied."""
    import json

    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    path = works.recap_path(series_id, chapter_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": "A recap", "model": "m", "created_at": "now"})
    )
    loaded = works.read_recap(series_id, chapter_id)
    assert loaded is not None
    assert loaded.detail == "standard"
    assert loaded.instruction == ""


def test_narration_roundtrip(tmp_path, monkeypatch, series_id, chapter_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    meta = works.NarrationMetadata(text="A narration", model="m", created_at=utcnow())
    works.write_narration(series_id, chapter_id, meta)
    loaded = works.read_narration(series_id, chapter_id)
    assert loaded is not None
    assert loaded.text == "A narration"


def test_translation_roundtrip(tmp_path, monkeypatch, series_id, chapter_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    meta = works.TranslationMetadata(
        pages=10, model="m", created_at=utcnow(), bubbles=[{"x": 1}]
    )
    works.write_translation(series_id, chapter_id, meta)
    loaded = works.read_translation(series_id, chapter_id)
    assert loaded is not None
    assert loaded.bubbles == [{"x": 1}]


def test_video_metadata_roundtrip(tmp_path, monkeypatch, series_id, chapter_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    meta = works.VideoMetadata(
        kind="recap",
        duration_s=120.0,
        model="m",
        tts_engine="kokoro",
        created_at=utcnow(),
    )
    works.write_video_metadata(series_id, chapter_id, meta)
    loaded = works.read_video_metadata(series_id, chapter_id, "recap")
    assert loaded is not None
    assert loaded.duration_s == 120.0


def test_video_file_paths(tmp_path, monkeypatch, series_id, chapter_id):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    recap_path = works.video_file_path(series_id, chapter_id, "recap")
    narr_path = works.video_file_path(series_id, chapter_id, "narration")
    tiktok_path = works.video_file_path(series_id, None, "tiktok")
    assert recap_path.name == "out.mp4"
    assert "video-recap" in str(recap_path)
    assert "video-narration" in str(narr_path)
    assert "tiktok" in str(tiktok_path)


def test_list_work_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(
        works.WorkMetadata(
            id="s1", title="A", source="mangadex", source_id="s1", added_at=utcnow()
        )
    )
    works.write_work_metadata(
        works.WorkMetadata(
            id="s2", title="B", source="mangadex", source_id="s2", added_at=utcnow()
        )
    )
    dirs = works.list_work_dirs()
    assert len(dirs) == 2
    # Work dirs are human-readable slugs; ids resolve to them transparently.
    assert {d.name for d in dirs} == {"a", "b"}
    assert works.work_dir("s1").name == "a"
    assert works.work_dir("s2").name == "b"


def _work(series_id: str, title: str) -> works.WorkMetadata:
    return works.WorkMetadata(
        id=series_id, title=title, source="mangadex", source_id=series_id,
        added_at=utcnow(),
    )


def _chapter(chapter_id: str, num: float | None) -> works.ChapterMetadata:
    return works.ChapterMetadata(
        id=chapter_id, chapter_num=num, title=None, lang="en", pages=None,
        published_at=None, fetched_at=utcnow(),
    )


def test_new_works_get_slug_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(_work("01JABC", "One Piece"))
    works.write_chapter_metadata("01JABC", _chapter("01JCH1", 1.0))
    works.write_chapter_metadata("01JABC", _chapter("01JCH2", 1.5))
    works.write_chapter_metadata("01JABC", _chapter("01JCH3", None))

    assert works.work_dir("01JABC") == tmp_path / "works" / "one-piece"
    assert works.chapter_dir("01JABC", "01JCH1").name == "ch-001"
    assert works.chapter_dir("01JABC", "01JCH2").name == "ch-001.5"
    # Chapter with no number falls back to a slug of its id (case-insensitive
    # filesystems may report the id's original case).
    assert works.chapter_dir("01JABC", "01JCH3").name.lower() == "01jch3"
    # Artifacts land under the slug dirs.
    assert str(works.recap_path("01JABC", "01JCH1")).startswith(
        str(tmp_path / "works" / "one-piece" / "chapters" / "ch-001")
    )


def test_slug_collision_gets_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(_work("s1", "One Piece"))
    works.write_work_metadata(_work("s2", "One Piece"))
    assert works.work_dir("s1").name == "one-piece"
    assert works.work_dir("s2").name == "one-piece-2"


def test_legacy_id_named_dirs_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    # Pre-existing id-named layout with metadata inside.
    cdir = tmp_path / "works" / "01JABC" / "chapters" / "01JCH1"
    cdir.mkdir(parents=True)
    (tmp_path / "works" / "01JABC" / "work.json").write_text(
        json.dumps(_work("01JABC", "One Piece").to_dict())
    )
    (cdir / "chapter.json").write_text(json.dumps(_chapter("01JCH1", 1.0).to_dict()))

    assert works.work_dir("01JABC").name == "01JABC"
    assert works.chapter_dir("01JABC", "01JCH1") == cdir
    # Rewrites stay in the legacy dirs (no rename on write).
    works.write_chapter_metadata("01JABC", _chapter("01JCH1", 1.0))
    assert works.chapter_dir("01JABC", "01JCH1") == cdir


def test_remove_work_removes_slug_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(_work("s1", "One Piece"))
    works.remove_work("s1")
    assert not (tmp_path / "works" / "one-piece").exists()
    assert works.read_work_metadata("s1") is None
