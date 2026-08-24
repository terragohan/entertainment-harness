"""One-time migration from the legacy split layout to data/works/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entertainment_harness import db
from entertainment_harness.library import migrate, works
from entertainment_harness.db import utcnow


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EH_NO_AUTO_MIGRATE", "1")
    return tmp_path


def _legacy_manga_dir(root: Path, sid: str, cid: str) -> Path:
    return root / "manga" / sid / cid


def _legacy_video_dir(root: Path, sid: str, cid: str) -> Path:
    return root / "videos" / sid / cid


def test_migrate_moves_manga_pages_and_videos(root):
    sid = "s1"
    cid = "ch-1"

    # Legacy source pages.
    legacy_manga = _legacy_manga_dir(root, sid, cid)
    legacy_manga.mkdir(parents=True)
    (legacy_manga / "page-001.jpg").write_bytes(b"page")

    # Legacy translated overlay pages.
    legacy_translated = legacy_manga / "translated"
    legacy_translated.mkdir(parents=True)
    (legacy_translated / "page-001.png").write_bytes(b"translated-page")

    # Legacy videos.
    legacy_video = _legacy_video_dir(root, sid, cid)
    legacy_video.mkdir(parents=True)
    (legacy_video / "out.mp4").write_bytes(b"recap-video")
    legacy_narration = legacy_video / "narration"
    legacy_narration.mkdir(parents=True)
    (legacy_narration / "out.mp4").write_bytes(b"narration-video")
    legacy_tiktok = root / "videos" / sid / "tiktok"
    legacy_tiktok.mkdir(parents=True)
    (legacy_tiktok / "out.mp4").write_bytes(b"tiktok-video")

    # Minimal DB representing the legacy state.
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at, kind)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, "Test Manga", "mangadex", sid, "ongoing", utcnow(), "manga"),
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " published_at, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (cid, sid, 1.0, "Ch 1", "en", 1, utcnow(), utcnow()),
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES (?, ?, ?, ?)",
        (cid, "A recap.", utcnow(), "fake"),
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES (?, ?, ?, ?)",
        (cid, "A narration.", utcnow(), "fake"),
    )
    conn.execute(
        "INSERT INTO translations (chapter_id, pages, created_at)"
        " VALUES (?, ?, ?)",
        (cid, 1, utcnow()),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path, duration_s,"
        " created_at, tts_engine, model, kind, video_gen_provider)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, 1.0, 1.0, str(legacy_video / "out.mp4"), 60.0, utcnow(),
         "fake", "m", "recap", "local"),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path, duration_s,"
        " created_at, tts_engine, model, kind, video_gen_provider)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, 1.0, 1.0, str(legacy_narration / "out.mp4"), 120.0, utcnow(),
         "fake", "m", "narration", "local"),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path, duration_s,"
        " created_at, tts_engine, model, kind, video_gen_provider)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sid, None, None, str(legacy_tiktok / "out.mp4"), 90.0, utcnow(),
         "fake", "m", "tiktok", "local"),
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES (?, ?, ?)",
        (sid, 1.0, utcnow()),
    )
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES (?, ?, ?)",
        (sid, "Story so far.", 1.0),
    )
    conn.commit()
    conn.close()

    assert migrate.migrate() == 1

    # Work metadata.
    assert works.work_metadata_path(sid).exists()
    meta = works.read_work_metadata(sid)
    assert meta is not None
    assert meta.title == "Test Manga"
    assert meta.progress.last_read_chapter == 1.0
    assert meta.context.rolling_summary == "Story so far."

    # Chapter metadata.
    assert works.chapter_metadata_path(sid, cid).exists()
    chapter_meta = works.read_chapter_metadata(sid, cid)
    assert chapter_meta is not None
    assert chapter_meta.chapter_num == 1.0

    # Source pages.
    assert (works.source_dir(sid, cid) / "page-001.jpg").read_bytes() == b"page"

    # Translated pages + bubble metadata.
    assert (works.translated_dir(sid, cid) / "page-001.png").read_bytes() == (
        b"translated-page"
    )
    assert works.translation_path(sid, cid).exists()

    # Recap / narration metadata.
    assert works.read_recap(sid, cid) is not None
    assert works.read_narration(sid, cid) is not None

    # Videos.
    assert (works.video_recap_dir(sid, cid) / "out.mp4").read_bytes() == (
        b"recap-video"
    )
    assert (works.video_narration_dir(sid, cid) / "out.mp4").read_bytes() == (
        b"narration-video"
    )
    assert works.video_file_path(sid, cid, "recap").exists()
    assert works.video_file_path(sid, cid, "narration").exists()
    assert works.read_video_metadata(sid, cid, "recap") is not None
    assert works.read_video_metadata(sid, cid, "narration") is not None

    # Tiktok.
    assert (works.tiktok_dir(sid) / "out.mp4").read_bytes() == b"tiktok-video"
    assert works.read_video_metadata(sid, None, "tiktok") is not None


def test_migrate_moves_imported_book(root):
    sid = "book-1"

    legacy_books = root / "books" / sid
    legacy_books.mkdir(parents=True)
    (legacy_books / "cover.jpg").write_bytes(b"cover")
    (legacy_books / "ch-001.txt").write_bytes(b"chapter text")

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at, kind)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, "Test Book", "import", sid, "completed", utcnow(), "book"),
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("ch-001", sid, 1.0, "Chapter 1", "en", 0, utcnow()),
    )
    conn.commit()
    conn.close()

    assert migrate.migrate() == 1

    assert works.work_metadata_path(sid).exists()
    assert (works.work_dir(sid) / "cover.jpg").read_bytes() == b"cover"
    assert (works.source_dir(sid, "ch-001") / "ch-001.txt").read_bytes() == (
        b"chapter text"
    )


def test_migrate_is_noop_when_works_already_exists(root):
    sid = "s1"
    works.write_work_metadata(
        works.WorkMetadata(
            id=sid, title="Existing", source="mangadex", source_id=sid,
            added_at=utcnow(), kind="manga",
        )
    )
    assert migrate.migrate() == 0


def test_migrate_moves_weights_to_cache_dir(root):
    cache = root / "cache"
    models = root / "models" / "hf"
    models.mkdir(parents=True)
    (models / "weights.gguf").write_bytes(b"w")
    tts = root / "tts"
    tts.mkdir()
    (tts / "kokoro-v1.0.onnx").write_bytes(b"k")
    (tts / "voices").mkdir()
    (tts / "voices" / "narrator.wav").write_bytes(b"v")

    migrate.migrate()

    assert (cache / "models" / "hf" / "weights.gguf").read_bytes() == b"w"
    assert (cache / "tts" / "kokoro-v1.0.onnx").read_bytes() == b"k"
    assert (root / "voices" / "narrator.wav").read_bytes() == b"v"
    assert not (root / "models").exists()
    assert not (root / "tts").exists()


def test_migrate_cache_layout_keeps_existing_targets(root):
    cache = root / "cache"
    (cache / "models").mkdir(parents=True)
    old = root / "models"
    old.mkdir()
    (old / "stale").write_bytes(b"old")

    migrate.migrate()

    assert old.exists()  # target existed; leave source alone


def test_cache_dir_env_precedence(monkeypatch, tmp_path):
    from entertainment_harness.config import cache_dir

    monkeypatch.delenv("EH_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert cache_dir() == Path.home() / ".cache" / "entertainment-harness"

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "entertainment-harness"

    monkeypatch.setenv("EH_CACHE_DIR", str(tmp_path / "eh"))
    assert cache_dir() == tmp_path / "eh"


def test_migrate_renames_id_named_work_dirs(root, monkeypatch):
    # Legacy id-named works layout with metadata inside.
    cdir = root / "works" / "01JABC" / "chapters" / "01JCH1"
    cdir.mkdir(parents=True)
    (root / "works" / "01JABC" / "work.json").write_text(
        json.dumps(
            works.WorkMetadata(
                id="01JABC", title="One Piece", source="mangadex",
                source_id="01JABC", added_at=utcnow(),
            ).to_dict()
        )
    )
    (cdir / "chapter.json").write_text(
        json.dumps(
            works.ChapterMetadata(
                id="01JCH1", chapter_num=1.0, title=None, lang="en",
                pages=None, published_at=None, fetched_at=utcnow(),
            ).to_dict()
        )
    )

    migrate.migrate()

    assert works.work_dir("01JABC") == root / "works" / "one-piece"
    assert (root / "works" / "one-piece" / "chapters" / "ch-001").is_dir()
    assert not (root / "works" / "01JABC").exists()
