"""Tests for rebuilding the SQLite index from the filesystem."""

from __future__ import annotations

import pytest

from entertainment_harness import db
from entertainment_harness.library import index, works
from entertainment_harness.db import utcnow


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    return db.connect()


def test_index_rebuilds_series_and_chapters(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(
        works.WorkMetadata(
            id="s1", title="Series", source="mangadex", source_id="s1",
            added_at=utcnow(), kind="manga",
        )
    )
    works.write_chapter_metadata(
        "s1",
        works.ChapterMetadata(
            id="c1", chapter_num=1.0, title="Ch 1", lang="en", pages=10,
            published_at=utcnow(), fetched_at=utcnow(),
        ),
    )
    conn.close()

    count = index.index_works(tmp_path)
    assert count == 1

    conn = db.connect()
    series = conn.execute("SELECT * FROM series WHERE id = ?", ("s1",)).fetchone()
    assert series["title"] == "Series"
    assert series["kind"] == "manga"
    chapter = conn.execute("SELECT * FROM chapters WHERE id = ?", ("c1",)).fetchone()
    assert chapter["chapter_num"] == 1.0
    conn.close()


def test_index_rebuilds_recaps_and_narrations(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(
        works.WorkMetadata(
            id="s1", title="Series", source="mangadex", source_id="s1",
            added_at=utcnow(),
        )
    )
    works.write_chapter_metadata(
        "s1",
        works.ChapterMetadata(
            id="c1", chapter_num=1.0, title="Ch 1", lang="en", pages=10,
            published_at=utcnow(), fetched_at=utcnow(),
        ),
    )
    works.write_recap("s1", "c1", works.RecapMetadata("summary", "m", utcnow()))
    works.write_narration("s1", "c1", works.NarrationMetadata("text", "m", utcnow()))
    conn.close()

    index.index_works(tmp_path)

    conn = db.connect()
    recap = conn.execute("SELECT * FROM recaps WHERE chapter_id = ?", ("c1",)).fetchone()
    # the legacy narration.json folds into recaps as detail='full' on open
    # (db.fold_narrations_into_recaps): full supersedes the standard summary
    assert recap["summary"] == "text"
    assert recap["detail"] == "full"
    narration = conn.execute(
        "SELECT * FROM narrations WHERE chapter_id = ?", ("c1",)
    ).fetchone()
    assert narration["text"] == "text"
    conn.close()


def test_index_clears_stale_rows(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    # Seed DB with a stale series.
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at, kind)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("old", "Old", "mangadex", "old", None, utcnow(), "manga"),
    )
    conn.commit()
    conn.close()

    # No works on disk.
    index.index_works(tmp_path)

    conn = db.connect()
    rows = conn.execute("SELECT * FROM series").fetchall()
    assert len(rows) == 0
    conn.close()


def test_index_preserves_recap_attribution(conn, tmp_path, monkeypatch):
    """recaps.instruction and recaps.standalone survive an index rebuild."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(
        works.WorkMetadata(
            id="s1", title="Series", source="mangadex", source_id="s1",
            added_at=utcnow(),
        )
    )
    works.write_chapter_metadata(
        "s1",
        works.ChapterMetadata(
            id="c1", chapter_num=1.0, title="Ch 1", lang="en", pages=10,
            published_at=utcnow(), fetched_at=utcnow(),
        ),
    )
    works.write_recap(
        "s1", "c1",
        works.RecapMetadata(
            "summary", "m", utcnow(), detail="detailed",
            instruction="skip the cold open", standalone=True,
        ),
    )
    conn.close()

    index.index_works(tmp_path)

    conn = db.connect()
    recap = conn.execute(
        "SELECT instruction, standalone FROM recaps WHERE chapter_id = 'c1'"
    ).fetchone()
    assert recap["instruction"] == "skip the cold open"
    assert recap["standalone"] == 1
    conn.close()
