"""Library tests: temp-dir SQLite (via EH_DATA_DIR), respx-mocked MangaDex."""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from entertainment_harness import db, library
from entertainment_harness.config import data_dir
from entertainment_harness.library import LibraryError

BASE = "https://api.mangadex.org"
MANGA_ID = "835feda4-2db0-4753-8249-4575a3ceffe2"

MANGA_RESPONSE = {
    "result": "ok",
    "data": {
        "id": MANGA_ID,
        "attributes": {
            "title": {"en": "Kenja no Mago"},
            "altTitles": [{"en": "Wise Man's Grandchild"}],
            "year": 2016,
            "status": "ongoing",
        },
    },
}

FEED_RESPONSE = {
    "result": "ok",
    "total": 3,
    "data": [
        {
            "id": "ch-1",
            "attributes": {
                "chapter": "1",
                "title": "Ch 1",
                "translatedLanguage": "en",
                "pages": 53,
                "publishAt": "2026-01-01T00:00:00+00:00",
            },
        },
        {
            "id": "ch-1-pt",
            "attributes": {
                "chapter": "1",
                "title": "Ch 1",
                "translatedLanguage": "pt-br",
                "pages": 53,
                "publishAt": "2026-01-01T00:00:00+00:00",
            },
        },
        {
            "id": "ch-2",
            "attributes": {
                "chapter": "2",
                "title": "Ch 2",
                "translatedLanguage": "en",
                "pages": 54,
                "publishAt": "2026-01-02T00:00:00+00:00",
            },
        },
    ],
}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    connection = db.connect()
    yield connection
    connection.close()


@respx.mock
def test_add_series_by_id(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    series = library.add_series(conn, MANGA_ID)
    assert series.title == "Kenja no Mago"
    row = conn.execute("SELECT * FROM series WHERE id = ?", (MANGA_ID,)).fetchone()
    assert row["source"] == "mangadex"
    assert row["status"] == "ongoing"
    assert "Wise Man's Grandchild" in row["alt_titles"]


@respx.mock
def test_add_series_by_search_takes_top_result(conn):
    respx.get(f"{BASE}/manga").mock(
        return_value=Response(
            200, json={"result": "ok", "total": 1, "data": [MANGA_RESPONSE["data"]]}
        )
    )
    series = library.add_series(conn, "Kenja no Mago")
    assert series.id == MANGA_ID


@respx.mock
def test_add_series_rejects_duplicates(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    with pytest.raises(LibraryError, match="already in the library"):
        library.add_series(conn, MANGA_ID)


@respx.mock
def test_sync_chapters_upserts(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    respx.get(f"{BASE}/manga/{MANGA_ID}/feed").mock(
        return_value=Response(200, json=FEED_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    count = library.sync_chapters(conn, MANGA_ID)
    assert count == 2  # pt-br chapter filtered out (default lang "en")
    rows = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ? ORDER BY chapter_num", (MANGA_ID,)
    ).fetchall()
    assert [r["id"] for r in rows] == ["ch-1", "ch-2"]
    assert rows[0]["chapter_num"] == 1.0
    assert rows[0]["lang"] == "en"
    assert rows[0]["pages"] == 53

    # re-sync updates instead of duplicating
    count = library.sync_chapters(conn, MANGA_ID)
    assert count == 2
    total = conn.execute("SELECT COUNT(*) AS n FROM chapters").fetchone()["n"]
    assert total == 2


@respx.mock
def test_sync_chapters_force_removes_stale_local_chapters(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    respx.get(f"{BASE}/manga/{MANGA_ID}/feed").mock(
        return_value=Response(200, json=FEED_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    library.sync_chapters(conn, MANGA_ID)

    # Simulate a chapter that was removed from the source but still exists locally.
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " published_at, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("ch-3", MANGA_ID, 3.0, "Ch 3", "en", 10, "2026-01-03T00:00:00+00:00", "now"),
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at) VALUES (?, ?, ?)",
        ("ch-3", "old recap", "now"),
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at) VALUES (?, ?, ?)",
        ("ch-3", "old narration", "now"),
    )
    conn.commit()

    # Without force the stale chapter remains.
    count = library.sync_chapters(conn, MANGA_ID)
    assert count == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM chapters").fetchone()["n"] == 3

    # With force the stale chapter and its recap/narration are removed.
    count = library.sync_chapters(conn, MANGA_ID, force=True)
    assert count == 2
    rows = conn.execute(
        "SELECT id FROM chapters WHERE series_id = ? ORDER BY chapter_num", (MANGA_ID,)
    ).fetchall()
    assert [r["id"] for r in rows] == ["ch-1", "ch-2"]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM recaps WHERE chapter_id = ?", ("ch-3",)
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM narrations WHERE chapter_id = ?", ("ch-3",)
    ).fetchone()["n"] == 0


@respx.mock
def test_sync_chapters_writes_works_metadata(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    respx.get(f"{BASE}/manga/{MANGA_ID}/feed").mock(
        return_value=Response(200, json=FEED_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    library.sync_chapters(conn, MANGA_ID)

    from entertainment_harness.library import works

    meta = works.read_work_metadata(MANGA_ID)
    assert meta is not None
    assert meta.id == MANGA_ID
    assert meta.title == "Kenja no Mago"
    assert meta.source == "mangadex"
    chapter = works.read_chapter_metadata(MANGA_ID, "ch-1")
    assert chapter is not None
    assert chapter.chapter_num == 1.0
    assert chapter.lang == "en"


@respx.mock
def test_sync_chapters_other_language(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    respx.get(f"{BASE}/manga/{MANGA_ID}/feed").mock(
        return_value=Response(200, json=FEED_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    count = library.sync_chapters(conn, MANGA_ID, langs=["pt-br"])
    assert count == 1
    row = conn.execute("SELECT * FROM chapters WHERE series_id = ?", (MANGA_ID,)).fetchone()
    assert row["id"] == "ch-1-pt"


def test_sync_unknown_series_raises(conn):
    with pytest.raises(LibraryError):
        library.sync_chapters(conn, "no-such-series")


@respx.mock
def test_resolve_series_by_title_substring(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    row = library.resolve_series(conn, "Kenja")
    assert row["id"] == MANGA_ID
    with pytest.raises(LibraryError, match="No series matches"):
        library.resolve_series(conn, "Nonexistent")


def test_progress_roundtrip(conn):
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, added_at)"
        " VALUES ('s1', 'S', 'mangadex', 's1', 'now')"
    )
    assert library.get_progress(conn, "s1") is None
    library.advance_progress(conn, "s1", 1.0)
    assert library.get_progress(conn, "s1") == 1.0
    library.advance_progress(conn, "s1", 2.5)
    assert library.get_progress(conn, "s1") == 2.5


def test_remove_series_deletes_rows_and_files(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, added_at)"
        " VALUES ('s1', 'Dupe', 'mangadex', 's1', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('c1', 's1', 1.0, 'Ch 1', 'en', 2, 'now')"
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at) VALUES ('c1', 'r', 'now')"
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at) VALUES ('c1', 'n', 'now')"
    )
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES ('s1', 'ctx', 1.0)"
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 1.0, 'now')"
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path, created_at)"
        " VALUES ('s1', 1.0, 1.0, '/tmp/x.mp4', 'now')"
    )
    conn.execute(
        "INSERT INTO translations (chapter_id, pages, created_at)"
        " VALUES ('c1', 2, 'now')"
    )
    conn.commit()
    from entertainment_harness.library import works

    page_dir = works.source_dir("s1", "c1")
    page_dir.mkdir(parents=True)
    (page_dir / "page-001.jpg").write_bytes(b"")

    library.remove_series(conn, "s1", log=lambda m: None)

    for table in ("series", "chapters", "recaps", "narrations", "series_context",
                  "progress", "videos", "translations"):
        assert conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == 0
    assert not works.work_dir("s1").exists()


def test_remove_series_keep_files(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, added_at)"
        " VALUES ('s1', 'Dupe', 'mangadex', 's1', 'now')"
    )
    conn.commit()
    from entertainment_harness.library import works

    work_dir = works.work_dir("s1")
    work_dir.mkdir(parents=True)

    library.remove_series(conn, "s1", delete_files=False, log=lambda m: None)
    assert work_dir.exists()
    with pytest.raises(LibraryError):
        library.remove_series(conn, "s1")


@respx.mock
def test_sync_chapters_multiple_langs(conn):
    respx.get(f"{BASE}/manga/{MANGA_ID}").mock(
        return_value=Response(200, json=MANGA_RESPONSE)
    )
    respx.get(f"{BASE}/manga/{MANGA_ID}/feed").mock(
        return_value=Response(200, json=FEED_RESPONSE)
    )
    library.add_series(conn, MANGA_ID)
    count = library.sync_chapters(conn, MANGA_ID, langs=["en", "pt-br"])
    assert count == 3
    rows = conn.execute(
        "SELECT lang FROM chapters WHERE series_id = ? ORDER BY lang", (MANGA_ID,)
    ).fetchall()
    assert [r["lang"] for r in rows] == ["en", "en", "pt-br"]


def test_load_config_langs_and_backwards_compat(tmp_path):
    from entertainment_harness.config import load_config

    new_config = tmp_path / "config.toml"
    new_config.write_text('[library]\nlangs = ["en", "es"]\n')
    cfg = load_config(new_config)
    assert cfg.library.langs == ["en", "es"]

    old_config = tmp_path / "old_config.toml"
    old_config.write_text('[library]\nlang = "pt-br"\n')
    cfg = load_config(old_config)
    assert cfg.library.langs == ["pt-br"]

    default_config = tmp_path / "default_config.toml"
    default_config.write_text("[hardware]\n")
    cfg = load_config(default_config)
    assert cfg.library.langs == ["en"]
