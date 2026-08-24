"""db.py schema/migration tests for the unify-recap-narrate Phase 1 merge:
the additive recaps.detail ALTER and the one-time narrations fold-in
(db.fold_narrations_into_recaps), incl. idempotency. The legacy narrations
table and series_context must survive untouched.
"""

from __future__ import annotations

import pytest

from entertainment_harness import db


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    yield conn
    conn.close()


def _seed_series(conn):
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, ?, 'en', 9, 'now')",
        [("ch-1", 1.0, "C1"), ("ch-2", 2.0, "C2"), ("ch-3", 3.0, "C3")],
    )
    conn.commit()


def _narrations(conn):
    return {
        r["chapter_id"]: r["text"]
        for r in conn.execute("SELECT * FROM narrations")
    }


def _recaps(conn):
    return {
        r["chapter_id"]: (r["summary"], r["detail"], r["model"], r["created_at"])
        for r in conn.execute("SELECT * FROM recaps")
    }


def test_recaps_detail_column_defaults_to_standard(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(recaps)")}
    assert "detail" in cols
    _seed_series(conn)
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('ch-1', 's', 'now', 'm')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()
    assert row["detail"] == "standard"


def test_recaps_instruction_column_defaults_to_empty(conn):
    """recaps.instruction (Phase 5 steering attribution) is an additive ALTER
    defaulting to '' for rows written without one."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(recaps)")}
    assert "instruction" in cols
    _seed_series(conn)
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('ch-1', 's', 'now', 'm')"
    )
    conn.commit()
    row = conn.execute(
        "SELECT instruction FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()
    assert row["instruction"] == ""


def test_connect_migrations_are_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    db.connect().close()
    conn = db.connect()  # second open: every ALTER is a guarded no-op
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(recaps)")}
    assert "detail" in cols
    conn.close()


def test_fold_narration_only_chapter_gets_full_recap(conn):
    _seed_series(conn)
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES ('ch-1', 'full retelling', '2026-01-01T00:00:00+00:00', 'nm')"
    )
    conn.commit()

    db.fold_narrations_into_recaps(conn)

    row = conn.execute(
        "SELECT * FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()
    assert row["detail"] == "full"
    assert row["summary"] == "full retelling"
    assert row["model"] == "nm"
    assert row["created_at"] == "2026-01-01T00:00:00+00:00"
    # the narrations row stays behind, untouched
    assert _narrations(conn) == {"ch-1": "full retelling"}


def test_fold_both_artifacts_chapter_full_supersedes(conn):
    """A chapter with both a recap and a narration keeps ONE recaps row,
    upgraded in place: the narration text (and its provenance) wins."""
    _seed_series(conn)
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('ch-1', 'standard summary', '2026-01-01T00:00:00+00:00', 'rm')"
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES ('ch-1', 'full retelling', '2026-02-02T00:00:00+00:00', 'nm')"
    )
    conn.commit()

    db.fold_narrations_into_recaps(conn)

    rows = conn.execute(
        "SELECT * FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchall()
    assert len(rows) == 1  # upgraded in place, not duplicated
    row = rows[0]
    assert row["detail"] == "full"
    assert row["summary"] == "full retelling"  # standard summary discarded
    assert row["model"] == "nm"
    assert row["created_at"] == "2026-02-02T00:00:00+00:00"
    assert _narrations(conn) == {"ch-1": "full retelling"}


def test_fold_keeps_unnarrated_recaps_at_standard(conn):
    _seed_series(conn)
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('ch-1', 'standard summary', 'now', 'rm')"
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES ('ch-2', 'full retelling', 'now', 'nm')"
    )
    conn.commit()

    db.fold_narrations_into_recaps(conn)

    assert _recaps(conn) == {
        "ch-1": ("standard summary", "standard", "rm", "now"),
        "ch-2": ("full retelling", "full", "nm", "now"),
    }


def test_fold_is_idempotent(conn):
    _seed_series(conn)
    conn.executemany(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES (?, 'standard summary', 'now', 'rm')",
        [("ch-1",), ("ch-2",)],
    )
    conn.executemany(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES (?, 'full retelling', 'then', 'nm')",
        [("ch-2",), ("ch-3",)],
    )
    conn.commit()

    db.fold_narrations_into_recaps(conn)
    first = (_recaps(conn), _narrations(conn))
    db.fold_narrations_into_recaps(conn)  # second run: no-op
    assert (_recaps(conn), _narrations(conn)) == first
    assert _recaps(conn) == {
        "ch-1": ("standard summary", "standard", "rm", "now"),
        "ch-2": ("full retelling", "full", "nm", "then"),
        "ch-3": ("full retelling", "full", "nm", "then"),
    }


def test_fold_leaves_series_context_untouched(conn):
    _seed_series(conn)
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES ('s1', 'story so far', 3.0)"
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES ('ch-2', 'full retelling', 'then', 'nm')"
    )
    conn.commit()

    db.fold_narrations_into_recaps(conn)

    row = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    assert (row["rolling_summary"], row["through_chapter"]) == ("story so far", 3.0)


def test_connect_succeeds_while_another_connection_holds_write_lock(
    tmp_path, monkeypatch
):
    """An uncommitted write transaction on one connection (an `eh serve` run
    between LLM calls) must not break other connect()s: with nothing to fold
    connect() stays read-only, and a pending fold defers instead of raising
    "database is locked" (it lands on a later open)."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    blocker = db.connect()
    _seed_series(blocker)
    blocker.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES ('ch-1', 'full retelling', 'then', 'nm')"
    )
    blocker.commit()
    # Hold the write lock with an uncommitted INSERT.
    blocker.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s2', 'T2', 'mangadex', 'y', 'ongoing', 'now')"
    )

    conn = db.connect()  # must not raise; the pending fold defers
    assert _recaps(conn) == {}  # nothing folded while locked
    conn.close()

    blocker.commit()
    blocker.close()
    conn = db.connect()  # lock released: the fold lands now
    assert _recaps(conn) == {
        "ch-1": ("full retelling", "full", "nm", "then"),
    }
    conn.close()
