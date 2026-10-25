"""D1 store tests (store/d1.py): REST client against respx-mocked API,
snapshot pull, offline diff, flush + baseline advance. No network.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from entertainment_harness.store import d1


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    for var in ("D1_ACCOUNT_ID", "D1_DATABASE_ID", "D1_API_TOKEN",
                "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def cfg() -> d1.D1Config:
    return d1.D1Config(account_id="acc", database_id="db1", token="tok")


API = "https://api.cloudflare.com/client/v4/accounts/acc/d1/database/db1/query"


def _ok(rows, changed=None):
    return httpx.Response(200, json={
        "success": True,
        "errors": [],
        "result": [{"results": rows, "success": True, "meta": {}}],
    })


# --- config -----------------------------------------------------------------


def test_config_from_env(root, monkeypatch):
    assert not d1.D1Config.from_env().available()
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("D1_DATABASE_ID", "db1")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    env = d1.D1Config.from_env()
    assert env.available()
    assert env.account_id == "acc" and env.database_id == "db1"


# --- REST client --------------------------------------------------------------


@respx.mock
def test_query_returns_rows(root, cfg):
    route = respx.post(API).mock(return_value=_ok([{"id": "s1", "n": 1}]))
    rows = d1.query(cfg, "SELECT * FROM series")
    assert rows == [{"id": "s1", "n": 1}]
    body = json.loads(route.calls[0].request.content)
    assert body == {"sql": "SELECT * FROM series", "params": []}
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"


@respx.mock
def test_query_api_error_raises(root, cfg):
    respx.post(API).mock(return_value=httpx.Response(400, json={
        "success": False,
        "errors": [{"code": 7500, "message": "no such table: nope: SQLITE_ERROR"}],
        "result": [],
    }))
    with pytest.raises(d1.D1Error, match="no such table"):
        d1.query(cfg, "SELECT * FROM nope")


# --- snapshot pull --------------------------------------------------------------


@respx.mock
def test_snapshot_pull_stages_rows(root, cfg):
    def route(request):
        sql = json.loads(request.content)["sql"]
        if "sqlite_master" in sql:
            return _ok([{"name": "series"}, {"name": "chapters"}])
        if "FROM series" in sql:
            return _ok([{
                "id": "s1", "title": "Demo", "alt_titles": None,
                "source": "mangadex", "source_id": "x", "status": None,
                "added_at": "2026-01-01T00:00:00+00:00", "kind": "manga",
            }])
        if "FROM chapters" in sql:
            return _ok([{
                "id": "c1", "series_id": "s1", "chapter_num": 1.0,
                "title": None, "lang": "en", "pages": 10,
                "published_at": None, "fetched_at": "2026-01-01T00:00:00+00:00",
            }])
        raise AssertionError(f"unexpected sql: {sql}")

    respx.post(API).mock(side_effect=route)
    dest = root / "harness.db"
    baseline = root / "jobs" / "j1" / "baseline.db"
    rows = d1.snapshot_pull(cfg, dest, baseline)
    assert rows == 2
    import sqlite3

    for path in (dest, baseline):
        conn = sqlite3.connect(path)
        assert conn.execute("SELECT title FROM series WHERE id='s1'").fetchone()[0] == "Demo"
        assert conn.execute("SELECT pages FROM chapters WHERE id='c1'").fetchone()[0] == 10
        conn.close()


@respx.mock
def test_snapshot_pull_fresh_database(root, cfg):
    respx.post(API).mock(return_value=_ok([]))  # sqlite_master: no tables
    dest = root / "harness.db"
    rows = d1.snapshot_pull(cfg, dest, root / "b.db")
    assert rows == 0
    assert dest.exists()  # schema-only staging file


# --- diff (offline) --------------------------------------------------------------


def _seed(path, rows):
    conn = d1._schema_conn(path)
    conn.executemany(
        "INSERT INTO series (id, title, source, source_id, added_at)"
        " VALUES (?, ?, 'mangadex', 'x', 't')",
        rows,
    )
    conn.commit()
    conn.close()


def test_diff_insert_update_delete(root):
    local = root / "local.db"
    baseline = root / "base.db"
    _seed(baseline, [("keep", "Keep"), ("change", "Old"), ("drop", "Drop")])
    _seed(local, [("keep", "Keep"), ("change", "New"), ("new", "New Row")])
    ops = d1.diff_statements(local, baseline)
    inserts = [o for o in ops if o.sql.startswith("INSERT")]
    deletes = [o for o in ops if o.sql.startswith("DELETE")]
    # changed + new rows upserted together; untouched row excluded.
    assert len(inserts) == 1 and inserts[0].rows == 2
    assert "INSERT OR REPLACE INTO series" in inserts[0].sql
    assert inserts[0].params[0:2] == ["change", "New"]
    assert inserts[0].params[8:10] == ["new", "New Row"]
    assert deletes == [
        d1.FlushOp("series", "DELETE FROM series WHERE id = ?", ["drop"], 1)
    ]


def test_diff_orders_fk_safely(root):
    """Regression (cloud job 8f6b05faaec1, 2026-10-19): a vanished work's
    DELETEs went out in schema order — series first — and D1 rejected the
    series delete because its chapters/recaps were still present, aborting
    every insert queued behind it. Upserts now lead; deletes run
    children-before-parents."""
    import shutil
    import sqlite3

    local = root / "local.db"
    baseline = root / "base.db"
    _seed(local, [("s2", "New")])
    conn = d1._schema_conn(baseline)
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, added_at)"
        " VALUES ('s1', 'Gone', 'mangadex', 'x', 't')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, fetched_at)"
        " VALUES ('c1', 's1', 1.0, 't')"
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at)"
        " VALUES ('c1', 'sum', 't')"
    )
    conn.commit()
    conn.close()
    ops = d1.diff_statements(local, baseline)
    assert [(o.table, o.sql.split(" ")[0]) for o in ops] == [
        ("series", "INSERT"),
        ("recaps", "DELETE"),
        ("chapters", "DELETE"),
        ("series", "DELETE"),
    ]
    # The sequence applies cleanly against a FK-enforcing copy of the
    # baseline (the old parent-first order raised SQLITE_CONSTRAINT here).
    remote = root / "remote.db"
    shutil.copy(baseline, remote)
    conn = sqlite3.connect(remote)
    conn.execute("PRAGMA foreign_keys = ON")
    for op in ops:
        conn.execute(op.sql, op.params)
    conn.commit()
    rows = conn.execute("SELECT id FROM series").fetchall()
    assert [r[0] for r in rows] == ["s2"]
    assert conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM recaps").fetchone()[0] == 0
    conn.close()


def test_diff_respects_param_budget(root, monkeypatch):
    monkeypatch.setattr(d1, "MAX_PARAMS_PER_STATEMENT", 8)  # series: 8 cols
    local = root / "local.db"
    baseline = root / "base.db"
    _seed(baseline, [])
    _seed(local, [("a", "A"), ("b", "B"), ("c", "C")])
    ops = [o for o in d1.diff_statements(local, baseline)
           if o.sql.startswith("INSERT")]
    assert len(ops) == 3  # 8 params // 8 cols = 1 row per statement
    assert all(o.rows == 1 for o in ops)


def test_diff_live_db_against_staged_baseline(root, monkeypatch):
    """Regression (C2 cloud run, 2026-10-17): the job DB goes through
    db.connect() — SCHEMA plus additive column migrations — while the
    baseline came from SCHEMA alone, so the EXCEPT diff died on mismatched
    column counts and the D1 flush never landed. Both sides must now share
    the migrated schema."""
    monkeypatch.setenv("EH_NO_AUTO_MIGRATE", "1")
    from entertainment_harness import db

    local = root / "harness.db"
    conn = db.connect(local)
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, added_at)"
        " VALUES ('s1', 'Live', 'mangadex', 'x', 't')"
    )
    conn.commit()
    conn.close()
    baseline = root / "jobs" / "j1" / "baseline.db"
    d1._schema_conn(baseline).close()
    ops = d1.diff_statements(local, baseline)
    assert any(o.sql.startswith("INSERT OR REPLACE INTO series") for o in ops)


def test_diff_schema_drift_names_table_and_columns(root):
    """A baseline whose columns drifted from the job DB must fail loudly
    naming the table and both column lists (the EXCEPT it would otherwise
    die on names neither)."""
    import sqlite3

    local = root / "local.db"
    baseline = root / "base.db"
    _seed(local, [("s1", "Live")])
    _seed(baseline, [])
    conn = sqlite3.connect(baseline)
    conn.execute("ALTER TABLE series ADD COLUMN stale TEXT")
    conn.commit()
    conn.close()
    with pytest.raises(d1.D1Error, match=r"schema drift on 'series'") as exc:
        d1.diff_statements(local, baseline)
    assert "stale" in str(exc.value)
    assert "kind" in str(exc.value)  # the live column the stale file lacks


@respx.mock
def test_snapshot_pull_rebuilds_stale_files(root, cfg):
    """A persisted volume can leave a snapshot file written by an older
    schema; snapshot_pull deletes and rebuilds both files rather than
    layering the current schema on top of the stale one."""
    import sqlite3

    dest = root / "harness.db"
    baseline = root / "b.db"
    conn = sqlite3.connect(dest)
    conn.execute(
        "CREATE TABLE series (id TEXT PRIMARY KEY, stale TEXT)"
    )
    conn.commit()
    conn.close()
    respx.post(API).mock(return_value=_ok([]))  # no remote tables
    assert d1.snapshot_pull(cfg, dest, baseline) == 0
    conn = sqlite3.connect(dest)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(series)")]
    conn.close()
    assert "stale" not in cols
    assert "title" in cols


# --- flush --------------------------------------------------------------


@respx.mock
def test_flush_posts_and_advances_baseline(root, cfg):
    posted = []

    def route(request):
        body = json.loads(request.content)
        posted.append(body["sql"])
        return _ok([])

    respx.post(API).mock(side_effect=route)
    local = root / "local.db"
    baseline = root / "base.db"
    _seed(baseline, [("keep", "Keep")])
    _seed(local, [("keep", "Keep"), ("new", "New Row")])

    stats = d1.snapshot_flush(cfg, local, baseline)
    assert stats == {"series": {"upserted": 1, "deleted": 0}}
    assert any("INSERT OR REPLACE INTO series" in sql for sql in posted)

    # Baseline advanced: a second flush finds nothing to do and posts nothing.
    assert d1.snapshot_flush(cfg, local, baseline) == {}
    assert len(posted) == 1


def test_flush_failure_leaves_baseline(root, cfg, monkeypatch):
    local = root / "local.db"
    baseline = root / "base.db"
    _seed(baseline, [])
    _seed(local, [("a", "A")])

    def boom(cfg, sql, params=()):
        raise d1.D1Error("offline")

    monkeypatch.setattr(d1, "query", boom)
    with pytest.raises(d1.D1Error):
        d1.snapshot_flush(cfg, local, baseline)
    # Baseline not advanced: the next diff still carries the row.
    assert d1.diff_statements(local, baseline)


# --- jobs ledger ------------------------------------------------------------


@respx.mock
def test_ledger_update_sql(root, cfg):
    route = respx.post(API).mock(return_value=_ok([]))
    d1.ledger_update(cfg, "j1", status="running", current={"chapter": 2})
    body = json.loads(route.calls[0].request.content)
    assert body["sql"].startswith(
        "UPDATE jobs SET updated_at = ?, current_json = ?, status = ?"
    )
    assert body["sql"].endswith("WHERE id = ?")
    assert body["params"][1] == '{"chapter": 2}'
    assert body["params"][2:] == ["running", "j1"]


@respx.mock
def test_ledger_update_terminal_clears_cursor(root, cfg):
    route = respx.post(API).mock(return_value=_ok([]))
    d1.ledger_update(cfg, "j1", status="done")
    body = json.loads(route.calls[0].request.content)
    assert body["params"][1] is None  # stale cursor cleared
    assert "error = ?" not in body["sql"]  # untouched when None


@respx.mock
def test_ledger_stop_requested(root, cfg):
    respx.post(API).mock(return_value=_ok([{"stop_requested": 1}]))
    assert d1.ledger_stop_requested(cfg, "j1") is True
