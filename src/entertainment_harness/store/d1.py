"""Cloudflare D1 backend for the harness index (cloud execution, Track B).

The pipeline DB is a rebuildable cache of the works layout; in cloud runs
D1 holds the durable copy between jobs. Per-statement REST from inside a
minutes-long pipeline would be slow and would burn the rows-written budget,
so a job stages a local SQLite snapshot at start (snapshot_pull) and flushes
batched row diffs back at chapter boundaries (snapshot_flush). Pipelines
commit per chapter, so a second read connection sees boundary state — no
pipeline changes.

REST shape (verified against the live API 2026-10-16):
POST /accounts/{account}/d1/database/{db}/query {"sql": ..., "params": [...]}
-> {"success": true, "result": [{"results": [...row dicts...], "meta": ...}]};
a missing table fails the whole call with HTTP 400, code 7500 "no such
table". Tables are probed via sqlite_master instead of failing per table.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from entertainment_harness.db import DB_FILENAME, SCHEMA, create_schema, utcnow

# Conservative bound-parameter budget per statement: D1 caps bound
# parameters per query, so multi-row INSERTs are chunked well under it.
MAX_PARAMS_PER_STATEMENT = 90

_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)")

# Harness tables synced to D1, derived from db.SCHEMA so the list tracks the
# schema. The D1-only `jobs` ledger is control-plane state and is never
# snapshotted.
HARNESS_TABLES: tuple[str, ...] = tuple(_TABLE_RE.findall(SCHEMA))


class D1Error(Exception):
    """A D1 REST call failed — the message carries the API error detail."""


@dataclass
class D1Config:
    """Where the durable index lives in cloud runs. All values come from
    env (the Worker injects them into the container in C2)."""

    account_id: str = ""
    database_id: str = ""
    token: str = ""

    @classmethod
    def from_env(cls) -> "D1Config":
        return cls(
            account_id=os.environ.get("D1_ACCOUNT_ID")
            or os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
            database_id=os.environ.get("D1_DATABASE_ID", ""),
            token=os.environ.get("D1_API_TOKEN")
            or os.environ.get("CLOUDFLARE_API_TOKEN", ""),
        )

    def available(self) -> bool:
        return bool(self.account_id and self.database_id and self.token)


def query(cfg: D1Config, sql: str, params: list | tuple = ()) -> list[dict]:
    """Run one D1 statement over the REST API; returns result rows as dicts."""
    resp = httpx.post(
        f"https://api.cloudflare.com/client/v4/accounts/{cfg.account_id}"
        f"/d1/database/{cfg.database_id}/query",
        headers={"Authorization": f"Bearer {cfg.token}"},
        json={"sql": sql, "params": list(params)},
        timeout=30,
    )
    try:
        body = resp.json()
    except ValueError:
        raise D1Error(f"D1 query failed with HTTP {resp.status_code}") from None
    if resp.status_code != 200 or not body.get("success"):
        errors = body.get("errors") if isinstance(body, dict) else None
        detail = (
            "; ".join(str(e.get("message", "?")) for e in errors or [])
            or resp.text[:200]
        )
        raise D1Error(f"D1 query failed: {detail} (sql: {sql[:80]!r})")
    return body["result"][0]["results"]


def _schema_conn(path: Path) -> sqlite3.Connection:
    """Open a fresh SQLite file with the full harness schema (SCHEMA plus
    the additive column migrations, matching what db.connect() produces —
    diff_statements compares SELECT * across the live DB and these files,
    so the column sets must be identical)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    create_schema(conn)
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def _pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in sorted(info, key=lambda r: r["pk"]) if r["pk"]]


def snapshot_pull(cfg: D1Config, dest: Path, baseline: Path) -> int:
    """Materialize the D1 harness tables into two identical local files:
    `dest` (the DB the job runs against — the data dir's harness.db) and
    `baseline` (the diff reference for snapshot_flush). Returns the number
    of rows staged; 0 means a fresh database and the caller should
    bootstrap by indexing the staged works layout."""
    remote_tables = {
        r["name"]
        for r in query(cfg, "SELECT name FROM sqlite_master WHERE type='table'")
    }
    total = 0
    for path in (dest, baseline):
        # Rebuild from scratch: a persisted volume could carry a file written
        # by an older schema, and create_schema only adds missing columns to
        # an existing file (it never drops stale ones).
        path.unlink(missing_ok=True)
        conn = _schema_conn(path)
        try:
            for table in HARNESS_TABLES:
                if table not in remote_tables:
                    continue
                rows = query(cfg, f"SELECT * FROM {table}")
                cols = _columns(conn, table)
                placeholders = ", ".join("?" for _ in cols)
                conn.executemany(
                    f"INSERT OR REPLACE INTO {table}"
                    f" ({', '.join(cols)}) VALUES ({placeholders})",
                    [[row.get(c) for c in cols] for row in rows],
                )
                total += len(rows)
            conn.commit()
        finally:
            conn.close()
    return total // 2  # rows were written to both files


@dataclass
class FlushOp:
    table: str
    sql: str
    params: list = field(default_factory=list)
    rows: int = 0  # rows affected (for stats; deletes carry 1 each)


def diff_statements(local: Path, baseline: Path) -> list[FlushOp]:
    """Row-level diff of `local` vs `baseline` per harness table, as batched
    INSERT OR REPLACE statements (changed/new rows) plus per-row DELETEs
    (rows absent locally).

    Op order is FK-safe: upserts come first in schema order (parents before
    children, so a child row never lands before its parent), DELETEs last in
    reverse schema order (children before parents, so a row is never removed
    while D1 still holds its children — one bad delete otherwise aborts the
    flush and every insert queued behind it)."""
    conn = sqlite3.connect(local)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS base", (str(baseline),))
        upserts: list[FlushOp] = []
        deletes: list[FlushOp] = []
        for table in HARNESS_TABLES:
            cols = _columns(conn, table)
            base_cols = [
                r["name"] for r in conn.execute(f'PRAGMA base.table_info("{table}")')
            ]
            if cols != base_cols:
                raise D1Error(
                    f"schema drift on {table!r}: local columns {cols}"
                    f" vs baseline {base_cols}"
                    f" (rebuild the baseline; snapshot_pull deletes stale files)"
                )
            pk = _pk_columns(conn, table)
            if not pk:
                continue  # every harness table declares a primary key
            changed = conn.execute(
                f"SELECT * FROM main.{table}"
                f" EXCEPT SELECT * FROM base.{table}"
            ).fetchall()
            removed = conn.execute(
                f"SELECT {', '.join(pk)} FROM base.{table}"
                f" EXCEPT SELECT {', '.join(pk)} FROM main.{table}"
            ).fetchall()
            per_stmt = max(1, MAX_PARAMS_PER_STATEMENT // len(cols))
            col_list = ", ".join(cols)
            for i in range(0, len(changed), per_stmt):
                chunk = changed[i : i + per_stmt]
                values = ", ".join(
                    "(" + ", ".join("?" for _ in cols) + ")" for _ in chunk
                )
                upserts.append(FlushOp(
                    table,
                    f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES {values}",
                    [row[c] for row in chunk for c in cols],
                    rows=len(chunk),
                ))
            where = " AND ".join(f"{c} = ?" for c in pk)
            for row in removed:
                deletes.append(FlushOp(
                    table,
                    f"DELETE FROM {table} WHERE {where}",
                    [row[c] for c in pk],
                    rows=1,
                ))
        return upserts + deletes[::-1]
    finally:
        conn.close()


def snapshot_flush(cfg: D1Config, local: Path, baseline: Path) -> dict:
    """Push the local-vs-baseline row diff to D1, then advance the baseline
    (per-table contents are replaced with the current local ones, so the
    next flush sends only newer changes). On failure the baseline is left
    untouched and the next flush retries the full delta. Returns per-table
    {upserted, deleted} counts."""
    ops = diff_statements(local, baseline)
    stats: dict[str, dict[str, int]] = {}
    for op in ops:
        query(cfg, op.sql, op.params)
        entry = stats.setdefault(op.table, {"upserted": 0, "deleted": 0})
        entry["deleted" if op.sql.startswith("DELETE") else "upserted"] += op.rows
    if ops:
        conn = sqlite3.connect(local)
        try:
            conn.execute("ATTACH DATABASE ? AS base", (str(baseline),))
            with conn:
                for table in HARNESS_TABLES:
                    conn.execute(f"DELETE FROM base.{table}")
                    conn.execute(
                        f"INSERT INTO base.{table} SELECT * FROM main.{table}"
                    )
            conn.execute("DETACH DATABASE base")
        finally:
            conn.close()
    return stats


def default_paths(job_id: str, root: Path) -> tuple[Path, Path]:
    """(local DB, baseline) for a job: the data dir's harness.db and a
    per-job baseline under jobs/<id>/."""
    return root / DB_FILENAME, root / "jobs" / job_id / "baseline.db"


# --- jobs ledger -------------------------------------------------------------
# The D1-only control-plane table (never snapshotted — see HARNESS_TABLES).
# The Worker owns rows at submit time; the job container updates status,
# cursor, and error over REST, and polls the stop flag at chapter boundaries.


def ledger_update(
    cfg: D1Config,
    job_id: str,
    *,
    status: str | None = None,
    current: dict | None = None,
    error: str | None = None,
) -> None:
    """Update a jobs ledger row. The cursor is always rewritten (None clears
    it) so a terminal write leaves no stale chapter position behind."""
    sets = ["updated_at = ?", "current_json = ?"]
    params: list = [utcnow(), json.dumps(current) if current else None]
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if error is not None:
        sets.append("error = ?")
        params.append(error)
    params.append(job_id)
    query(cfg, f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)


def ledger_stop_requested(cfg: D1Config, job_id: str) -> bool:
    """Whether POST /api/jobs/{id}/stop has flagged this job."""
    rows = query(cfg, "SELECT stop_requested FROM jobs WHERE id = ?", [job_id])
    return bool(rows and rows[0].get("stop_requested"))
