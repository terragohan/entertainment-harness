"""SQLite schema + helpers (stdlib sqlite3). DB lives at <data_dir>/harness.db.

Schema is created on open (migration-on-open); see docs/design.md "SQLite schema".
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from entertainment_harness.config import data_dir

DB_FILENAME = "harness.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    alt_titles TEXT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    status TEXT,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chapters (
    id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    chapter_num REAL,
    title TEXT,
    lang TEXT,
    pages INTEGER,
    published_at TEXT,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    series_id TEXT PRIMARY KEY REFERENCES series(id),
    last_read_chapter REAL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recaps (
    id INTEGER PRIMARY KEY,
    chapter_id TEXT NOT NULL UNIQUE REFERENCES chapters(id),
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT
);
CREATE TABLE IF NOT EXISTS narrations (
    id INTEGER PRIMARY KEY,
    chapter_id TEXT NOT NULL UNIQUE REFERENCES chapters(id),
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT
);
CREATE TABLE IF NOT EXISTS series_context (
    series_id TEXT PRIMARY KEY REFERENCES series(id),
    rolling_summary TEXT NOT NULL,
    through_chapter REAL
);
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    from_chapter REAL,
    to_chapter REAL,
    path TEXT NOT NULL,
    duration_s REAL,
    created_at TEXT NOT NULL,
    tts_engine TEXT,
    model TEXT,
    video_gen_provider TEXT DEFAULT 'local',
    wiped_at TEXT
);
CREATE TABLE IF NOT EXISTS online_summaries (
    series_id TEXT PRIMARY KEY REFERENCES series(id),
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    summary TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    steering_prompt TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS translations (
    chapter_id TEXT PRIMARY KEY REFERENCES chapters(id),
    pages INTEGER,
    model TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    role TEXT NOT NULL DEFAULT '',
    first_seen REAL,
    last_seen REAL,
    origin TEXT NOT NULL DEFAULT 'observed',
    edited INTEGER NOT NULL DEFAULT 0,
    UNIQUE(series_id, name)
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fold_pending(conn: sqlite3.Connection) -> bool:
    """Read-only check for un-folded narrations (keeps connect() from taking
    a write lock on every open — see fold_narrations_into_recaps)."""
    return bool(
        conn.execute(
            "SELECT EXISTS("
            " SELECT 1 FROM narrations n"
            "  LEFT JOIN recaps r ON r.chapter_id = n.chapter_id"
            "  WHERE r.chapter_id IS NULL"
            ") OR EXISTS("
            " SELECT 1 FROM recaps"
            "  WHERE detail != 'full'"
            "  AND chapter_id IN (SELECT chapter_id FROM narrations)"
            ")"
        ).fetchone()[0]
    )


def fold_narrations_into_recaps(conn: sqlite3.Connection) -> None:
    """Fold legacy narrations rows into recaps (unify-recap-narrate Phase 1).

    A chapter with only a narration gets a new recaps row at detail='full'
    (summary/model/created_at taken from the narration); a chapter with both
    keeps its recaps row, upgraded in place to detail='full' — the narration
    text supersedes the standard summary (its model/created_at follow the
    text, since the artifact is now the narration). The narrations table
    itself is never touched: it stays behind fully populated. series_context
    is not touched either (each chapter's events were folded into the rolling
    summary when its recap/narration was first stored).

    Idempotent: the INSERT only fills chapters still missing a recaps row and
    the UPDATE only touches rows not yet at detail='full', so a second run is
    a no-op.

    The write only happens when a read-only pre-check finds un-folded
    narrations: connect() calls this on every open, and an unconditional
    INSERT/UPDATE would make every read-only caller fail with "database is
    locked" whenever a long-running pipeline holds a write transaction (e.g.
    an `eh serve` run between LLM calls). For the same reason a locked
    database defers the fold to the next open rather than raising.
    """
    if not _fold_pending(conn):
        return
    try:
        conn.execute(
            "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
            " SELECT n.chapter_id, n.text, n.created_at, n.model, 'full'"
            " FROM narrations n LEFT JOIN recaps r ON r.chapter_id = n.chapter_id"
            " WHERE r.chapter_id IS NULL"
        )
        conn.execute(
            "UPDATE recaps SET"
            " summary = (SELECT text FROM narrations"
            "  WHERE narrations.chapter_id = recaps.chapter_id),"
            " model = COALESCE((SELECT model FROM narrations"
            "  WHERE narrations.chapter_id = recaps.chapter_id), model),"
            " created_at = (SELECT created_at FROM narrations"
            "  WHERE narrations.chapter_id = recaps.chapter_id),"
            " detail = 'full'"
            " WHERE detail != 'full'"
            " AND chapter_id IN (SELECT chapter_id FROM narrations)"
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        conn.rollback()
        if "locked" not in str(exc):
            raise
        # Another connection holds the write lock; the fold stays pending
        # and runs on a later connect().


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (creating + migrating if needed) the harness database."""
    path = db_path if db_path is not None else data_dir() / DB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # series.kind: manga (default) | book | comic — imported via 'eh import'.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(series)")}
    if "kind" not in cols:
        conn.execute(
            "ALTER TABLE series ADD COLUMN kind TEXT NOT NULL DEFAULT 'manga'"
        )
        conn.commit()
    # video generation provider attribution (local | runway | ...)
    video_cols = {r["name"] for r in conn.execute("PRAGMA table_info(videos)")}
    if "video_gen_provider" not in video_cols:
        conn.execute(
            "ALTER TABLE videos ADD COLUMN video_gen_provider TEXT DEFAULT 'local'"
        )
        conn.commit()
    # video kind: recap (default) | narration — set by 'eh recap' (grain-dependent)
    if "kind" not in video_cols:
        conn.execute(
            "ALTER TABLE videos ADD COLUMN kind TEXT NOT NULL DEFAULT 'recap'"
        )
        conn.commit()
    # wiped_at: set by 'eh wipe' when a rendered video's files are deleted;
    # cleared on the next render.
    if "wiped_at" not in video_cols:
        conn.execute("ALTER TABLE videos ADD COLUMN wiped_at TEXT")
        conn.commit()
    # recaps.detail: gist | brief | standard (default) | detailed | full —
    # the artifact grain from the unify-recap-narrate merge; narrations fold
    # into recaps as detail='full'.
    recap_cols = {r["name"] for r in conn.execute("PRAGMA table_info(recaps)")}
    if "detail" not in recap_cols:
        conn.execute(
            "ALTER TABLE recaps ADD COLUMN detail TEXT NOT NULL DEFAULT 'standard'"
        )
        conn.commit()
    # recaps.instruction: the steering direction the artifact was generated
    # with ('' = none). Attribution only — changing it never invalidates
    # existing artifacts.
    if "instruction" not in recap_cols:
        conn.execute(
            "ALTER TABLE recaps ADD COLUMN instruction TEXT NOT NULL DEFAULT ''"
        )
        conn.commit()
    # recaps.standalone: 1 when the artifact was generated without the rolling
    # story-so-far (recap/narrate --video gap-fill chapters behind the read
    # frontier — folding them would double-fold the context tape).
    if "standalone" not in recap_cols:
        conn.execute(
            "ALTER TABLE recaps ADD COLUMN standalone INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()
    # One-time narrations fold-in; self-guarding (no-op once every narrated
    # chapter has a full-detail recaps row), so it is safe to run on every
    # open — e.g. after an 'eh index' rebuild repopulates narrations.
    fold_narrations_into_recaps(conn)
    # One-time migration from the legacy split layout to data/works/.
    if os.environ.get("EH_NO_AUTO_MIGRATE") not in ("1", "true", "yes"):
        from entertainment_harness.library import migrate

        migrate.migrate(conn)
    return conn


# --- character registry (character-bible initiative) -------------------------


def get_characters(conn: sqlite3.Connection, series_id: str) -> list[sqlite3.Row]:
    """The work's cast registry, ordered for prompt injection (most recently
    seen first)."""
    return conn.execute(
        "SELECT * FROM characters WHERE series_id = ?"
        " ORDER BY last_seen DESC, name",
        (series_id,),
    ).fetchall()


def upsert_character(
    conn: sqlite3.Connection,
    series_id: str,
    name: str,
    *,
    aliases: list[str] | None = None,
    role: str = "",
    chapter_num: float | None = None,
    origin: str = "observed",
    edited: bool = False,
) -> None:
    """Insert or update one registry row by (series_id, name). Aliases union
    with the stored ones; a non-empty role/last_seen wins over the old. The
    caller merges semantics (pipelines/characters.py); this is persistence."""
    existing = conn.execute(
        "SELECT * FROM characters WHERE series_id = ? AND name = ?",
        (series_id, name),
    ).fetchone()
    alias_set = list(dict.fromkeys(aliases or []))
    if existing is None:
        conn.execute(
            "INSERT INTO characters"
            " (series_id, name, aliases, role, first_seen, last_seen, origin,"
            "  edited)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                series_id, name, json.dumps(alias_set), role,
                chapter_num, chapter_num, origin, 1 if edited else 0,
            ),
        )
        return
    merged = list(
        dict.fromkeys([*json.loads(existing["aliases"]), *alias_set])
    )
    last_seen = chapter_num
    if existing["last_seen"] is not None:
        last_seen = max(existing["last_seen"], chapter_num or existing["last_seen"])
    conn.execute(
        "UPDATE characters SET aliases = ?, role = ?, last_seen = ?,"
        " origin = ?, edited = ? WHERE id = ?",
        (
            json.dumps(merged),
            role or existing["role"],
            last_seen,
            origin if edited else existing["origin"],
            1 if (edited or existing["edited"]) else 0,
            existing["id"],
        ),
    )


def replace_characters(
    conn: sqlite3.Connection, series_id: str, cast: list[dict]
) -> None:
    """Replace the whole registry (the UI's save). Every row lands as
    origin='user', edited=1 — user-touched entries are extractor-locked."""
    conn.execute("DELETE FROM characters WHERE series_id = ?", (series_id,))
    for entry in cast:
        conn.execute(
            "INSERT INTO characters"
            " (series_id, name, aliases, role, first_seen, last_seen, origin,"
            "  edited)"
            " VALUES (?, ?, ?, ?, ?, ?, 'user', 1)",
            (
                series_id,
                entry["name"],
                json.dumps(list(entry.get("aliases") or [])),
                entry.get("role") or "",
                entry.get("first_seen"),
                entry.get("last_seen"),
            ),
        )
    conn.commit()
