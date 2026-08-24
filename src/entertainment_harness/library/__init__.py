"""Collection-manager logic: add series, sync chapters, reading progress."""

from __future__ import annotations

import sqlite3

from entertainment_harness.library import works
from entertainment_harness.db import utcnow
from entertainment_harness.sources import Series, get_client, looks_like_id
from entertainment_harness.sources.mangadex import MangaDexClient

SOURCE_MANGADEX = "mangadex"


class LibraryError(Exception):
    pass


def _parse_chapter_num(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def add_series(
    conn: sqlite3.Connection,
    query: str,
    client: MangaDexClient | None = None,
    source: str = SOURCE_MANGADEX,
) -> Series:
    """Add a series by id or by title search (top result) on the given source."""
    client = client or get_client(source)
    if looks_like_id(source, query):
        series = client.get_series(query)
    else:
        results = client.search(query)
        if not results:
            raise LibraryError(f"No {source} results for {query!r}")
        series = results[0]

    existing = conn.execute(
        "SELECT id FROM series WHERE source = ? AND source_id = ?",
        (source, series.id),
    ).fetchone()
    if existing:
        raise LibraryError(f"Series {series.title!r} is already in the library")

    now = utcnow()
    conn.execute(
        "INSERT INTO series (id, title, alt_titles, source, source_id, status, added_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            series.id,
            series.title,
            ", ".join(series.alt_titles),
            source,
            series.id,
            series.status,
            now,
        ),
    )
    conn.commit()
    works.write_work_metadata(
        works.WorkMetadata(
            id=series.id,
            title=series.title,
            source=source,
            source_id=series.id,
            added_at=now,
            kind="manga",
            alt_titles=list(series.alt_titles),
            status=series.status,
        )
    )
    return series


def sync_chapters(
    conn: sqlite3.Connection,
    series_id: str,
    client: MangaDexClient | None = None,
    langs: list[str] | None = None,
    force: bool = False,
) -> int:
    """Fetch the chapter list from the source into the DB. Returns row count.

    Only chapters in one of ``langs`` are kept: the API filters server-side,
    and the results are filtered again here so a stale/off-language row can
    never enter the library.

    With ``force=True``, local chapters in any of ``langs`` that no longer
    exist on the remote source are removed so the library matches the upstream
    list.
    """
    row = conn.execute(
        "SELECT id, source FROM series WHERE id = ?", (series_id,)
    ).fetchone()
    if row is None:
        raise LibraryError(f"Series {series_id!r} is not in the library")
    if client is None:
        client = get_client(row["source"])
    if langs is None:
        langs = ["en"]
    if not langs:
        return 0

    chapters = [c for c in client.chapters(series_id, langs=langs) if c.lang in langs]
    now = utcnow()
    for c in chapters:
        chapter_num = _parse_chapter_num(c.chapter)
        conn.execute(
            "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
            " published_at, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET chapter_num = excluded.chapter_num,"
            " title = excluded.title, lang = excluded.lang, pages = excluded.pages,"
            " published_at = excluded.published_at, fetched_at = excluded.fetched_at",
            (
                c.id,
                series_id,
                chapter_num,
                c.title,
                c.lang,
                c.pages,
                c.published_at,
                now,
            ),
        )
        works.write_chapter_metadata(
            series_id,
            works.ChapterMetadata(
                id=c.id,
                chapter_num=chapter_num,
                title=c.title,
                lang=c.lang,
                pages=c.pages,
                published_at=c.published_at,
                fetched_at=now,
            ),
        )
    if force:
        remote_ids = {c.id for c in chapters}
        placeholders = ", ".join("?" for _ in langs)
        local_ids = [
            r["id"]
            for r in conn.execute(
                f"SELECT id FROM chapters WHERE series_id = ? AND lang IN ({placeholders})",
                (series_id,) + tuple(langs),
            ).fetchall()
        ]
        for chapter_id in local_ids:
            if chapter_id not in remote_ids:
                conn.execute("DELETE FROM translations WHERE chapter_id = ?", (chapter_id,))
                conn.execute("DELETE FROM recaps WHERE chapter_id = ?", (chapter_id,))
                conn.execute("DELETE FROM narrations WHERE chapter_id = ?", (chapter_id,))
                conn.execute("DELETE FROM chapters WHERE id = ?", (chapter_id,))
    conn.commit()
    return len(chapters)


def resolve_series(conn: sqlite3.Connection, series_arg: str) -> sqlite3.Row:
    """Resolve a CLI series argument (id prefix or title substring) to a row."""
    row = conn.execute("SELECT * FROM series WHERE id = ?", (series_arg,)).fetchone()
    if row is None:
        rows = conn.execute(
            "SELECT * FROM series WHERE id LIKE ? OR title LIKE ?",
            (f"{series_arg}%", f"%{series_arg}%"),
        ).fetchall()
        if len(rows) == 1:
            row = rows[0]
        elif not rows:
            raise LibraryError(f"No series matches {series_arg!r}")
        else:
            titles = ", ".join(r["title"] for r in rows)
            raise LibraryError(f"Ambiguous series {series_arg!r}: {titles}")
    return row


def parse_chapter_spec(spec: str) -> list[tuple[float, float]]:
    """Parse a chapter selector like "1-3,4,6-10" into inclusive ranges.

    Chapter numbers are floats (side chapters like 9.5 or 1.2 are valid), so
    ranges are returned as (lo, hi) pairs to match against rather than
    enumerated. Singles parse to (n, n).
    """
    ranges: list[tuple[float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise LibraryError(f"Empty entry in chapter spec {spec!r}")
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = float(lo_s), float(hi_s)
            except ValueError:
                raise LibraryError(
                    f"Bad chapter range {part!r} in {spec!r}"
                ) from None
            if hi < lo:
                raise LibraryError(f"Reversed chapter range {part!r}")
            ranges.append((lo, hi))
        else:
            try:
                n = float(part)
            except ValueError:
                raise LibraryError(f"Bad chapter number {part!r} in {spec!r}") from None
            ranges.append((n, n))
    if not ranges:
        raise LibraryError("Empty chapter spec")
    return ranges


def chapter_in_spec(chapter_num: float | None, ranges: list[tuple[float, float]]) -> bool:
    return chapter_num is not None and any(
        lo <= chapter_num <= hi for lo, hi in ranges
    )


def get_progress(conn: sqlite3.Connection, series_id: str) -> float | None:
    row = conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = ?", (series_id,)
    ).fetchone()
    return row["last_read_chapter"] if row else None


def remove_series(
    conn: sqlite3.Connection,
    series_id: str,
    delete_files: bool = True,
    log=lambda m: None,
) -> None:
    """Delete a series and everything attached to it: DB rows (chapters,
    recaps, narrations, context, progress, videos, translations) and, unless
    delete_files=False, its data dirs (pages, book text, videos)."""
    row = conn.execute(
        "SELECT id, title FROM series WHERE id = ?", (series_id,)
    ).fetchone()
    if row is None:
        raise LibraryError(f"Series {series_id!r} is not in the library")
    chapter_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM chapters WHERE series_id = ?", (series_id,)
        ).fetchall()
    ]
    for table, column in (
        ("translations", "chapter_id"),
        ("recaps", "chapter_id"),
        ("narrations", "chapter_id"),
    ):
        for chapter_id in chapter_ids:
            conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (chapter_id,))
    conn.execute("DELETE FROM chapters WHERE series_id = ?", (series_id,))
    conn.execute("DELETE FROM series_context WHERE series_id = ?", (series_id,))
    conn.execute("DELETE FROM progress WHERE series_id = ?", (series_id,))
    conn.execute("DELETE FROM videos WHERE series_id = ?", (series_id,))
    conn.execute("DELETE FROM series WHERE id = ?", (series_id,))
    conn.commit()
    if delete_files:
        works.remove_work(series_id)
        log(f"  deleted {works.work_dir(series_id)}")
    log(f"Removed {row['title']!r} ({len(chapter_ids)} chapters).")


def advance_progress(
    conn: sqlite3.Connection, series_id: str, chapter_num: float | None
) -> None:
    now = utcnow()
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES (?, ?, ?)"
        " ON CONFLICT(series_id) DO UPDATE SET last_read_chapter ="
        " excluded.last_read_chapter, updated_at = excluded.updated_at",
        (series_id, chapter_num, now),
    )
    conn.commit()
    meta = works.read_work_metadata(series_id)
    if meta is not None:
        meta.progress = works.WorkProgress(
            last_read_chapter=chapter_num, updated_at=now
        )
        works.write_work_metadata(meta)
