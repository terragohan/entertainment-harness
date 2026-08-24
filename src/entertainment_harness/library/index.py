"""Rebuild the SQLite index from the filesystem.

The DB is a queryable cache. `index_works()` scans `data/works/` and recreates
all rows. Callers can use this after moving works around or after a fresh
install.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.config import data_dir
from entertainment_harness.db import connect, utcnow


def _clear_index(conn: sqlite3.Connection) -> None:
    """Remove all rows from derived tables (keep schema)."""
    for table in (
        "online_summaries",
        "translations",
        "videos",
        "narrations",
        "recaps",
        "series_context",
        "progress",
        "chapters",
        "series",
    ):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def index_work(conn: sqlite3.Connection, work_dir: Path) -> None:
    """Index a single work directory into the DB."""
    meta = works.read_work_metadata_at(work_dir)
    if meta is None:
        return

    conn.execute(
        "INSERT INTO series (id, title, alt_titles, source, source_id, status,"
        " added_at, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET title = excluded.title,"
        " alt_titles = excluded.alt_titles, source = excluded.source,"
        " source_id = excluded.source_id, status = excluded.status,"
        " added_at = excluded.added_at, kind = excluded.kind",
        (
            meta.id,
            meta.title,
            ", ".join(meta.alt_titles),
            meta.source,
            meta.source_id,
            meta.status,
            meta.added_at,
            meta.kind,
        ),
    )

    if meta.progress.updated_at is not None:
        conn.execute(
            "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
            " VALUES (?, ?, ?) ON CONFLICT(series_id) DO UPDATE SET"
            " last_read_chapter = excluded.last_read_chapter,"
            " updated_at = excluded.updated_at",
            (
                meta.id,
                meta.progress.last_read_chapter,
                meta.progress.updated_at,
            ),
        )

    if meta.context.rolling_summary:
        conn.execute(
            "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
            " VALUES (?, ?, ?) ON CONFLICT(series_id) DO UPDATE SET"
            " rolling_summary = excluded.rolling_summary,"
            " through_chapter = excluded.through_chapter",
            (
                meta.id,
                meta.context.rolling_summary,
                meta.context.through_chapter,
            ),
        )

    if meta.online_summary is not None:
        osum = meta.online_summary
        conn.execute(
            "INSERT INTO online_summaries (series_id, provider, query, summary,"
            " sources_json, steering_prompt, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(series_id) DO UPDATE SET provider = excluded.provider,"
            " query = excluded.query, summary = excluded.summary,"
            " sources_json = excluded.sources_json,"
            " steering_prompt = excluded.steering_prompt,"
            " created_at = excluded.created_at",
            (
                meta.id,
                osum.provider,
                osum.query,
                osum.summary,
                json.dumps(osum.sources),
                osum.steering_prompt,
                osum.created_at or utcnow(),
            ),
        )

    for chapter_dir in works.list_chapter_dirs_at(work_dir):
        chapter_meta = works.read_chapter_metadata_at(chapter_dir)
        if chapter_meta is None:
            continue
        conn.execute(
            "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
            " published_at, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET series_id = excluded.series_id,"
            " chapter_num = excluded.chapter_num, title = excluded.title,"
            " lang = excluded.lang, pages = excluded.pages,"
            " published_at = excluded.published_at, fetched_at = excluded.fetched_at",
            (
                chapter_meta.id,
                meta.id,
                chapter_meta.chapter_num,
                chapter_meta.title,
                chapter_meta.lang,
                chapter_meta.pages,
                chapter_meta.published_at,
                chapter_meta.fetched_at,
            ),
        )

        recap = works.read_recap(meta.id, chapter_meta.id)
        if recap is not None:
            conn.execute(
                "INSERT INTO recaps (chapter_id, summary, created_at, model,"
                " detail, instruction, standalone)"
                " VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(chapter_id) DO"
                " UPDATE SET summary = excluded.summary,"
                " created_at = excluded.created_at, model = excluded.model,"
                " detail = excluded.detail, instruction = excluded.instruction,"
                " standalone = excluded.standalone",
                (chapter_meta.id, recap.summary, recap.created_at, recap.model,
                 recap.detail, recap.instruction, int(recap.standalone)),
            )

        narration = works.read_narration(meta.id, chapter_meta.id)
        if narration is not None:
            conn.execute(
                "INSERT INTO narrations (chapter_id, text, created_at, model)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(chapter_id) DO UPDATE SET"
                " text = excluded.text, created_at = excluded.created_at,"
                " model = excluded.model",
                (chapter_meta.id, narration.text, narration.created_at, narration.model),
            )

        translation = works.read_translation(meta.id, chapter_meta.id)
        if translation is not None:
            conn.execute(
                "INSERT INTO translations (chapter_id, pages, model, created_at)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(chapter_id) DO UPDATE SET"
                " pages = excluded.pages, model = excluded.model,"
                " created_at = excluded.created_at",
                (
                    chapter_meta.id,
                    translation.pages,
                    translation.model,
                    translation.created_at,
                ),
            )

        for path, kind, name in works.list_video_files(meta.id, chapter_meta.id):
            video_meta = works.read_video_metadata(meta.id, chapter_meta.id, kind)
            if video_meta is None:
                video_meta = works.VideoMetadata(
                    kind=kind,
                    duration_s=None,
                    model=None,
                    tts_engine=None,
                    created_at=utcnow(),
                )
            conn.execute(
                "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
                " duration_s, created_at, tts_engine, model, kind,"
                " video_gen_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT DO UPDATE SET path = excluded.path,"
                " duration_s = excluded.duration_s, created_at = excluded.created_at,"
                " tts_engine = excluded.tts_engine, model = excluded.model,"
                " kind = excluded.kind, video_gen_provider = excluded.video_gen_provider",
                (
                    meta.id,
                    chapter_meta.chapter_num,
                    chapter_meta.chapter_num,
                    str(path),
                    video_meta.duration_s,
                    video_meta.created_at,
                    video_meta.tts_engine,
                    video_meta.model,
                    kind,
                    video_meta.video_gen_provider or "local",
                ),
            )

    # Tiktok video.
    tiktok_video = works.read_video_metadata(meta.id, None, "tiktok")
    if tiktok_video is not None:
        path = works.video_file_path(meta.id, None, "tiktok")
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, tts_engine, model, kind,"
            " video_gen_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT DO UPDATE SET path = excluded.path,"
            " duration_s = excluded.duration_s, created_at = excluded.created_at,"
            " tts_engine = excluded.tts_engine, model = excluded.model,"
            " kind = excluded.kind, video_gen_provider = excluded.video_gen_provider",
            (
                meta.id,
                None,
                None,
                str(path),
                tiktok_video.duration_s,
                tiktok_video.created_at,
                tiktok_video.tts_engine,
                tiktok_video.model,
                "tiktok",
                tiktok_video.video_gen_provider or "local",
            ),
        )

    conn.commit()


def index_works(root: Path | None = None) -> int:
    """Rebuild the entire DB index from `data/works/`.

    Returns the number of works indexed.
    """
    root = root if root is not None else data_dir()
    conn = connect(root / "harness.db")
    _clear_index(conn)
    count = 0
    for work_dir in works.list_work_dirs():
        index_work(conn, work_dir)
        count += 1
    conn.close()
    return count
