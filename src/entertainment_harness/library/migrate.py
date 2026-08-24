"""One-time migration from the legacy split layout to resource-centric works/.

The legacy layout kept files under data/manga/, data/books/, and data/videos/
with SQLite as the source of truth. The new layout co-locates everything under
data/works/<series-id>/ with metadata files. This module migrates existing data
when the new works/ directory is empty but legacy directories or DB rows exist.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from entertainment_harness import db
from entertainment_harness.library import works
from entertainment_harness.config import cache_dir, data_dir


def _needs_migration() -> bool:
    """Return True if legacy data exists and no works have been created yet."""
    root = data_dir()
    works_root = root / "works"
    if works_root.is_dir() and any(works_root.iterdir()):
        return False
    legacy = (root / "manga", root / "books", root / "videos")
    return any(d.is_dir() and any(d.iterdir()) for d in legacy)


def _move_or_copy(src: Path, dest: Path) -> None:
    """Move src to dest if src exists. Falls back to copy + remove."""
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    try:
        shutil.move(str(src), str(dest))
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest)
            shutil.rmtree(src)
        else:
            shutil.copy2(src, dest)
            src.unlink()


def _apply_renames(renames: list[tuple[Path, Path]]) -> None:
    """Rename dirs, two-phase via temp names so swaps can't collide."""
    temps = []
    for src, dst in renames:
        tmp = src.with_name(f".{src.name}.renaming")
        src.rename(tmp)
        temps.append((tmp, dst))
    for tmp, dst in temps:
        tmp.rename(dst)


def _migrate_work_dir_names() -> bool:
    """Rename ULID/id-named work and chapter dirs to human-readable names.

    Works become a slug of the title (`one-piece`); chapters become `ch-NNN`.
    Ids stay in work.json / chapter.json; the works resolution layer keeps
    every caller working. Returns True when anything was renamed.
    """
    root = data_dir() / "works"
    if not root.is_dir():
        return False

    chapter_renames: list[tuple[Path, Path]] = []
    work_renames: list[tuple[Path, Path]] = []
    taken: set[str] = set()
    for wdir in sorted(root.iterdir()):
        if not wdir.is_dir():
            continue
        meta = works.read_work_metadata_at(wdir)
        if meta is None:
            taken.add(wdir.name)
            continue
        for cdir in (wdir / "chapters").iterdir() if (wdir / "chapters").is_dir() else []:
            if not cdir.is_dir():
                continue
            cmeta = works.read_chapter_metadata_at(cdir)
            if cmeta is None:
                continue
            name = works.chapter_dir_name(cmeta.chapter_num, cmeta.id)
            if name != cdir.name and not (cdir.parent / name).exists():
                chapter_renames.append((cdir, cdir.parent / name))
        slug = works.slugify(meta.title)
        name = slug
        suffix = 2
        while name in taken and name != wdir.name:
            name = f"{slug}-{suffix}"
            suffix += 1
        taken.add(name)
        if name != wdir.name:
            work_renames.append((wdir, wdir.parent / name))

    if not chapter_renames and not work_renames:
        return False
    _apply_renames(chapter_renames)
    _apply_renames(work_renames)
    works.refresh_paths()
    return True


def _migrate_cache_layout() -> None:
    """Move re-downloadable weights out of the data dir into cache_dir().

    The voice library (user-provided reference samples) moves the other way:
    out of tts/ into data_dir()/voices/. Skipped per-item when the target
    already exists, so re-runs are safe.
    """
    root = data_dir()
    cache = cache_dir()
    if root.resolve() == cache.resolve():
        return
    voices_src, voices_dest = root / "tts" / "voices", root / "voices"
    if voices_src.exists() and not voices_dest.exists():
        _move_or_copy(voices_src, voices_dest)
    for name in ("models", "tts", "colorize"):
        src, dest = root / name, cache / name
        if src.exists() and not dest.exists():
            _move_or_copy(src, dest)


def _migrate_chapter(
    conn: sqlite3.Connection,
    series_id: str,
    chapter: sqlite3.Row,
) -> None:
    chapter_id = chapter["id"]
    chapter_dir = works.chapter_dir(series_id, chapter_id)
    chapter_dir.mkdir(parents=True, exist_ok=True)

    works.write_chapter_metadata(
        series_id,
        works.ChapterMetadata(
            id=chapter_id,
            chapter_num=chapter["chapter_num"],
            title=chapter["title"],
            lang=chapter["lang"],
            pages=chapter["pages"],
            published_at=chapter["published_at"],
            fetched_at=chapter["fetched_at"],
        ),
    )

    # Source pages / book text.
    legacy_manga = data_dir() / "manga" / series_id / chapter_id
    if legacy_manga.is_dir():
        dest_source = works.source_dir(series_id, chapter_id)
        for item in legacy_manga.iterdir():
            if item.name in ("translated", "metadata.json"):
                continue
            _move_or_copy(item, dest_source / item.name)

    legacy_book_txt = data_dir() / "books" / series_id / f"ch-{int(chapter['chapter_num']):03d}.txt"
    if legacy_book_txt.is_file():
        _move_or_copy(legacy_book_txt, works.source_dir(series_id, chapter_id) / legacy_book_txt.name)

    # Translated overlay pages.
    legacy_translated = legacy_manga / "translated"
    if legacy_translated.is_dir():
        dest_translated = works.translated_dir(series_id, chapter_id)
        for item in legacy_translated.iterdir():
            _move_or_copy(item, dest_translated / item.name)
        # Preserve legacy translation.json bubble data if present.
        legacy_translation_json = legacy_translated / "translation.json"
        if legacy_translation_json.is_file():
            try:
                raw = json.loads(legacy_translation_json.read_text(encoding="utf-8"))
                bubbles = raw.get("bubbles", [])
            except (json.JSONDecodeError, OSError):
                bubbles = []
        else:
            bubbles = []
        trans_row = conn.execute(
            "SELECT pages, model, created_at FROM translations WHERE chapter_id = ?",
            (chapter_id,),
        ).fetchone()
        if trans_row is not None:
            works.write_translation(
                series_id,
                chapter_id,
                works.TranslationMetadata(
                    pages=trans_row["pages"],
                    model=trans_row["model"],
                    created_at=trans_row["created_at"],
                    bubbles=bubbles,
                ),
            )

    # Recap.
    recap_row = conn.execute(
        "SELECT summary, model, created_at FROM recaps WHERE chapter_id = ?",
        (chapter_id,),
    ).fetchone()
    if recap_row is not None:
        works.write_recap(
            series_id,
            chapter_id,
            works.RecapMetadata(
                summary=recap_row["summary"],
                model=recap_row["model"],
                created_at=recap_row["created_at"],
            ),
        )

    # Narration.
    narr_row = conn.execute(
        "SELECT text, model, created_at FROM narrations WHERE chapter_id = ?",
        (chapter_id,),
    ).fetchone()
    if narr_row is not None:
        works.write_narration(
            series_id,
            chapter_id,
            works.NarrationMetadata(
                text=narr_row["text"],
                model=narr_row["model"],
                created_at=narr_row["created_at"],
            ),
        )

    # Videos.
    legacy_video_dir = data_dir() / "videos" / series_id / chapter_id
    if legacy_video_dir.is_dir():
        for kind, dest_name in (
            ("recap", works.VIDEO_RECAP_DIR),
            ("narration", works.VIDEO_NARRATION_DIR),
        ):
            if kind == "recap":
                src_dir = legacy_video_dir
            else:
                src_dir = legacy_video_dir / "narration"
            if not src_dir.is_dir():
                continue
            dest_dir = chapter_dir / dest_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                # The narration subdir is handled in the next iteration.
                if kind == "recap" and item.name == "narration":
                    continue
                _move_or_copy(item, dest_dir / item.name)

            video_row = conn.execute(
                "SELECT duration_s, model, tts_engine, video_gen_provider,"
                " created_at FROM videos WHERE series_id = ? AND kind = ?"
                " AND from_chapter = ? AND to_chapter = ?",
                (series_id, kind, chapter["chapter_num"], chapter["chapter_num"]),
            ).fetchone()
            if video_row is not None:
                works.write_video_metadata(
                    series_id,
                    chapter_id,
                    works.VideoMetadata(
                        kind=kind,
                        duration_s=video_row["duration_s"],
                        model=video_row["model"],
                        tts_engine=video_row["tts_engine"],
                        video_gen_provider=video_row["video_gen_provider"] or "local",
                        created_at=video_row["created_at"],
                    ),
                )


def _migrate_series(conn: sqlite3.Connection, series_id: str) -> None:
    series = conn.execute("SELECT * FROM series WHERE id = ?", (series_id,)).fetchone()
    if series is None:
        return

    work_dir = works.work_dir(series_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    progress = conn.execute(
        "SELECT last_read_chapter, updated_at FROM progress WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    progress_meta = works.WorkProgress(
        last_read_chapter=progress["last_read_chapter"] if progress else None,
        updated_at=progress["updated_at"] if progress else None,
    )

    context = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    context_meta = works.WorkContext(
        rolling_summary=context["rolling_summary"] if context else "",
        through_chapter=context["through_chapter"] if context else None,
    )

    online = conn.execute(
        "SELECT provider, query, summary, sources_json, steering_prompt, created_at"
        " FROM online_summaries WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    online_meta = None
    if online is not None:
        try:
            sources = json.loads(online["sources_json"] or "[]")
        except json.JSONDecodeError:
            sources = []
        online_meta = works.OnlineSummary(
            provider=online["provider"],
            query=online["query"],
            summary=online["summary"],
            sources=sources,
            steering_prompt=online["steering_prompt"] or "",
            created_at=online["created_at"],
        )

    works.write_work_metadata(
        works.WorkMetadata(
            id=series_id,
            title=series["title"],
            source=series["source"],
            source_id=series["source_id"],
            added_at=series["added_at"],
            kind=series["kind"],
            alt_titles=(series["alt_titles"] or "").split(", ") if series["alt_titles"] else [],
            status=series["status"],
            progress=progress_meta,
            context=context_meta,
            online_summary=online_meta,
        )
    )

    # Imported book/comic cover.
    for subdir in ("books", "manga"):
        legacy_root = data_dir() / subdir / series_id
        if legacy_root.is_dir():
            for cover in legacy_root.glob("cover.*"):
                _move_or_copy(cover, work_dir / cover.name)

    chapters = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ? ORDER BY chapter_num ASC",
        (series_id,),
    ).fetchall()
    for chapter in chapters:
        _migrate_chapter(conn, series_id, chapter)

    # Tiktok whole-work video.
    legacy_tiktok = data_dir() / "videos" / series_id / "tiktok"
    if legacy_tiktok.is_dir():
        dest_tiktok = works.tiktok_dir(series_id)
        dest_tiktok.mkdir(parents=True, exist_ok=True)
        for item in legacy_tiktok.iterdir():
            _move_or_copy(item, dest_tiktok / item.name)
        tiktok_row = conn.execute(
            "SELECT duration_s, model, tts_engine, video_gen_provider, created_at"
            " FROM videos WHERE series_id = ? AND kind = 'tiktok'",
            (series_id,),
        ).fetchone()
        if tiktok_row is not None:
            works.write_video_metadata(
                series_id,
                None,
                works.VideoMetadata(
                    kind="tiktok",
                    duration_s=tiktok_row["duration_s"],
                    model=tiktok_row["model"],
                    tts_engine=tiktok_row["tts_engine"],
                    video_gen_provider=tiktok_row["video_gen_provider"] or "local",
                    created_at=tiktok_row["created_at"],
                ),
            )


def migrate(conn: sqlite3.Connection | None = None) -> int:
    """Migrate legacy layout to works/ layout if needed.

    Also relocates cache-able model weights into cache_dir() and renames
    id-named work dirs to human-readable names. Returns the number of series
    migrated.
    """
    _migrate_cache_layout()
    migrated = 0
    if _needs_migration():
        own_conn = conn is None
        if own_conn:
            # Open the DB directly to avoid recursion with db.connect().
            path = data_dir() / db.DB_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(db.SCHEMA)
        try:
            series_ids = [
                r["id"] for r in conn.execute("SELECT id FROM series ORDER BY added_at ASC").fetchall()
            ]
            for series_id in series_ids:
                _migrate_series(conn, series_id)
            conn.commit()
            migrated = len(series_ids)
        finally:
            if own_conn:
                conn.close()
    if _migrate_work_dir_names():
        # Absolute paths in the videos table went stale; rebuild the index.
        from entertainment_harness.library import index

        index.index_works()
    return migrated
