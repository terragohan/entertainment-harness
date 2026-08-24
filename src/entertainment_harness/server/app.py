"""FastAPI app factory for `eh serve` (see docs/design.md "Local server").

All endpoints live under /api. Everything reads/writes through the same
seams as the CLI: db.connect() + library/works.py for library data and
video paths, config.load_config/save_config for config.toml, and the
sources registry for source state. The app binds nothing itself — the CLI
runs it under uvicorn.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from entertainment_harness import db, library
from entertainment_harness.config import (
    CONFIG_FILENAME,
    DEFAULT_ENABLED_SOURCES,
    ConfigWriteError,
    data_dir,
    load_config,
    save_config,
)
from entertainment_harness.library import works
from entertainment_harness.server.auto import AutoSupervisor
from entertainment_harness.server.castbuilds import (
    CastBuildConflict,
    CastBuildManager,
)
from entertainment_harness.server.exports import ExportConflict, ExportManager
from entertainment_harness.server.runs import (
    Run,
    RunConflict,
    RunManager,
    RunOptions,
)

SSE_WAIT_S = 15.0  # max block waiting for new events before re-checking


class RunRequest(BaseModel):
    """POST /api/runs body: the headless subset of `eh recap` flags."""

    work: str
    all_chapters: bool = False
    chapter: float | None = None
    chapters: str | None = None
    max_chapters: int = 500
    skip_done: bool = False
    video: bool = False
    translated: bool = False
    thinking: str | None = None
    detail: str | None = None
    instruction: str | None = None
    voice: str | None = None
    tts_engine: str | None = None
    colorize: bool = False
    compress: str | None = None
    video_mode: str | None = None
    panel_first: bool | None = None
    skip_preflight: bool = False


class SourcesUpdate(BaseModel):
    enabled: list[str]


class AutoRequest(BaseModel):
    """PUT /api/works/{work}/auto body: the per-work auto-process toggle
    plus the modifiers stored for the runs it starts. The scope is
    optional: neither `chapters` nor `all_chapters` = the default pending
    selection (from the read mark onward, gaps behind it filled in)."""

    enabled: bool
    detail: str | None = None
    instruction: str | None = None
    skip_preflight: bool = False
    video_mode: str | None = None  # kenburns | scroll; None = [video] mode=
    chapters: str | None = None  # optional "--chapters"-style scope
    all_chapters: bool = False  # optional scope: every synced chapter


class CharacterEntry(BaseModel):
    """One character-registry row as the work view edits it."""

    name: str
    aliases: list[str] = []
    role: str = ""


class CharactersUpdate(BaseModel):
    """PUT /api/works/{work}/characters body: replaces the whole registry."""

    characters: list[CharacterEntry]


def _config_path():
    return data_dir() / CONFIG_FILENAME


def _config_payload() -> dict:
    """The effective config for the settings editor. `raw` (the whole parsed
    config.toml, kept for plugin merges) is implementation detail — sending
    it would duplicate the payload and invite edits to a bogus [raw] table.
    """
    payload = asdict(load_config())
    payload.pop("raw", None)
    return payload


def _sse_frame(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event)}\n\n"


def _wait_for_events(run: Run, idx: int) -> None:
    with run.cond:
        run.cond.wait_for(
            lambda: len(run.events) > idx or run.status != "running",
            timeout=SSE_WAIT_S,
        )


async def _event_stream(run: Run):
    """Replay the buffered events, then stream live until the terminal
    run-done/run-error event (so late subscribers see the whole run)."""
    idx = 0
    while True:
        with run.cond:
            chunk = list(run.events[idx:])
            idx = len(run.events)
            terminal = run.status != "running"
        for event in chunk:
            yield _sse_frame(event)
        if terminal:
            return
        if not chunk:
            await asyncio.to_thread(_wait_for_events, run, idx)


def _ordered_enabled(enabled: list[str]) -> list[str]:
    """Built-ins first in registry order, then extras in their given order."""
    from entertainment_harness.sources import REGISTRY

    builtins = REGISTRY.builtin_names()
    ordered = [n for n in builtins if n in enabled]
    ordered += [n for n in enabled if n not in builtins]
    return ordered


def _sources_payload() -> dict:
    from entertainment_harness.sources import REGISTRY, is_source_enabled

    config = load_config()
    return {
        "sources": [
            {
                "name": name,
                "builtin": not REGISTRY.is_entry_point(name),
                "enabled": is_source_enabled(name, config),
            }
            for name in REGISTRY.names()
        ]
    }


def _validate_run_request(req: RunRequest) -> None:
    from entertainment_harness.pipelines.judge import THINKING_LEVELS
    from entertainment_harness.pipelines.recap import DETAIL_LEVELS

    if req.max_chapters < 1:
        raise HTTPException(422, "max_chapters must be at least 1")
    if req.chapters is not None:
        from entertainment_harness.library import LibraryError, parse_chapter_spec

        try:
            parse_chapter_spec(req.chapters)
        except LibraryError as exc:
            raise HTTPException(422, f"chapters: {exc}") from exc
    if req.thinking is not None and req.thinking not in THINKING_LEVELS:
        raise HTTPException(
            422, f"thinking must be one of: {', '.join(THINKING_LEVELS)}"
        )
    if req.detail is not None and req.detail not in DETAIL_LEVELS:
        raise HTTPException(
            422, f"detail must be one of: {', '.join(DETAIL_LEVELS)}"
        )
    if req.compress is not None:
        from entertainment_harness.video.compress import PRESETS

        if req.compress not in PRESETS:
            raise HTTPException(
                422, f"compress must be one of: {', '.join(PRESETS)}"
            )
    if req.video_mode is not None:
        from entertainment_harness.video.pipeline import VIDEO_MODES

        if req.video_mode not in VIDEO_MODES:
            raise HTTPException(
                422, f"video_mode must be one of: {', '.join(VIDEO_MODES)}"
            )


def create_app() -> FastAPI:
    app = FastAPI(title="entertainment-harness")
    # The Electrobun webview has an opaque origin; the server only ever
    # listens on localhost, so permissive CORS is safe here.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    runs = RunManager()
    app.state.runs = runs
    supervisor = AutoSupervisor(runs)
    supervisor.load()
    app.state.auto = supervisor
    exports = ExportManager()
    app.state.exports = exports
    cast_builds = CastBuildManager()
    app.state.cast_builds = cast_builds

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/library")
    def get_library() -> dict:
        with db.connect() as conn:
            series_rows = conn.execute(
                "SELECT s.*,"
                " (SELECT last_read_chapter FROM progress p"
                "  WHERE p.series_id = s.id) AS last_read"
                " FROM series s ORDER BY s.title"
            ).fetchall()
            result = []
            for s in series_rows:
                chapters = conn.execute(
                    "SELECT c.id, c.chapter_num, c.title, c.lang, c.pages,"
                    " r.detail AS artifact_detail,"
                    " v.kind AS video_kind, v.duration_s AS video_duration,"
                    " v.wiped_at AS video_wiped"
                    " FROM chapters c"
                    " LEFT JOIN recaps r ON r.chapter_id = c.id"
                    # Several videos rows per chapter are legal (one per
                    # kind); pick a single one — usable beats wiped, then
                    # newest — so the payload never lists a chapter twice.
                    " LEFT JOIN videos v ON v.id = ("
                    "  SELECT v2.id FROM videos v2"
                    "  WHERE v2.series_id = c.series_id"
                    "   AND v2.from_chapter = c.chapter_num"
                    "   AND v2.to_chapter = c.chapter_num"
                    "  ORDER BY (v2.wiped_at IS NOT NULL), v2.id DESC"
                    "  LIMIT 1)"
                    " WHERE c.series_id = ? ORDER BY c.chapter_num",
                    (s["id"],),
                ).fetchall()
                chapter_list = []
                for c in chapters:
                    has_video = (
                        c["video_kind"] is not None and c["video_wiped"] is None
                    )
                    num = c["chapter_num"]
                    chapter_list.append({
                        "id": c["id"],
                        "chapter_num": num,
                        "title": c["title"],
                        "lang": c["lang"],
                        "pages": c["pages"],
                        "detail": c["artifact_detail"],
                        "has_recap": c["artifact_detail"] is not None,
                        "has_video": has_video,
                        "video": (
                            {"kind": c["video_kind"],
                             "duration_s": c["video_duration"]}
                            if has_video else None
                        ),
                        "stream": (
                            f"/api/videos/{s['id']}/{num:g}/stream"
                            if has_video else None
                        ),
                    })
                result.append({
                    "id": s["id"],
                    "title": s["title"],
                    "status": s["status"],
                    "kind": s["kind"],
                    "source": s["source"],
                    "last_read": s["last_read"],
                    "auto": supervisor.is_enabled(s["id"]),
                    "chapters": chapter_list,
                })
        return {"works": result}

    @app.get("/api/videos/{work}/{chapter}/stream")
    def stream_video(work: str, chapter: float):
        """Stream a chapter's mp4. Range support (206 partial content) comes
        from Starlette's FileResponse — WebKit <video> needs it for seeking.
        Only locally-present files are served (no store pull); wiped or
        missing files 404."""
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
            video = conn.execute(
                "SELECT * FROM videos WHERE series_id = ?"
                " AND from_chapter = ? AND to_chapter = ?"
                # Multiple rows per chapter are legal (one per kind): the
                # usable one wins, so a wiped row can't shadow a playable
                # video with a 404.
                " ORDER BY (wiped_at IS NOT NULL), id DESC LIMIT 1",
                (row["id"], chapter, chapter),
            ).fetchone()
            if video is None:
                raise HTTPException(
                    404, f"No video for chapter {chapter:g}"
                )
            if video["wiped_at"]:
                raise HTTPException(
                    404, f"Video for chapter {chapter:g} was wiped"
                    " — re-render it with 'eh recap --video'"
                )
            ch = conn.execute(
                "SELECT id FROM chapters WHERE series_id = ?"
                " AND chapter_num = ?",
                (row["id"], chapter),
            ).fetchone()
            if ch is None:
                raise HTTPException(
                    404, f"Chapter {chapter:g} is not in the library"
                )
        # Resolve the path the same way 'eh play' does: the works layout is
        # authoritative, and a compressed copy is the deliverable when one
        # was made. Ids come from DB rows, never from raw path joins.
        meta = works.read_video_metadata(row["id"], ch["id"], video["kind"])
        suffix = (
            f"out-{meta.compress}.mp4"
            if meta is not None and meta.compress else "out.mp4"
        )
        path = works.video_file_path(
            row["id"], ch["id"], video["kind"], suffix=suffix
        )
        if not path.is_file():
            raise HTTPException(
                404, "Video file is not present locally; pull it with"
                " 'eh play' or re-render"
            )
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.post("/api/works/{work}/export", status_code=202)
    def start_export(work: str) -> dict:
        """Assemble the work's playable chapter videos into one mp4 in the
        background (copy for a single chapter, lossless concat otherwise)
        and land it in ~/Downloads/EntertainmentHarness. Poll
        GET /api/exports for progress; one active export per work."""
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
        try:
            job = exports.start(row["id"], row["title"])
        except ExportConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/exports")
    def list_exports() -> dict:
        return {"exports": [j.to_dict() for j in exports.list()]}

    @app.post("/api/runs", status_code=201)
    def start_run(req: RunRequest) -> dict:
        _validate_run_request(req)
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, req.work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
        options = RunOptions(**req.model_dump())
        try:
            run = runs.start(row["id"], row["title"], options)
        except RunConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return run.to_dict()

    @app.get("/api/runs")
    def list_runs() -> dict:
        return {"runs": [run.to_dict() for run in runs.list()]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"No run {run_id!r}")
        return run.to_dict()

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict:
        """Cooperatively stop a running run. Idempotent: stopping a finished
        run is a no-op (the run returns as-is)."""
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"No run {run_id!r}")
        run.request_stop()
        return run.to_dict()

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str):
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, f"No run {run_id!r}")
        return StreamingResponse(
            _event_stream(run),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.put("/api/works/{work}/auto")
    def set_auto(work: str, req: AutoRequest) -> dict:
        """The per-work auto-process toggle. Enabling persists the toggle
        and its modifiers (detail/instruction/skip_preflight/video_mode) and
        starts a run immediately when the work has unfinished chapters;
        disabling persists and cooperatively stops any active run for the
        work. The toggle never auto-resumes: state loaded from disk at
        startup is disarmed, so after a backend restart the user must flip
        it on again (the settings are kept)."""
        from entertainment_harness.pipelines.recap import DETAIL_LEVELS
        from entertainment_harness.video.pipeline import VIDEO_MODES

        if req.detail is not None and req.detail not in DETAIL_LEVELS:
            raise HTTPException(
                422, f"detail must be one of: {', '.join(DETAIL_LEVELS)}"
            )
        if req.video_mode is not None and req.video_mode not in VIDEO_MODES:
            raise HTTPException(
                422, f"video_mode must be one of: {', '.join(VIDEO_MODES)}"
            )
        if req.chapters and req.all_chapters:
            raise HTTPException(
                422, "chapters and all_chapters are mutually exclusive"
            )
        if req.chapters:
            try:
                library.parse_chapter_spec(req.chapters)
            except library.LibraryError as exc:
                raise HTTPException(422, str(exc)) from exc
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
        return supervisor.set(
            row["id"],
            req.enabled,
            {
                "detail": req.detail,
                "instruction": req.instruction,
                "skip_preflight": req.skip_preflight,
                "video_mode": req.video_mode,
                "chapters": req.chapters,
                "all_chapters": req.all_chapters,
            },
        )

    def _character_payload(row) -> dict:
        return {
            "name": row["name"],
            "aliases": json.loads(row["aliases"]),
            "role": row["role"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "origin": row["origin"],
            "edited": bool(row["edited"]),
        }

    @app.get("/api/works/{work}/characters")
    def get_characters(work: str) -> dict:
        """The work's character registry (character-bible initiative),
        ordered most-recently-seen first."""
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
            return {
                "characters": [
                    _character_payload(r)
                    for r in db.get_characters(conn, row["id"])
                ]
            }

    @app.put("/api/works/{work}/characters")
    def put_characters(work: str, req: CharactersUpdate) -> dict:
        """Replace the whole registry from the work view's Characters
        section. Every saved row lands origin='user', edited=1, so the
        extractor refines but never rewrites user-touched entries."""
        entries = []
        for entry in req.characters:
            name = entry.name.strip()
            if not name:
                raise HTTPException(422, "character names must not be empty")
            entries.append(
                {
                    "name": name,
                    "aliases": [a.strip() for a in entry.aliases if a.strip()],
                    "role": entry.role.strip(),
                }
            )
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
            db.replace_characters(conn, row["id"], entries)
            return {
                "characters": [
                    _character_payload(r)
                    for r in db.get_characters(conn, row["id"])
                ]
            }

    @app.post("/api/works/{work}/characters/rebuild", status_code=202)
    def start_cast_build(work: str) -> dict:
        """Re-extract the work's character registry from its stored chapter
        artifacts in the background (the work view's Rebuild button — the
        same fold as `eh cast --rebuild`). REPLACES the whole registry,
        manual edits included. Poll GET /api/cast-builds for progress; one
        active build per work."""
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work)
            except library.LibraryError as exc:
                raise HTTPException(404, str(exc)) from exc
            has_artifacts = conn.execute(
                "SELECT 1 FROM recaps r JOIN chapters c"
                " ON r.chapter_id = c.id WHERE c.series_id = ? LIMIT 1",
                (row["id"],),
            ).fetchone()
        if has_artifacts is None:
            raise HTTPException(
                422, "No recaps yet — nothing to rebuild the cast from."
            )
        try:
            job = cast_builds.start(row["id"], row["title"])
        except CastBuildConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/cast-builds")
    def list_cast_builds() -> dict:
        return {"builds": [j.to_dict() for j in cast_builds.list()]}

    @app.get("/api/config")
    def get_config() -> dict:
        return _config_payload()

    @app.put("/api/config")
    def put_config(updates: dict[str, Any]) -> dict:
        try:
            save_config(_config_path(), updates)
        except ConfigWriteError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _config_payload()

    @app.get("/api/sources")
    def get_sources() -> dict:
        return _sources_payload()

    @app.put("/api/sources")
    def put_sources(update: SourcesUpdate) -> dict:
        from entertainment_harness.sources import REGISTRY

        names = REGISTRY.names()
        unknown = [n for n in update.enabled if n not in names]
        if unknown:
            raise HTTPException(
                422, f"Unknown sources: {', '.join(unknown)}"
                f" (registered: {', '.join(names)})"
            )
        try:
            save_config(
                _config_path(),
                {"sources": {"enabled": _ordered_enabled(update.enabled)}},
            )
        except ConfigWriteError as exc:
            raise HTTPException(422, str(exc)) from exc
        return _sources_payload()

    @app.post("/api/sources/reset")
    def reset_sources() -> dict:
        save_config(
            _config_path(),
            {"sources": {"enabled": list(DEFAULT_ENABLED_SOURCES)}},
        )
        return _sources_payload()

    return app
