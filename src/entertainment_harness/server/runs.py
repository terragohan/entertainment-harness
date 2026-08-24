"""Run manager: background recap runs for the HTTP server.

A run executes the same flow as `eh recap` (select chapters, pre-flight
guard, recap_series with a per-chapter video callback, then backfill videos
for recapped chapters that lack one) in a worker thread, reporting through
ServerProgress. One active run per work; runs are tracked by id with a
running|done|error status and a buffered event log for SSE replay.

Differences from the CLI, by design (headless): no Rich console (everything
goes through the progress reporter), no --dry-run, and a steering
instruction is used literally — the @file form is not resolved.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field

from entertainment_harness import db, library
from entertainment_harness.config import data_dir, load_config
from entertainment_harness.db import utcnow
from entertainment_harness.server.progress import RunCancelled, ServerProgress


class RunConflict(Exception):
    """A run is already active for this work (POST /api/runs -> 409)."""


class RunError(Exception):
    """A run-level failure reported as the run-error terminal event."""


@dataclass
class RunOptions:
    """The headless subset of `eh recap` flags (see POST /api/runs)."""

    work: str
    all_chapters: bool = False
    chapter: float | None = None
    chapters: str | None = None  # "--chapters" spec, e.g. "1-3,4"
    max_chapters: int = 500
    skip_done: bool = False  # drop recapped(+videod) chapters from the scope
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
    panel_first: bool | None = None  # None = [video] panel_first=
    skip_preflight: bool = False


#: Non-log events that move the run's current-stage cursor. Polling clients
#: (nav indicator, library badges) render this instead of streaming SSE.
_CURRENT_EVENTS = frozenset(
    {"run-start", "chapter-start", "stage", "video-ready", "chapter-done"}
)


@dataclass
class Run:
    id: str
    work: str
    title: str
    options: RunOptions
    status: str = "running"  # running | done | error | cancelled
    error: str | None = None
    created_at: str = field(default_factory=utcnow)
    events: list[dict] = field(default_factory=list)
    current: dict | None = None  # latest _CURRENT_EVENTS event, minus noise
    done_count: int = 0  # chapter-done events seen (for run-cancelled)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    cond: threading.Condition = field(default_factory=threading.Condition)

    def _event(self, event: str, **fields) -> dict:
        return {
            "event": event,
            "run_id": self.id,
            "work": self.work,
            "ts": utcnow(),
            **fields,
        }

    def emit(self, event: str, **fields) -> None:
        """Append a non-terminal event (called from the worker thread)."""
        with self.cond:
            self.events.append(self._event(event, **fields))
            if event == "chapter-done":
                self.done_count += 1
            if event in _CURRENT_EVENTS:
                self.current = {
                    "event": event,
                    "chapter": fields.get("chapter"),
                    "stage": fields.get("stage"),
                    "detail": fields.get("detail"),
                }
            self.cond.notify_all()

    def request_stop(self) -> None:
        """Ask the worker to unwind at the next structural progress call."""
        self.stop_requested.set()

    def finish(self, chapters: int) -> None:
        with self.cond:
            self.events.append(self._event("run-done", chapters=chapters))
            self.status = "done"
            self.current = None
            self.cond.notify_all()

    def cancel(self) -> None:
        with self.cond:
            self.events.append(self._event(
                "run-cancelled", chapters=self.done_count
            ))
            self.status = "cancelled"
            self.current = None
            self.cond.notify_all()

    def fail(self, error: str) -> None:
        with self.cond:
            self.events.append(self._event("run-error", message=error))
            self.status = "error"
            self.error = error
            self.current = None
            self.cond.notify_all()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "work": self.work,
            "title": self.title,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "current": self.current,
            "options": asdict(self.options),
        }


def _make_chapter_video(conn, row, config, profile, options: RunOptions,
                        progress: ServerProgress, panel_first: bool):
    """The per-chapter video callback, mirroring the CLI's: (re)build the
    chapter's one video (kind follows the artifact's detail; panel_first
    forces narration) and push it to the store when configured."""
    from entertainment_harness import store
    from entertainment_harness.library import works
    from entertainment_harness.pipelines.recap import video_kind_for_detail
    from entertainment_harness.video.pipeline import build_video

    def make_video(chapter_id: str) -> None:
        chapter_row = conn.execute(
            "SELECT * FROM chapters WHERE id = ?", (chapter_id,)
        ).fetchone()
        artifact = conn.execute(
            "SELECT detail FROM recaps WHERE chapter_id = ?", (chapter_id,)
        ).fetchone()
        kind = (
            "narration" if panel_first
            else video_kind_for_detail(artifact["detail"])
            if artifact is not None
            else "recap"
        )
        existing = conn.execute(
            "SELECT id FROM videos WHERE series_id = ?"
            " AND from_chapter = ? AND to_chapter = ? AND kind = ?",
            (row["id"], chapter_row["chapter_num"],
             chapter_row["chapter_num"], kind),
        ).fetchone()
        if existing is not None:
            # artifacts were built from the previous recap/narration — rebuild
            conn.execute("DELETE FROM videos WHERE id = ?", (existing["id"],))
            video_dir = works.video_dir_for_kind(row["id"], chapter_id, kind)
            if video_dir.is_dir():
                shutil.rmtree(video_dir)
                video_dir.mkdir(parents=True)
        out = build_video(
            conn, row, chapter_row, config, profile,
            voice=options.voice, engine_name=options.tts_engine,
            colorize=True if options.colorize else None,
            translated=True if options.translated else None,
            compress=options.compress, video_mode=options.video_mode,
            panel_first=panel_first,
            progress=progress, log=progress.log,
        )
        if store.is_configured(config):
            try:
                store.push_video(config, row["id"], chapter_id, log=progress.log)
            except store.StoreError as exc:
                progress.log(f"Store push failed: {exc}")
        progress.log(f"Video ready: {out}")

    return make_video


def _run_flow(run: Run, options: RunOptions, progress: ServerProgress) -> int:
    """The `eh recap` flow (cli._run_chapter_pipeline) minus the TTY bits.
    Returns the number of chapters recapped."""
    from entertainment_harness.cli import _chapters_missing_video
    from entertainment_harness.hardware import probe, snapshot
    from entertainment_harness.pipelines.recap import (
        DETAIL_LEVELS,
        chapter_needs_work,
        recap_series,
        select_chapters,
    )
    from entertainment_harness.preflight import check, plan_for_recap

    config = load_config()
    if options.thinking is not None:
        config.pipeline.thinking = options.thinking
    if options.detail is not None:
        config.pipeline.detail = options.detail
    if options.instruction is not None:
        config.pipeline.instructions = options.instruction
    if config.pipeline.detail not in DETAIL_LEVELS:
        raise RunError(
            f"[pipeline] detail must be one of: {', '.join(DETAIL_LEVELS)}"
        )
    detail = config.pipeline.detail
    panel_first = (
        config.video.panel_first if options.panel_first is None
        else options.panel_first
    )
    profile = probe(config.hardware.budget_gb)

    with db.connect() as conn:
        row = library.resolve_series(conn, options.work)
        # fill_gaps mirrors `eh recap --video` (cli._run_chapter_pipeline):
        # chapters behind the read frontier with no artifact and no usable
        # video join the run as standalone gap-fills.
        chapters = select_chapters(
            conn, row, config, detail=detail, chapter_num=options.chapter,
            chapters_spec=options.chapters,
            all_chapters=options.all_chapters,
            max_chapters=options.max_chapters, translated=options.translated,
            fill_gaps=options.video, verb="recap", log=progress.log,
        )
        if options.skip_done and chapters:
            # Same filter recap_series applies, so the pre-flight plan (and
            # its model/RAM estimates) covers only what will actually run.
            chapters = [
                c for c in chapters
                if chapter_needs_work(
                    conn, row["id"], c, detail=detail,
                    want_video=options.video,
                )
            ]
        if chapters and config.preflight.enabled and not options.skip_preflight:
            snap = snapshot(data_dir(), config.hardware.budget_gb)
            plan = plan_for_recap(
                config, profile, row, chapters,
                video=options.video, translated=options.translated,
            )
            issues = check(plan, snap, config.preflight)
            if issues:
                raise RunError(
                    "Pre-flight check failed: " + "; ".join(issues)
                    + " (set skip_preflight to bypass)"
                )

        make_video = _make_chapter_video(
            conn, row, config, profile, options, progress, panel_first
        )

        def on_recap(chapter_id: str) -> None:
            make_video(chapter_id)
            chapter_row = conn.execute(
                "SELECT chapter_num FROM chapters WHERE id = ?", (chapter_id,)
            ).fetchone()
            num = chapter_row["chapter_num"]
            video = conn.execute(
                "SELECT kind, duration_s FROM videos WHERE series_id = ?"
                " AND from_chapter = ? AND to_chapter = ?",
                (row["id"], num, num),
            ).fetchone()
            if video is not None:
                run.emit(
                    "video-ready", chapter=num, kind=video["kind"],
                    duration_s=video["duration_s"],
                    stream=f"/api/videos/{row['id']}/{num:g}/stream",
                )

        done = recap_series(
            conn, row, config, profile, all_chapters=options.all_chapters,
            chapter_num=options.chapter, chapters_spec=options.chapters,
            max_chapters=options.max_chapters,
            translated=options.translated,
            skip_done=options.skip_done,
            fill_gaps=options.video,
            thinking=config.pipeline.thinking,
            detail=detail,
            instruction=config.pipeline.instructions,
            on_recap=on_recap if options.video else None,
            progress=progress, log=progress.log,
        )

        if options.video:
            # Backfill: recapped chapters still lacking a playable video.
            missing = _chapters_missing_video(
                conn, row["id"], table="recaps",
                translated=options.translated, langs=config.library.langs,
            )
            for chapter_row in missing:
                progress.log(
                    f"Backfilling video for recapped chapter"
                    f" {chapter_row['chapter_num']:g}..."
                )
                on_recap(chapter_row["id"])
    return len(done)


def _execute(run: Run, options: RunOptions) -> None:
    progress = ServerProgress(run)
    try:
        chapters = _run_flow(run, options, progress)
    except RunCancelled:
        run.cancel()
        return
    except Exception as exc:
        run.fail(str(exc))
        return
    run.finish(chapters)


class RunManager:
    """Tracks runs by id; enforces one active run per work."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def start(self, series_id: str, title: str, options: RunOptions) -> Run:
        with self._lock:
            for existing in self._runs.values():
                if existing.work == series_id and existing.status == "running":
                    raise RunConflict(
                        f"A run is already active for {title!r}"
                        f" (run {existing.id})"
                    )
            run = Run(
                id=uuid.uuid4().hex[:12],
                work=series_id,
                title=title,
                options=options,
            )
            self._runs[run.id] = run
        thread = threading.Thread(
            target=_execute, args=(run, options),
            name=f"eh-run-{run.id}", daemon=True,
        )
        thread.start()
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list(self) -> list[Run]:
        return list(reversed(self._runs.values()))
