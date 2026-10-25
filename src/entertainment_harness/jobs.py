"""Headless job runner for cloud containers (animation-director Track B).

`eh job-run --spec <json>` executes one pipeline job with no TTY: stage the
work in from the R2 store in two phases (metadata — work.json, the cover,
every JSON — then just the pages of the chapters the run will touch, so a
large work fits the container disk; mp4s never cross at stage-in), rebuild
the SQLite index from the staged layout (library.index.index_works on a
fresh store, an additive index_work into a populated D1 snapshot), run the
same flow as POST /api/runs (server.runs._run_flow), stage out. Events
stream as JSONL on stdout and to EH_JOB_EVENTS (default
<data_dir>/jobs/<id>/events.jsonl) — the same event dicts the local
server's SSE emits, so clients render either source unchanged.

The cloud profile (config.apply_cloud_profile, EH_CLOUD=1) forces
openai_compat roles + the R2 store and disables the local-hardware
pre-flight guard; the container image sets it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from entertainment_harness.config import data_dir, load_config
from entertainment_harness.db import connect, utcnow
from entertainment_harness.library import works
from entertainment_harness.server.progress import RunCancelled, ServerProgress
from entertainment_harness import store
from entertainment_harness.store import d1


class JobSpecError(ValueError):
    """Unusable job spec — the message says what to fix."""


@dataclass
class JobSpec:
    """One headless job. `options` carries the RunOptions fields (the
    headless `eh recap` flags; see server.runs.RunOptions)."""

    work: str  # series id or slug
    kind: str = "recap"  # only "recap" runs are supported so far (C0)
    options: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def from_dict(cls, data: dict) -> "JobSpec":
        if not isinstance(data, dict):
            raise JobSpecError("Job spec must be a JSON object.")
        work = data.get("work")
        if not isinstance(work, str) or not work:
            raise JobSpecError('Job spec needs a non-empty "work" (id or slug).')
        kind = str(data.get("kind", "recap"))
        if kind != "recap":
            raise JobSpecError(
                f"Unknown job kind {kind!r}; only 'recap' is supported."
            )
        options = data.get("options", {})
        if not isinstance(options, dict):
            raise JobSpecError('"options" must be a JSON object.')
        if "work" in options:
            raise JobSpecError(
                '"work" belongs at the spec top level, not inside "options".'
            )
        return cls(
            work=work,
            kind=kind,
            options=dict(options),
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            created_at=str(data.get("created_at") or utcnow()),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_spec(value: str) -> JobSpec:
    """Parse a --spec argument: a file path, '-' for stdin, or inline JSON."""
    if value == "-":
        text = sys.stdin.read()
    elif os.path.exists(value):
        text = Path(value).read_text(encoding="utf-8")
    else:
        text = value
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JobSpecError(f"Job spec is not valid JSON: {exc}") from exc
    return JobSpec.from_dict(data)


class JobRun:
    """The duck-typed Run surface server.runs._run_flow and ServerProgress
    use (emit + stop_requested), writing every event as a JSONL line — to
    stdout (container logs) and to the events file. No subscribers, so no
    threading.Condition; terminal events are written by finish/cancel/fail.
    """

    def __init__(self, spec: JobSpec, events_path: Path | None = None) -> None:
        self.spec = spec
        self.id = spec.id
        self.work = spec.work
        self.status = "running"  # running | done | error | cancelled
        self.error: str | None = None
        self.current: dict | None = None
        self.done_count = 0
        self.stop_requested = threading.Event()
        override = os.environ.get("EH_JOB_EVENTS", "")
        self.events_path = (
            Path(override)
            if override
            else events_path
            or data_dir() / "jobs" / spec.id / "events.jsonl"
        )
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields) -> None:
        record = {"event": event, "run_id": self.id, "work": self.work,
                  "ts": utcnow(), **fields}
        if event == "chapter-done":
            self.done_count += 1
        if event in ("run-start", "chapter-start", "stage", "video-ready",
                     "chapter-done"):
            self.current = {
                "event": event,
                "chapter": fields.get("chapter"),
                "stage": fields.get("stage"),
                "detail": fields.get("detail"),
            }
        line = json.dumps(record)
        print(line, flush=True)
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log(self, msg) -> None:
        self.emit("log", message=str(msg))

    def request_stop(self) -> None:
        self.stop_requested.set()

    def finish(self, chapters: int) -> None:
        self.emit("run-done", chapters=chapters)
        self.status = "done"
        self.current = None

    def cancel(self) -> None:
        self.emit("run-cancelled", chapters=self.done_count)
        self.status = "cancelled"
        self.current = None

    def fail(self, error: str) -> None:
        self.emit("run-error", message=error)
        self.status = "error"
        self.error = error
        self.current = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "work": self.work,
            "kind": self.spec.kind,
            "status": self.status,
            "error": self.error,
            "created_at": self.spec.created_at,
            "current": self.current,
            "options": self.spec.options,
        }


def _run_options(spec: JobSpec):
    """Build server.runs.RunOptions from the spec, rejecting unknown keys."""
    from entertainment_harness.server.runs import RunOptions

    valid = {f.name for f in fields(RunOptions)} - {"work"}
    unknown = sorted(set(spec.options) - valid)
    if unknown:
        raise JobSpecError(
            "Unknown job option(s): " + ", ".join(unknown)
            + f" (valid: {', '.join(sorted(valid))})"
        )
    return RunOptions(work=spec.work, **spec.options)


class _FlushOnChapterDone(ServerProgress):
    """ServerProgress plus a callback after each chapter-done event.

    Pipelines commit per chapter, so boundary state is visible to the
    flush's separate read connection without touching the pipeline's.
    """

    def __init__(self, run, flush) -> None:
        super().__init__(run)
        self._flush = flush

    def chapter_done(self, output_desc: str = "") -> None:
        super().chapter_done(output_desc)
        self._flush()


class _CloudSink:
    """Chapter-boundary and terminal side effects for cloud runs: flush the
    D1 snapshot diff, push events.jsonl to the store, update the jobs
    ledger, and poll the cooperative stop flag (set by
    POST /api/jobs/{id}/stop). Remote failures are logged and retried at
    the next boundary — never fatal to the pipeline."""

    def __init__(self, config, run, d1_cfg, db_path, baseline) -> None:
        self._config = config
        self._run = run
        self._d1 = d1_cfg
        self._db_path = db_path
        self._baseline = baseline

    def flush_snapshot(self) -> None:
        try:
            stats = d1.snapshot_flush(self._d1, self._db_path, self._baseline)
        except Exception as exc:
            # Non-fatal: the next boundary retries the full delta.
            self._run.log(f"D1 flush failed (will retry): {exc}")
            return
        changed = {
            t: c for t, c in stats.items() if c["upserted"] or c["deleted"]
        }
        if changed:
            self._run.log(f"D1 flush: {changed}")

    def push_events(self) -> None:
        if not store.is_configured(self._config):
            return
        try:
            store.backend(self._config).upload(
                self._config,
                [(self._run.events_path, f"jobs/{self._run.id}/events.jsonl")],
            )
        except Exception as exc:
            self._run.log(f"Events push failed (will retry): {exc}")

    def stop_requested(self) -> bool:
        try:
            return d1.ledger_stop_requested(self._d1, self._run.id)
        except Exception:
            return False

    def boundary(self) -> None:
        self.flush_snapshot()
        self.push_events()
        try:
            d1.ledger_update(self._d1, self._run.id, current=self._run.current)
        except Exception as exc:
            self._run.log(f"Ledger update failed (will retry): {exc}")
        if self.stop_requested():
            self._run.log("Stop requested via the jobs ledger.")
            self._run.request_stop()

    def terminal(self) -> None:
        self.push_events()
        try:
            d1.ledger_update(
                self._d1,
                self._run.id,
                status=self._run.status,
                error=self._run.error,
            )
        except Exception as exc:
            self._run.log(f"Final ledger update failed: {exc}")


def _register_store_videos(series_id: str, remote_sizes: dict) -> int:
    """Insert videos rows for chapter videos that exist in the store but
    were not pulled (mp4s never cross at stage-in). The skip checks
    (chapter_needs_work, _chapters_missing_video) are pure DB queries, so
    the rows keep skip_done and the video backfill truthful without the
    files on disk; serving the file is a local concern. Returns the number
    of rows added."""
    from entertainment_harness.library import works

    prefix = f"works/{series_id}/chapters/"
    kinds = {
        works.VIDEO_RECAP_DIR: "recap",
        works.VIDEO_NARRATION_DIR: "narration",
    }
    added = 0
    with connect() as conn:
        for key in remote_sizes:
            if not key.startswith(prefix) or not key.endswith("/out.mp4"):
                continue
            rest = key[len(prefix):].split("/")
            if len(rest) != 3:
                continue
            chapter_id, kind = rest[0], kinds.get(rest[1])
            if kind is None:
                continue
            chapter = conn.execute(
                "SELECT chapter_num FROM chapters WHERE id = ?", (chapter_id,)
            ).fetchone()
            if chapter is None:
                continue
            existing = conn.execute(
                "SELECT 1 FROM videos WHERE series_id = ? AND from_chapter = ?"
                " AND to_chapter = ? AND kind = ?",
                (series_id, chapter["chapter_num"], chapter["chapter_num"],
                 kind),
            ).fetchone()
            if existing is not None:
                continue
            vmeta = works.read_video_metadata(series_id, chapter_id, kind)
            conn.execute(
                "INSERT INTO videos (series_id, from_chapter, to_chapter,"
                " path, duration_s, created_at, tts_engine, model, kind,"
                " video_gen_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    series_id, chapter["chapter_num"], chapter["chapter_num"],
                    str(works.video_file_path(series_id, chapter_id, kind)),
                    vmeta.duration_s if vmeta else None,
                    vmeta.created_at if vmeta else utcnow(),
                    vmeta.tts_engine if vmeta else None,
                    vmeta.model if vmeta else None,
                    kind,
                    (vmeta.video_gen_provider if vmeta else None) or "local",
                ),
            )
            added += 1
    return added


def _run_scope(spec: JobSpec, log) -> set[str]:
    """Ids of every chapter the flow will touch: the selected chapters plus,
    with video on, the missing-video backfill set. Computed on the pre-run
    index, which is a superset of the post-run backfill (chapters recapped
    in this run get their video from on_recap), so staging in exactly these
    chapters' pages covers everything the flow reads."""
    from entertainment_harness import library
    from entertainment_harness.cli import _chapters_missing_video
    from entertainment_harness.server import runs as server_runs

    options = _run_options(spec)
    config, detail = server_runs.config_for_run(options)
    with connect() as conn:
        row = library.resolve_series(conn, spec.work)
        chapters = server_runs.select_run_chapters(
            conn, row, config, options, detail=detail, log=log
        )
        ids = {c["id"] for c in chapters}
        if options.video:
            ids |= {
                c["id"]
                for c in _chapters_missing_video(
                    conn, row["id"], table="recaps",
                    translated=options.translated, langs=config.library.langs,
                )
            }
    return ids


def run_job(spec: JobSpec) -> int:
    """Execute one job: stage-in -> index -> run -> stage-out.

    Returns the process exit code: 0 done/cancelled, 1 error.
    """
    from entertainment_harness.library import index as library_index
    from entertainment_harness.server import runs as server_runs
    from entertainment_harness.store import staging

    config = load_config()
    run = JobRun(spec)
    code = 0

    # D1 durable index (env-configured; the Worker injects these in C2):
    # stage a local snapshot + flush baseline, then push row diffs back at
    # chapter boundaries and once more at the end.
    d1_cfg = d1.D1Config.from_env()
    sink = None
    indexed_from_d1 = False
    if d1_cfg.available():
        db_path, baseline = d1.default_paths(spec.id, data_dir())
        staged_rows = d1.snapshot_pull(d1_cfg, db_path, baseline)
        indexed_from_d1 = staged_rows > 0
        run.log(f"D1 snapshot staged ({staged_rows} row(s)).")
        sink = _CloudSink(config, run, d1_cfg, db_path, baseline)

    if sink is not None and sink.stop_requested():
        # Stopped while still queued — don't start the flow at all.
        run.log("Stop requested before the run started.")
        run.cancel()
    else:
        try:
            remote_sizes: dict[str, int] = {}
            if store.is_configured(config):
                # Two-phase stage-in: metadata first (indexable state —
                # work.json, the cover, every JSON), then just the pages of
                # the chapters the run will touch. Whole-work pulls can
                # exceed the container disk; mp4s never cross at stage-in.
                meta_n, remote_sizes = staging.pull_work_matching(
                    config, spec.work, staging.is_metadata_key, log=run.log
                )
                run.log(
                    f"Staged in {meta_n} metadata file(s) for {spec.work}."
                )
            else:
                run.log("Store not configured; running against local data only.")
            if indexed_from_d1:
                run.log("Index restored from D1 snapshot.")
                # Re-index the staged work additively (index_works stays
                # skipped — it clears every table, turning remote rows into
                # flush DELETEs). index_work upserts, so a known work costs
                # a no-op rewrite while a partial snapshot — e.g. a series
                # row flushed by a job whose later ops failed, leaving no
                # chapters behind — heals. Rows reach D1 via the flush.
                wdir = works.work_dir(spec.work)
                if wdir.is_dir():
                    with connect() as conn:
                        library_index.index_work(conn, wdir)
                    run.log(
                        f"Indexed staged work {spec.work!r} into the snapshot."
                    )
            else:
                # Fresh D1 (or none): bootstrap by scanning the staged layout.
                # Never rescan over a populated snapshot — index_works clears
                # every table and the flush would turn remote rows into DELETEs.
                run.log(f"Indexed {library_index.index_works()} work(s).")
            if remote_sizes:
                # mp4s were not pulled, but the skip checks are pure DB
                # queries — register store-held videos so skip_done and the
                # missing-video backfill see them.
                registered = _register_store_videos(spec.work, remote_sizes)
                if registered:
                    run.log(
                        f"Registered {registered} store-held video(s) in the"
                        " index."
                    )
                scope = _run_scope(spec, run.log)
                if scope:
                    staging.pull_work_matching(
                        config, spec.work, staging.scope_key_filter(scope),
                        log=run.log,
                    )
                    run.log(f"Staged in pages for {len(scope)} chapter(s).")
            progress = (
                _FlushOnChapterDone(run, sink.boundary)
                if sink
                else ServerProgress(run)
            )
            chapters = server_runs._run_flow(run, _run_options(spec), progress)
        except RunCancelled:
            run.cancel()
        except Exception as exc:
            run.fail(str(exc))
            code = 1
        else:
            run.finish(chapters)
    if sink is not None:
        sink.flush_snapshot()
    if store.is_configured(config):
        try:
            pushed = staging.push_work(config, spec.work, log=run.log)
            run.log(f"Staged out {pushed} file(s) for {spec.work}.")
        except store.StoreError as exc:
            run.log(f"Stage-out failed: {exc}")
            if run.status == "done":
                run.status = "error"
                run.error = f"Stage-out failed: {exc}"
            code = 1
    if sink is not None:
        sink.terminal()
    return code


def main(spec_arg: str) -> int:
    """`eh job-run` entry point: parse the spec and run it."""
    return run_job(load_spec(spec_arg))
