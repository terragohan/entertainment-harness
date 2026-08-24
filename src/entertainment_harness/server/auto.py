"""Auto-process supervisor: the per-work "process unfinished" toggle.

The supervisor owns one bit of state per work — "keep processing this work's
unfinished chapters in the background" — persisted as JSON at
`data_dir()/auto.json` so the toggle and its modifiers survive both window
closes and backend restarts. Enabling a work starts a run immediately when
one is needed (and none is active); a daemon Timer loop re-checks every
`poll_s` seconds so a run that finished or errored is followed up (finished
→ start the next batch, errored → turn the toggle off so a broken config
can't loop).

The toggle is *armed* only by an explicit `set(enabled=True)` call (i.e. the
user flipping it on). State loaded from disk at startup is disarmed: after a
backend restart, enabled works do NOT auto-resume — the user must toggle
again. This keeps a crash loop (e.g. a dying TTS stack that kills the
process before the error-recheck can persist "off") from re-arming itself
on every launch, while the settings themselves (scope, detail, instruction)
are still there to re-enable with.

"Unfinished" mirrors exactly what a run would do (see `_run_flow` in
runs.py): the gap-aware pending set (`pending_chapters` at the configured
detail, with the configured langs, `fill_gaps=True` — chapters behind the
read frontier with no artifact and no usable video included) plus recapped
chapters lacking a playable video (`_chapters_missing_video`, the `--video`
backfill set). Runs are started with the default (scope-less) selection —
that IS the gap-aware pending set — plus `video=True` and the modifiers
stored with the toggle (detail, instruction, skip_preflight, video_mode).

The toggle can also carry an optional scope: a `--chapters`-style spec or
`all_chapters`. Scoped runs select exactly that scope with `skip_done=True`,
so only the unfinished chapters inside it are processed; `has_unfinished`
mirrors that per-scope check instead of the scope-less one.

`armed` is session state only: it is never written to auto.json (the disk
format is unchanged) and `load()` starts every entry disarmed.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from entertainment_harness import db
from entertainment_harness.config import data_dir, load_config
from entertainment_harness.pipelines.recap import (
    DETAIL_LEVELS,
    chapter_needs_work,
    pending_chapters,
    select_chapters,
)
from entertainment_harness.server.runs import RunManager, RunOptions

STATE_FILENAME = "auto.json"

#: One-shot reconcile this long after a supervisor-started run, so a failed
#: run turns the toggle off promptly instead of waiting for the next tick.
_ERROR_RECHECK_S = 2.0


def state_path() -> Path:
    return data_dir() / STATE_FILENAME


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


class AutoSupervisor:
    """Owns the persisted per-work auto state and the reconcile loop.

    `set()` is the only mutation entry. The Timer loop is daemon=True and
    started lazily on the first enable; tests can pass a large `poll_s` and
    call `reconcile()` directly.
    """

    def __init__(self, run_manager: RunManager, poll_s: float = 30.0) -> None:
        self.runs = run_manager
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}
        self._timer: threading.Timer | None = None

    # --- persistence -------------------------------------------------------

    def load(self) -> None:
        """Read auto.json; a missing or corrupt file means empty state.
        Everything loaded starts disarmed (see the module docstring) — only
        an explicit `set(enabled=True)` re-arms a work after a restart."""
        try:
            raw = json.loads(state_path().read_text())
        except (OSError, ValueError):
            raw = {}
        with self._lock:
            self._state = {
                str(work_id): {
                    "enabled": bool(entry.get("enabled")),
                    "armed": False,
                    "options": {
                        "detail": entry.get("options", {}).get("detail"),
                        "instruction": entry.get("options", {}).get("instruction"),
                        "skip_preflight": bool(
                            entry.get("options", {}).get("skip_preflight")
                        ),
                        "video_mode": entry.get("options", {}).get("video_mode"),
                        "chapters": entry.get("options", {}).get("chapters"),
                        "all_chapters": bool(
                            entry.get("options", {}).get("all_chapters")
                        ),
                    },
                }
                for work_id, entry in raw.items()
                if isinstance(entry, dict)
            }

    def _persist(self) -> None:
        # `armed` is session state — persist only the durable fields so the
        # on-disk format is unchanged.
        _write_atomic(
            state_path(),
            {
                work_id: {
                    "enabled": entry["enabled"],
                    "options": entry["options"],
                }
                for work_id, entry in self._state.items()
            },
        )

    def is_enabled(self, work_id: str) -> bool:
        """The user-facing toggle state: enabled AND armed. Loaded state is
        disarmed, so after a restart this reads False until the user
        re-toggles the work on."""
        with self._lock:
            entry = self._state.get(work_id)
            return bool(entry and entry["enabled"] and entry["armed"])

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._state))

    # --- the toggle --------------------------------------------------------

    def set(
        self,
        work_id: str,
        enabled: bool,
        options: dict | None = None,
    ) -> dict:
        """The only mutation entry. Persists the new state; on enable arms
        the toggle and starts a run immediately when the work has unfinished
        chapters and none is active; on disable disarms it and cooperatively
        stops any active run. Returns `{work, enabled, run}` (run = the
        started run's dict, or null)."""
        options = dict(options or {})
        run_dict: dict | None = None
        with self._lock:
            self._state[work_id] = {
                "enabled": bool(enabled),
                "armed": bool(enabled),
                "options": {
                    "detail": options.get("detail"),
                    "instruction": options.get("instruction"),
                    "skip_preflight": bool(options.get("skip_preflight")),
                    "video_mode": options.get("video_mode"),
                    "chapters": options.get("chapters"),
                    "all_chapters": bool(options.get("all_chapters")),
                },
            }
            self._persist()
            if enabled:
                self._ensure_timer()
                if not self._active_run(work_id):
                    run = self._start_if_unfinished(work_id)
                    if run is not None:
                        run_dict = run.to_dict()
            else:
                active = self._active_run(work_id)
                if active is not None:
                    active.request_stop()
        return {"work": work_id, "enabled": enabled, "run": run_dict}

    # --- reconcile loop ----------------------------------------------------

    def reconcile(self) -> None:
        """For each enabled+armed work with no active run: disable it if its
        latest run errored (a broken config must not loop), else start a run
        when there is unfinished work. Disarmed entries (loaded at startup,
        or toggled off) are skipped. Runs entirely under the lock; per-work
        failures are logged to stderr and never break the loop."""
        import sys

        with self._lock:
            for work_id, entry in list(self._state.items()):
                if not entry["enabled"] or not entry["armed"]:
                    continue
                try:
                    latest = self._latest_run(work_id)
                    if latest is not None and latest.status == "error":
                        entry["enabled"] = False
                        self._persist()
                        continue
                    if self._active_run(work_id) is not None:
                        continue
                    self._start_if_unfinished(work_id)
                except Exception as exc:  # noqa: BLE001 — never kill the loop
                    print(
                        f"auto: reconcile failed for {work_id}: {exc}",
                        file=sys.stderr,
                    )

    def _tick(self) -> None:
        try:
            self.reconcile()
        finally:
            with self._lock:
                if any(e["armed"] for e in self._state.values()):
                    self._start_timer()

    def _start_timer(self) -> None:
        # Caller holds the lock.
        self._timer = threading.Timer(self.poll_s, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _ensure_timer(self) -> None:
        # Caller holds the lock.
        if self._timer is None or not self._timer.is_alive():
            self._start_timer()

    def _recheck_soon(self) -> None:
        """One-shot reconcile shortly after a supervisor-started run, so a
        terminal error turns the toggle off without waiting for the next
        poll tick."""
        timer = threading.Timer(_ERROR_RECHECK_S, self.reconcile)
        timer.daemon = True
        timer.start()

    # --- run plumbing ------------------------------------------------------

    def _latest_run(self, work_id: str):
        # runs.list() is newest-first.
        return next((r for r in self.runs.list() if r.work == work_id), None)

    def _active_run(self, work_id: str):
        return next(
            (
                r
                for r in self.runs.list()
                if r.work == work_id and r.status == "running"
            ),
            None,
        )

    def _start_if_unfinished(self, work_id: str):
        """Start a run when the work has unfinished chapters; None otherwise.
        Caller holds the lock."""
        from entertainment_harness import library

        config = load_config()
        with db.connect() as conn:
            try:
                row = library.resolve_series(conn, work_id)
            except library.LibraryError:
                return None  # work vanished from the library — skip it
            if not has_unfinished(conn, row["id"], config, self._state[work_id]):
                return None
        options = self._state[work_id]["options"]
        # A scoped run selects exactly the stored scope and skips the
        # chapters already recapped(+videod) inside it — the toggle's job is
        # unfinished chapters, within whatever scope is chosen.
        scoped = bool(options.get("chapters") or options.get("all_chapters"))
        run = self.runs.start(
            row["id"],
            row["title"],
            RunOptions(
                work=row["id"],
                all_chapters=bool(options.get("all_chapters")),
                chapters=options.get("chapters"),
                skip_done=scoped,
                video=True,
                detail=options["detail"],
                instruction=options["instruction"],
                skip_preflight=options["skip_preflight"],
                video_mode=options.get("video_mode"),
            ),
        )
        self._recheck_soon()
        return run


def has_unfinished(conn, series_id: str, config, entry: dict | None = None) -> bool:
    """True when a supervisor run with the stored options would have
    something to do. Scope-less: pending recaps at the configured grain
    (and langs), chapters behind the read frontier with no artifact and no
    usable video (the `--video` gap bucket), or recapped chapters missing a
    playable video (the backfill set). Scoped (`chapters`/`all_chapters` in
    the entry's options): any chapter inside the scope that
    `chapter_needs_work` would keep (a scoped run is skip_done, so done
    chapters — inside or outside the scope — don't count)."""
    from entertainment_harness.cli import _chapters_missing_video

    entry = entry or {}
    options = entry.get("options") or {}
    detail = options.get("detail") or config.pipeline.detail
    if detail not in DETAIL_LEVELS:
        detail = "standard"
    if options.get("chapters") or options.get("all_chapters"):
        series = conn.execute(
            "SELECT * FROM series WHERE id = ?", (series_id,)
        ).fetchone()
        if series is None:
            return False
        todo = select_chapters(
            conn, series, config, detail=detail, chapter_num=None,
            chapters_spec=options.get("chapters"),
            all_chapters=bool(options.get("all_chapters")),
            max_chapters=500, translated=False, verb="recap",
            log=lambda *args, **kwargs: None,
        )
        return any(
            chapter_needs_work(
                conn, series_id, c, detail=detail, want_video=True
            )
            for c in todo
        )
    if pending_chapters(
        conn, series_id, langs=config.library.langs, detail=detail,
        fill_gaps=True,
    ):
        return True
    return bool(
        _chapters_missing_video(
            conn, series_id, table="recaps", translated=False,
            langs=config.library.langs,
        )
    )
