"""Cast rebuild: re-extract a work's character registry from its stored
chapter artifacts in the background — the work view's Rebuild button,
running the same fold as `eh cast --rebuild` (`characters.rebuild_cast`).

Jobs run on daemon threads, one active build per work, and are pollable via
`GET /api/cast-builds`. They are in-memory like exports: gone when the
backend restarts (a restart mid-build simply leaves the registry at
whatever the last committed chapter fold produced).
"""

from __future__ import annotations

import sys
import threading
import uuid
from dataclasses import dataclass, field

from entertainment_harness import db
from entertainment_harness.db import utcnow

LOG_TAIL = 8  # log lines kept for the polling UI


class CastBuildConflict(Exception):
    """A cast build is already active for this work (POST -> 409)."""


@dataclass
class CastBuildJob:
    id: str
    work: str  # series id
    title: str
    status: str = "running"  # running | done | error
    count: int = 0  # characters in the rebuilt registry
    error: str | None = None
    log: list[str] = field(default_factory=list)  # tail of the fold's log
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "work": self.work,
            "title": self.title,
            "status": self.status,
            "count": self.count,
            "error": self.error,
            "log": list(self.log),
            "created_at": self.created_at,
        }


def _run_build(job: CastBuildJob, series_id: str) -> None:
    def log(msg: str) -> None:
        print(f"cast-build: {msg}", file=sys.stderr)
        job.log.append(msg)
        del job.log[:-LOG_TAIL]

    try:
        from entertainment_harness.config import load_config
        from entertainment_harness.hardware import probe
        from entertainment_harness.pipelines import characters as chars_mod

        config = load_config()
        profile = probe(config.hardware.budget_gb)
        thinking = config.pipeline.thinking or "medium"
        with db.connect() as conn:
            series = conn.execute(
                "SELECT * FROM series WHERE id = ?", (series_id,)
            ).fetchone()
            job.count = chars_mod.rebuild_cast(
                conn, series, config, profile, thinking=thinking, log=log
            )
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 — terminal state for the poller
        job.error = str(exc)
        job.status = "error"


class CastBuildManager:
    """Tracks cast-build jobs by id; one active build per work."""

    def __init__(self) -> None:
        self._jobs: dict[str, CastBuildJob] = {}
        self._lock = threading.Lock()

    def start(self, series_id: str, title: str) -> CastBuildJob:
        with self._lock:
            for existing in self._jobs.values():
                if existing.work == series_id and existing.status == "running":
                    raise CastBuildConflict(
                        f"A cast rebuild is already active for {title!r}"
                        f" (job {existing.id})"
                    )
            job = CastBuildJob(
                id=uuid.uuid4().hex[:12], work=series_id, title=title
            )
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=_run_build, args=(job, series_id),
            name=f"eh-cast-build-{job.id}", daemon=True,
        )
        thread.start()
        return job

    def list(self) -> list[CastBuildJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at, reverse=True,
            )
