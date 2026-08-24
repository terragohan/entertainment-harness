"""Video export: assemble a work's chapter videos into one mp4 in the
background and land it in the user's Downloads folder.

An export resolves every playable chapter video (the same
usable-beats-wiped, compressed-deliverable-first resolution as the stream
endpoint), then either copies the single part or losslessly concatenates
several (mixed formats re-encode at 720p — see `video.assemble.concat_mp4s`,
shared with `eh video concat`). Jobs run on daemon threads, one active
export per work, and are pollable via `GET /api/exports`; the Electrobun
main process polls them so it can fire a native notification even when the
window is closed.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from entertainment_harness import db
from entertainment_harness.db import utcnow
from entertainment_harness.library import works

#: Destination directory override (tests, smoke); default
#: ~/Downloads/EntertainmentHarness.
DEST_DIR_ENV = "EH_EXPORT_DIR"


class ExportConflict(Exception):
    """An export is already active for this work (POST -> 409)."""


@dataclass
class ExportJob:
    id: str
    work: str  # series id
    title: str
    status: str = "running"  # running | done | error
    dest: str = ""
    total: int = 0  # chapter videos assembled
    skipped: int = 0  # chapters with no usable local video
    error: str | None = None
    created_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "work": self.work,
            "title": self.title,
            "status": self.status,
            "dest": self.dest,
            "total": self.total,
            "skipped": self.skipped,
            "error": self.error,
            "created_at": self.created_at,
        }


def dest_dir() -> Path:
    override = os.environ.get(DEST_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / "Downloads" / "EntertainmentHarness"


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "export"


def _resolve_parts(conn, series_id: str) -> tuple[list[tuple[float, Path]], int]:
    """Playable chapter videos (chapter_num, path) in chapter order, plus
    the count of chapters left out (no usable local file)."""
    rows = conn.execute(
        "SELECT c.id AS chapter_id, c.chapter_num, v.kind, v.wiped_at"
        " FROM chapters c"
        # One row per chapter even with several videos rows (one per kind
        # is legal): usable beats wiped, then newest.
        " LEFT JOIN videos v ON v.id = ("
        "  SELECT v2.id FROM videos v2"
        "  WHERE v2.series_id = c.series_id"
        "   AND v2.from_chapter = c.chapter_num"
        "   AND v2.to_chapter = c.chapter_num"
        "  ORDER BY (v2.wiped_at IS NOT NULL), v2.id DESC LIMIT 1)"
        " WHERE c.series_id = ? AND c.chapter_num IS NOT NULL"
        " ORDER BY c.chapter_num",
        (series_id,),
    ).fetchall()
    parts: list[tuple[float, Path]] = []
    skipped = 0
    for row in rows:
        if row["kind"] is None or row["wiped_at"] is not None:
            skipped += 1
            continue
        candidates = []
        meta = works.read_video_metadata(
            series_id, row["chapter_id"], row["kind"]
        )
        if meta is not None and meta.compress:
            candidates.append(works.video_file_path(
                series_id, row["chapter_id"], row["kind"],
                suffix=f"out-{meta.compress}.mp4",
            ))
        candidates.append(
            works.video_file_path(series_id, row["chapter_id"], row["kind"])
        )
        part = next((c for c in candidates if c.is_file()), None)
        if part is None:
            skipped += 1
            continue
        parts.append((row["chapter_num"], part))
    return parts, skipped


def _run_export(job: ExportJob, series_id: str, title: str) -> None:
    log = lambda msg: print(f"export: {msg}", file=sys.stderr)  # noqa: E731
    try:
        with db.connect() as conn:
            parts, skipped = _resolve_parts(conn, series_id)
        job.skipped = skipped
        if not parts:
            raise RuntimeError(
                "no playable local videos for this work — re-render chapters"
                " or pull them from the store first"
            )
        first, last = parts[0][0], parts[-1][0]
        name = (
            f"{_slug(title)}-ch{first:g}.mp4" if first == last
            else f"{_slug(title)}-ch{first:g}-{last:g}.mp4"
        )
        dest = dest_dir() / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        job.dest = str(dest)
        if len(parts) == 1:
            log(f"copying {parts[0][1]} -> {dest}")
            shutil.copyfile(parts[0][1], dest)
        else:
            log(f"assembling {len(parts)} videos -> {dest}")
            from entertainment_harness.video.assemble import concat_mp4s

            concat_mp4s([p for _, p in parts], dest, log=log)
        job.total = len(parts)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 — terminal state for the poller
        job.error = str(exc)
        job.status = "error"


class ExportManager:
    """Tracks export jobs by id; one active export per work."""

    def __init__(self) -> None:
        self._jobs: dict[str, ExportJob] = {}
        self._lock = threading.Lock()

    def start(self, series_id: str, title: str) -> ExportJob:
        with self._lock:
            for existing in self._jobs.values():
                if existing.work == series_id and existing.status == "running":
                    raise ExportConflict(
                        f"An export is already active for {title!r}"
                        f" (job {existing.id})"
                    )
            job = ExportJob(id=uuid.uuid4().hex[:12], work=series_id,
                            title=title)
            self._jobs[job.id] = job
        thread = threading.Thread(
            target=_run_export, args=(job, series_id, title),
            name=f"eh-export-{job.id}", daemon=True,
        )
        thread.start()
        return job

    def list(self) -> list[ExportJob]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda j: j.created_at, reverse=True,
            )
