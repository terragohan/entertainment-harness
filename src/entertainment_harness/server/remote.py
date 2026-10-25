"""Remote run control (animation-director Track B, C3): a RunManager-shaped
client for the cloud Worker.

With [cloud] enabled, `eh serve` swaps the local RunManager for
RemoteRunManager, so the /api/runs endpoints, the SSE event stream, and the
auto-toggle supervisor work unchanged while jobs execute on Cloudflare
containers. Differences from local runs:

- jobs carry a "queued" status until the container starts, and events lag:
  the container pushes events.jsonl to the store at chapter boundaries, so
  the SSE stream is a poll of that file, not a live feed;
- the auto toggle is one-shot per arming (see auto.py reconcile): it
  submits one job and disarms on any terminal status, because the local
  "unfinished" check reads the local index while the job's truth is the D1
  index in the cloud.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict

import httpx

from entertainment_harness import db
from entertainment_harness.config import load_config
from entertainment_harness.library import works
from entertainment_harness.store import StoreError, staging
from entertainment_harness.store import r2 as store_r2
from entertainment_harness.server.runs import RunConflict, RunOptions

#: Terminal ledger status -> the closing SSE event a subscriber expects.
TERMINAL = {"done": "run-done", "error": "run-error", "cancelled": "run-cancelled"}
TERMINAL_EVENTS = frozenset(TERMINAL.values())


class RemoteError(Exception):
    """The worker rejected the call or could not be reached."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _current_token() -> str:
    """Resolve the submit token for each call, so a token saved through the
    settings UI takes effect without an `eh serve` restart. The config value
    wins; the EH_CLOUD_TOKEN env var is the fallback."""
    try:
        return load_config().cloud.resolved_token
    except Exception:
        return os.environ.get("EH_CLOUD_TOKEN", "")


class RemoteRun:
    """A Run-shaped view of one jobs-ledger row (title filled from the
    local index when the work exists locally)."""

    def __init__(self, manager: "RemoteRunManager", job: dict) -> None:
        self._manager = manager
        self.id = job["id"]
        self.work = job["work"]
        self.title = job.get("title") or manager._title_for(job["work"])
        self.status = job["status"]
        self.error = job.get("error")
        self.created_at = job.get("created_at", "")
        self.current = job.get("current")
        options = dict(job.get("options") or {})
        try:
            # Normalize to the full RunOptions shape the UI reads.
            options = asdict(RunOptions(work=self.work, **options))
        except TypeError:
            pass  # forward the sparse dict rather than dropping the job
        self.options = options

    def request_stop(self) -> None:
        self._manager.stop(self.id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "work": self.work,
            "title": self.title,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "current": self.current,
            "options": self.options,
        }


class RemoteRunManager:
    """Duck-typed RunManager talking to the cloud Worker's jobs API."""

    #: Tells auto.py's reconcile to disarm a toggle after one terminal job
    #: instead of re-checking unfinished work against the local index.
    one_shot_runs = True

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    @classmethod
    def from_config(cls, config) -> "RemoteRunManager":
        if not config.cloud.resolved_token:
            import sys

            print(
                "cloud: [cloud] enabled but no submit token is set — put it in"
                " [cloud] token (Settings → API keys) or export"
                " EH_CLOUD_TOKEN; run-control endpoints will 502 until then",
                file=sys.stderr,
            )
        return cls(config.cloud.worker_url)

    # --- HTTP ----------------------------------------------------------------

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        token = _current_token()
        if not token:
            raise RemoteError(
                "cloud: no submit token — set the cloud worker token in"
                " Settings → API keys ([cloud] token) or export"
                " EH_CLOUD_TOKEN"
            )
        try:
            resp = httpx.request(
                method,
                f"{self._base}{path}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=self._timeout,
                **kw,
            )
        except httpx.HTTPError as exc:
            raise RemoteError(f"cloud worker unreachable: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            raise RemoteError(
                f"cloud worker: {resp.status_code}: {detail}",
                status=resp.status_code,
            )
        return resp

    def _title_for(self, work_id: str) -> str:
        try:
            with db.connect() as conn:
                row = conn.execute(
                    "SELECT title FROM series WHERE id = ?", (work_id,)
                ).fetchone()
        except Exception:
            return work_id
        return row["title"] if row else work_id

    # --- RunManager surface ----------------------------------------------------

    def list(self) -> list[RemoteRun]:
        resp = self._request("GET", "/api/jobs")
        return [RemoteRun(self, job) for job in resp.json()["jobs"]]

    def get(self, run_id: str) -> RemoteRun | None:
        try:
            resp = self._request("GET", f"/api/jobs/{run_id}")
        except RemoteError as exc:
            if exc.status == 404:
                return None
            raise
        return RemoteRun(self, resp.json()["job"])

    def start(self, work_id: str, title: str, options: RunOptions) -> RemoteRun:
        self._stage_up(work_id)
        opts = asdict(options)
        opts.pop("work", None)
        try:
            resp = self._request(
                "POST", "/api/jobs",
                json={"work": work_id, "kind": "recap", "options": opts},
            )
        except RemoteError as exc:
            if exc.status == 409:
                raise RunConflict(str(exc)) from exc
            raise
        job = resp.json()["job"]
        job.setdefault("title", title)
        return RemoteRun(self, job)

    def _stage_up(self, work_id: str) -> None:
        """Push the work's local files to R2 so the container can stage them
        in. Skipped with a stderr note when the work has no local dir or R2
        is unconfigured (the cloud may already hold it from an earlier job);
        an upload failure aborts the submit — a half-staged work would fail
        in the container anyway."""
        import sys

        if not works.work_dir(work_id).is_dir():
            print(
                f"cloud: no local work dir for {work_id} — submitting without"
                " stage-up",
                file=sys.stderr,
            )
            return
        config = load_config()
        if not store_r2.is_ready(config):
            print(
                "cloud: R2 store not configured (Settings → API keys or R2_*"
                " env vars) — submitting without stage-up; the job needs the"
                " work already in R2",
                file=sys.stderr,
            )
            return
        try:
            staging.stage_up(
                config, work_id,
                log=lambda m: print(f"cloud: {m}", file=sys.stderr),
            )
        except StoreError as exc:
            raise RemoteError(f"stage-up failed: {exc}") from exc

    def stop(self, run_id: str) -> None:
        """Cooperative stop. A 409 (job no longer active) mirrors the local
        idempotent stop: a finished run is already stopped."""
        try:
            self._request("POST", f"/api/jobs/{run_id}/stop")
        except RemoteError as exc:
            if exc.status != 409:
                raise

    # --- events (for the SSE bridge) ------------------------------------------

    def events(self, run_id: str) -> list[dict]:
        """The job's events.jsonl as parsed dicts; [] while the file doesn't
        exist yet (the container first pushes it at a chapter boundary)."""
        try:
            resp = self._request("GET", f"/api/jobs/{run_id}/events")
        except RemoteError as exc:
            if exc.status == 404:
                return []
            raise
        return [
            json.loads(line) for line in resp.text.splitlines() if line.strip()
        ]


def _sse_frame(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event)}\n\n"


async def remote_event_stream(
    manager: RemoteRunManager, run_id: str, poll_s: float = 2.0
):
    """SSE-shaped replay of a cloud job's events.jsonl. Ends on the job's
    terminal event; when the ledger reaches a terminal status without one
    (hard crash, reconciled row), one settle cycle lands the final push,
    then a synthetic closing frame is emitted so subscribers don't hang."""
    seen = 0
    while True:
        events = await asyncio.to_thread(manager.events, run_id)
        chunk = events[seen:]
        seen = len(events)
        for event in chunk:
            yield _sse_frame(event)
        if any(e.get("event") in TERMINAL_EVENTS for e in chunk):
            return
        run = await asyncio.to_thread(manager.get, run_id)
        if run is None or run.status in TERMINAL:
            await asyncio.sleep(poll_s)
            events = await asyncio.to_thread(manager.events, run_id)
            chunk = events[seen:]
            for event in chunk:
                yield _sse_frame(event)
            if any(e.get("event") in TERMINAL_EVENTS for e in chunk):
                return
            status = run.status if run is not None else "error"
            yield _sse_frame({
                "event": TERMINAL[status],
                "run_id": run_id,
                "work": run.work if run else "",
                "ts": "",
                "message": (run.error if run else None) or f"job {status}",
            })
            return
        await asyncio.sleep(poll_s)
