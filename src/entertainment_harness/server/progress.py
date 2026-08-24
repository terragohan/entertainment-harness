"""Server-side progress reporter: the duck-typed pipeline interface from
ui.py (start/chapter_start/stage/log/chapter_done), but each call becomes a
JSON-able event dict buffered on the run for SSE subscribers.

Pipelines run in a worker thread while SSE subscribers live on the asyncio
loop, so delivery goes through a threading.Condition on the Run: emit()
appends under the lock and notifies; the SSE generator replays the buffer
then blocks (via asyncio.to_thread) for more events. No event-loop handle is
captured, so this is safe under both uvicorn and the FastAPI TestClient.
"""

from __future__ import annotations

from typing import Any


class RunCancelled(Exception):
    """Raised through the progress interface when the run is stopped.

    Cancellation is cooperative: the flag is checked at structural progress
    callbacks (start/chapter_start/stage/chapter_done), so an in-flight model
    or ffmpeg call runs to completion and the run unwinds at the next stage
    or chapter boundary. `log` never raises — it is called from error paths.
    """


class ServerProgress:
    """Drop-in for ui.PipelineUI; `run` supplies emit (thread-safe) and the
    stop flag. Structural callbacks raise RunCancelled once a stop has been
    requested; `log` never does (pipelines call it from error paths)."""

    def __init__(self, run) -> None:
        self._run = run
        self._chapter: float | None = None

    def _check(self) -> None:
        if self._run.stop_requested.is_set():
            raise RunCancelled()

    def start(self, n_chapters: int) -> None:
        self._check()
        self._run.emit("run-start", chapters=n_chapters)

    def chapter_start(self, chapter_num: float) -> None:
        self._check()
        self._chapter = chapter_num
        self._run.emit("chapter-start", chapter=chapter_num)

    def stage(self, name: str, detail: str = "") -> None:
        self._check()
        self._run.emit("stage", chapter=self._chapter, stage=name, detail=detail)

    def log(self, msg: Any) -> None:
        self._run.emit("log", chapter=self._chapter, message=str(msg))

    def chapter_done(self, output_desc: str = "") -> None:
        self._check()
        self._run.emit("chapter-done", chapter=self._chapter, output=output_desc)
        self._chapter = None
