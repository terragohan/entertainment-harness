"""Live pipeline UI: a Rich progress display pinned at the bottom of the
terminal while translate/recap/video stages run above it as log lines.

Two bars: overall chapter progress, and the current chapter's stage (with a
free-text detail like "page 3/16" or "narration"). Pipelines report through
the duck-typed interface start/chapter_start/stage/log/chapter_done — they
accept progress=None, so tests and library use never construct this.

On a non-terminal (piped/redirected output) the bars are disabled and log
lines print plainly, keeping CI/background runs readable.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


class PipelineUI:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._live = self.console.is_terminal
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[chapter]}"),
            TextColumn("{task.fields[stage]}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            disable=not self._live,
        )
        self._overall = None
        self._current = None

    # context manager: bars live for the duration of the run
    def __enter__(self) -> "PipelineUI":
        self._progress.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._progress.__exit__(*exc)

    def start(self, n_chapters: int) -> None:
        if self._live:
            self._overall = self._progress.add_task(
                "chapters", total=max(n_chapters, 1),
                chapter="overall", stage="",
            )

    def chapter_start(self, chapter_num: float) -> None:
        if self._current is not None:
            self._progress.remove_task(self._current)
        if self._live:
            self._current = self._progress.add_task(
                "stage", total=None,
                chapter=f"ch {chapter_num:g}", stage="starting",
            )

    def stage(self, name: str, detail: str = "") -> None:
        if self._live and self._current is not None:
            self._progress.update(
                self._current, stage=f"{name}: {detail}" if detail else name
            )

    def log(self, msg: str) -> None:
        # prints above the pinned bars when live; plain print otherwise
        self._progress.console.print(msg)

    def chapter_done(self, output_desc: str = "") -> None:
        if self._current is not None:
            self._progress.remove_task(self._current)
            self._current = None
        if self._live and self._overall is not None:
            self._progress.advance(self._overall)
        if output_desc:
            self.log(f"  output: {output_desc}")
