"""PipelineUI: non-terminal fallback prints plainly and never crashes; the
duck-typed interface matches what the pipelines call."""

from __future__ import annotations

import io

from rich.console import Console

from entertainment_harness.ui import PipelineUI


def _make_ui() -> tuple[PipelineUI, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    return PipelineUI(console), buf


def test_non_terminal_logs_plainly():
    ui, buf = _make_ui()
    with ui:
        ui.start(2)
        ui.chapter_start(26.0)
        ui.stage("translate", "page 3/16")
        ui.log("  page 3/16: 4 bubble(s).")
        ui.chapter_done("recap stored in library DB")
        ui.chapter_start(27.0)
        ui.stage("recap", "pages 1-4 of 20")
        ui.chapter_done()
    out = buf.getvalue()
    assert "page 3/16: 4 bubble(s)." in out
    assert "output: recap stored in library DB" in out


def test_chapter_done_without_output_is_quiet():
    ui, buf = _make_ui()
    with ui:
        ui.start(1)
        ui.chapter_start(1.0)
        ui.chapter_done()
    assert "output:" not in buf.getvalue()


def test_no_start_still_works():
    # pipelines only call start() when chapters are selected; a bare UI must
    # still accept stage/log calls (e.g. video backfill without recaps)
    ui, buf = _make_ui()
    with ui:
        ui.log("Backfilling video for recapped chapter 3...")
        ui.stage("video", "script")
    assert "Backfilling video" in buf.getvalue()
