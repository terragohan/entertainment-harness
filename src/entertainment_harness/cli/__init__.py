"""Typer CLI; entry point `eh`."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape

from entertainment_harness import db, library
from entertainment_harness.config import data_dir, load_config
from entertainment_harness.hardware import probe, snapshot
from entertainment_harness.library import works
from entertainment_harness.models.base import ModelError
from entertainment_harness.plugins import PluginError
from entertainment_harness.preflight import (
    check,
    format_plan,
    plan_for_recap,
)

app = typer.Typer(help="Entertainment Harness: local manga recaps and videos.")
models_app = typer.Typer(help="Inspect and manage local models.")
app.add_typer(models_app, name="models")
voices_app = typer.Typer(help="Manage cloned voices for the qwen3 TTS engine.")
app.add_typer(voices_app, name="voices")
sources_app = typer.Typer(help="List, enable, and disable content sources.")
app.add_typer(sources_app, name="sources")
plugin_app = typer.Typer(help="Inspect and conformance-check plugins.")
app.add_typer(plugin_app, name="plugin")
console = Console()
err_console = Console(stderr=True)


def load_instruction(value: str) -> str:
    """Resolve a steering instruction: a leading `@` reads the instruction
    from the named file (curl-style; relative paths resolve from the cwd),
    anything else passes through as the inline instruction. File contents are
    stripped, so an empty file means no instruction."""
    if not value.startswith("@"):
        return value
    path = Path(value[1:])
    try:
        return path.read_text().strip()
    except OSError as exc:
        err_console.print(
            f"[red]Cannot read instruction file '{path}':"
            f" {exc.strerror or exc}[/red]"
        )
        raise typer.Exit(code=1) from exc


def _pipeline_config(
    thinking: str | None,
    vision_model: str | None,
    text_model: str | None,
    backend: str | None,
    quant: str | None,
    detail: str | None = None,
    instruction: str | None = None,
):
    """Load config and apply the shared model/thinking/detail/instruction CLI
    overrides."""
    from entertainment_harness.pipelines.judge import THINKING_LEVELS
    from entertainment_harness.pipelines.recap import DETAIL_LEVELS

    config = load_config()
    if thinking is not None:
        if thinking not in THINKING_LEVELS:
            err_console.print(
                f"[red]--thinking must be one of: {', '.join(THINKING_LEVELS)}[/red]"
            )
            raise typer.Exit(code=1)
        config.pipeline.thinking = thinking
    if detail is not None:
        if detail not in DETAIL_LEVELS:
            err_console.print(
                f"[red]--detail must be one of: {', '.join(DETAIL_LEVELS)}[/red]"
            )
            raise typer.Exit(code=1)
        config.pipeline.detail = detail
    if instruction is not None:
        config.pipeline.instructions = instruction
    if vision_model:
        config.models.vision.model = vision_model
    if text_model:
        config.models.text.model = text_model
    if backend:
        config.models.vision.backend = backend
        config.models.text.backend = backend
    if quant:
        config.models.vision.quant = quant
    return config


def _validate_pipeline_opts(
    all_chapters: bool, chapter: float | None,
    compress: str | None, video_mode: str | None, max_chapters: int,
) -> None:
    """Shared option validation for `eh recap`."""
    from entertainment_harness.video.compress import PRESETS
    from entertainment_harness.video.pipeline import VIDEO_MODES

    if max_chapters < 1:
        err_console.print("[red]--max-chapters must be at least 1[/red]")
        raise typer.Exit(code=1)
    if compress is not None and compress not in PRESETS:
        err_console.print(
            f"[red]--compress must be one of: {', '.join(PRESETS)}[/red]"
        )
        raise typer.Exit(code=1)
    if video_mode is not None and video_mode not in VIDEO_MODES:
        err_console.print(
            f"[red]--video-mode must be one of: {', '.join(VIDEO_MODES)}[/red]"
        )
        raise typer.Exit(code=1)


def _run_preflight_check(
    plan,
    config,
    *,
    dry_run: bool,
    skip_preflight: bool,
) -> None:
    """Build snapshot, check plan, and either exit (dry-run) or raise.

    Returns normally when the plan passes or is skipped."""
    snap = snapshot(data_dir(), config.hardware.budget_gb)

    if dry_run:
        console.print("[bold]Pre-flight plan[/bold]")
        console.print(format_plan(plan, snap))
        issues = check(plan, snap, config.preflight)
        if issues:
            console.print("[yellow]Issues:[/yellow]")
            for issue in issues:
                console.print(f"  - {issue}")
        raise typer.Exit(code=0 if not issues else 1)

    if not config.preflight.enabled or skip_preflight:
        return

    issues = check(plan, snap, config.preflight)
    if issues:
        err_console.print("[red]Pre-flight check failed:[/red]")
        for issue in issues:
            err_console.print(f"  - {issue}")
        err_console.print("Use --skip-preflight to bypass this guard.")
        raise typer.Exit(code=1)


def _make_chapter_video(conn, row, config, profile, ui, *,
                        voice, tts_engine, colorize, translated, compress,
                        video_mode, panel_first=False):
    """Return the callback that (re)builds one chapter's video and pushes it
    to the store when configured. The video kind follows the artifact's
    detail (full -> narration, else recap); panel_first forces narration.
    build_video supersedes the other kind's video, so one video per chapter
    remains."""
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
            voice=voice, engine_name=tts_engine,
            colorize=True if colorize else None,
            translated=True if translated else None,
            compress=compress, video_mode=video_mode,
            panel_first=panel_first,
            progress=ui, log=ui.log,
        )
        from entertainment_harness import store

        if store.is_configured(config):
            try:
                store.push_video(config, row["id"], chapter_id, log=ui.log)
            except store.StoreError as exc:
                ui.log(f"Store push failed: {exc}")
        console.print(f"[bold green]{out}[/bold green]")

    return make_video


def _chapters_missing_video(conn, series_id, *, table, translated, langs,
                            detail=None):
    """Artifacted chapters lacking a playable video. One video per chapter at
    the artifact's grain, so a videos row of EITHER kind satisfies the
    chapter; wiped rows (file deleted, row kept with wiped_at set) count as
    missing: those need a re-render just as much as chapters with no videos
    row. `table` is the artifact table joined for chapter selection
    ("recaps" post-merge); when `detail` is given, only chapters whose
    artifact is at that grain count."""
    if translated or not langs:
        lang_filter = ""
        params: tuple = (series_id,)
    else:
        placeholders = ", ".join("?" for _ in langs)
        lang_filter = f" AND c.lang IN ({placeholders})"
        params = (series_id,) + tuple(langs)
    translated_join = (
        " JOIN translations t ON t.chapter_id = c.id" if translated else ""
    )
    detail_filter = ""
    if detail is not None:
        detail_filter = " AND n.detail = ?"
        params = params + (detail,)
    return conn.execute(
        f"SELECT c.* FROM chapters c JOIN {table} n ON n.chapter_id = c.id"
        + translated_join +
        " LEFT JOIN videos v ON v.series_id = c.series_id"
        "  AND v.from_chapter = c.chapter_num"
        "  AND v.to_chapter = c.chapter_num"
        " WHERE c.series_id = ?"
        + lang_filter + detail_filter +
        " GROUP BY c.id"
        " HAVING COUNT(v.id) = 0 OR COUNT(v.id) = SUM(v.wiped_at IS NOT NULL)"
        " ORDER BY c.chapter_num ASC",
        params,
    ).fetchall()


def _run_chapter_pipeline(
    series: str,
    all_chapters: bool,
    chapter: float | None,
    chapters: str | None,
    max_chapters: int,
    video: bool,
    translated: bool,
    thinking: str | None,
    voice: str | None,
    tts_engine: str | None,
    colorize: bool,
    compress: str | None,
    video_mode: str | None,
    panel_first: bool | None,
    vision_model: str | None,
    text_model: str | None,
    backend: str | None,
    quant: str | None,
    dry_run: bool,
    skip_preflight: bool,
    *,
    detail: str | None,  # --detail flag; None = [pipeline] detail config default
    instruction: str | None = None,  # --instruction flag; None = [pipeline]
                                     # instructions config default
) -> None:
    """Body of the `eh recap` command (the single chapter pipeline)."""
    if chapters is not None:
        if chapter is not None or all_chapters:
            err_console.print(
                "[red]--chapters cannot be combined with --chapter or"
                " --all[/red]"
            )
            raise typer.Exit(code=1)
        try:
            library.parse_chapter_spec(chapters)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    _validate_pipeline_opts(all_chapters, chapter, compress, video_mode,
                            max_chapters)
    config = _pipeline_config(thinking, vision_model, text_model, backend, quant,
                              detail=detail, instruction=instruction)
    # Resolve @file here rather than in _pipeline_config: only recap consumes
    # instructions, so an unreadable @path must not break the other commands
    # that load pipeline config.
    config.pipeline.instructions = load_instruction(config.pipeline.instructions)
    # Tri-state flag: no --panel-first/--no-panel-first → [video] panel_first=
    # (on by default). Resolve once here so the callback's kind derivation and
    # build_video both see the effective value.
    panel_first = config.video.panel_first if panel_first is None else panel_first
    profile = probe(config.hardware.budget_gb)

    from entertainment_harness.pipelines.recap import (
        DETAIL_LEVELS,
        recap_series,
        select_chapters,
    )
    from entertainment_harness.ui import PipelineUI
    from entertainment_harness.video.script import VideoError

    if config.pipeline.detail not in DETAIL_LEVELS:
        # Can only come from config.toml here — a --detail flag value was
        # already validated in _pipeline_config.
        err_console.print(
            f"[red][pipeline] detail must be one of:"
            f" {', '.join(DETAIL_LEVELS)}[/red]"
        )
        raise typer.Exit(code=1)
    detail = config.pipeline.detail  # resolved grain: CLI flag > config default

    with db.connect() as conn, PipelineUI(console) as ui:
        try:
            row = library.resolve_series(conn, series)
            selected = select_chapters(
                conn, row, config, detail=detail, chapter_num=chapter,
                chapters_spec=chapters, all_chapters=all_chapters,
                max_chapters=max_chapters,
                translated=translated, fill_gaps=video, verb="recap", log=ui.log,
            )
            if selected:
                plan = plan_for_recap(
                    config, profile, row, selected,
                    video=video, translated=translated,
                )
                _run_preflight_check(
                    plan, config, dry_run=dry_run,
                    skip_preflight=skip_preflight,
                )
            make_video = _make_chapter_video(
                conn, row, config, profile, ui,
                voice=voice, tts_engine=tts_engine, colorize=colorize,
                translated=translated, compress=compress, video_mode=video_mode,
                panel_first=panel_first,
            )
            recap_series(
                conn, row, config, profile, all_chapters=all_chapters,
                chapter_num=chapter, chapters_spec=chapters,
                max_chapters=max_chapters,
                translated=translated,
                thinking=config.pipeline.thinking,
                detail=detail,
                instruction=config.pipeline.instructions,
                fill_gaps=video,
                on_recap=make_video if video else None,
                progress=ui, log=ui.log,
            )

            if video:
                # Backfill: artifacted chapters lacking any video (one per
                # chapter; the kind follows each artifact's detail grain).
                missing = _chapters_missing_video(
                    conn, row["id"], table="recaps",
                    translated=translated, langs=config.library.langs,
                )
                for chapter_row in missing:
                    ui.log(
                        f"Backfilling video for recapped chapter"
                        f" {chapter_row['chapter_num']:g}..."
                    )
                    make_video(chapter_row["id"])
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        except (VideoError, ModelError, PluginError) as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1) from exc


@app.command()
def recap(
    series: str,
    all_chapters: bool = typer.Option(
        False, "--all", help="re-recap EVERY synced chapter in order,"
        " overwriting existing recaps (context/progress unchanged)"
    ),
    chapter: float | None = typer.Option(
        None, "--chapter", "-c",
        help="force-(re)recap one chapter; context and progress unchanged",
    ),
    chapters: str = typer.Option(
        None, "--chapters",
        help="force-(re)recap exactly these synced chapters, e.g."
        " '1-3,4,6-10' (side chapters like 9.5 match by containment);"
        " context and progress unchanged; chapters behind the read"
        " frontier with no artifact are generated standalone (story-so-far"
        " withheld); cannot be combined with --chapter or --all",
    ),
    max_chapters: int = typer.Option(
        500, "--max-chapters", help="cap how many chapters one run processes"
    ),
    video: bool = typer.Option(
        False, "--video",
        help="render each chapter's video right after its recap, then backfill"
        " videos for recapped chapters that lack one; chapters behind the"
        " read frontier that have no artifact and no video are gap-filled:"
        " narrated standalone (story-so-far withheld) so their videos can"
        " be built",
    ),
    translated: bool = typer.Option(
        False, "--translated",
        help="translate each chapter into the configured language first"
        " (any source language eligible), then recap the translated pages",
    ),
    thinking: str | None = typer.Option(
        None, "--thinking",
        help="critiquing depth: low (no judge), medium (default),"
        " high (vision-verified judging, more attempts)",
    ),
    detail: str | None = typer.Option(
        None, "--detail",
        help="artifact detail: gist (2-4 sentences), brief (one paragraph),"
        " standard (default), detailed (every notable beat),"
        " full (complete in-order retelling)"
        " (default: [pipeline] detail=)",
    ),
    instruction: str | None = typer.Option(
        None, "--instruction", "-i",
        help="steering direction for the artifact (content to skip, voice,"
        " emphasis); @file reads it from a file; recorded on the artifact,"
        " never invalidates it (default: [pipeline] instructions=)",
    ),
    voice: str | None = typer.Option(None, "--voice", help="TTS voice (with --video)"),
    tts_engine: str | None = typer.Option(
        None, "--tts-engine", help="kokoro, say, or qwen3 (voice cloning; with --video)"
    ),
    colorize: bool = typer.Option(
        False, "--colorize",
        help="colorize B&W pages with DDColor (with --video, opt-in)",
    ),
    compress: str | None = typer.Option(
        None, "--compress",
        help="compress output video to a quality preset:"
        " hd, balanced, small (default: [video] compress=)",
    ),
    video_mode: str | None = typer.Option(
        None, "--video-mode",
        help="video presentation: kenburns (default), scroll, slideshow,"
        " cards (caption cards; the only mode for books), panels, motion,"
        " animate (AI-generated panel frames; uses [frames] provider),"
        " sequence (panel outpainted to video size, AI frames chained into"
        " motion; uses [sequence] provider)",
    ),
    panel_first: bool | None = typer.Option(
        None, "--panel-first/--no-panel-first",
        help="script the video from the chapter's cached panel beats"
        " (with --video; segments born with their panel spans, no page"
        " assignment or grounding judge; forces the narration kind)."
        " ON BY DEFAULT; --no-panel-first uses the grounded"
        " assign+grounding path instead (default: [video] panel_first=)",
    ),
    vision_model: str | None = typer.Option(None, "--vision-model"),
    text_model: str | None = typer.Option(None, "--text-model"),
    backend: str | None = typer.Option(
        None, "--backend", help="ollama, huggingface, or openai_compat"
    ),
    quant: str | None = typer.Option(None, "--quant"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="print the computed resource plan and exit without running",
    ),
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight",
        help="bypass the pre-flight resource guard",
    ),
) -> None:
    """Recap all pending chapters at the configured detail grain (default),
    EVERY synced chapter with --all, or one chapter with --chapter.

    Single pipeline command: --translated runs the translation stage first,
    --video renders the chapter's video after each artifact. --detail picks
    the artifact grain (gist < brief < standard < detailed < full); chapters
    with a lower-grain artifact are upgraded in place.
    """
    _run_chapter_pipeline(
        series, all_chapters, chapter, chapters, max_chapters, video,
        translated, thinking,
        voice, tts_engine, colorize, compress, video_mode, panel_first,
        vision_model, text_model,
        backend, quant, dry_run, skip_preflight,
        detail=detail, instruction=instruction,
    )


import entertainment_harness.cli.library as _library_commands  # noqa: E402, F401
from . import models as _models_commands  # noqa: E402, F401
from . import plugins as _plugins_commands  # noqa: E402, F401
from . import serve as _serve_commands  # noqa: E402, F401
from . import sources as _sources_commands  # noqa: E402, F401
from . import video as _video_commands  # noqa: E402, F401
from . import voices as _voices_commands  # noqa: E402, F401

# Importing the cli.library submodule above rebinds this package's `library`
# attribute; restore the root library package for the commands defined here.
from entertainment_harness import library  # noqa: E402, F401
