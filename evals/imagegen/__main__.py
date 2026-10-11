"""`uv run python -m evals.imagegen` — the frame-gen eval harness CLI.

Commands: run an experiment TOML into a self-contained run dir, list the
experiments with estimated generation counts (a cost preview), re-render
a run dir's gallery offline. See evals/README.md.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from evals.imagegen.experiment import (
    ExperimentError,
    estimated_generations,
    load_experiment,
    load_experiments,
)

app = typer.Typer(
    help="Frame-gen evals: TOML-declared experiments over the"
         " animated-frame image-generation seams. See evals/README.md.",
)
console = Console()
err_console = Console(stderr=True)

DEFAULT_EXPERIMENTS_DIR = Path("evals/experiments")


@app.command()
def run(
    experiment: Path = typer.Argument(..., help="path to an experiment TOML"),
    out: Path = typer.Option(
        Path("evals/runs"), "--out", "-o", help="parent dir for the run dir"
    ),
    judge: bool | None = typer.Option(
        None, "--judge/--no-judge",
        help="override the experiment's judge flag",
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model",
        help="vision model id on the openai_compat backend (e.g."
             " openai/gpt-6-luna) — no config edit, no ollama",
    ),
    clips: bool = typer.Option(
        False, "--clips", help="build per-arm preview MP4s (needs ffmpeg)"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="run only the first N arm×panel pairs"
    ),
    gen_timeout: float = typer.Option(
        900.0, "--gen-timeout",
        help="hard deadline per panel generation call, in seconds"
             " (0 disables; a call is up to max_frames images)",
    ),
) -> None:
    """Run an experiment: frames + manifest.json + index.html in a new
    run dir under --out. A failing arm is recorded, never fatal."""
    from evals.imagegen.runner import run_experiment

    if limit is not None and limit < 1:
        err_console.print("[red]--limit must be at least 1[/red]")
        raise typer.Exit(code=1)
    if gen_timeout < 0:
        err_console.print("[red]--gen-timeout must be >= 0[/red]")
        raise typer.Exit(code=1)
    try:
        exp = load_experiment(experiment)
    except ExperimentError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    run_dir = run_experiment(
        exp, out=out, judge=judge, judge_model=judge_model, clips=clips,
        limit=limit, gen_timeout=gen_timeout or None, log=console.print,
    )
    console.print(f"[bold green]{run_dir}[/bold green]")
    gallery = run_dir / "index.html"
    if gallery.is_file():
        console.print(f"[bold green]{gallery}[/bold green]")


@app.command(name="list")
def list_cmd(
    dir: Path = typer.Option(
        DEFAULT_EXPERIMENTS_DIR, "--dir", help="experiments directory"
    ),
) -> None:
    """Experiments with estimated generation counts (a cost preview)."""
    try:
        experiments = load_experiments(dir)
    except ExperimentError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not experiments:
        err_console.print(f"[red]no experiment TOMLs in {dir}[/red]")
        raise typer.Exit(code=1)
    table = Table("experiment", "mode", "panels", "arms", "est. generations")
    for exp in experiments:
        table.add_row(
            exp.name, exp.mode, str(len(exp.panels)), str(len(exp.arms)),
            str(estimated_generations(exp)),
        )
    console.print(table)
    console.print(
        "[dim]estimates = frame_count(duration, knobs) × panels, summed"
        " over arms; a frame-animation critic can double its arm at worst[/dim]"
    )


@app.command()
def render(
    run_dir: Path = typer.Argument(
        ..., help="an existing run dir with manifest.json"
    ),
    clips: bool = typer.Option(
        False, "--clips", help="(re)build missing preview MP4s too"
    ),
) -> None:
    """Re-render index.html from a run dir's manifest — offline."""
    from evals.imagegen.gallery import render_gallery

    if not (run_dir / "manifest.json").is_file():
        err_console.print(f"[red]{run_dir}: no manifest.json[/red]")
        raise typer.Exit(code=1)
    out = render_gallery(run_dir, clips=clips, log=console.print)
    console.print(f"[bold green]{out}[/bold green]")


if __name__ == "__main__":
    app()
