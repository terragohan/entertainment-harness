"""`eh models` and `eh quantize` commands."""

from __future__ import annotations

import typer
from rich.table import Table

from entertainment_harness.cli import (
    _run_preflight_check,
    app,
    console,
    err_console,
    models_app,
)
from entertainment_harness.config import load_config
from entertainment_harness.hardware import probe
from entertainment_harness.models.base import ModelError
from entertainment_harness.models.registry import fit_verdict, get_adapter
from entertainment_harness.plugins import PluginError
from entertainment_harness.preflight import plan_for_quantize

@models_app.callback(invoke_without_command=True)
def models_main(
    ctx: typer.Context,
    backend: str = typer.Option(
        "ollama", "--backend", "-b", help="ollama, huggingface, or openai_compat"
    ),
) -> None:
    """Show detected hardware and locally available models with fit verdicts."""
    if ctx.invoked_subcommand is not None:
        return
    config = load_config()
    profile = probe(config.hardware.budget_gb)

    hw = Table(title="Hardware profile", show_header=False)
    hw.add_column("key", style="bold")
    hw.add_column("value")
    hw.add_row("chip", profile.chip)
    hw.add_row("total RAM", f"{profile.total_ram_gb:.1f} GB")
    hw.add_row("safe budget", f"{profile.budget_gb:.1f} GB")
    hw.add_row("GPU backend", profile.gpu_backend)
    console.print(hw)

    for role_name, role in (("vision", config.models.vision), ("text", config.models.text)):
        console.print(f"configured {role_name}: {role.model} (backend {role.backend})"
                      + (f", pinned quant {role.quant}" if role.quant else ""))

    try:
        adapter = get_adapter(backend, config)
        available = adapter.list_available()
    except Exception as exc:  # backend not running / unreachable / no API key
        err_console.print(f"[yellow]Could not query {backend}: {exc}[/yellow]")
        return

    remote = getattr(adapter, "remote", False)
    table = Table(title=f"{'Remote' if remote else 'Local'} {backend} models")
    table.add_column("name", style="bold")
    table.add_column("params")
    if not remote:
        table.add_column("quant")
        table.add_column("size")
        table.add_column("verdict")
    for info in available:
        row = [info.name, f"{info.params:g}B" if info.params else "?"]
        if not remote:
            verdict = fit_verdict(info.size_bytes, profile.budget_bytes)
            style = {"fits": "green", "tight": "yellow", "too large": "red"}.get(verdict, "")
            row += [
                info.quant or "?",
                f"{info.size_gb:.1f} GB" if info.size_gb is not None else "?",
                f"[{style}]{verdict}[/{style}]" if style else verdict,
            ]
        table.add_row(*row)
    console.print(table)


@models_app.command("rm")
def models_rm(
    model: str,
    backend: str = typer.Option(
        "ollama", "--backend", "-b", help="ollama or huggingface (remote backends have nothing to delete)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation"),
) -> None:
    """Delete the local copy of a model (re-downloadable via ensure/pull)."""
    try:
        adapter = get_adapter(backend, load_config())
    except (ModelError, PluginError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not yes and not typer.confirm(f"Delete local model {model!r} ({backend})?"):
        raise typer.Abort()
    try:
        adapter.remove(model)
    except ModelError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Removed {model} ({backend}).")


@app.command()
def quantize(
    repo: str,
    quant: str = typer.Option(..., "--quant", "-q", help="e.g. q4_k_m"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the cost confirmation"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="print the computed resource plan and exit without running",
    ),
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight",
        help="bypass the pre-flight resource guard",
    ),
) -> None:
    """Locally quantize a HF GGUF repo's FP16/BF16 source (last resort —
    published quants are always preferred)."""
    config = load_config()
    profile = probe(config.hardware.budget_gb)
    plan = plan_for_quantize(config, profile, repo, quant)
    _run_preflight_check(plan, config, dry_run=dry_run, skip_preflight=skip_preflight)

    from entertainment_harness.models.quantize import QuantizeError, quantize as run

    try:
        out = run(
            repo, quant,
            confirm=(lambda msg: True) if yes else typer.confirm,
        )
    except QuantizeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]{out}[/bold green]")
    console.print(
        f"Use it with: --backend huggingface --<role>-model {repo}:{quant.upper()}"
    )
