"""`eh plugins` and `eh plugin check` commands."""

from __future__ import annotations

import typer
from rich.table import Table

from entertainment_harness.cli import app, console, err_console, plugin_app
from entertainment_harness.config import load_config


def _registries():
    from entertainment_harness.models.registry import REGISTRY as backends
    from entertainment_harness.search import REGISTRY as search_providers
    from entertainment_harness.sources import REGISTRY as sources
    from entertainment_harness.video.colorize import REGISTRY as colorizers
    from entertainment_harness.video.frames import REGISTRY as frame_animators
    from entertainment_harness.video.gen import REGISTRY as video_gens
    from entertainment_harness.video.tts import REGISTRY as tts_engines

    return (
        ("model backends", backends),
        ("sources", sources),
        ("tts engines", tts_engines),
        ("video generators", video_gens),
        ("frame animators", frame_animators),
        ("colorizers", colorizers),
        ("search providers", search_providers),
    )


@app.command()
def plugins(
    check: bool = typer.Option(
        False, "--check",
        help="health pass: load every plugin and run the conformance kit on it",
    ),
) -> None:
    """List registered plugins per category and which is active per config."""
    config = load_config()
    judge_backend = config.models.judge.backend
    active = {
        "model backends": {
            config.models.vision.backend, config.models.text.backend, judge_backend,
        },
        # sources show enabled/disabled state instead (see [sources].enabled)
        "sources": set(),
        "tts engines": {config.video.tts_engine},
        "video generators": {config.video_gen.provider},
        "frame animators": {config.frames.provider},
        "colorizers": {config.colorize.provider},
        "search providers": {config.search.provider},
    }
    failures = 0
    for label, registry in _registries():
        try:
            names = registry.names()
        except Exception as exc:  # broken entry-point discovery etc.
            err_console.print(f"[yellow]Could not list {label}: {exc}[/yellow]")
            failures += 1
            continue
        table = Table(title=label)
        table.add_column("plugin", style="bold")
        table.add_column("origin")
        table.add_column("capabilities")
        table.add_column("active")
        if check:
            table.add_column("health")
        for name in names:
            origin = (
                "[cyan]entry point[/cyan]" if registry.is_entry_point(name)
                else "built-in"
            )
            if label == "sources":
                from entertainment_harness.sources import is_source_enabled

                is_active = (
                    "[green]enabled[/green]"
                    if is_source_enabled(name, config)
                    else "[red]disabled[/red]"
                )
            else:
                is_active = "[green]yes[/green]" if name in active[label] else ""
            caps = "—"
            health: str | None = None
            try:
                cls = registry.load(name)
                declared = getattr(cls, "capabilities", frozenset())
                caps = ", ".join(sorted(declared)) or "—"
                if check:
                    from entertainment_harness.testing import check_plugin

                    problems = check_plugin(cls, registry.category)
                    if problems:
                        failures += 1
                        health = "[red]" + "; ".join(problems) + "[/red]"
                    else:
                        health = "[green]ok[/green]"
            except Exception as exc:  # broken lazy import / entry point
                if check:
                    failures += 1
                    health = f"[red]load failed: {exc}[/red]"
            row = [name, origin, caps, is_active]
            if check:
                row.append(health or "[green]ok[/green]")
            table.add_row(*row)
        console.print(table)
    if check and failures:
        err_console.print(f"[red]{failures} plugin problem(s) found[/red]")
        raise typer.Exit(code=1)


@plugin_app.command()
def check(
    target: str = typer.Argument(
        ..., help="plugin class as a dotted path: pkg.mod:Class"
    ),
    category: str = typer.Option(
        ..., "--category", "-c",
        help="plugin category: model_backend, sources, tts, video_gen,"
        " frames, colorize, search",
    ),
) -> None:
    """Conformance-check a plugin class (the same kit `eh plugins --check` runs)."""
    import importlib

    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        err_console.print(
            f"[red]Target must be a dotted path like pkg.mod:Class, got {target!r}[/red]"
        )
        raise typer.Exit(code=1)
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        err_console.print(f"[red]Cannot import {target!r}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    from entertainment_harness.testing import check_plugin

    problems = check_plugin(cls, category)
    if not problems:
        console.print(f"[green]{target} conforms to the {category} contract[/green]")
        return
    err_console.print(f"[red]{target} does not conform:[/red]")
    for problem in problems:
        err_console.print(f"  - {problem}")
    raise typer.Exit(code=1)
