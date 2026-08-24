"""`eh sources` commands: list, enable, disable, reset content sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import typer
from rich.table import Table

from entertainment_harness.cli import console, err_console, sources_app
from entertainment_harness.config import (
    CONFIG_FILENAME,
    DEFAULT_ENABLED_SOURCES,
    ConfigWriteError,
    data_dir,
    load_config,
    save_config,
)


def _config_path() -> Path:
    return data_dir() / CONFIG_FILENAME


def _ordered_enabled(enabled: Iterable[str]) -> list[str]:
    """Built-ins first in registry order, then extras in their given order."""
    from entertainment_harness.sources import REGISTRY

    enabled = list(enabled)
    builtins = REGISTRY.builtin_names()
    ordered = [n for n in builtins if n in enabled]
    ordered += [n for n in enabled if n not in builtins]
    return ordered


def _write_enabled(enabled: list[str]) -> None:
    try:
        save_config(_config_path(), {"sources": {"enabled": enabled}})
    except ConfigWriteError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def _require_registered(name: str) -> None:
    from entertainment_harness.sources import REGISTRY

    names = REGISTRY.names()
    if name not in names:
        err_console.print(
            f"[red]Unknown source {name!r}; available: {', '.join(names)}[/red]"
        )
        raise typer.Exit(code=1)


@sources_app.callback(invoke_without_command=True)
def sources_list(ctx: typer.Context) -> None:
    """List registered sources with their enabled/disabled state."""
    if ctx.invoked_subcommand is not None:
        return
    from entertainment_harness.sources import REGISTRY, is_source_enabled

    config = load_config()
    table = Table(title="Content sources")
    table.add_column("source", style="bold")
    table.add_column("origin")
    table.add_column("state")
    for name in REGISTRY.names():
        origin = (
            "[cyan]entry point[/cyan]" if REGISTRY.is_entry_point(name)
            else "built-in"
        )
        state = (
            "[green]enabled[/green]" if is_source_enabled(name, config)
            else "[red]disabled[/red]"
        )
        table.add_row(name, origin, state)
    console.print(table)
    console.print(f"[dim]Toggle with: eh sources enable|disable NAME"
                  f" (writes {_config_path()})[/dim]")


@sources_app.command("enable")
def sources_enable(name: str) -> None:
    """Enable a registered source (adds it to [sources].enabled)."""
    _require_registered(name)
    config = load_config()
    if name in config.sources.enabled:
        console.print(f"Source {name!r} is already enabled.")
        return
    _write_enabled(_ordered_enabled([*config.sources.enabled, name]))
    console.print(f"[bold green]Source {name!r} enabled.[/bold green]")


@sources_app.command("disable")
def sources_disable(name: str) -> None:
    """Disable a built-in source (removes it from [sources].enabled)."""
    from entertainment_harness.sources import REGISTRY

    _require_registered(name)
    if REGISTRY.is_entry_point(name):
        err_console.print(
            f"[red]{name!r} is a third-party plugin; [sources].enabled only"
            f" gates built-in sources ({', '.join(REGISTRY.builtin_names())}).[/red]"
        )
        raise typer.Exit(code=1)
    config = load_config()
    if name not in config.sources.enabled:
        console.print(f"Source {name!r} is already disabled.")
        return
    enabled = [n for n in config.sources.enabled if n != name]
    _write_enabled(_ordered_enabled(enabled))
    console.print(f"[bold yellow]Source {name!r} disabled.[/bold yellow]")


@sources_app.command("reset")
def sources_reset() -> None:
    """Restore [sources].enabled to the defaults (all built-in sources)."""
    _write_enabled(list(DEFAULT_ENABLED_SOURCES))
    console.print(
        f"[bold green]Sources reset to defaults:[/bold green]"
        f" {', '.join(DEFAULT_ENABLED_SOURCES)}"
    )
