"""Library commands: search, chapters, add, remove, sync, list, rebuild-index, show, import."""

from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from entertainment_harness import db, library
from entertainment_harness.cli import app, console, err_console
from entertainment_harness.config import data_dir, load_config
from entertainment_harness.library import index

@app.command()
def search(
    title: str,
    source: str = typer.Option("mangadex", "--source", "-s", help="mangadex or weebcentral"),
) -> None:
    """Search a source for a series by title."""
    from entertainment_harness.sources import get_client

    try:
        client = get_client(source)
    except ValueError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    results = client.search(title)
    if not results:
        console.print("No results.")
        raise typer.Exit(code=1)

    table = Table(title=f"{source} results for {title!r}")
    table.add_column("id", style="bold")
    table.add_column("title")
    table.add_column("year")
    table.add_column("status")
    for s in results:
        table.add_row(s.id, s.title, str(s.year or "?"), s.status or "?")
    console.print(table)


@app.command()
def chapters(
    manga_id: str,
    lang: str | None = typer.Option(None, "--lang", help="translatedLanguage filter"),
    source: str = typer.Option("mangadex", "--source", "-s", help="mangadex or weebcentral"),
) -> None:
    """List hosted chapters for a series."""
    from entertainment_harness.sources import get_client

    try:
        client = get_client(source)
    except ValueError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    result = client.chapters(manga_id, langs=[lang] if lang else None)
    if not result:
        console.print("No chapters found.")
        raise typer.Exit(code=1)

    table = Table(title=f"Chapters of {manga_id}" + (f" [{lang}]" if lang else ""))
    table.add_column("chapter")
    table.add_column("lang")
    table.add_column("pages", justify="right")
    table.add_column("title")
    for c in result:
        table.add_row(c.chapter or "-", c.lang, str(c.pages), c.title or "")
    console.print(table)
    console.print(f"{len(result)} chapters")


@app.command()
def add(
    query: str,
    source: str = typer.Option("mangadex", "--source", "-s", help="mangadex or weebcentral"),
) -> None:
    """Add a series to the library by id or title search."""
    with db.connect() as conn:
        try:
            series = library.add_series(conn, query, source=source)
        except (library.LibraryError, ValueError) as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1) from exc
    console.print(f"Added [bold]{series.title}[/bold] ({series.id}, {source})")


@app.command()
def remove(
    series: str,
    keep_files: bool = typer.Option(
        False, "--keep-files", help="delete DB rows only; keep downloaded"
        " pages, book text, and videos on disk"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation"),
) -> None:
    """Remove a series and its recaps/narrations/videos from the library (e.g.
    a duplicate). Store backups, if any, are left untouched."""
    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
            counts = {
                name: conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE {column} = ?",
                    (row["id"],),
                ).fetchone()["n"]
                for name, table, column in (
                    ("chapters", "chapters", "series_id"),
                    ("videos", "videos", "series_id"),
                )
            }
            recaps = conn.execute(
                "SELECT COUNT(*) AS n FROM recaps r JOIN chapters c"
                " ON r.chapter_id = c.id WHERE c.series_id = ?",
                (row["id"],),
            ).fetchone()["n"]
            narrations = conn.execute(
                "SELECT COUNT(*) AS n FROM narrations n JOIN chapters c"
                " ON n.chapter_id = c.id WHERE c.series_id = ?",
                (row["id"],),
            ).fetchone()["n"]
            label = f"{row['title']!r} ({row['id'][:8]}, {row['source']})"
            if not yes and not typer.confirm(
                f"Remove {label}: {counts['chapters']} chapters, {recaps}"
                f" recaps, {narrations} narrations, {counts['videos']} videos"
                + (", DB rows only" if keep_files else ", including local files")
                + "?"
            ):
                raise typer.Abort()
            library.remove_series(
                conn, row["id"], delete_files=not keep_files, log=console.print
            )
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc


@app.command()
def sync(
    series: str | None = typer.Argument(None),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="delete local chapters in the configured language that are no longer"
        " present on the source",
    ),
    lang: str | None = typer.Option(
        None, "--lang", "-l",
        help="sync only this source language (overrides [library] langs)",
    ),
) -> None:
    """Fetch chapter lists from the source into the DB (all series if omitted).
    By default only chapters in [library] langs are kept; use --lang to fetch a
    specific language (e.g. pt-br) when a series is not available in your
    preferred languages."""
    config = load_config()
    langs = [lang] if lang else config.library.langs
    with db.connect() as conn:
        rows = (
            [library.resolve_series(conn, series)]
            if series
            else conn.execute("SELECT * FROM series").fetchall()
        )
        if not rows:
            err_console.print("[yellow]Library is empty.[/yellow]")
            raise typer.Exit(code=1)
        for row in rows:
            try:
                count = library.sync_chapters(
                    conn, row["id"], langs=langs, force=force
                )
            except ValueError as exc:
                err_console.print(f"[red]{escape(str(exc))}[/red]")
                raise typer.Exit(code=1) from exc
            console.print(
                f"{row['title']}: {count} chapters synced [{', '.join(langs)}]"
            )


@app.command(name="list")
def list_series() -> None:
    """Show the library with read/recap status and artifact detail."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM chapters c WHERE c.series_id = s.id)"
            " AS chapter_count,"
            " (SELECT COUNT(*) FROM recaps r JOIN chapters c ON r.chapter_id = c.id"
            "  WHERE c.series_id = s.id) AS recap_count,"
            " (SELECT last_read_chapter FROM progress p WHERE p.series_id = s.id)"
            " AS last_read"
            " FROM series s"
        ).fetchall()
        detail_counts = conn.execute(
            "SELECT c.series_id AS sid, r.detail AS detail, COUNT(*) AS n"
            " FROM recaps r JOIN chapters c ON r.chapter_id = c.id"
            " GROUP BY c.series_id, r.detail"
        ).fetchall()
    if not rows:
        console.print("Library is empty.")
        return

    from entertainment_harness.pipelines.recap import DETAIL_LEVELS

    grains: dict[str, dict[str, int]] = {}
    for d in detail_counts:
        grains.setdefault(d["sid"], {})[d["detail"]] = d["n"]

    def breakdown(series_id: str) -> str:
        per_grain = grains.get(series_id, {})
        return ", ".join(
            f"{grain} {per_grain[grain]}"
            for grain in DETAIL_LEVELS
            if per_grain.get(grain)
        ) or "-"

    table = Table(title="Library")
    table.add_column("id")
    table.add_column("title", style="bold")
    table.add_column("status")
    table.add_column("chapters", justify="right")
    table.add_column("read", justify="right")
    table.add_column("recaps", justify="right")
    table.add_column("detail")
    for r in rows:
        table.add_row(
            r["id"][:8],
            r["title"],
            r["status"] or "?",
            str(r["chapter_count"]),
            f"{r['last_read']:g}" if r["last_read"] is not None else "-",
            str(r["recap_count"]),
            breakdown(r["id"]),
        )
    console.print(table)


@app.command()
def rebuild_index() -> None:
    """Rebuild the SQLite index from the filesystem (data/works/)."""
    count = index.index_works(data_dir())
    console.print(f"Indexed {count} work(s) from {data_dir() / 'works'}.")


@app.command()
def show(
    series: str,
    chapter: float | None = typer.Option(None, "--chapter", "-c"),
    narration: bool = typer.Option(
        False, "--narration",
        help="deprecated: the chapter's artifact is shown at whatever detail"
        " it has (detail 'full' is the old narration)",
    ),
) -> None:
    """Print the stored chapter artifact(s) for a series: the recaps row at
    whatever detail grain it has (detail 'full' = the old narration)."""
    if narration:
        err_console.print(
            "[yellow]Note: '--narration' is deprecated — 'eh show' prints the"
            " chapter's artifact at its current detail (full = the old"
            " narration).[/yellow]"
        )
    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        query = (
            "SELECT c.chapter_num, r.summary AS body, r.detail, r.model,"
            " r.created_at, r.instruction, r.standalone FROM recaps r"
            " JOIN chapters c ON r.chapter_id = c.id WHERE c.series_id = ?"
        )
        params: tuple = (row["id"],)
        if chapter is not None:
            query += " AND c.chapter_num = ?"
            params = (row["id"], chapter)
        rows = conn.execute(query + " ORDER BY c.chapter_num", params).fetchall()
    if not rows:
        console.print("No recaps yet.")
        raise typer.Exit(code=1)
    for r in rows:
        header = (
            f"[bold]Chapter {r['chapter_num']:g}[/bold] "
            f"[dim]({r['detail']}, {r['model']}, {r['created_at']})[/dim]"
        )
        instruction = r["instruction"].strip()
        if instruction:
            if len(instruction) > 60:
                instruction = instruction[:57].rstrip() + "..."
            header += f" [dim]— instruction: \"{escape(instruction)}\"[/dim]"
        if r["standalone"]:
            header += (
                " [dim]— standalone (generated without story-so-far"
                " context)[/dim]"
            )
        console.print(header)
        console.print(r["body"])
        console.print()


@app.command(name="cast")
def cast_cmd(
    series: str,
    rebuild: bool = typer.Option(
        False, "--rebuild",
        help="re-extract the registry from every stored chapter artifact,"
        " in chapter order (backfill for works with history)",
    ),
    thinking: str = typer.Option(
        "medium", "--thinking",
        help="critiquing depth for the rebuild's cast judge: low | medium | high",
    ),
) -> None:
    """Show a work's character registry — the names later chapters'
    recaps/narrations are allowed to use (character bible). The registry
    fills automatically as chapters are recapped."""
    import json as _json

    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        if rebuild:
            _rebuild_cast(conn, row, thinking)
            return
        rows = db.get_characters(conn, row["id"])
    if not rows:
        console.print(
            "No characters registered yet — the registry fills as chapters"
            " are recapped (or use --rebuild for existing recaps)."
        )
        raise typer.Exit(code=1)
    table = Table(title=f"Cast of {row['title']!r}")
    table.add_column("name", style="bold")
    table.add_column("aliases")
    table.add_column("role")
    table.add_column("seen")
    for r in rows:
        aliases = ", ".join(_json.loads(r["aliases"]))
        seen = f"ch {r['first_seen']:g}"
        if r["last_seen"] != r["first_seen"]:
            seen += f"–{r['last_seen']:g}"
        if r["edited"]:
            seen += " (edited)"
        table.add_row(r["name"], aliases, r["role"], seen)
    console.print(table)


def _rebuild_cast(conn, row, thinking: str) -> None:
    """`eh cast --rebuild`: re-extract the registry from every stored
    chapter artifact, oldest-first. The fold itself lives in
    characters.rebuild_cast, shared with the server's cast-build jobs."""
    from entertainment_harness.hardware import probe
    from entertainment_harness.pipelines.characters import rebuild_cast

    config = load_config()
    profile = probe(config.hardware.budget_gb)
    try:
        count = rebuild_cast(
            conn, row, config, profile, thinking=thinking, log=console.print
        )
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Registry rebuilt: {count} character(s).")


@app.command(name="import")
def import_cmd(
    source: str = typer.Argument(
        ..., help="Local file/folder or direct-file URL (EPUB/TXT/MD book,"
        " CBZ/CBR/image-folder comic)"
    ),
    title: str | None = typer.Option(None, "--title", "-t"),
    kind: str | None = typer.Option(
        None, "--kind", "-k", help="book or comic; default: detect from extension"
    ),
) -> None:
    """Import a book or comic into the library from a file or URL."""
    from entertainment_harness.library.importer import ImportFailure, import_work

    config = load_config()
    with db.connect() as conn:
        try:
            series = import_work(
                conn, config, source, title=title, kind=kind, log=console.print
            )
        except ImportFailure as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print(
        f"Imported [bold]{series['title']}[/bold] ({series['id'][:8]},"
        f" {series['kind']}). Next: eh recap {series['id'][:8]} --all"
    )
