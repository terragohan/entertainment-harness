"""Video commands: tiktok, online-summary, play, compress, anime-scene."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import typer

from entertainment_harness import db, library
from entertainment_harness.cli import (
    _pipeline_config,
    _run_preflight_check,
    app,
    console,
    err_console,
)
from entertainment_harness.config import load_config
from entertainment_harness.hardware import probe
from entertainment_harness.library import works
from entertainment_harness.models.base import ModelError
from entertainment_harness.plugins import PluginError
from entertainment_harness.preflight import PreflightError, plan_for_tiktok

@app.command()
def anime_scene(
    series: str,
    chapter: float = typer.Option(..., "--chapter", "-c"),
    pages: str = typer.Option(
        ..., "--pages", "-p", help="contiguous source pages, e.g. '8-10'"
    ),
    instruction: str = typer.Option(
        ..., "--instruction", "-i",
        help="free-form direction for the adaptation (tone, emphasis, limits)",
    ),
    stop_after: str | None = typer.Option(
        None, "--stop-after", help="plan | keyframes: stop before paid stages"
    ),
    regenerate_shot: list[int] | None = typer.Option(
        None, "--regenerate-shot",
        help="force-regenerate one shot's keyframe+clip (repeatable)",
    ),
    vision_model: str | None = typer.Option(None, "--vision-model"),
    backend: str | None = typer.Option(
        None, "--backend", help="planner backend: ollama, huggingface, openai_compat"
    ),
) -> None:
    """Adapt one contiguous manga scene into a <=30s anime scene: guided
    storyboard (scene.json, human-editable) -> Runway gen4_image keyframes ->
    3-5 generated 5s shots -> one 16:9 mp4. Cached per stage; paid Runway
    calls are never made for cached artifacts. See
    docs/initiatives/manga-to-anime-scene/."""
    from entertainment_harness.anime.pipeline import build_scene, parse_page_spec
    from entertainment_harness.anime.scene import SceneError

    config = _pipeline_config(None, vision_model, None, backend, None)
    profile = probe(config.hardware.budget_gb)

    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
            chapter_row = conn.execute(
                "SELECT * FROM chapters WHERE series_id = ? AND chapter_num = ?",
                (row["id"], chapter),
            ).fetchone()
            if chapter_row is None:
                raise library.LibraryError(
                    f"{row['title']}: chapter {chapter:g} is not in the library"
                    " — run 'eh sync' first."
                )
            manga_dir = works.source_dir(row["id"], chapter_row["id"])
            image_files = sorted(
                p for p in manga_dir.iterdir()
                if p.is_file() and p.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp"}
            ) if manga_dir.is_dir() else []
            if not image_files:
                raise SceneError(
                    f"Chapter {chapter:g} pages are not cached"
                    " — run 'eh recap' first."
                )
            selected = parse_page_spec(pages, len(image_files))
            page_paths = {n: image_files[n - 1] for n in selected}
            out = build_scene(
                row["id"], chapter_row["id"], chapter, row["title"],
                page_paths, config, profile, instruction,
                stop_after=stop_after,
                regenerate_shots=set(regenerate_shot or []),
                log=console.print,
            )
        except (library.LibraryError, SceneError, ModelError, PluginError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print(f"[bold green]{out}[/bold green]")


@app.command()
def tiktok(
    title: str,
    voice: str | None = typer.Option(None, "--voice"),
    tts_engine: str | None = typer.Option(
        None, "--tts-engine", help="kokoro, say, or qwen3 (voice cloning)"
    ),
    vision_model: str | None = typer.Option(None, "--vision-model"),
    text_model: str | None = typer.Option(None, "--text-model"),
    backend: str | None = typer.Option(
        None, "--backend", help="ollama, huggingface, or openai_compat"
    ),
    quant: str | None = typer.Option(None, "--quant"),
    video_gen: str | None = typer.Option(None, "--video-gen", help="local or runway"),
    steering_prompt: str | None = typer.Option(None, "--steering-prompt", "-p"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="print the computed resource plan and exit without running",
    ),
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight",
        help="bypass the pre-flight resource guard",
    ),
) -> None:
    """Render a whole-work vertical short-form (1080x1920, ~60-90s) summary
    video from a title. Searches the web for summaries, then builds the video
    with generated caption cards. With --video-gen runway, generates clips via
    Runway ML instead of local ffmpeg/Ken-Burns."""
    import hashlib

    from entertainment_harness.db import utcnow
    from entertainment_harness.models.registry import get_text_model
    from entertainment_harness.search.pipeline import summarize_and_store

    config = _pipeline_config(None, vision_model, text_model, backend, quant)
    profile = probe(config.hardware.budget_gb)

    plan = plan_for_tiktok(config, profile)
    _run_preflight_check(plan, config, dry_run=dry_run, skip_preflight=skip_preflight)

    from entertainment_harness.video.script import VideoError
    from entertainment_harness.video.short import build_short

    series_id = "search-" + hashlib.sha256(title.encode()).hexdigest()[:16]

    with db.connect() as conn:
        try:
            row = conn.execute(
                "SELECT * FROM series WHERE id = ?", (series_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO series (id, title, alt_titles, source, source_id,"
                    " status, added_at, kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (series_id, title, "", "search", title, None, utcnow(), "book"),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM series WHERE id = ?", (series_id,)
                ).fetchone()

            text = get_text_model(config, profile)
            summarize_and_store(
                conn, row, config, text.adapter, text.info.name,
                steering_prompt=steering_prompt or "",
                log=console.print,
            )
            out = build_short(
                conn, row, config, profile,
                voice=voice, engine_name=tts_engine,
                video_gen_provider=video_gen,
                steering_prompt=steering_prompt,
                log=console.print,
            )
        except (library.LibraryError, VideoError, ModelError, PluginError,
                RuntimeError, PreflightError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print(f"[bold green]{out}[/bold green]")


@app.command()
def online_summary(
    series: str,
    provider: str | None = typer.Option(None, "--provider", help="duckduckgo"),
    steering_prompt: str | None = typer.Option(None, "--steering-prompt", "-p"),
    text_model: str | None = typer.Option(None, "--text-model"),
    backend: str | None = typer.Option(
        None, "--backend", help="ollama, huggingface, or openai_compat"
    ),
) -> None:
    """Fetch online summaries (blogs, Reddit, YouTube transcripts) for a series
    and store a synthesized rolling summary for use with 'eh tiktok <title>'."""
    config = _pipeline_config(None, None, text_model, backend, None)
    if provider:
        config.search.provider = provider
    profile = probe(config.hardware.budget_gb)

    from entertainment_harness.models.registry import get_text_model
    from entertainment_harness.search.pipeline import summarize_and_store
    from entertainment_harness.video.script import VideoError

    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
            text = get_text_model(config, profile)
            summarize_and_store(
                conn, row, config, text.adapter, text.info.name,
                steering_prompt=steering_prompt or "",
                log=console.print,
            )
        except (library.LibraryError, VideoError, PluginError, RuntimeError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print("[bold green]Online summary stored.[/bold green]")


@app.command()
def play(
    series: str,
    chapter: float | None = typer.Option(
        None, "--chapter", "-c", help="default: the latest rendered video"
    ),
    short: bool = typer.Option(
        False, "--tiktok", help="play the whole-work short-form video"
    ),
    narration: bool = typer.Option(
        False, "--narration",
        help="deprecated: the chapter's video is found regardless of kind;"
        " with this flag only narration-kind videos are considered",
    ),
) -> None:
    """Open a chapter's rendered video in the system player (one video per
    chapter; the narration kind wins ties while both kinds exist). If the
    file was cleaned locally, it is pulled back from the configured store
    first."""
    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        series_id = row["id"]
        if short:
            video = conn.execute(
                "SELECT * FROM videos WHERE series_id = ?"
                " AND from_chapter IS NULL AND to_chapter IS NULL",
                (series_id,),
            ).fetchone()
        else:
            # One video per chapter regardless of kind; the narration kind
            # (detail 'full') wins when a chapter has both.
            query = (
                "SELECT * FROM videos WHERE series_id = ?"
                " AND from_chapter IS NOT NULL"
            )
            params: tuple = (series_id,)
            if narration:
                query += " AND kind = 'narration'"
            if chapter is not None:
                query += " AND from_chapter = ? AND to_chapter = ?"
                params = params + (chapter, chapter)
            video = conn.execute(
                query + " ORDER BY from_chapter DESC, (kind = 'narration') DESC",
                params,
            ).fetchone()

        if video is None:
            msg = "No video yet"
            if chapter is not None:
                msg += f" for chapter {chapter:g}"
            if narration:
                # With the kind restriction on, a recap-kind video may exist.
                recap_query = (
                    "SELECT 1 FROM videos WHERE series_id = ?"
                    " AND kind = 'recap' AND from_chapter IS NOT NULL"
                )
                recap_params: tuple = (series_id,)
                if chapter is not None:
                    recap_query += " AND from_chapter = ? AND to_chapter = ?"
                    recap_params = (series_id, chapter, chapter)
                if conn.execute(recap_query, recap_params).fetchone():
                    msg += " — only a recap video exists; drop --narration to play it."
                else:
                    msg += " — run 'eh recap --detail full --video' first."
            else:
                msg += " — run 'eh recap --video' first."
            err_console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=1)

        # Resolve the authoritative local path from the works layout.
        video_kind = "tiktok" if short else video["kind"]
        if short:
            chapter_id = None
        else:
            ch = conn.execute(
                "SELECT id FROM chapters WHERE series_id = ? AND chapter_num = ?",
                (series_id, video["from_chapter"]),
            ).fetchone()
            if ch is None:
                err_console.print(
                    f"[red]Chapter {video['from_chapter']:g} is not in the library.[/red]"
                )
                raise typer.Exit(code=1)
            chapter_id = ch["id"]

        def _works_video_path(series_id: str, chapter_id: str | None, kind: str) -> Path:
            meta = works.read_video_metadata(series_id, chapter_id, kind)
            suffix = "out.mp4"
            if meta is not None and meta.compress:
                suffix = f"out-{meta.compress}.mp4"
            return works.video_file_path(series_id, chapter_id, kind, suffix=suffix)

        path = _works_video_path(series_id, chapter_id, video_kind)

        if video["wiped_at"] and not path.exists():
            render_cmd = (
                "eh recap --detail full --video"
                if video_kind == "narration" else "eh recap --video"
            )
            err_console.print(
                f"[red]This video was wiped on {video['wiped_at'][:10]} —"
                f" re-render it with '{render_cmd}"
                + (f" --chapter {video['from_chapter']:g}'." if chapter_id else "'.")
                + "[/red]"
            )
            raise typer.Exit(code=1)

        if not path.exists():
            # The configured store is the default backing: pull the video back.
            from entertainment_harness import store

            config = load_config()
            if store.is_configured(config):
                chapter_key = "tiktok" if short else chapter_id
                if chapter_key is not None:
                    try:
                        store.pull_video(config, series_id, chapter_key)
                    except store.StoreError as exc:
                        err_console.print(f"[yellow]{exc}[/yellow]")
                    if path.exists():
                        console.print(f"Restored from the store: {path}")
    if not path.exists():
        err_console.print(f"[red]Video file is missing: {path}[/red]")
        raise typer.Exit(code=1)

    opener = "open" if platform.system() == "Darwin" else "xdg-open"
    label = (
        "tiktok"
        if short
        else f"chapter {video['from_chapter']:g}"
    )
    console.print(f"Playing {label} ({video['duration_s']:.0f}s): {path}")
    subprocess.run([opener, str(path)], check=True)


@app.command()
def concat(
    series: str,
    kind: str = typer.Option(
        "narration", "--kind", "-k", help="narration or recap"
    ),
    preset: str | None = typer.Option(
        None, "--preset",
        help="use out-<preset>.mp4 copies when present (e.g. balanced);"
        " default: any compressed copy, else the master",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="destination mp4; default: works/<slug>/<kind>-full.mp4"
    ),
) -> None:
    """Concatenate a series' chapter videos into one long mp4, in chapter order.

    When every selected chapter has identical codec/resolution/fps (always
    true for videos rendered with the same config) the concat is lossless
    (stream copy, no re-encode). Mixed formats fall back to a balanced
    re-encode of the whole run."""
    import json
    import subprocess
    import tempfile

    if kind not in ("narration", "recap"):
        err_console.print("[red]--kind must be 'narration' or 'recap'[/red]")
        raise typer.Exit(code=1)

    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        series_id = row["id"]

        dir_name = (
            works.VIDEO_NARRATION_DIR if kind == "narration" else works.VIDEO_RECAP_DIR
        )
        parts: list[Path] = []
        wiped = 0
        for cdir in works.list_chapter_dirs(series_id):
            cmeta = works.read_chapter_metadata_at(cdir)
            if cmeta is None:
                continue
            workdir = cdir / dir_name
            meta = works.read_video_metadata(series_id, cmeta.id, kind)
            if meta is not None and meta.wiped_at:
                wiped += 1
                continue
            candidates = []
            if preset:
                candidates.append(workdir / f"out-{preset}.mp4")
            elif meta is not None and meta.compress:
                candidates.append(workdir / f"out-{meta.compress}.mp4")
            candidates.append(workdir / "out.mp4")
            part = next((c for c in candidates if c.exists()), None)
            if part is not None:
                parts.append(part)

    if len(parts) < 2:
        err_console.print(
            f"[red]Need at least 2 {kind} videos to concatenate"
            f" (found {len(parts)}).[/red]"
        )
        raise typer.Exit(code=1)

    dest = out or works.work_dir(series_id) / f"{kind}-full.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"Concatenating {len(parts)} {kind} videos -> {dest}")
    if wiped:
        console.print(f"[dim]skipping {wiped} wiped chapter(s)[/dim]")
    from entertainment_harness.video.assemble import concat_mp4s
    from entertainment_harness.video.script import VideoError

    try:
        concat_mp4s(parts, dest, log=console.print)
    except VideoError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    hours = ""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(dest)],
            capture_output=True, text=True,
        )
        secs = float(r.stdout.strip())
        hours = f" ({secs / 3600:.1f} h)"
    except (ValueError, IndexError):
        pass
    console.print(
        f"[bold green]Wrote {dest} — {len(parts)} chapters{hours},"
        f" {dest.stat().st_size / 1e9:.2f} GB[/bold green]"
    )


@app.command()
def compress(
    series: str | None = typer.Argument(None, help="series (id/title); default: all"),
    chapter: float | None = typer.Option(
        None, "--chapter", "-c", help="only this chapter number"
    ),
    preset: str | None = typer.Option(
        None, "--preset",
        help="compression preset; default: [video] compress= or 'balanced'",
    ),
    prune: bool = typer.Option(
        False, "--prune",
        help="also delete clips/segment WAVs and, per keep_master, masters",
    ),
) -> None:
    """Compress rendered chapter videos that don't have an out-<preset>.mp4 yet.

    Encodes from the full-quality master when present (or from the newest
    compressed copy when the master was already pruned). Prints per-chapter
    before/after sizes and a total."""
    from entertainment_harness.video.compress import PRESETS, compress as compress_video
    from entertainment_harness.video.pipeline import prune_intermediates
    from entertainment_harness.video.script import VideoError

    config = load_config()
    preset = preset or config.video.compress or "balanced"
    if preset not in PRESETS:
        err_console.print(
            f"[red]Unknown preset {preset!r}; expected one of: {', '.join(PRESETS)}[/red]"
        )
        raise typer.Exit(code=1)

    with db.connect() as conn:
        if series is not None:
            try:
                rows = [library.resolve_series(conn, series)]
            except library.LibraryError as exc:
                err_console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1) from exc
        else:
            rows = conn.execute("SELECT * FROM series ORDER BY added_at").fetchall()

        total_before = total_after = 0
        done = skipped = 0
        for row in rows:
            for cdir in works.list_chapter_dirs(row["id"]):
                cmeta = works.read_chapter_metadata_at(cdir)
                if cmeta is None or (
                    chapter is not None and cmeta.chapter_num != chapter
                ):
                    continue
                for kind, dir_name in (
                    ("recap", works.VIDEO_RECAP_DIR),
                    ("narration", works.VIDEO_NARRATION_DIR),
                ):
                    workdir = cdir / dir_name
                    master = workdir / "out.mp4"
                    dest = workdir / f"out-{preset}.mp4"
                    if not workdir.is_dir() or (not master.exists() and not dest.exists()
                                                and not any(workdir.glob("out-*.mp4"))):
                        continue
                    if dest.exists():
                        skipped += 1
                        continue
                    copies = sorted(
                        workdir.glob("out-*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True,
                    )
                    source = master if master.exists() else copies[0]
                    try:
                        compress_video(source, preset, workdir)
                    except VideoError as exc:
                        err_console.print(f"[red]{exc}[/red]")
                        raise typer.Exit(code=1) from exc
                    before, after = source.stat().st_size, dest.stat().st_size
                    total_before += before
                    total_after += after
                    done += 1
                    console.print(
                        f"  {row['title']} ch {cmeta.chapter_num:g} {kind}:"
                        f" {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB"
                    )
                    if prune:
                        prune_intermediates(workdir)
                        if not config.video.keep_master and master.exists():
                            master.unlink()
                    meta = works.read_video_metadata(row["id"], cmeta.id, kind)
                    if meta is not None:
                        meta.compress = preset
                        works.write_video_metadata(row["id"], cmeta.id, meta)
                    conn.execute(
                        "UPDATE videos SET path = ? WHERE series_id = ?"
                        " AND from_chapter = ? AND kind = ?",
                        (str(dest), row["id"], cmeta.chapter_num, kind),
                    )
        conn.commit()

    state_note = ""
    if prune and not config.video.keep_master:
        state_note = " (masters pruned)"
    elif prune:
        state_note = " (intermediates pruned)"
    console.print(
        f"[bold green]Compressed {done} video(s), skipped {skipped} already"
        f" compressed{state_note}.[/bold green]"
    )
    if total_before:
        console.print(
            f"[bold green]Total: {total_before / 1e9:.1f} GB ->"
            f" {total_after / 1e9:.1f} GB[/bold green]"
        )


@app.command()
def wipe(
    series: str,
    chapters: str = typer.Option(
        ..., "--chapters", "-c",
        help="chapter selector, e.g. '1-3,4,6-10' (side chapters like 9.5 ok)",
    ),
    kind: str = typer.Option(
        "all", "--kind", "-k", help="narration, recap, or all (default)"
    ),
) -> None:
    """Delete rendered video files for chapters and mark them wiped.

    Wiped chapters keep their script.json/render_state.json caches, so a
    later 'eh recap --video' re-renders from cached stages (which
    also clears the wiped mark). Chapters with no artifact at all cannot be
    re-rendered — wiping their only video is irreversible — so each one
    gets an explicit warning. 'eh play' reports wiped chapters
    distinctly; 'eh concat' skips them."""
    from entertainment_harness.db import utcnow

    if kind not in ("all", "narration", "recap"):
        err_console.print("[red]--kind must be 'narration', 'recap', or 'all'[/red]")
        raise typer.Exit(code=1)
    try:
        ranges = library.parse_chapter_spec(chapters)
    except library.LibraryError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    kinds = (
        (("recap", works.VIDEO_RECAP_DIR), ("narration", works.VIDEO_NARRATION_DIR))
        if kind == "all"
        else (
            (kind, works.VIDEO_NARRATION_DIR if kind == "narration"
             else works.VIDEO_RECAP_DIR),
        )
    )

    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, series)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        series_id = row["id"]

        now = utcnow()
        wiped = empty = 0
        freed = 0
        for cdir in works.list_chapter_dirs(series_id):
            cmeta = works.read_chapter_metadata_at(cdir)
            if cmeta is None or not library.chapter_in_spec(cmeta.chapter_num, ranges):
                continue
            chapter_had = False
            for k, dir_name in kinds:
                workdir = cdir / dir_name
                if not workdir.is_dir():
                    continue
                files = sorted(workdir.glob("out*.mp4"))
                meta = works.read_video_metadata(series_id, cmeta.id, k)
                if not conn.execute(
                    "SELECT 1 FROM recaps WHERE chapter_id = ?", (cmeta.id,)
                ).fetchone() and not conn.execute(
                    "SELECT 1 FROM narrations WHERE chapter_id = ?", (cmeta.id,)
                ).fetchone():
                    # The artifact this video was rendered from is gone, so
                    # there is no cached script to re-render from — 'eh wipe'
                    # is destroying the only remaining form of the chapter's
                    # story. Gap fill can mint a fresh standalone artifact
                    # (story-so-far withheld), but the original is lost.
                    console.print(
                        f"  [yellow]{row['title']} ch {cmeta.chapter_num:g}"
                        f" {k}: no recap or narration on file — the original"
                        " artifact is gone; this video cannot be re-rendered"
                        " from cache ('eh recap --video' would replace it"
                        " with a standalone one)[/yellow]"
                    )
                if not files:
                    # Files already gone (e.g. deleted by hand) but the
                    # metadata still claims a video: mark it wiped too.
                    if meta is not None and meta.wiped_at is None:
                        chapter_had = True
                        meta.wiped_at = now
                        works.write_video_metadata(series_id, cmeta.id, meta)
                        conn.execute(
                            "UPDATE videos SET wiped_at = ? WHERE series_id = ?"
                            " AND from_chapter = ? AND kind = ?",
                            (now, series_id, cmeta.chapter_num, k),
                        )
                        wiped += 1
                        console.print(
                            f"  {row['title']} ch {cmeta.chapter_num:g} {k}:"
                            " already gone — marked wiped"
                        )
                    continue
                chapter_had = True
                size = sum(f.stat().st_size for f in files)
                for f in files:
                    f.unlink()
                meta = works.read_video_metadata(series_id, cmeta.id, k)
                if meta is not None:
                    meta.wiped_at = now
                    works.write_video_metadata(series_id, cmeta.id, meta)
                conn.execute(
                    "UPDATE videos SET wiped_at = ? WHERE series_id = ?"
                    " AND from_chapter = ? AND kind = ?",
                    (now, series_id, cmeta.chapter_num, k),
                )
                wiped += 1
                freed += size
                console.print(
                    f"  {row['title']} ch {cmeta.chapter_num:g} {k}:"
                    f" wiped {size / 1e6:.0f} MB"
                )
            if not chapter_had:
                empty += 1
        conn.commit()

    console.print(
        f"[bold green]Wiped {wiped} video(s), freed {freed / 1e9:.2f} GB.[/bold green]"
    )
    if empty:
        console.print(f"[dim]{empty} chapter(s) in range had no videos.[/dim]")
