"""`eh voices` commands."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import typer
from rich.table import Table

from entertainment_harness.cli import console, err_console, voices_app


def _play_paths(paths: list[Path]) -> None:
    """Play audio files sequentially (afplay on macOS, else the OS opener)."""
    if platform.system() == "Darwin":
        for path in paths:
            console.print(f"Playing {path}")
            subprocess.run(["afplay", str(path)], check=True)
    else:
        for path in paths:
            subprocess.run(["xdg-open", str(path)], check=True)

@voices_app.command("add")
def voices_add(
    name: str,
    audio: Path | None = typer.Option(
        None, "--audio", "-a",
        help="reference sample (wav/mp3/m4a, 3+ s, one speaker); omit for"
             " interactive recording",
    ),
    text: str | None = typer.Option(
        None, "--text", "-t", help="exact transcript of the sample (recommended)"
    ),
    x_vector: bool = typer.Option(
        False, "--x-vector", help="skip the transcript (lower cloning quality)"
    ),
    preview: bool = typer.Option(
        True, "--preview/--no-preview",
        help="synthesize sample clips with the new voice so you can audition it",
    ),
    play: bool = typer.Option(
        False, "--play", help="play the preview clips once generated"
    ),
) -> None:
    """Register a voice to clone with --tts-engine qwen3.

    Without --audio, runs interactive onboarding: records from your
    microphone (macOS), with playback and re-record until you're happy."""
    from entertainment_harness.video import voices as voice_lib
    from entertainment_harness.video.script import VideoError

    if audio is None:
        audio, text = _onboarding_record(name, text, x_vector)

    try:
        voice = voice_lib.add_voice(name, audio, text, x_vector=x_vector)
    except VideoError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if voice.duration_s < voice_lib.MIN_REF_SECONDS:
        err_console.print(
            f"[yellow]Warning: sample is only {voice.duration_s:.1f}s;"
            f" {voice_lib.MIN_REF_SECONDS:.0f}s+ of clean speech clones better.[/yellow]"
        )
    mode = "ICL (transcript)" if voice.transcript else "x-vector only"
    console.print(
        f"[bold green]Voice {name!r} added[/bold green]"
        f" ({voice.duration_s:.1f}s, {mode})."
        f" Use it with: --tts-engine qwen3 --voice {name}"
    )
    if preview:
        _run_previews(name, play=play)


@voices_app.command("preview")
def voices_preview(
    name: str,
    play: bool = typer.Option(
        False, "--play", help="play the preview clips once generated"
    ),
) -> None:
    """(Re)generate sample clips for a registered voice so you can audition it."""
    _run_previews(name, play=play)


def _run_previews(name: str, play: bool) -> None:
    from entertainment_harness.video import voices as voice_lib
    from entertainment_harness.video.script import VideoError

    try:
        paths = voice_lib.generate_previews(name, log=lambda m: console.print(m))
    except VideoError as exc:
        # The voice itself is fine — previews are regenerable, so warn
        # instead of failing the command that got us here.
        err_console.print(
            f"[yellow]Previews failed: {exc}[/yellow]"
            f"\n[yellow]Retry later with: eh voices preview {name}[/yellow]"
        )
        return
    console.print(f"[bold green]Previews for {name!r}:[/bold green]")
    for i, path in enumerate(paths):
        console.print(f"  {i}. {voice_lib.PREVIEW_LINES[i]}")
        console.print(f"     {path}")
    console.print("Play them with: eh voices preview %s --play" % name)
    if play:
        _play_paths(paths)


def _onboarding_record(
    name: str, text: str | None, x_vector: bool
) -> tuple[Path, str | None]:
    """Interactive reference recording (macOS). Returns (audio_path, transcript)
    ready to pass to add_voice."""
    import tempfile

    from entertainment_harness.video import voices as voice_lib

    console.print(f"[bold]Let's record a reference sample for {name!r}.[/bold]")
    if x_vector:
        console.print(
            "Speak naturally for up to 15 seconds — any clean, single-speaker"
            " speech works."
        )
        line: str | None = None
    else:
        line = (text or voice_lib.PREVIEW_LINES[0]).strip()
        console.print("Read this line aloud — say it 2-3 times, natural pace:")
        console.print(f'  "{line}"')
        console.print(
            "[dim]ICL cloning needs the transcript to match your words exactly,"
            " which is why we hand you the line.[/dim]"
        )
    while True:
        typer.prompt(
            "\nPress Enter to start recording",
            default="", show_default=False,
        )
        dest = Path(tempfile.mkstemp(suffix=".m4a", prefix=f"eh-voice-{name}-")[1])
        console.print("Recording… speak now (up to 15 s, Ctrl-C to cancel)")
        dest = voice_lib.record_reference(dest)
        console.print(f"Recorded {dest.stat().st_size / 1e6:.1f} MB.")
        if platform.system() == "Darwin" and typer.confirm(
            "Play it back?", default=False
        ):
            subprocess.run(["afplay", str(dest)], check=True)
        if not typer.confirm("Re-record?", default=False):
            return dest, line


@voices_app.command("list")
def voices_list() -> None:
    """List registered voices."""
    from entertainment_harness.video import voices as voice_lib

    voices = voice_lib.list_voices()
    if not voices:
        console.print("No voices registered. Add one with: eh voices add NAME --audio sample.wav --text \"...\"")
        return
    table = Table(title="Cloned voices (eh voices)")
    table.add_column("name", style="bold")
    table.add_column("sample")
    table.add_column("mode")
    for voice in voices:
        mode = "ICL (transcript)" if voice.transcript else "x-vector only"
        table.add_row(voice.name, f"{voice.duration_s:.1f}s", mode)
    console.print(table)


@voices_app.command("remove")
def voices_remove(name: str) -> None:
    """Delete a registered voice."""
    from entertainment_harness.video import voices as voice_lib
    from entertainment_harness.video.script import VideoError

    try:
        voice_lib.remove_voice(name)
    except VideoError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Removed voice {name!r}.")
