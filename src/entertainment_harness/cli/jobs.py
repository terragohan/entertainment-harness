"""`eh job-run`: headless pipeline job runner (cloud container entrypoint).

See entertainment_harness.jobs — events stream as JSONL on stdout, so
container logs are the event feed.
"""

from __future__ import annotations

import typer

from entertainment_harness.cli import app, err_console


@app.command("job-run")
def job_run(
    spec: str = typer.Option(
        ..., "--spec",
        help="job spec JSON: a file path, '-' for stdin, or an inline JSON"
        " object — {\"work\": <series id or slug>, \"kind\": \"recap\","
        " \"options\": {...eh recap flags...}}",
    ),
) -> None:
    """Run one pipeline job headless from a JSON spec and exit.

    Stage the work in from the configured store, rebuild the index from the
    works layout, run the headless recap flow, stage out. With EH_CLOUD=1
    the config loads the cloud profile (openai_compat roles, R2 store, no
    pre-flight guard). Events stream as JSONL on stdout and to EH_JOB_EVENTS
    (default <data_dir>/jobs/<id>/events.jsonl). Exit code: 0 done or
    cancelled, 1 error, 2 bad spec.
    """
    from entertainment_harness import jobs

    try:
        code = jobs.main(spec)
    except jobs.JobSpecError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    raise typer.Exit(code=code)
