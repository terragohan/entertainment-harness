"""`eh cloud`: submit pipeline jobs to the Cloudflare control plane and
observe them (animation-director Track B, phase C3).

The Worker (infra/entertainment-harness/src/worker.ts, in the sibling infra
repo) owns the jobs ledger in D1 and runs each job on a container; these
commands are the local submit/observe path. Auth
is a single bearer token: [cloud] worker_url + [cloud] token in config.toml
(env fallback: EH_CLOUD_TOKEN).
"""

from __future__ import annotations

import json

import httpx
import typer
from rich.markup import escape
from rich.table import Table

from entertainment_harness import db, library
from entertainment_harness.cli import app, console, err_console
from entertainment_harness.config import data_dir, load_config
from entertainment_harness.library import works
from entertainment_harness.store import StoreError, staging
from entertainment_harness.store import r2 as store_r2

cloud_app = typer.Typer(help="Submit and manage cloud pipeline jobs.")
app.add_typer(cloud_app, name="cloud")


def _endpoint() -> tuple[str, str]:
    """(worker_url, bearer token); exits with guidance when unconfigured."""
    cloud = load_config().cloud
    url = cloud.worker_url.rstrip("/")
    token = cloud.resolved_token
    if not url:
        err_console.print(
            "[red]Cloud worker not configured: set worker_url in [cloud] in"
            f" {data_dir() / 'config.toml'}.[/red]"
        )
        raise typer.Exit(code=1)
    if not token:
        err_console.print(
            "[red]No cloud submit token: set token in [cloud] in"
            f" {data_dir() / 'config.toml'} (Settings → API keys) or export"
            " EH_CLOUD_TOKEN.[/red]"
        )
        raise typer.Exit(code=1)
    return url, token


def _request(method: str, path: str, **kw) -> httpx.Response:
    url, token = _endpoint()
    try:
        resp = httpx.request(
            method,
            f"{url}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            **kw,
        )
    except httpx.HTTPError as exc:
        err_console.print(f"[red]Cloud worker unreachable: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    if resp.status_code == 401:
        err_console.print(
            "[red]Unauthorized: the cloud token does not match the worker's"
            " secret — check [cloud] token / EH_CLOUD_TOKEN.[/red]"
        )
        raise typer.Exit(code=1)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        err_console.print(f"[red]{resp.status_code}: {escape(str(detail))}[/red]")
        raise typer.Exit(code=1)
    return resp


@cloud_app.command()
def submit(
    work: str,
    chapter: float | None = typer.Option(
        None, "--chapter", "-c", help="run one chapter"
    ),
    all_chapters: bool = typer.Option(
        False, "--all", help="run every synced chapter"
    ),
    video: bool = typer.Option(
        False, "--video", help="render each chapter's video after its recap"
    ),
    translated: bool = typer.Option(
        False, "--translated", help="translate pages before recapping"
    ),
    detail: str | None = typer.Option(None, "--detail", help="artifact grain"),
    instruction: str | None = typer.Option(
        None, "--instruction", "-i", help="steering instruction"
    ),
    options: str | None = typer.Option(
        None, "--options",
        help="extra RunOptions fields as a JSON object (see server.runs."
        " RunOptions); the named flags above win on conflicts",
    ),
) -> None:
    """Submit a recap job to run on Cloudflare; prints the job id."""
    opts: dict = {}
    if options is not None:
        try:
            parsed = json.loads(options)
        except json.JSONDecodeError as exc:
            err_console.print(f"[red]--options is not valid JSON: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        if not isinstance(parsed, dict):
            err_console.print("[red]--options must be a JSON object[/red]")
            raise typer.Exit(code=1)
        opts = parsed
    if chapter is not None:
        opts["chapter"] = chapter
    if all_chapters:
        opts["all_chapters"] = True
    if video:
        opts["video"] = True
    if translated:
        opts["translated"] = True
    if detail is not None:
        opts["detail"] = detail
    if instruction is not None:
        opts["instruction"] = instruction
    # The submit stages the work up from the local library — resolve locally
    # first so a typo fails here, not after a container boot.
    with db.connect() as conn:
        try:
            row = library.resolve_series(conn, work)
        except library.LibraryError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    work = row["id"]
    config = load_config()
    if not works.work_dir(work).is_dir():
        err_console.print(
            f"[yellow]No local work dir for {work} — submitting without"
            " stage-up; the job needs the work already in R2.[/yellow]"
        )
    elif not store_r2.is_ready(config):
        err_console.print(
            "[yellow]R2 store not configured ([store.r2] keys or R2_* env"
            " vars) — submitting without stage-up; the job needs the work"
            " already in R2.[/yellow]"
        )
    else:
        try:
            staging.stage_up(
                config, work, log=lambda m: console.print(f"[dim]{m}[/dim]")
            )
        except StoreError as exc:
            err_console.print(f"[red]Stage-up failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc
    resp = _request(
        "POST", "/api/jobs",
        json={"work": work, "kind": "recap", "options": opts},
    )
    job = resp.json()["job"]
    console.print(f"[bold green]queued[/bold green] job {job['id']} for {work}")
    console.print(f"  eh cloud status {job['id']}  ·  eh cloud logs {job['id']}")


@cloud_app.command()
def status(job_id: str | None = typer.Argument(None)) -> None:
    """List recent cloud jobs, or show one job's detail."""
    if job_id is None:
        jobs = _request("GET", "/api/jobs").json()["jobs"]
        if not jobs:
            console.print("No cloud jobs yet.")
            return
        table = Table("id", "status", "work", "updated")
        for j in jobs:
            table.add_row(j["id"], j["status"], j["work"], j["updated_at"])
        console.print(table)
        return
    job = _request("GET", f"/api/jobs/{job_id}").json()["job"]
    console.print(f"[bold]{job['id']}[/bold]  {job['status']}")
    console.print(f"  work:    {job['work']} ({job['kind']})")
    console.print(f"  options: {json.dumps(job['options'])}")
    if job.get("current"):
        cur = job["current"]
        bits = ", ".join(
            f"{k}={v}" for k, v in cur.items() if v is not None
        )
        console.print(f"  current: {bits}")
    if job.get("error"):
        console.print(f"  [red]error: {escape(job['error'])}[/red]")
    if job.get("stop_requested"):
        console.print("  stop requested (takes effect at the next boundary)")
    console.print(f"  created: {job['created_at']}")
    console.print(f"  updated: {job['updated_at']}")


@cloud_app.command()
def stop(job_id: str) -> None:
    """Ask a queued/running job to stop at the next chapter boundary."""
    _request("POST", f"/api/jobs/{job_id}/stop")
    console.print(
        f"Stop requested for {job_id}; the job stops at its next chapter"
        " boundary (immediately if still queued)."
    )


@cloud_app.command()
def logs(job_id: str) -> None:
    """Print a job's event stream (JSONL pushed to the store by the job)."""
    resp = _request("GET", f"/api/jobs/{job_id}/events")
    for line in resp.text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ts = str(rec.get("ts", ""))[11:19]
        event = rec.get("event", "?")
        detail = (
            rec.get("message")
            or rec.get("detail")
            or (f"chapter {rec['chapter']:g}" if rec.get("chapter") is not None else "")
            or ""
        )
        console.print(f"{ts} {escape(str(event)):<14} {escape(str(detail))}")
