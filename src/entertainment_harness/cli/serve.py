"""`eh serve`: localhost HTTP server for the desktop UI.

The first stdout line is machine-readable for the parent process that
spawned us (Electrobun): `listening <port>`. Everything else (uvicorn
logs, errors) goes to stderr. See docs/design.md "Local server".
"""

from __future__ import annotations

import asyncio

import typer

from entertainment_harness.cli import app, err_console


@app.command()
def serve(
    port: int = typer.Option(
        0, "--port", help="port to listen on; 0 = ephemeral (the OS picks"
        " one, reported on the 'listening' line)"
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Serve the local HTTP API (library, video streaming, runs, config)
    for the desktop UI. Prints 'listening <port>' as the first stdout line
    once bound; all other output goes to stderr."""
    import uvicorn

    from entertainment_harness.server.app import create_app

    config = uvicorn.Config(
        create_app(), host=host, port=port, log_level="info"
    )
    server = uvicorn.Server(config)
    try:
        asyncio.run(_serve(server))
    except KeyboardInterrupt:
        pass


async def _serve(server) -> None:
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        if serve_task.done():
            # startup failed (e.g. port in use); surface the exception
            await serve_task
            return
        await asyncio.sleep(0.01)
    sock = server.servers[0].sockets[0]
    actual_port = sock.getsockname()[1]
    print(f"listening {actual_port}", flush=True)
    err_console.print(
        f"[dim]eh serve: API on http://{server.config.host}:{actual_port}"
        "/api (Ctrl+C to stop)[/dim]"
    )
    await serve_task
