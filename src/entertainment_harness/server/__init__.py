"""Localhost HTTP bridge for the desktop UI (`eh serve`).

FastAPI app exposing the library, range-streamed chapter videos, background
recap runs with SSE progress, and config/sources editing — all read/write
through the same db/works/config seams the CLI uses. See docs/design.md
"Local server".
"""
