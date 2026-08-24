"""Backing store for rendered videos: pluggable remote backend.

Rendered videos are the harness's heavy, expensive-to-regenerate artifacts —
pages are re-creatable from the source, so only videos are stored. When a
store is configured it is the default backing for the pipeline:
`eh recap --video` pushes each chapter's rendered video after building it,
and `eh play` pulls a missing video back before giving up.

Configure the backend in config.toml:

    [store]
    provider = "hf"               # or "r2"
    repo = "owner/manga-archive"  # hf: private HF dataset repo (HF_TOKEN auth)

    [store.r2]                    # r2: Cloudflare R2 (S3-compatible) bucket
    bucket = "manga-archive"
    account_id = "..."
    access_key_id = "..."         # or R2_ACCESS_KEY_ID env var
    secret_access_key = "..."     # or R2_SECRET_ACCESS_KEY env var

Remote layout is id-keyed (stable across local slug renames):
  works/<series-id>/chapters/<chapter-id>/video-recap/out.mp4
  works/<series-id>/chapters/<chapter-id>/video-narration/out.mp4
  works/<series-id>/tiktok/out.mp4
Only final mp4s are pushed, never intermediate clips. Pulls land in the
id-named layout and are reconciled into the local slug dirs.
"""

from __future__ import annotations

import shutil
import types
from collections.abc import Callable
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.config import Config, data_dir

VIDEO_PREFIX = "works"


class StoreError(Exception):
    pass


def _backend(config: Config) -> types.ModuleType:
    provider = config.store.provider
    if provider == "hf":
        from entertainment_harness.store import hf

        return hf
    if provider == "r2":
        from entertainment_harness.store import r2

        return r2
    raise StoreError(
        f"Unknown store provider {provider!r}; expected 'hf' or 'r2'"
        f" ([store] provider in {data_dir() / 'config.toml'})."
    )


def is_configured(config: Config) -> bool:
    """Whether a usable remote store is configured."""
    if config.store.provider == "r2":
        return bool(config.store.r2.bucket)
    return bool(config.store.repo)


def _collect_chapter_mp4s(series_id: str, chapter_id: str) -> list[tuple[Path, str]]:
    """(path, chapter-relative key) pairs for a chapter's rendered mp4s."""
    pairs: list[tuple[Path, str]] = []
    for dir_name, d in (
        (works.VIDEO_RECAP_DIR, works.video_recap_dir(series_id, chapter_id)),
        (works.VIDEO_NARRATION_DIR, works.video_narration_dir(series_id, chapter_id)),
    ):
        if d.is_dir():
            pairs.extend(
                (p, f"{dir_name}/{p.name}") for p in sorted(d.glob("*.mp4"))
            )
    return pairs


def _collect_tiktok_mp4s(series_id: str) -> list[tuple[Path, str]]:
    d = works.tiktok_dir(series_id)
    if d.is_dir():
        return [(p, p.name) for p in sorted(d.glob("*.mp4"))]
    return []


def push_video(config: Config, series_id: str, chapter_key: str,
               log: Callable[[str], None] = print) -> int:
    """Upload the rendered mp4(s) for one chapter (or "tiktok") — both recap
    (video-recap/) and narration (video-narration/) kinds. Returns the number
    pushed."""
    if chapter_key == "tiktok":
        files = _collect_tiktok_mp4s(series_id)
        prefix = f"{VIDEO_PREFIX}/{series_id}/tiktok/"
    else:
        files = _collect_chapter_mp4s(series_id, chapter_key)
        prefix = f"{VIDEO_PREFIX}/{series_id}/chapters/{chapter_key}/"
    if not files:
        raise StoreError(
            f"No rendered video to push for {series_id}/{chapter_key}."
        )
    _backend(config).upload(
        config,
        [(p, f"{prefix}{rel}") for p, rel in files],
        log,
    )
    log(f"Pushed {len(files)} video(s) for {chapter_key} to the store.")
    return len(files)


def pull_video(config: Config, series_id: str, chapter_key: str) -> list[Path]:
    """Restore the rendered mp4(s) for one chapter (or "tiktok") into the
    local cache. Returns the restored files (empty when absent remotely)."""
    prefix = (
        f"{VIDEO_PREFIX}/{series_id}/tiktok/"
        if chapter_key == "tiktok"
        else f"{VIDEO_PREFIX}/{series_id}/chapters/{chapter_key}/"
    )
    _backend(config).download(config, prefix)
    _reconcile_landing(series_id, chapter_key)
    if chapter_key == "tiktok":
        return [p for p, _ in _collect_tiktok_mp4s(series_id)]
    return [p for p, _ in _collect_chapter_mp4s(series_id, chapter_key)]


def _reconcile_landing(series_id: str, chapter_key: str) -> None:
    """Move files pulled into the id-named remote layout over to the resolved
    (possibly slug-named) local directories."""
    landing = works.works_root() / series_id
    if landing == works.work_dir(series_id) or not landing.is_dir():
        return
    if chapter_key == "tiktok":
        pairs = [(landing / works.TIKTOK_DIR, works.tiktok_dir(series_id))]
    else:
        pairs = [(
            landing / "chapters" / chapter_key,
            works.chapter_dir(series_id, chapter_key),
        )]
    for src, dst in pairs:
        if not src.is_dir():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.move(str(item), str(dst / item.name))
        src.rmdir()
    # Prune now-empty id-named parents.
    for d in (landing / "chapters", landing):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
