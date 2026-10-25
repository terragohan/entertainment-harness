"""Whole-work staging over the remote store (cloud execution, Track B).

(Named staging.py rather than works.py: store/__init__.py imports
library.works as `works`, which would shadow a same-named submodule.)

`store/__init__.py` pushes/pulls final mp4s per chapter; cloud jobs need the
whole work dir staged in before a run and staged out after. The remote
layout stays id-keyed — `works/<series-id>/chapters/<chapter-id>/...` — so
local slug renames never invalidate the remote copy, and pulled id-named
dirs resolve directly (works.work_dir/chapter_dir fall back to id-named
directories).

Regenerable intermediates are excluded from pushes: video concat inputs
(`clips/`, `seg-*.wav` — pruned locally once a final video exists anyway),
caches, and OS noise. Work-root `*.mp4` files never cross: the works layout
has no root-level video artifact (chapter videos live under `chapters/`,
shorts under `tiktok/`), so a stray assembled deliverable there (e.g. a
hand-concatenated recap-full.mp4) is not staging's concern.

`pull_work_matching()` is the selective half of pulls: cloud containers
stage in two phases (metadata first, then just the in-scope chapters'
sources) because whole-work pulls can exceed the container disk. Pushes
are incremental (same-size keys skipped) when the backend exposes
`list_sizes`. `stage_up()` is the submit-side half: it forces the R2
provider (cloud containers always stage from R2) so the local submit path
can push a work's sources before queueing its job.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from entertainment_harness.config import Config
from entertainment_harness.library import works
from entertainment_harness.store import VIDEO_PREFIX, StoreError, backend

#: Directory names never pushed (regenerable intermediates / caches).
EXCLUDED_DIRS = frozenset({"clips", "__pycache__"})
#: File-name patterns never pushed.
EXCLUDED_FILES = (".DS_Store",)


def _excluded(path: Path) -> bool:
    return (
        any(part in EXCLUDED_DIRS for part in path.parts)
        or path.name in EXCLUDED_FILES
        or path.name.startswith("seg-") and path.suffix == ".wav"
        # Work-root mp4s are not works-layout artifacts (chapter videos live
        # under chapters/, shorts under tiktok/) — never stage stray
        # assembled deliverables like a hand-made recap-full.mp4.
        or len(path.parts) == 1 and path.suffix == ".mp4"
    )


def _chapter_ids_by_dir(series_id: str) -> dict[str, str]:
    """Local chapter dir name -> chapter id, from chapter.json files."""
    mapping: dict[str, str] = {}
    base = works.work_dir(series_id) / "chapters"
    if base.is_dir():
        for d in base.iterdir():
            if not d.is_dir():
                continue
            try:
                meta = json.loads(
                    (d / works.CHAPTER_METADATA_FILENAME).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(meta, dict) and meta.get("id"):
                mapping[d.name] = str(meta["id"])
    return mapping


def collect_work_files(series_id: str) -> list[tuple[Path, str]]:
    """(local path, id-keyed object key) pairs for the work's pushable files.

    Local chapter dirs are slug-named (ch-NNN); keys are rewritten to the
    chapter id from chapter.json so the remote layout matches the per-chapter
    video layout. Dirs without chapter.json keep their local name.
    """
    wdir = works.work_dir(series_id)
    if not wdir.is_dir():
        raise StoreError(f"No local work directory for {series_id!r}.")
    dir_to_id = _chapter_ids_by_dir(series_id)
    pairs: list[tuple[Path, str]] = []
    for path in sorted(p for p in wdir.rglob("*") if p.is_file()):
        rel = path.relative_to(wdir)
        if _excluded(rel):
            continue
        parts = list(rel.parts)
        if len(parts) > 1 and parts[0] == "chapters":
            parts[1] = dir_to_id.get(parts[1], parts[1])
        pairs.append((path, "/".join([VIDEO_PREFIX, series_id, *parts])))
    return pairs


def push_work(config: Config, series_id: str,
              log: Callable[[str], None] = print) -> int:
    """Upload the work's files (minus regenerable intermediates) to the
    store. Returns the number of files uploaded.

    Incremental when the backend exposes `list_sizes` (R2 does): a key whose
    remote size matches the local file is already uploaded and skipped, so
    re-pushes and stage-outs only send what changed."""
    files = collect_work_files(series_id)
    if not files:
        raise StoreError(f"Nothing to push for {series_id!r}.")
    be = backend(config)
    skipped = 0
    list_sizes = getattr(be, "list_sizes", None)
    if list_sizes is not None:
        remote = list_sizes(config, f"{VIDEO_PREFIX}/{series_id}/")
        kept = [(p, k) for p, k in files if remote.get(k) != p.stat().st_size]
        skipped = len(files) - len(kept)
        files = kept
    be.upload(config, files, log)
    note = f" ({skipped} unchanged skipped)" if skipped else ""
    log(f"Pushed {len(files)} file(s) for {series_id} to the store.{note}")
    return len(files)


def stage_up(config: Config, series_id: str,
             log: Callable[[str], None] = print) -> int:
    """Stage a local work up to R2 so a cloud job can run it.

    Cloud containers always pull works from R2, so the store provider is
    forced to r2 here regardless of the local [store] provider. Returns the
    number of files uploaded (incremental via push_work). Raises StoreError
    when the work has no local directory or R2 is unconfigured — check
    r2.is_ready(config) first when skipping is acceptable.
    """
    from entertainment_harness.store import r2 as store_r2

    forced = replace(config, store=replace(config.store, provider="r2"))
    if not store_r2.is_ready(forced):
        raise StoreError(
            "R2 store is not configured: cloud jobs stage works through R2 —"
            " set [store.r2] keys (Settings → API keys) or the R2_* env vars."
        )
    return push_work(forced, series_id, log=log)


def is_metadata_key(rel: str) -> bool:
    """Indexable state under a work prefix: work.json, the cover image, and
    every JSON (chapter/recap/translation/video metadata, panels, scripts).
    Pages and mp4s are the bulk; they are not metadata."""
    return (
        rel == works.WORK_METADATA_FILENAME
        or rel.startswith("cover.")
        or rel.endswith(".json")
    )


def scope_key_filter(chapter_ids) -> Callable[[str], bool]:
    """keep(rel) matching just the given chapters' page dirs (source/ and
    translated/) — the only heavy inputs a recap job reads."""
    prefixes = tuple(
        f"chapters/{cid}/{d}/"
        for cid in chapter_ids
        for d in (works.SOURCE_DIR, works.TRANSLATED_DIR)
    )
    return lambda rel: rel.startswith(prefixes)


def pull_work(config: Config, series_id: str,
              log: Callable[[str], None] = print) -> int:
    """Download everything stored under the work's prefix into the local
    data dir (id-named layout; resolution falls back to id-named dirs).
    Returns the number of files restored."""
    backend(config).download(config, f"{VIDEO_PREFIX}/{series_id}/")
    works.refresh_paths()
    landing = works.works_root() / series_id
    count = (
        sum(1 for p in landing.rglob("*") if p.is_file())
        if landing.is_dir()
        else 0
    )
    log(f"Pulled {count} file(s) for {series_id} from the store.")
    return count


def pull_work_matching(config: Config, series_id: str, keep,
                       log: Callable[[str], None] = print
                       ) -> tuple[int, dict[str, int]]:
    """Download only the store keys under the work's prefix for which
    `keep(rel)` is true (rel is the key relative to works/<series-id>/).
    Returns (files downloaded, the full remote key -> size listing) so the
    caller can reuse the listing (jobs.py derives store-held videos from
    it). Backends without listing support fall back to a whole-work pull
    and return an empty listing."""
    be = backend(config)
    list_sizes = getattr(be, "list_sizes", None)
    if list_sizes is None:
        count = pull_work(config, series_id, log=log)
        return count, {}
    prefix = f"{VIDEO_PREFIX}/{series_id}/"
    sizes = list_sizes(config, prefix)
    keys = sorted(k for k in sizes if keep(k[len(prefix):]))
    if keys:
        be.download_keys(config, keys)
    works.refresh_paths()
    log(f"Pulled {len(keys)} of {len(sizes)} file(s) for {series_id}"
        " from the store.")
    return len(keys), sizes

