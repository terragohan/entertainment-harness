"""Cloudflare R2 (S3-compatible) store backend.

Configure the bucket in config.toml (credentials fall back to env vars):

    [store]
    provider = "r2"

    [store.r2]
    bucket = "manga-archive"    # or R2_BUCKET
    account_id = "..."          # or R2_ACCOUNT_ID — https://<account_id>.r2.cloudflarestorage.com
    access_key_id = "..."       # or R2_ACCESS_KEY_ID
    secret_access_key = "..."   # or R2_SECRET_ACCESS_KEY

Remote layout mirrors the local works layout:
  works/<series-id>/chapters/<chapter-key>/video-recap/out.mp4
  works/<series-id>/chapters/<chapter-key>/video-narration/out.mp4
  works/<series-id>/tiktok/out.mp4
(final mp4s only; narration videos keep their video-narration/ subdir).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from entertainment_harness.config import Config, data_dir
from entertainment_harness.store import StoreError


def _client(config: Config):
    """Return (s3_client, bucket). Raises StoreError when unconfigured."""
    r2 = config.store.r2
    bucket = r2.bucket or os.environ.get("R2_BUCKET", "")
    account_id = r2.account_id or os.environ.get("R2_ACCOUNT_ID", "")
    access_key = r2.access_key_id or os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key = r2.secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY", "")
    missing = []
    if not bucket:
        missing.append("bucket (or the R2_BUCKET env var)")
    if not account_id:
        missing.append("account_id (or the R2_ACCOUNT_ID env var)")
    if not access_key:
        missing.append("access_key_id (or the R2_ACCESS_KEY_ID env var)")
    if not secret_key:
        missing.append("secret_access_key (or the R2_SECRET_ACCESS_KEY env var)")
    if missing:
        raise StoreError(
            "R2 store is not configured: set " + ", ".join(missing)
            + f" in [store.r2] in {data_dir() / 'config.toml'}."
        )
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return client, bucket


def is_ready(config: Config) -> bool:
    """All four settings resolvable (config or env) — unlike _client, never
    raises; callers use this to skip R2-dependent work gracefully."""
    r2 = config.store.r2
    return all([
        r2.bucket or os.environ.get("R2_BUCKET"),
        r2.account_id or os.environ.get("R2_ACCOUNT_ID"),
        r2.access_key_id or os.environ.get("R2_ACCESS_KEY_ID"),
        r2.secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY"),
    ])


def _list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def list_sizes(config: Config, prefix: str) -> dict[str, int]:
    """Object key -> size for everything under a prefix. Drives incremental
    pushes: a same-size key is treated as already uploaded."""
    client, bucket = _client(config)
    sizes: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj.get("Size", -1)
    return sizes


def upload(config: Config, files: list[tuple[Path, str]],
           log: Callable[[str], None] = print) -> None:
    """Upload (local path, object key) pairs to the bucket."""
    client, bucket = _client(config)
    for local, key in files:
        client.upload_file(str(local), bucket, key)


def _download_keys(client, bucket: str, keys) -> None:
    for key in keys:
        dest = data_dir() / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))


def download_keys(config: Config, keys) -> None:
    """Download an explicit key list into the local data dir."""
    client, bucket = _client(config)
    try:
        _download_keys(client, bucket, list(keys))
    except Exception as exc:
        raise StoreError(f"Could not pull from r2 bucket {bucket!r}: {exc}") from exc


def download(config: Config, prefix: str) -> None:
    """Download everything under an object prefix into the local data dir."""
    client, bucket = _client(config)
    try:
        _download_keys(client, bucket, _list_keys(client, bucket, prefix))
    except Exception as exc:
        raise StoreError(f"Could not pull from r2 bucket {bucket!r}: {exc}") from exc
