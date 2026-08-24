"""HF Hub dataset-repo store backend.

Auth comes from the HF_TOKEN environment variable (or `huggingface-cli
login`). The repo is created as private on first push. Configure the target
repo in config.toml:

    [store]
    provider = "hf"
    repo = "owner/manga-archive"
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from entertainment_harness.config import Config, data_dir
from entertainment_harness.store import StoreError


def _repo(config: Config) -> str:
    if not config.store.repo:
        raise StoreError(
            "No store repo configured. Set [store] repo = \"owner/name\" in"
            f" {data_dir() / 'config.toml'} and authenticate with HF_TOKEN."
        )
    return config.store.repo


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def upload(config: Config, files: list[tuple[Path, str]],
           log: Callable[[str], None] = print) -> None:
    """Upload (local path, repo path) pairs to the dataset repo."""
    repo = _repo(config)
    api = _api()
    try:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for local, key in files:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=key,
                repo_id=repo,
                repo_type="dataset",
            )
    except Exception as exc:
        raise StoreError(f"Could not push to {repo!r}: {exc}") from exc


def download(config: Config, prefix: str) -> None:
    """Download everything under a repo prefix into the local data dir."""
    repo = _repo(config)
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo, repo_type="dataset",
            allow_patterns=f"{prefix}*", local_dir=str(data_dir()),
        )
    except Exception as exc:
        raise StoreError(f"Could not pull from {repo!r}: {exc}") from exc
