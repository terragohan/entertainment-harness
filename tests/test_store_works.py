"""Whole-work staging tests (store/staging.py): id-keyed push layout,
intermediate exclusion, pull restore — fake boto3 client, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entertainment_harness import store
from entertainment_harness.config import Config
from entertainment_harness.library import works
from entertainment_harness.store import r2 as store_r2
from entertainment_harness.store import staging


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    return tmp_path


def _config() -> Config:
    config = Config()
    config.store.provider = "r2"
    config.store.r2.bucket = "bucket"
    return config


class FakePaginator:
    def __init__(self, client) -> None:
        self.client = client

    def paginate(self, Bucket, Prefix, Delimiter=None):
        yield {
            "Contents": [
                {"Key": k, "Size": len(self.client.objects[k])}
                for k in sorted(self.client.objects)
                if k.startswith(Prefix)
            ]
        }


class FakeS3:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.uploads: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self)

    def upload_file(self, src, bucket, key):
        self.objects[key] = Path(src).read_bytes()
        self.uploads.append(key)

    def download_file(self, bucket, key, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.objects[key])


def _fake_client(monkeypatch, objects=None) -> FakeS3:
    client = FakeS3(objects)
    monkeypatch.setattr(store_r2, "_client", lambda config: (client, "bucket"))
    return client


def _seed_work(root: Path, sid: str = "s1") -> Path:
    """A work with a slug-named dir, one chapter, a recap, and a rendered
    video with regenerable intermediates."""
    wdir = works.work_dir(sid)
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "work.json").write_text(json.dumps({
        "id": sid, "title": "Demo", "source": "mangadex", "source_id": "x",
        "added_at": "2026-01-01T00:00:00+00:00",
    }))
    ch = wdir / "chapters" / "ch-001"
    (ch / "source").mkdir(parents=True)
    (ch / "chapter.json").write_text(json.dumps({
        "id": "ch-1", "chapter_num": 1, "title": None, "lang": "en",
        "pages": 1, "published_at": None, "fetched_at": "2026-01-01T00:00:00+00:00",
    }))
    (ch / "source" / "001.png").write_bytes(b"page")
    (ch / "recap.json").write_text(json.dumps({
        "summary": "s", "model": "m", "created_at": "2026-01-01T00:00:00+00:00",
    }))
    video = ch / "video-recap"
    (video / "clips").mkdir(parents=True)
    (video / "out.mp4").write_bytes(b"video")
    (video / "clips" / "clip-00-0.mp4").write_bytes(b"clip")
    (video / "seg-0.wav").write_bytes(b"wav")
    return wdir


def test_push_work_uses_id_keyed_layout(root, monkeypatch):
    client = _fake_client(monkeypatch)
    _seed_work(root)
    count = staging.push_work(_config(), "s1", log=lambda m: None)
    assert count == 5
    assert client.uploads == sorted(client.uploads)
    assert set(client.uploads) == {
        "works/s1/chapters/ch-1/chapter.json",
        "works/s1/chapters/ch-1/recap.json",
        "works/s1/chapters/ch-1/source/001.png",
        "works/s1/chapters/ch-1/video-recap/out.mp4",
        "works/s1/work.json",
    }


def test_push_work_excludes_intermediates(root, monkeypatch):
    client = _fake_client(monkeypatch)
    _seed_work(root)
    staging.push_work(_config(), "s1", log=lambda m: None)
    assert not any("clips/" in k or k.endswith("seg-0.wav") for k in client.uploads)


def test_push_work_missing_work_rejected(root, monkeypatch):
    _fake_client(monkeypatch)
    with pytest.raises(store.StoreError, match="No local work directory"):
        staging.push_work(_config(), "nope", log=lambda m: None)


def test_pull_work_restores_files(root, monkeypatch):
    _fake_client(monkeypatch, {
        "works/s1/work.json": b"{}",
        "works/s1/chapters/ch-1/source/001.png": b"page",
    })
    count = staging.pull_work(_config(), "s1", log=lambda m: None)
    assert count == 2
    landing = works.works_root() / "s1"
    assert (landing / "work.json").read_bytes() == b"{}"
    assert (landing / "chapters" / "ch-1" / "source" / "001.png").read_bytes() == b"page"


def test_pull_work_empty_when_absent(root, monkeypatch):
    _fake_client(monkeypatch)
    assert staging.pull_work(_config(), "s9", log=lambda m: None) == 0


# --- incremental pushes + stage_up -------------------------------------------

R2_ENV = ("R2_BUCKET", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
          "R2_SECRET_ACCESS_KEY")


def _r2_ready_config(provider: str = "r2") -> Config:
    config = _config()
    config.store.provider = provider
    config.store.r2.account_id = "acct"
    config.store.r2.access_key_id = "ak"
    config.store.r2.secret_access_key = "sk"
    return config


def test_push_work_skips_unchanged_remote_keys(root, monkeypatch):
    """A same-size remote key is already uploaded; only changed files go."""
    client = _fake_client(monkeypatch)
    wdir = _seed_work(root)
    assert staging.push_work(_config(), "s1", log=lambda m: None) == 5
    # Second push: nothing changed.
    assert staging.push_work(_config(), "s1", log=lambda m: None) == 0
    # A changed file (different size) is re-uploaded, alone.
    (wdir / "chapters" / "ch-001" / "recap.json").write_bytes(
        b'{"summary": "a much longer summary than before"}'
    )
    assert staging.push_work(_config(), "s1", log=lambda m: None) == 1
    assert client.uploads[-1] == "works/s1/chapters/ch-1/recap.json"


def test_stage_up_forces_r2_provider(root, monkeypatch):
    """Cloud containers stage from R2, so stage_up pushes through the r2
    backend even when the local [store] provider is hf."""
    client = _fake_client(monkeypatch)
    _seed_work(root)
    config = _r2_ready_config(provider="hf")
    count = staging.stage_up(config, "s1", log=lambda m: None)
    assert count == 5
    assert config.store.provider == "hf"  # the passed config is not mutated
    assert "works/s1/work.json" in client.objects


def test_stage_up_unconfigured_raises_actionably(root, monkeypatch):
    for var in R2_ENV:
        monkeypatch.delenv(var, raising=False)
    _seed_work(root)
    with pytest.raises(store.StoreError, match="not configured"):
        staging.stage_up(Config(), "s1", log=lambda m: None)


def test_is_ready_checks_config_then_env(root, monkeypatch):
    for var in R2_ENV:
        monkeypatch.delenv(var, raising=False)
    assert not store_r2.is_ready(Config())
    assert store_r2.is_ready(_r2_ready_config())
    monkeypatch.setenv("R2_BUCKET", "b")
    monkeypatch.setenv("R2_ACCOUNT_ID", "a")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s")
    assert store_r2.is_ready(Config())


# --- selective pulls (scoped stage-in) -----------------------------------------


def test_push_work_never_stages_work_root_mp4(root, monkeypatch):
    """A stray assembled deliverable at the work root (recap-full.mp4) is
    not a works-layout artifact and never crosses the store."""
    client = _fake_client(monkeypatch)
    wdir = _seed_work(root)
    (wdir / "recap-full.mp4").write_bytes(b"assembled")
    assert staging.push_work(_config(), "s1", log=lambda m: None) == 5
    assert not any(k.endswith("recap-full.mp4") for k in client.uploads)


def test_is_metadata_key_and_scope_filter():
    assert staging.is_metadata_key("work.json")
    assert staging.is_metadata_key("cover.jpg")
    assert staging.is_metadata_key("chapters/ch-1/recap.json")
    assert staging.is_metadata_key("chapters/ch-1/video-recap/video.json")
    assert not staging.is_metadata_key("chapters/ch-1/source/001.png")
    assert not staging.is_metadata_key("chapters/ch-1/video-recap/out.mp4")
    keep = staging.scope_key_filter({"ch-1"})
    assert keep("chapters/ch-1/source/001.png")
    assert keep("chapters/ch-1/translated/001.png")
    assert not keep("chapters/ch-2/source/001.png")
    assert not keep("chapters/ch-1/video-recap/out.mp4")
    assert not keep("work.json")


def test_pull_work_matching_downloads_only_kept_keys(root, monkeypatch):
    _fake_client(monkeypatch, {
        "works/s1/work.json": b"{}",
        "works/s1/chapters/ch-1/chapter.json": b"{}",
        "works/s1/chapters/ch-1/source/001.png": b"page",
        "works/s1/chapters/ch-1/video-recap/out.mp4": b"video",
    })
    count, listing = staging.pull_work_matching(
        _config(), "s1", staging.is_metadata_key, log=lambda m: None
    )
    assert count == 2
    assert len(listing) == 4  # the full listing comes back for reuse
    landing = works.works_root() / "s1"
    assert (landing / "work.json").is_file()
    assert (landing / "chapters" / "ch-1" / "chapter.json").is_file()
    assert not (landing / "chapters" / "ch-1" / "source" / "001.png").exists()
    count2, _ = staging.pull_work_matching(
        _config(), "s1", staging.scope_key_filter({"ch-1"}),
        log=lambda m: None,
    )
    assert count2 == 1  # the source page; the mp4 never crosses
    assert (landing / "chapters" / "ch-1" / "source" / "001.png").is_file()
    assert not (landing / "chapters" / "ch-1" / "video-recap").exists()
