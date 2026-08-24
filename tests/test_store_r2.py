"""R2 store-backend tests: config parsing, dispatch, and S3 operations with a
fake boto3 client — no network, no credentials needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from entertainment_harness import store
from entertainment_harness.store import r2 as store_r2
from entertainment_harness.config import Config, load_config


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    return tmp_path


def _config(**r2_kwargs) -> Config:
    config = Config()
    config.store.provider = "r2"
    for key, value in r2_kwargs.items():
        setattr(config.store.r2, key, value)
    return config


class FakePaginator:
    def __init__(self, client) -> None:
        self.client = client

    def paginate(self, Bucket, Prefix, Delimiter=None):
        yield {
            "Contents": [
                {"Key": k} for k in sorted(self.client.objects)
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
        Path(dest).write_bytes(self.objects[key])


def _fake_client(monkeypatch, objects=None) -> FakeS3:
    client = FakeS3(objects)
    monkeypatch.setattr(store_r2, "_client", lambda config: (client, "bucket"))
    return client


# --- config -------------------------------------------------------------------


def test_config_defaults_to_hf(root):
    (root / "config.toml").write_text('[store]\nrepo = "me/manga-archive"\n')
    config = load_config()
    assert config.store.provider == "hf"
    assert config.store.r2.bucket == ""


def test_config_loads_r2_settings(root):
    (root / "config.toml").write_text(
        '[store]\nprovider = "r2"\n'
        "[store.r2]\n"
        'bucket = "manga-archive"\n'
        'account_id = "abc123"\n'
        'access_key_id = "key"\n'
        'secret_access_key = "secret"\n'
    )
    config = load_config()
    assert config.store.provider == "r2"
    assert config.store.r2.bucket == "manga-archive"
    assert config.store.r2.account_id == "abc123"


def test_unknown_provider_rejected(root):
    _seed_video(root, "s1")
    config = Config()
    config.store.provider = "s3"
    with pytest.raises(store.StoreError, match="Unknown store provider"):
        store.push_video(config, "s1", "ch-1", log=lambda m: None)


def test_is_configured(root):
    config = _config()
    assert not store.is_configured(config)  # r2 without a bucket
    config.store.r2.bucket = "b"
    assert store.is_configured(config)


# --- client settings ------------------------------------------------------------


def test_client_requires_bucket_and_keys(root, monkeypatch):
    for var in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(store.StoreError, match="R2 store is not configured"):
        store_r2._client(_config())
    with pytest.raises(store.StoreError, match="R2 store is not configured"):
        store_r2._client(_config(bucket="b", account_id="a"))


def test_client_uses_env_credential_fallback(root, monkeypatch):
    captured = {}

    class FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            captured.update(kwargs)
            return FakeS3()

    monkeypatch.setenv("R2_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "env-secret")
    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)
    client, bucket = store_r2._client(_config(bucket="b", account_id="acc"))
    assert bucket == "b"
    assert captured["endpoint_url"] == "https://acc.r2.cloudflarestorage.com"
    assert captured["aws_access_key_id"] == "env-key"
    assert captured["aws_secret_access_key"] == "env-secret"


# --- video push/pull (via the facade) --------------------------------------------


def _seed_video(root: Path, sid: str, chapter: str = "ch-1") -> Path:
    from entertainment_harness.library import works

    video_dir = works.video_recap_dir(sid, chapter)
    video_dir.mkdir(parents=True)
    (video_dir / "out.mp4").write_bytes(b"video")
    (video_dir / "clips").mkdir()
    (video_dir / "clips" / "clip-00-0.mp4").write_bytes(b"clip")
    return video_dir


def test_push_video_uploads_only_final_mp4s(root, monkeypatch):
    client = _fake_client(monkeypatch)
    _seed_video(root, "s1")
    count = store.push_video(_config(), "s1", "ch-1", log=lambda m: None)
    assert count == 1
    assert client.uploads == ["works/s1/chapters/ch-1/video-recap/out.mp4"]


def test_pull_video_restores_missing_file(root, monkeypatch):
    _fake_client(
        monkeypatch,
        {"works/s1/chapters/ch-1/video-recap/out.mp4": b"restored-video"},
    )
    restored = store.pull_video(_config(), "s1", "ch-1")
    assert [p.name for p in restored] == ["out.mp4"]
    assert restored[0].read_bytes() == b"restored-video"


def test_pull_video_empty_when_absent(root, monkeypatch):
    _fake_client(monkeypatch)
    assert store.pull_video(_config(), "s1", "ch-9") == []


def test_push_video_includes_narration_subdir(root, monkeypatch):
    from entertainment_harness.library import works

    client = _fake_client(monkeypatch)
    video_dir = _seed_video(root, "s1")
    narration = works.video_narration_dir("s1", "ch-1")
    narration.mkdir(parents=True)
    (narration / "out.mp4").write_bytes(b"narration-video")
    count = store.push_video(_config(), "s1", "ch-1", log=lambda m: None)
    assert count == 2
    assert client.uploads == [
        "works/s1/chapters/ch-1/video-recap/out.mp4",
        "works/s1/chapters/ch-1/video-narration/out.mp4",
    ]


def test_pull_video_restores_narration_subdir(root, monkeypatch):
    from entertainment_harness.library import works

    _fake_client(
        monkeypatch,
        {
            "works/s1/chapters/ch-1/video-recap/out.mp4": b"recap-video",
            "works/s1/chapters/ch-1/video-narration/out.mp4": b"narration-video",
        },
    )
    restored = store.pull_video(_config(), "s1", "ch-1")
    rel = sorted(
        str(p.relative_to(works.chapter_dir("s1", "ch-1"))) for p in restored
    )
    assert rel == ["video-narration/out.mp4", "video-recap/out.mp4"]
    assert (
        works.video_narration_dir("s1", "ch-1") / "out.mp4"
    ).read_bytes() == b"narration-video"
