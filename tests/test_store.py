"""Store tests: video backup to a HF dataset repo. huggingface_hub is faked —
no network, no token needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entertainment_harness import store
from entertainment_harness.store import hf as store_hf
from entertainment_harness.config import Config, load_config


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    return tmp_path


def _config(repo: str | None = "me/manga-archive") -> Config:
    config = Config()
    config.store.repo = repo
    return config


def _seed_video(root: Path, sid: str, chapter: str = "ch-1") -> Path:
    from entertainment_harness.library import works

    video_dir = works.video_recap_dir(sid, chapter)
    video_dir.mkdir(parents=True)
    (video_dir / "out.mp4").write_bytes(b"video")
    (video_dir / "script.json").write_text("{}")
    clips = video_dir / "clips"
    clips.mkdir()
    (clips / "clip-00-0.mp4").write_bytes(b"clip")
    return video_dir


class FakeApi:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.uploads: list[dict] = []

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        self.created.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private}
        )

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None):
        self.uploads.append(
            {"repo_id": repo_id, "path_in_repo": path_in_repo}
        )


def _fake_api(monkeypatch) -> FakeApi:
    api = FakeApi()
    monkeypatch.setattr(store_hf, "_api", lambda: api)
    return api


# --- config -------------------------------------------------------------------


def test_config_loads_store_repo(root):
    (root / "config.toml").write_text('[store]\nrepo = "me/manga-archive"\n')
    assert load_config().store.repo == "me/manga-archive"


def test_repo_required(root):
    _seed_video(root, "s1")
    with pytest.raises(store.StoreError, match="No store repo configured"):
        store.push_video(_config(repo=None), "s1", "ch-1", log=lambda m: None)


def test_is_configured(root):
    assert not store.is_configured(Config())
    assert store.is_configured(_config())


# --- video push/pull ----------------------------------------------------------


def test_push_video_uploads_only_final_mp4s(root, monkeypatch):
    api = _fake_api(monkeypatch)
    _seed_video(root, "s1")
    count = store.push_video(_config(), "s1", "ch-1", log=lambda m: None)
    assert count == 1
    assert api.created == [
        {"repo_id": "me/manga-archive", "repo_type": "dataset", "private": True}
    ]
    # intermediate clips and scripts are not pushed
    assert [u["path_in_repo"] for u in api.uploads] == [
        "works/s1/chapters/ch-1/video-recap/out.mp4"
    ]


def test_push_video_requires_render(root, monkeypatch):
    _fake_api(monkeypatch)
    with pytest.raises(store.StoreError, match="No rendered video"):
        store.push_video(_config(), "s1", "ch-1", log=lambda m: None)


def test_pull_video_restores_missing_file(root, monkeypatch):
    _fake_api(monkeypatch)

    def fake_snapshot(repo, repo_type=None, allow_patterns=None, local_dir=None):
        assert allow_patterns == "works/s1/chapters/ch-1/*"
        dest = Path(local_dir) / "works/s1/chapters/ch-1/video-recap"
        dest.mkdir(parents=True)
        (dest / "out.mp4").write_bytes(b"restored-video")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    restored = store.pull_video(_config(), "s1", "ch-1")
    assert [p.name for p in restored] == ["out.mp4"]
    assert restored[0].read_bytes() == b"restored-video"


def test_pull_video_empty_when_absent(root, monkeypatch):
    _fake_api(monkeypatch)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: str(root)
    )
    assert store.pull_video(_config(), "s1", "ch-9") == []


def test_push_video_includes_narration_subdir(root, monkeypatch):
    from entertainment_harness.library import works

    api = _fake_api(monkeypatch)
    video_dir = _seed_video(root, "s1")
    narration = works.video_narration_dir("s1", "ch-1")
    narration.mkdir(parents=True)
    (narration / "out.mp4").write_bytes(b"narration-video")
    count = store.push_video(_config(), "s1", "ch-1", log=lambda m: None)
    assert count == 2
    assert [u["path_in_repo"] for u in api.uploads] == [
        "works/s1/chapters/ch-1/video-recap/out.mp4",
        "works/s1/chapters/ch-1/video-narration/out.mp4",
    ]


def test_pull_video_restores_narration_subdir(root, monkeypatch):
    _fake_api(monkeypatch)

    def fake_snapshot(repo, repo_type=None, allow_patterns=None, local_dir=None):
        dest = Path(local_dir) / "works/s1/chapters/ch-1"
        (dest / "video-recap").mkdir(parents=True)
        (dest / "video-narration").mkdir(parents=True)
        (dest / "video-recap" / "out.mp4").write_bytes(b"recap-video")
        (dest / "video-narration" / "out.mp4").write_bytes(b"narration-video")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    restored = store.pull_video(_config(), "s1", "ch-1")
    rel = sorted(
        str(p.relative_to(root / "works" / "s1" / "chapters" / "ch-1")) for p in restored
    )
    assert rel == ["video-narration/out.mp4", "video-recap/out.mp4"]
