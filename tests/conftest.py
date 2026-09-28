"""Shared fakes and fixtures for the recap pipeline tests (all detail
grains, including full = the narration grain).

Only the recap variants live here: the translate, video, short, and
search test files use adapters with a different call-recording API (int
counters, canned output strings, real image writes) and keep their own
local fakes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Rich snapshots COLUMNS into Console instances at construction, and the CLI
# builds its console at import time. CI runners export COLUMNS=0, which would
# freeze every console at width 0 and swallow all output — pin a wide
# terminal before any test module imports the package.
os.environ["COLUMNS"] = "200"

# Typer forces terminal/color output when GITHUB_ACTIONS is set (its
# rich_utils FORCE_TERMINAL), which splits flag names across styled spans
# ('-\x1b[0m\x1b[1;36m-thinking') and breaks plain-text help assertions.
# This must be set before typer.rich_utils is imported.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

from entertainment_harness import db
from entertainment_harness.pipelines import recap as recap_mod
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection

GB = 10**9


@pytest.fixture(autouse=True)
def disable_auto_migrate(monkeypatch):
    """Tests control their own data layout; skip startup auto-migration."""
    monkeypatch.setenv("EH_NO_AUTO_MIGRATE", "1")


@pytest.fixture(autouse=True)
def isolate_cache_dir(tmp_path, monkeypatch):
    """Keep cache_dir() hermetic: never touch the real ~/.cache."""
    monkeypatch.setenv("EH_CACHE_DIR", str(tmp_path / "cache"))


class FakeAdapter:
    """In-memory ModelAdapter: records calls, returns canned text."""

    name = "fake"

    def __init__(self, model_name: str, batch_label: str = "summary") -> None:
        self.model_name = model_name
        self.batch_label = batch_label
        self.calls: list[dict] = []
        self.ensured: list[str] = []

    def supports(self, model: str) -> ModelInfo:
        return ModelInfo(model, self.name, params=8.0, quant="Q8_0", size_bytes=5 * GB)

    def ensure(self, model: str, quant: str | None = None) -> None:
        self.ensured.append(model)

    def list_available(self) -> list[ModelInfo]:
        return [self.supports(self.model_name)]

    def generate(self, model: str, prompt: str, images: list[Path] | None = None) -> str:
        self.calls.append({"model": model, "prompt": prompt, "images": images or []})
        if images:
            return f"[batch {self.batch_label} of {len(images)} pages]"
        return f"[text output {len(self.calls)}]"


class FakeClient:
    """Stands in for MangaDexClient.download_pages: writes empty page files."""

    def __init__(self, pages_per_chapter: dict[str, int]) -> None:
        self.pages_per_chapter = pages_per_chapter
        self.downloaded: list[str] = []

    def download_pages(self, chapter_id: str, dest_dir: Path) -> list[Path]:
        self.downloaded.append(chapter_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return [
            dest_dir / f"page-{i:03d}.jpg"
            for i in range(1, self.pages_per_chapter[chapter_id] + 1)
        ]


class ScriptedJudge(FakeAdapter):
    """Judge adapter with per-stage verdict queues (chapter output vs
    context calls), defaulting to pass when the queue runs out."""

    def __init__(self, verdicts=(), context_verdicts=()) -> None:
        super().__init__("fake-judge:4b")
        self.verdicts = list(verdicts)
        self.context_verdicts = list(context_verdicts)

    def generate(self, model, prompt, images=None) -> str:
        self.calls.append({"model": model, "prompt": prompt, "images": []})
        queue = (
            self.context_verdicts
            if "Updated story so far" in prompt
            else self.verdicts
        )
        return queue.pop(0) if queue else '{"pass": true}'


@pytest.fixture
def patch_judge(monkeypatch):
    """Return a helper that installs a fake judge into recap.setup_models."""

    def patch(judge) -> None:
        info = ModelInfo("fake-judge:4b", "fake", 4.0, "Q8_0", 3 * GB)
        monkeypatch.setattr(
            recap_mod, "get_judge_model",
            lambda config, profile: Selection(adapter=judge, info=info),
        )

    return patch


@pytest.fixture
def batch_label() -> str:
    """Canned label for per-image-batch outputs; overridden per test module."""
    return "summary"


@pytest.fixture
def harness(tmp_path, monkeypatch, batch_label, patch_judge):
    """Manga series s1 (ch-1: 9 pages, ch-2: 4 pages) with fake models.

    Model resolution lives in recap.setup_models, so every grain of the
    recap pipeline is served by patching recap_mod here.
    """
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, ?, ?, ?, 'now')",
        [("ch-1", 1.0, "Ch 1", "en", 9), ("ch-2", 2.0, "Ch 2", "en", 4)],
    )
    conn.commit()

    vision = FakeAdapter("fake-vision:8b", batch_label=batch_label)
    text = FakeAdapter("fake-text:4b", batch_label=batch_label)
    info_v = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    info_t = ModelInfo("fake-text:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        recap_mod, "get_vision_model",
        lambda config, profile: Selection(adapter=vision, info=info_v),
    )
    monkeypatch.setattr(
        recap_mod, "get_text_model",
        lambda config, profile: Selection(adapter=text, info=info_t),
    )
    patch_judge(ScriptedJudge())
    profile = HardwareProfile("fake chip", 24 * GB, 16 * GB, "cpu")
    client = FakeClient({"ch-1": 9, "ch-2": 4})
    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()
    yield conn, vision, text, client, series, profile
    conn.close()
