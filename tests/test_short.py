"""Short-form (TikTok) whole-work video tests: script prompt/parse, caption
cards, page flattening across chapters, and build_short orchestration with
faked models/TTS/assembly. No network, Ollama, or ffmpeg.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from PIL import Image

from entertainment_harness import db
from entertainment_harness.library import works
from entertainment_harness.config import Config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection
from entertainment_harness.video import short
from entertainment_harness.video.cards import render_cards
from entertainment_harness.video.script import Segment, VideoError

GB = 10**9

SCRIPT_JSON = (
    '[{"text": "What if your grandpa was the world\'s greatest mage?",'
    ' "moment": "Hook"},'
    ' {"text": "Shin grows up training in the woods.", "moment": "Training"}]'
)


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


def _write_page(path: Path, size: tuple[int, int] = (100, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


class FakeAdapter:
    def __init__(self, output: str, model_name: str) -> None:
        self.output = output
        self.model_name = model_name
        self.calls = 0

    def supports(self, model: str) -> ModelInfo:
        return ModelInfo(model, "fake", 4.0, "Q8_0", 3 * GB)

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def list_available(self) -> list[ModelInfo]:
        return [self.supports(self.model_name)]

    def generate(self, model, prompt, images=None) -> str:
        self.calls += 1
        return self.output


class FakeEngine:
    name = "fake"
    default_voice = "fake-voice"

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def synthesize(self, text: str, voice: str, dest: Path) -> None:
        self.synthesized.append(text)
        _write_wav(dest, seconds=0.5)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A recapped two-chapter manga with pages on disk."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    for cid, num, pages in (("ch-1", 1.0, 3), ("ch-2", 2.0, 2)):
        conn.execute(
            "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
            " fetched_at) VALUES (?, 's1', ?, 'Ch', 'en', ?, 'now')",
            (cid, num, pages),
        )
        conn.execute(
            "INSERT INTO recaps (chapter_id, summary, created_at, model)"
            " VALUES (?, 'recap', 'now', 'fake')",
            (cid,),
        )
        for i in range(1, pages + 1):
            _write_page(works.source_dir("s1", cid) / f"page-{i:03d}.jpg")
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES ('s1', 'Shin trains under Merlin and joins the academy.', 2.0)"
    )
    conn.commit()

    text = FakeAdapter(SCRIPT_JSON, "fake-text:4b")
    vision = FakeAdapter('{"0": [1], "1": [4]}', "fake-vision:8b")
    info = ModelInfo("fake:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        short, "get_text_model", lambda c, p: Selection(adapter=text, info=info)
    )
    monkeypatch.setattr(
        short, "get_vision_model", lambda c, p: Selection(adapter=vision, info=info)
    )
    engine = FakeEngine()
    monkeypatch.setattr(short, "get_engine", lambda name, config=None: engine)

    assembled: list[dict] = []

    def fake_assemble(segments, page_paths, workdir, resolution, log,
                      min_page_seconds=2.5, credits=None):
        assembled.append(
            {"n": len(segments), "resolution": resolution,
             "min_page_seconds": min_page_seconds, "pages": list(page_paths)}
        )
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(short, "assemble", fake_assemble)

    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()
    profile = HardwareProfile("fake", 24 * GB, 16 * GB, "cpu")
    yield conn, series, profile, text, vision, engine, assembled
    conn.close()


def test_short_script_prompt(harness):
    class RecordingAdapter(FakeAdapter):
        def __init__(self):
            super().__init__(SCRIPT_JSON, "fake-text:4b")
            self.prompts: list[str] = []

        def generate(self, model, prompt, images=None) -> str:
            self.prompts.append(prompt)
            return super().generate(model, prompt, images)

    adapter = RecordingAdapter()
    segments = short.generate_short_script(
        adapter, "fake-text:4b", "The summary.", "Test Manga"
    )
    assert segments[0].text.startswith("What if")
    prompt = adapter.prompts[0]
    assert "The summary." in prompt
    assert "hook" in prompt.lower()
    assert '"Test Manga"' in prompt


def test_flatten_pages_orders_across_chapters(harness):
    conn, *_ = harness
    paths = short._flatten_pages(conn, "s1")
    assert [p.name for p in paths] == [
        "page-001.jpg", "page-002.jpg", "page-003.jpg",  # ch-1
        "page-001.jpg", "page-002.jpg",  # ch-2
    ]
    assert paths[0].parent.parent.name == "ch-1"
    assert paths[3].parent.parent.name == "ch-2"


def test_build_short_end_to_end_and_cached(harness):
    conn, series, profile, text, vision, engine, assembled = harness
    out = short.build_short(conn, series, Config(), profile, log=lambda m: None)

    assert out == works.video_file_path("s1", None, "tiktok")
    assert engine.synthesized == [
        "What if your grandpa was the world's greatest mage?",
        "Shin grows up training in the woods.",
    ]
    assert assembled[0]["resolution"] == (1080, 1920)
    assert assembled[0]["min_page_seconds"] == short.MIN_PAGE_SECONDS
    assert vision.calls == 1  # 5 pages -> single contact sheet
    row = conn.execute(
        "SELECT * FROM videos WHERE series_id = 's1'"
    ).fetchone()
    assert row["from_chapter"] is None and row["to_chapter"] is None
    assert row["duration_s"] == pytest.approx(1.0)
    state = json.loads((out.parent / "render_state.json").read_text())
    assert state == {"format": "tiktok", "video_gen": "local", "credits": True}

    # second run: everything cached
    out2 = short.build_short(conn, series, Config(), profile, log=lambda m: None)
    assert out2 == out
    assert text.calls == 1
    assert vision.calls == 1
    assert len(assembled) == 1
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM videos WHERE series_id = 's1'"
    ).fetchone()["n"]
    assert count == 1


def test_build_short_requires_recaps(harness):
    conn, series, profile, *_ = harness
    conn.execute("DELETE FROM series_context WHERE series_id = 's1'")
    conn.execute("DELETE FROM recaps")
    conn.commit()
    with pytest.raises(VideoError, match="eh recap"):
        short.build_short(conn, series, Config(), profile, log=lambda m: None)


def test_build_short_allows_online_summary_without_recaps(harness):
    conn, series, profile, text, vision, engine, assembled = harness
    conn.execute("DELETE FROM recaps")
    conn.execute(
        "INSERT INTO online_summaries (series_id, provider, query, summary,"
        " sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("s1", "duckduckgo", "Test Manga", "Online summary.", "[]", "now"),
    )
    conn.commit()

    out = short.build_short(conn, series, Config(), profile, log=lambda m: None)
    assert out.exists()
    assert text.calls == 1
    assert vision.calls == 1


def test_render_cards(tmp_path):
    segments = [
        Segment(index=0, text="Hook line here.", moment="Hook"),
        Segment(index=1, text="Second beat.", moment="Training"),
    ]
    cards = render_cards(segments, "Test Book", tmp_path, size=(1080, 1920))
    assert len(cards) == 2
    for card in cards:
        with Image.open(card) as img:
            assert img.size == (1080, 1920)
    # cached second call returns the same paths without rewriting
    again = render_cards(segments, "Test Book", tmp_path, size=(1080, 1920))
    assert again == cards


def test_build_short_book_uses_cards(harness, monkeypatch):
    conn, series, profile, text, vision, engine, assembled = harness
    conn.execute("UPDATE series SET kind = 'book' WHERE id = 's1'")
    conn.commit()
    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()

    out = short.build_short(conn, series, Config(), profile, log=lambda m: None)
    assert vision.calls == 0  # no page picking for books
    cards = assembled[0]["pages"]
    assert len(cards) == 2
    assert all("cards" in p.parts and p.exists() for p in cards)
    assert out.exists()


def test_short_script_prompt_includes_steering():
    class RecordingAdapter:
        def __init__(self):
            self.prompts: list[str] = []
        def generate(self, model, prompt, images=None):
            self.prompts.append(prompt)
            return SCRIPT_JSON

    adapter = RecordingAdapter()
    segments = short.generate_short_script(
        adapter, "fake", "Summary.", "Test Manga", steering_prompt="focus on comedy"
    )
    assert segments[0].text.startswith("What if")
    prompt = adapter.prompts[0]
    assert "focus on comedy" in prompt
    assert "Steering direction" in prompt


def test_build_short_with_runway_provider(harness, monkeypatch, tmp_path):
    conn, series, profile, text, vision, engine, assembled = harness

    generated_clips: list[Path] = []

    class FakeVideoGen:
        name = "runway"
        animated = True
        def generate_segment(self, image, segment, duration, workdir):
            clip = workdir / f"seg-runway-{segment.index:02d}.mp4"
            clip.write_bytes(b"runway-clip")
            generated_clips.append(clip)
            return clip

    monkeypatch.setattr(
        short, "get_video_gen_provider", lambda name, cfg: FakeVideoGen()
    )
    monkeypatch.setattr(short, "video_duration", lambda path: 0.5)

    muxed: dict = {}

    def fake_mux_clips(segments, clips, workdir, log, credits=None):
        out = workdir / "out.mp4"
        out.write_bytes(b"muxed-runway")
        muxed["clips"] = clips
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(short, "mux_clips", fake_mux_clips)

    out = short.build_short(
        conn, series, Config(), profile,
        video_gen_provider="runway", log=lambda m: None,
    )

    assert out.exists()
    assert len(generated_clips) == 2
    assert len(muxed["clips"]) == 2
    row = conn.execute(
        "SELECT * FROM videos WHERE series_id = 's1'"
    ).fetchone()
    assert row["video_gen_provider"] == "runway"
    state = json.loads((out.parent / "render_state.json").read_text())
    assert state["video_gen"] == "runway"
