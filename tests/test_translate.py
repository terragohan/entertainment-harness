"""Translate-pipeline tests: fake vision adapter (canned bubble JSON),
temp-dir SQLite. No network, no models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import cv2

from entertainment_harness import db
from entertainment_harness.config import Config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection
from entertainment_harness.pipelines.judge import RenderIssue
from entertainment_harness.pipelines.translate import (
    BoxOverride,
    Bubble,
    TranslateError,
    _bubble_mask,
    _overrides_from_issues,
    _render_with_judge,
    _snap_box,
    parse_bubbles,
    render_page,
    translate_chapters,
)

GB = 10**9


# --- parse_bubbles ------------------------------------------------------------


def test_parse_bubbles_clean():
    raw = '[{"box": [0.1, 0.2, 0.3, 0.4], "original": "OLA",'
    raw += ' "translation": "HELLO"}]'
    bubbles = parse_bubbles(raw)
    assert bubbles == [Bubble([0.1, 0.2, 0.3, 0.4], "OLA", "HELLO")]


def test_parse_bubbles_tolerates_fences_and_preamble():
    raw = 'Here you go:\n```json\n[{"box": [0, 0, 0.5, 0.5],'
    raw += ' "original": "x", "translation": "y"}]\n```'
    assert len(parse_bubbles(raw)) == 1


def test_parse_bubbles_drops_garbage():
    raw = """[
      {"box": [0.1, 0.1, 0.4, 0.4], "original": "a", "translation": "ok"},
      {"box": [0.5, 0.5, 0.2, 0.2], "original": "b", "translation": "inverted"},
      {"box": [-0.5, 0.0, 2.0, 0.4], "original": "c", "translation": "clamped"},
      {"box": [0.001, 0.001, 0.002, 0.002], "original": "d", "translation": "speck"},
      {"box": [0.1, 0.1, 0.2, 0.2], "original": "e", "translation": ""},
      {"translation": "no box"},
      "not a dict"
    ]"""
    bubbles = parse_bubbles(raw)
    assert [b.translation for b in bubbles] == ["ok", "clamped"]
    assert bubbles[1].box == [0.0, 0.0, 1.0, 0.4]


def test_parse_bubbles_rejects_non_array():
    with pytest.raises(TranslateError):
        parse_bubbles('{"note": "no array here"}')
    with pytest.raises(TranslateError):
        parse_bubbles("no json here")


def test_parse_bubbles_repairs_newlines_in_strings():
    raw = '[{"box": [0.1, 0.2, 0.3, 0.4], "original": "OLA", "translation": "Hello\nworld"}]'
    bubbles = parse_bubbles(raw)
    assert len(bubbles) == 1
    assert bubbles[0].translation == "Hello\nworld"


def test_parse_bubbles_repairs_trailing_commas():
    raw = '[{"box": [0.1, 0.2, 0.3, 0.4], "original": "a", "translation": "b"},]'
    bubbles = parse_bubbles(raw)
    assert len(bubbles) == 1


def test_parse_bubbles_repairs_fenced_malformed_json():
    raw = 'Here:\n```json\n[{"box": [0.1,0.2,0.3,0.4], "original": "a", "translation": "line1\nline2"},]\n```'
    bubbles = parse_bubbles(raw)
    assert len(bubbles) == 1
    assert bubbles[0].translation == "line1\nline2"


def test_parse_bubbles_salvages_individual_objects_when_array_is_broken():
    # Missing comma between objects and a trailing comma
    raw = '[{"box": [0.1, 0.2, 0.3, 0.4], "original": "a", "translation": "ok"}' \
          ' {"box": [0.5, 0.5, 0.6, 0.6], "original": "b", "translation": "bad",}' \
          ' {"box": [0.7, 0.7, 0.8, 0.8], "original": "c", "translation": "fine"}]'
    bubbles = parse_bubbles(raw)
    assert len(bubbles) == 3
    assert [b.translation for b in bubbles] == ["ok", "bad", "fine"]


def test_parse_bubbles_includes_raw_snippet_on_failure():
    raw = '[not json at all]'
    with pytest.raises(TranslateError) as exc_info:
        parse_bubbles(raw, page_label="page 3 of chapter 7")
    message = str(exc_info.value)
    assert "page 3 of chapter 7" in message
    assert "Raw snippet:" in message
    assert raw in message


def test_parse_bubbles_returns_empty_list_for_empty_array():
    assert parse_bubbles("[]") == []
    assert parse_bubbles("```json\n[]\n```") == []


def test_parse_bubbles_normalizes_pixel_coordinates():
    # Some models ignore "normalized 0.0-1.0" and return pixel coords; with
    # image_size given those are normalized instead of clamped to nothing.
    raw = '[{"box": [45, 78, 321, 290], "original": "OLA", "translation": "HI"}]'
    bubbles = parse_bubbles(raw, image_size=(563, 800))
    assert len(bubbles) == 1
    box = bubbles[0].box
    assert box == pytest.approx([45 / 563, 78 / 800, 321 / 563, 290 / 800])
    # without image_size the old clamping behavior is kept
    clamped = parse_bubbles(raw)
    assert clamped == []  # collapses to a degenerate box and is dropped


# --- _snap_box ------------------------------------------------------------------


def test_snap_box_finds_white_bubble():
    img = np.full((200, 200), 100, dtype=np.uint8)  # gray art
    cv2.rectangle(img, (100, 50), (160, 110), 255, -1)  # white bubble
    (px1, py1, px2, py2), is_dark = _snap_box(img, [0.3, 0.1, 1.0, 0.9])
    assert not is_dark
    # cv2.rectangle's bottom-right is inclusive -> component spans 100..160
    assert (px1, py1) == (100, 50) and (px2, py2) == (161, 111)


def test_snap_box_dark_caption_kept_and_flagged():
    img = np.full((200, 200), 220, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 180), 20, -1)  # dark caption
    box, is_dark = _snap_box(img, [0.1, 0.1, 0.9, 0.9])
    assert is_dark
    assert box == (20, 20, 180, 180)  # 200px * [0.1, 0.9] box, unmodified


# --- _bubble_mask --------------------------------------------------------------


def test_bubble_mask_finds_bright_bubble():
    img = np.full((100, 100), 100, dtype=np.uint8)  # gray art
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)  # white bubble
    mask, is_dark = _bubble_mask(img)
    assert not is_dark
    assert cv2.countNonZero(mask) == 41 * 41  # filled rect from 30..70 inclusive


def test_bubble_mask_flags_dark_caption():
    img = np.full((100, 100), 30, dtype=np.uint8)
    mask, is_dark = _bubble_mask(img)
    assert is_dark
    assert cv2.countNonZero(mask) == img.size


def test_bubble_mask_falls_back_to_full_crop():
    img = np.full((100, 100), 180, dtype=np.uint8)  # no bright component, light
    mask, is_dark = _bubble_mask(img)
    assert not is_dark
    assert cv2.countNonZero(mask) == img.size


# --- render_page ----------------------------------------------------------------


def test_render_page_overlay(tmp_path):
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    src = tmp_path / "page-001.jpg"
    cv2.imwrite(str(src), img)
    dest = render_page(
        src, tmp_path / "out" / "page-001.png",
        [Bubble([0.1, 0.1, 0.6, 0.4], "OLA", "Hello there, this is a test.")],
    )
    from PIL import Image

    out = Image.open(dest)
    assert out.size == (300, 400)  # same dimensions
    assert dest.suffix == ".png"


# --- translate_chapters ---------------------------------------------------------


class FakeAdapter:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0
        self.prompts: list[str] = []

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def generate(self, model, prompt, images=None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.output


class ScriptedVision(FakeAdapter):
    """Vision adapter returning a different canned output per call."""

    def __init__(self, outputs: list[str]) -> None:
        super().__init__(outputs[0])
        self.outputs = list(outputs)

    def generate(self, model, prompt, images=None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.outputs.pop(0) if self.outputs else BUBBLES_JSON


class ScriptedJudge:
    """Judge adapter with a verdict queue, defaulting to pass."""

    def __init__(self, verdicts=()) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0
        self.prompts: list[str] = []

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def generate(self, model, prompt, images=None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.verdicts.pop(0) if self.verdicts else '{"pass": true}'


class FakeClient:
    def __init__(self, pages: int = 2) -> None:
        self.pages = pages

    def download_pages(self, chapter_id, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(1, self.pages + 1):
            p = dest_dir / f"page-{i:03d}.jpg"
            cv2.imwrite(str(p), np.full((100, 80, 3), 255, dtype=np.uint8))
            paths.append(p)
        return paths


BUBBLES_JSON = (
    '[{"box": [0.1, 0.1, 0.5, 0.4], "original": "BOM DIA"}]'
)

TRANSLATIONS_JSON = '[{"translation": "Good morning"}]'


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, ?, ?, 2, 'now')",
        [("ch-pt", 1.0, "Ch 1", "pt-br"), ("ch-en", 2.0, "Ch 2", "en")],
    )
    conn.commit()

    vision_adapter = FakeAdapter(BUBBLES_JSON)
    vision_info = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    translation_adapter = FakeAdapter(TRANSLATIONS_JSON)
    translation_info = ModelInfo("fake-translation:4b", "fake", 4.0, "Q8_0", 3 * GB)
    import entertainment_harness.pipelines.translate as translate_mod

    monkeypatch.setattr(
        translate_mod, "get_vision_model",
        lambda config, profile: Selection(adapter=vision_adapter, info=vision_info),
    )
    monkeypatch.setattr(
        translate_mod, "get_translation_model",
        lambda config, profile: Selection(adapter=translation_adapter, info=translation_info),
    )
    judge = ScriptedJudge()
    judge_info = ModelInfo("fake-judge:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        translate_mod, "get_judge_model",
        lambda config, profile: Selection(adapter=judge, info=judge_info),
    )
    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()
    profile = HardwareProfile("fake", 24 * GB, 16 * GB, "cpu")
    yield conn, series, profile, vision_adapter, translation_adapter
    conn.close()


def test_translates_first_untranslated_chapter(harness, tmp_path):
    conn, series, profile, vision_adapter, translation_adapter = harness
    done = translate_chapters(
        conn, series, Config(), profile, client=FakeClient(), log=lambda m: None
    )
    assert done == ["ch-pt"]
    out = tmp_path / "manga" / "s1" / "ch-pt" / "translated"
    assert (out / "page-001.png").exists()
    assert (out / "page-002.png").exists()
    record = json.loads((out / "translation.json").read_text())
    assert record["target_lang"] == "en"
    assert "fake-vision:8b" in record["model"]
    assert "fake-translation:4b" in record["model"]
    assert record["pages"][0]["bubbles"][0]["translation"] == "Good morning"
    row = conn.execute(
        "SELECT * FROM translations WHERE chapter_id = 'ch-pt'"
    ).fetchone()
    assert row["pages"] == 2


def test_skips_target_language_chapters(harness):
    conn, series, profile, vision_adapter, translation_adapter = harness
    done = translate_chapters(
        conn, series, Config(), profile, all_chapters=True,
        client=FakeClient(), log=lambda m: None,
    )
    assert done == ["ch-pt"]  # ch-en skipped: already English


def test_cached_pages_not_retranslated(harness):
    conn, series, profile, vision_adapter, translation_adapter = harness
    translate_chapters(conn, series, Config(), profile, client=FakeClient(),
                       log=lambda m: None)
    done = translate_chapters(conn, series, Config(), profile, force=False,
                              client=FakeClient(), log=lambda m: None)
    assert done == []  # translations row exists -> nothing pending
    # 2 pages * 2 vision calls (extraction + render judge) on the first run
    assert vision_adapter.calls == 4
    assert translation_adapter.calls == 2  # one translation call per page, first run only


def test_force_retranslates(harness):
    conn, series, profile, vision_adapter, translation_adapter = harness
    translate_chapters(conn, series, Config(), profile, client=FakeClient(),
                       log=lambda m: None)
    done = translate_chapters(conn, series, Config(), profile, chapter_num=1,
                              force=True, client=FakeClient(),
                              log=lambda m: None)
    assert done == ["ch-pt"]
    # 2 runs * 2 pages * 2 vision calls (extraction + render judge)
    assert vision_adapter.calls == 8
    assert translation_adapter.calls == 4


def test_unknown_chapter_raises(harness):
    conn, series, profile, _, _ = harness
    with pytest.raises(TranslateError, match="not synced"):
        translate_chapters(conn, series, Config(), profile, chapter_num=99,
                           client=FakeClient(), log=lambda m: None)


# --- translation judge loop ------------------------------------------------------

FAIL_LANG = '{"pass": false, "issues": ["translation left in pt-br: BOM DIA"]}'
GOOD_JSON = (
    '[{"box": [0.1, 0.1, 0.5, 0.4], "original": "BOM DIA"}]'
)
DEGENERATE_JSON = (
    '[{"box": [0.1, 0.1, 0.5, 0.4], "original": "BOM DIA"}]'
)


def _patch_judge(monkeypatch, judge):
    import entertainment_harness.pipelines.translate as translate_mod

    info = ModelInfo("fake-judge:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        translate_mod, "get_judge_model",
        lambda config, profile: Selection(adapter=judge, info=info),
    )


def test_judge_sees_original_translation_pairs(harness):
    conn, series, profile, _, _ = harness
    translate_chapters(conn, series, Config(), profile, client=FakeClient(),
                       log=lambda m: None)
    # judge called once per page (2 pages)
    import entertainment_harness.pipelines.translate as translate_mod
    judge = translate_mod.get_judge_model(Config(), profile).adapter
    assert judge.calls == 2
    assert "BOM DIA" in judge.prompts[0] and "Good morning" in judge.prompts[0]
    assert "pt-br" in judge.prompts[0]


def test_judge_rejection_retries_with_feedback(harness, monkeypatch):
    conn, series, profile, vision_adapter, translation_adapter = harness
    judge = ScriptedJudge(verdicts=[FAIL_LANG])  # page 1 fails once, then passes
    _patch_judge(monkeypatch, judge)
    done = translate_chapters(conn, series, Config(), profile,
                              client=FakeClient(), log=lambda m: None)
    assert done == ["ch-pt"]
    # page 1: extract + translate + render judge, retry translate;
    # page 2: extract + translate + render judge
    assert vision_adapter.calls == 4
    assert translation_adapter.calls == 3
    assert "rejected by the evaluator" in translation_adapter.prompts[1]
    assert "translation left in pt-br" in translation_adapter.prompts[1]


def test_judge_persistent_failure_keeps_first(harness, monkeypatch, tmp_path):
    conn, series, profile, _, _ = harness
    import entertainment_harness.pipelines.translate as translate_mod

    vision = ScriptedVision([GOOD_JSON, DEGENERATE_JSON, DEGENERATE_JSON])
    vision_info = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    monkeypatch.setattr(
        translate_mod, "get_vision_model",
        lambda config, profile: Selection(adapter=vision, info=vision_info),
    )
    # Translation model returns good then degenerate, so the first translation is kept.
    translation = ScriptedVision([TRANSLATIONS_JSON, '[{"translation": "I cannot translate this."}]', '[{"translation": "I cannot translate this."}]'])
    translation_info = ModelInfo("fake-translation:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        translate_mod, "get_translation_model",
        lambda config, profile: Selection(adapter=translation, info=translation_info),
    )
    judge = ScriptedJudge(verdicts=[FAIL_LANG] * 10)  # never passes
    _patch_judge(monkeypatch, judge)
    done = translate_chapters(
        conn, series, Config(), profile, client=FakeClient(pages=1),
        log=lambda m: None,
    )
    assert done == ["ch-pt"]  # stored anyway, with a warning
    assert translation.calls == 3  # MAX_ATTEMPTS, bounded
    record = json.loads(
        (tmp_path / "manga" / "s1" / "ch-pt" / "translated"
         / "translation.json").read_text()
    )
    # the FIRST attempt is kept: retries risk over-complying with a
    # mistaken critique (feedback poisoning)
    assert record["pages"][0]["bubbles"][0]["translation"] == "Good morning"


def test_no_judge_skips_judging(harness, monkeypatch):
    conn, series, profile, vision_adapter, translation_adapter = harness
    judge = ScriptedJudge()
    _patch_judge(monkeypatch, judge)
    done = translate_chapters(conn, series, Config(), profile, thinking="low",
                              client=FakeClient(), log=lambda m: None)
    assert done == ["ch-pt"]
    assert judge.calls == 0
    assert vision_adapter.calls == 2
    assert translation_adapter.calls == 2


def test_thinking_high_uses_text_judge_with_more_attempts(harness, monkeypatch):
    """On high the text judge still judges translations (attaching the page
    image made the vision judge degenerate), with a 5-attempt budget."""
    conn, series, profile, vision_adapter, _ = harness
    judge = ScriptedJudge(verdicts=[FAIL_LANG] * 10)  # never passes
    _patch_judge(monkeypatch, judge)
    done = translate_chapters(conn, series, Config(), profile, thinking="high",
                              client=FakeClient(), log=lambda m: None)
    assert done == ["ch-pt"]
    assert judge.calls == 10  # 2 pages * 5 attempts
    # the vision model only ever extracts — it never judges
    assert all("strict evaluator" not in p for p in vision_adapter.prompts)


# --- render-quality judge and adjustments -------------------------------------


def test_overrides_from_issues_expands_box_by_direction():
    issues = [RenderIssue(0, "original_text_visible", ["left", "bottom"])]
    overrides = _overrides_from_issues(issues, {})
    assert overrides[0].pad_left == 25
    assert overrides[0].pad_bottom == 25
    assert overrides[0].pad_right == 0
    assert overrides[0].pad_top == 0


def test_overrides_from_issues_are_cumulative_and_capped():
    base = {0: BoxOverride(pad_left=60)}
    issues = [RenderIssue(0, "original_text_visible", ["left"])]
    overrides = _overrides_from_issues(issues, base)
    # 60 + 25 would be 85, capped at MAX_EXTRA_PAD_PX (75)
    assert overrides[0].pad_left == 75


def test_overrides_from_issues_shrinks_font_on_overflow():
    issues = [RenderIssue(1, "text_overflow")]
    overrides = _overrides_from_issues(issues, {})
    assert overrides[1].font_size_delta == -2
    assert overrides[1].pad_right == 20
    assert overrides[1].pad_bottom == 20


class RenderJudgeAdapter:
    """Fake vision adapter that returns canned render verdicts."""

    name = "fake"

    def __init__(self, verdicts: tuple[str, ...] = ()) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0
        self.prompts: list[str] = []
        self.images: list[list[Path]] = []

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def generate(self, model: str, prompt: str, images: list[Path] | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.images.append(images or [])
        return self.verdicts.pop(0) if self.verdicts else '{"pass": true}'


def _make_page(tmp_path: Path) -> Path:
    """Create a blank white page for render-judge tests."""
    p = tmp_path / "page.png"
    cv2.imwrite(str(p), np.full((400, 300, 3), 255, dtype=np.uint8))
    return p


def test_render_with_judge_passes_on_first_attempt(tmp_path):
    page = _make_page(tmp_path)
    out = tmp_path / "out.png"
    adapter = RenderJudgeAdapter(['{"pass": true}'])
    _render_with_judge(
        adapter, "m", page,
        [Bubble([0.1, 0.1, 0.5, 0.4], "a", "Hello")],
        out,
        {"page": 4, "chapter": "1", "title": "Test"},
        log=lambda m: None,
    )
    assert adapter.calls == 1
    assert out.exists()


def test_render_with_judge_expands_box_on_leakage(tmp_path):
    page = _make_page(tmp_path)
    out = tmp_path / "out.png"
    adapter = RenderJudgeAdapter([
        '{"pass": false, "issues": [{"bubble_index": 0, "problem": "original_text_visible", "directions": ["left"]}]}',
        '{"pass": true}',
    ])
    _render_with_judge(
        adapter, "m", page,
        [Bubble([0.1, 0.1, 0.5, 0.4], "a", "Hello")],
        out,
        {"page": 4, "chapter": "1", "title": "Test"},
        log=lambda m: None,
    )
    assert adapter.calls == 2
    assert out.exists()


def test_render_with_judge_keeps_first_render_on_persistent_failure(tmp_path):
    page = _make_page(tmp_path)
    out = tmp_path / "out.png"
    # Judge always rejects; renderer should restore the first (unexpanded) render.
    adapter = RenderJudgeAdapter([
        '{"pass": false, "issues": [{"bubble_index": 0, "problem": "original_text_visible", "directions": ["left"]}]}',
        '{"pass": false, "issues": [{"bubble_index": 0, "problem": "original_text_visible", "directions": ["left"]}]}',
        '{"pass": false, "issues": [{"bubble_index": 0, "problem": "original_text_visible", "directions": ["left"]}]}',
    ])
    _render_with_judge(
        adapter, "m", page,
        [Bubble([0.1, 0.1, 0.5, 0.4], "a", "Hello")],
        out,
        {"page": 4, "chapter": "1", "title": "Test"},
        log=lambda m: None,
    )
    assert adapter.calls == 3
    assert out.exists()
