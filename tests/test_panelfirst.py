"""Panel-first narration tests (anchored-scroll Phase 4): beat flattening,
grouping parse/validation policy, judge loop, segment span-stamping, and
the panelfirst.json cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection
from entertainment_harness.video.panels import Panel, store_panels
from entertainment_harness.video.panelfirst import (
    Beat,
    _cache_key,
    _evidence_batches,
    build_panel_first_script,
    chapter_beats,
    group_beats,
    parse_groups,
    segments_from_groups,
)
from entertainment_harness.video.script import VideoError

GB = 10**9


class QueueAdapter:
    """Returns queued raw responses; records calls."""

    name = "fake"

    def __init__(self, responses=(), default="[]"):
        self.responses = list(responses)
        self.default = default
        self.calls = []
        self.ensured = []

    def ensure(self, model, quant=None):
        self.ensured.append(model)

    def generate(self, model, prompt, images=None):
        self.calls.append({"model": model, "prompt": prompt, "images": images or []})
        return self.responses.pop(0) if self.responses else self.default


def _sel(adapter, name="fake-text:4b"):
    return Selection(
        adapter=adapter,
        info=ModelInfo(name, "fake", params=4.0, quant="Q8_0", size_bytes=3 * GB),
    )


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))


def _seed_panels():
    store_panels("s1", "ch-1", {
        "page-001.jpg": [Panel([0.0, 0.0, 0.5, 0.5], "Shin wakes up."),
                         Panel([0.5, 0.0, 1.0, 0.5], "A dragon lands.")],
        "page-002.jpg": [Panel([0.1, 0.6, 0.9, 0.95], "They fight.")],
        "page-003.jpg": [Panel([0.0, 0.0, 1.0, 1.0], "The dust settles.")],
    })


PAGES = [Path(f"page-{i:03d}.jpg") for i in (1, 2, 3)]
SERIES = {"id": "s1", "title": "Test Manga"}
CHAPTER = {"id": "ch-1", "chapter_num": 1.0}

GROUPS_JSON = (
    '[{"from": 1, "to": 2, "moment": "arrival",'
    ' "text": "Shin wakes as a dragon lands."},'
    ' {"from": 3, "to": 4, "moment": "fight",'
    ' "text": "They fight until the dust settles."}]'
)
GROUPS_ALT_JSON = (
    '[{"from": 1, "to": 4, "moment": "everything",'
    ' "text": "A very different second attempt at the whole chapter."}]'
)


# --- chapter_beats ---


def test_chapter_beats_flattens_in_reading_order_and_pads(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    beats = chapter_beats(PAGES, "s1", "ch-1")
    assert [b.index for b in beats] == [1, 2, 3, 4]
    assert [b.page for b in beats] == [1, 1, 2, 3]
    assert [b.description for b in beats] == [
        "Shin wakes up.", "A dragon lands.", "They fight.", "The dust settles."
    ]
    assert beats[0].box == pytest.approx((0.0, 0.0, 0.52, 0.52))  # padded
    assert beats[1].box == pytest.approx((0.48, 0.0, 1.0, 0.52))  # clamped


def test_chapter_beats_skips_and_logs_panel_less_pages(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    logs = []
    pages = [Path("page-001.jpg"), Path("page-00x.jpg"), Path("page-002.jpg"),
             Path("page-003.jpg")]
    beats = chapter_beats(pages, "s1", "ch-1", log=logs.append)
    # The gap page shifts page numbers of everything after it.
    assert [b.page for b in beats] == [1, 1, 3, 4]
    assert any("page-00x.jpg" in m for m in logs)


# --- parse_groups (the partition policy) ---


def test_parse_groups_happy_path():
    groups = parse_groups(GROUPS_JSON, 4)
    assert groups == [
        {"from": 1, "to": 2, "moment": "arrival",
         "text": "Shin wakes as a dragon lands."},
        {"from": 3, "to": 4, "moment": "fight",
         "text": "They fight until the dust settles."},
    ]


def test_parse_groups_overlap_clamps_from():
    groups = parse_groups(
        '[{"from": 1, "to": 2, "text": "a"}, {"from": 2, "to": 3, "text": "b"}]',
        3,
    )
    assert [(g["from"], g["to"]) for g in groups] == [(1, 2), (3, 3)]


def test_parse_groups_overlap_emptied_is_dropped():
    groups = parse_groups(
        '[{"from": 1, "to": 2, "text": "a"}, {"from": 2, "to": 2, "text": "x"},'
        ' {"from": 3, "to": 3, "text": "b"}]',
        3,
    )
    assert [(g["from"], g["to"]) for g in groups] == [(1, 2), (3, 3)]


def test_parse_groups_gap_extends_previous_and_trailing_extends_last():
    groups = parse_groups(
        '[{"from": 1, "to": 1, "text": "a"}, {"from": 3, "to": 4, "text": "b"}]',
        5,
    )
    assert [(g["from"], g["to"]) for g in groups] == [(1, 2), (3, 5)]


def test_parse_groups_leading_gap_clamps_first_group_to_beat_one():
    groups = parse_groups('[{"from": 2, "to": 3, "text": "a"}]', 3)
    assert [(g["from"], g["to"]) for g in groups] == [(1, 3)]


def test_parse_groups_drops_invalid_items():
    groups = parse_groups(
        '["junk", {"from": "x", "to": 2, "text": "bad"},'
        ' {"from": 1, "to": 2},'
        ' {"from": 3, "to": 1, "text": "swapped"},'
        ' {"from": 1, "to": 2, "text": "good"}]',
        2,
    )
    assert groups == [{"from": 1, "to": 2, "moment": "", "text": "good"}]


def test_parse_groups_out_of_range_clamps_into_bounds():
    groups = parse_groups('[{"from": 0, "to": 99, "text": "a"}]', 4)
    assert [(g["from"], g["to"]) for g in groups] == [(1, 4)]


def test_parse_groups_nothing_usable_raises():
    with pytest.raises(VideoError):
        parse_groups("no JSON here at all", 3)
    with pytest.raises(VideoError):
        parse_groups("[]", 3)
    with pytest.raises(VideoError):
        parse_groups('[{"from": 1, "to": 2}]', 3)  # no text anywhere


def test_parse_groups_salvages_from_truncated_array():
    groups = parse_groups(
        '[{"from": 1, "to": 1, "moment": "a", "text": "one."},'
        ' {"from": 2, "to": 2, "mom',
        3,
    )
    assert [(g["from"], g["to"]) for g in groups] == [(1, 3)]  # trailing extend


def test_parse_groups_requires_beats():
    with pytest.raises(VideoError):
        parse_groups(GROUPS_JSON, 0)


# --- group_beats (model call + retries) ---


def _beats(n=4):
    return [
        Beat(i, 1, (0.0, 0.0, 1.0, 1.0), f"Panel {i}.") for i in range(1, n + 1)
    ]


def test_group_beats_retries_unusable_output_then_succeeds():
    adapter = QueueAdapter(["garbage", GROUPS_JSON])
    groups = group_beats(adapter, "m", _beats(), title="T", chapter_num=1.0)
    assert len(adapter.calls) == 2
    assert groups[0]["text"] == "Shin wakes as a dragon lands."


def test_group_beats_raises_after_max_attempts():
    adapter = QueueAdapter(["garbage", "still garbage", "[]"])
    with pytest.raises(VideoError, match="after 3 attempts"):
        group_beats(adapter, "m", _beats(), title="T", chapter_num=1.0)
    assert len(adapter.calls) == 3


def test_group_beats_prompt_lists_beats_and_honors_instruction_and_feedback():
    adapter = QueueAdapter([GROUPS_JSON])
    group_beats(
        adapter, "m", _beats(), title="T", chapter_num=1.0,
        instruction="Skip the gore.", feedback="\n\nREJECTED before.",
    )
    prompt = adapter.calls[0]["prompt"]
    assert "1. [page 1] Panel 1." in prompt
    assert "USER DIRECTION" in prompt and "Skip the gore." in prompt
    assert "REJECTED before." in prompt


# --- segments_from_groups ---


def test_segments_from_groups_stamps_spans_and_anchors():
    beats = [
        Beat(1, 1, (0.0, 0.0, 0.52, 0.52), "one"),
        Beat(2, 1, (0.48, 0.0, 1.0, 0.52), "two"),
        Beat(3, 2, (0.08, 0.58, 0.92, 0.97), "three"),
    ]
    groups = [
        {"from": 1, "to": 2, "moment": "a", "text": "First beat."},
        {"from": 3, "to": 3, "moment": "b", "text": "Second beat."},
    ]
    segments = segments_from_groups(groups, beats)
    assert [s.index for s in segments] == [0, 1]
    assert segments[0].pages == [1]
    assert segments[0].regions == [
        {"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.52, 0.52]},
        {"page": 1, "kind": "hold", "box": [0.48, 0.0, 1.0, 0.52]},
    ]
    assert segments[1].pages == [2]
    assert segments[1].regions == [
        {"page": 2, "kind": "hold", "box": [0.08, 0.58, 0.92, 0.97]}
    ]
    assert (segments[0].moment, segments[1].moment) == ("a", "b")


def test_segments_from_groups_split_pieces_restamp_the_span():
    beats = [Beat(1, 2, (0.0, 0.0, 1.0, 1.0), "one")]
    long_text = (
        " ".join(f"w{i}" for i in range(20)) + "."
        + " " + " ".join(f"x{i}" for i in range(20)) + "."
    )
    groups = [{"from": 1, "to": 1, "moment": "big", "text": long_text}]
    segments = segments_from_groups(groups, beats)
    assert len(segments) == 2  # 40 words > MAX_BEAT_WORDS: split at sentences
    for i, piece in enumerate(segments):
        assert piece.index == i
        assert piece.pages == [2]
        assert piece.regions == [
            {"page": 2, "kind": "hold", "box": [0.0, 0.0, 1.0, 1.0]}
        ]
        assert piece.moment == "big"


# --- evidence batches ---


def test_evidence_batches_group_descriptions_per_page():
    beats = [
        Beat(1, 1, (0, 0, 1, 1), "first"),
        Beat(2, 1, (0, 0, 1, 1), "second"),
        Beat(3, 3, (0, 0, 1, 1), "third"),
    ]
    assert _evidence_batches(beats) == [
        "Page 1:\n- first\n- second",
        "Page 3:\n- third",
    ]


# --- build_panel_first_script ---


def _real_pages(tmp_path, names=("page-001.jpg", "page-002.jpg", "page-003.jpg")):
    paths = []
    for name in names:
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 120), "white").save(path)
        paths.append(path)
    return paths


def test_build_panel_first_script_unjudged_and_cached(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON])
    segments, status = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), None,
        PAGES, SERIES, CHAPTER,
    )
    assert status == "unjudged"
    assert len(segments) == 2
    assert segments[0].pages == [1]
    assert segments[1].pages == [2, 3]
    assert len(segments[0].regions) == 2  # two panels in the first group
    assert (tmp_path / "works" / "s1" / "chapters" / "ch-1"
            / "panelfirst.json").exists()

    # Second run: the grouping model is never called.
    exploding = QueueAdapter()
    exploding.generate = lambda *a, **k: pytest.fail("regrouped")
    segments2, status2 = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(exploding), None,
        PAGES, SERIES, CHAPTER,
    )
    assert status2 == "cached"
    assert [s.text for s in segments2] == [s.text for s in segments]


def test_build_panel_first_script_instruction_rekeys_cache(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON, GROUPS_ALT_JSON])
    build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), None,
        PAGES, SERIES, CHAPTER,
    )
    assert len(text.calls) == 1
    segments, status = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), None,
        PAGES, SERIES, CHAPTER, instruction="Focus on the dragon.",
    )
    assert status == "unjudged"  # regrouped, not cached
    assert len(text.calls) == 2
    assert segments[0].text.startswith("A very different second attempt")
    assert "Focus on the dragon." in text.calls[1]["prompt"]


def test_build_panel_first_script_reextracted_panels_rekey_cache(
    tmp_path, monkeypatch
):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON, GROUPS_ALT_JSON])
    build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), None,
        PAGES, SERIES, CHAPTER,
    )
    store_panels("s1", "ch-1", {
        "page-001.jpg": [Panel([0.0, 0.0, 0.5, 0.5], "Shin wakes up CHANGED."),
                         Panel([0.5, 0.0, 1.0, 0.5], "A dragon lands.")],
        "page-002.jpg": [Panel([0.1, 0.6, 0.9, 0.95], "They fight.")],
        "page-003.jpg": [Panel([0.0, 0.0, 1.0, 1.0], "The dust settles.")],
    })
    _, status = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), None,
        PAGES, SERIES, CHAPTER,
    )
    assert status == "unjudged"  # description changed -> regroup
    assert len(text.calls) == 2


def test_build_panel_first_script_extracts_missing_pages(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    store_panels("s1", "ch-1", {
        "page-001.jpg": [Panel([0.0, 0.0, 0.5, 0.5], "Shin wakes up."),
                         Panel([0.5, 0.0, 1.0, 0.5], "A dragon lands.")],
        "page-002.jpg": [Panel([0.1, 0.6, 0.9, 0.95], "They fight.")],
    })
    panels_good = json.dumps([
        {"box": [0.0, 0.0, 1.0, 1.0], "description": "The dust settles."}
    ])
    vision = QueueAdapter([panels_good])
    text = QueueAdapter([GROUPS_JSON])
    pages = _real_pages(tmp_path)
    segments, status = build_panel_first_script(
        _sel(vision, "fake-vl:8b"), _sel(text), None, pages, SERIES, CHAPTER,
    )
    assert status == "unjudged"
    # Only the missing page hit the vision model.
    assert [c["images"][0].name for c in vision.calls] == ["page-003.jpg"]
    assert vision.ensured == ["fake-vl:8b"]
    assert len(segments) == 2
    assert segments[1].pages == [2, 3]


def test_build_panel_first_script_no_panels_raises(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    # Extraction runs (nothing cached) but yields nothing usable even via the
    # whole-page describe fallback ("[]" is rejected as JSON-ish).
    vision = QueueAdapter(["[]", "[]", "[]", "[]", "[]", "[]"])
    with pytest.raises(VideoError, match="no panels"):
        build_panel_first_script(
            _sel(vision, "fake-vl:8b"), _sel(QueueAdapter()), None,
            _real_pages(tmp_path), SERIES, CHAPTER,
        )


def test_build_panel_first_script_judged_pass(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON])
    judge = QueueAdapter(default='{"pass": true}')
    segments, status = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), _sel(judge, "fake-j:4b"),
        PAGES, SERIES, CHAPTER,
    )
    assert status == "passed"
    assert len(judge.calls) == 1
    prompt = judge.calls[0]["prompt"]
    assert "Page 1:\n- Shin wakes up.\n- A dragon lands." in prompt
    assert "Shin wakes as a dragon lands." in prompt  # the narration
    assert len(segments) == 2


def test_build_panel_first_script_judge_rejection_regenerates_with_feedback(
    tmp_path, monkeypatch
):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON, GROUPS_ALT_JSON])
    judge = QueueAdapter([
        '{"pass": false, "issues": ["misses the fight"]}',
        '{"pass": true}',
    ])
    segments, status = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), _sel(judge, "fake-j:4b"),
        PAGES, SERIES, CHAPTER,
    )
    assert status == "passed after 2 attempts"
    assert segments[0].text.startswith("A very different second attempt")
    assert "misses the fight" in text.calls[1]["prompt"]  # feedback flowed


def test_build_panel_first_script_persistent_judge_failure_keeps_first(
    tmp_path, monkeypatch
):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON, GROUPS_ALT_JSON])
    judge = QueueAdapter(default='{"pass": false, "issues": ["never happy"]}')
    segments, status = build_panel_first_script(
        _sel(QueueAdapter(), "fake-vl:8b"), _sel(text), _sel(judge, "fake-j:4b"),
        PAGES, SERIES, CHAPTER, max_attempts=2,
    )
    assert status == "kept first"
    assert len(text.calls) == 2
    # The FIRST (unpoisoned) attempt ships, not the last.
    assert segments[0].text == "Shin wakes as a dragon lands."


def test_build_panel_first_script_vision_verify_judges_against_pages(
    tmp_path, monkeypatch
):
    _env(tmp_path, monkeypatch)
    _seed_panels()
    text = QueueAdapter([GROUPS_JSON])
    vision = QueueAdapter(['{"pass": true}'])
    judge = QueueAdapter(default='{"pass": true}')
    pages = _real_pages(tmp_path)
    segments, status = build_panel_first_script(
        _sel(vision, "fake-vl:8b"), _sel(text), _sel(judge, "fake-j:4b"),
        pages, SERIES, CHAPTER, vision_verify=True,
    )
    assert status == "passed"
    assert judge.calls == []  # the text judge is bypassed
    assert len(vision.calls) == 1
    assert vision.calls[0]["images"]  # page samples went to the vision judge


def test_cache_key_ignores_boxes():
    a = [Beat(1, 1, (0.0, 0.0, 0.5, 0.5), "one")]
    b = [Beat(1, 1, (0.1, 0.1, 0.9, 0.9), "one")]
    assert _cache_key(a, "") == _cache_key(b, "")
    assert _cache_key(a, "x") != _cache_key(a, "")
