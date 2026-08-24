"""Grounding stage: parse/validation, judge repair loop + fallback, cache,
hit-rate stats. All model calls mocked."""

import json

import pytest
from PIL import Image

from entertainment_harness.video import regions as regions_mod
from entertainment_harness.video.grounding import (
    Anchor,
    MAX_GROUND_ATTEMPTS,
    GroundingError,
    draw_region_overlay,
    ground_page,
    ground_segments,
    grounding_path,
    judge_grounding,
    load_grounding,
    parse_ground_verdict,
    parse_grounding,
    store_grounding,
)
from entertainment_harness.video.panels import parse_panels
from entertainment_harness.video.script import Segment

PANELS = json.dumps([
    {"box": [0.05, 0.04, 0.95, 0.30], "description": "A valley at dawn."},
    {"box": [0.55, 0.35, 0.95, 0.60], "description": "Shin looks up."},
    {"box": [0.05, 0.35, 0.50, 0.60], "description": "A dragon appears."},
])
REGIONS = regions_mod.regions_for_panels(parse_panels(PANELS))


class RouterAdapter:
    """Routes by prompt marker: panel extraction / grounding / judge.
    Responses are queued per kind; records every call."""

    name = "fake"

    def __init__(self, panels=(), ground=(), judge=()):
        self.panels = list(panels)
        self.ground = list(ground)
        self.judge = list(judge)
        self.calls = []

    def generate(self, model, prompt, images=None):
        self.calls.append({"model": model, "prompt": prompt, "images": images or []})
        if "Divide the page into its panels" in prompt:
            return self.panels.pop(0) if self.panels else "[]"
        if "Verdict task" in prompt:
            return self.judge.pop(0) if self.judge else '{"pass": true}'
        return self.ground.pop(0) if self.ground else "{}"

    def prompts(self, marker):
        return [c["prompt"] for c in self.calls if marker in c["prompt"]]


def _page(path, size=(800, 1200)):
    Image.new("RGB", size, "white").save(path)
    return path


def _seg(index, text, pages, moment="m"):
    return Segment(index=index, text=text, moment=moment, pages=pages)


# --- parse_grounding ---


def test_parse_grounding_valid_converts_to_zero_based_sorted():
    out = parse_grounding('{"3": [3, 1, 1], "1": [2]}', {1, 3}, 3)
    assert out == {3: [0, 2], 1: [1]}


def test_parse_grounding_tolerates_fences_and_preamble():
    raw = 'Sure!\n```json\n{"0": [1]}\n```'
    assert parse_grounding(raw, {0}, 3) == {0: [0]}


def test_parse_grounding_drops_out_of_range_and_unknown():
    raw = json.dumps({
        "0": [0, 4, -1, "x", 2],   # only region 2 survives (1-based)
        "9": [1],                  # segment not shown in this batch
        "1": "not a list",
        "2": [],
    })
    assert parse_grounding(raw, {0, 1, 2}, 3) == {0: [1]}


def test_parse_grounding_raises_on_no_object():
    with pytest.raises(GroundingError, match="JSON object"):
        parse_grounding("I cannot help.", {0}, 3)


# --- parse_ground_verdict (strict: unparseable = rejection) ---


def test_parse_ground_verdict_pass_and_fail():
    assert parse_ground_verdict('{"pass": true}').passed
    v = parse_ground_verdict('{"pass": false, "issues": ["wrong panel"]}')
    assert not v.passed and v.issues == ["wrong panel"]


def test_parse_ground_verdict_garbage_is_a_rejection():
    # The inversion of judge.parse_verdict: a broken grounding judge must not
    # wave unverified choices through (safe over smooth).
    v = parse_ground_verdict("I think it looks fine")
    assert not v.passed
    assert "not valid JSON" in v.issues[0]


# --- overlay ---


def test_draw_region_overlay_writes_same_size_jpeg(tmp_path):
    page = _page(tmp_path / "p.jpg")
    dest = draw_region_overlay(page, REGIONS, tmp_path / "out" / "ov.jpg",
                               highlight=(1,))
    assert dest.exists()
    with Image.open(dest) as img:
        assert img.size == (800, 1200)


# --- ground_page (batched call) ---


def test_ground_page_batches_segments_in_one_call(tmp_path):
    adapter = RouterAdapter(ground=['{"0": [1], "1": [2]}'])
    segs = [_seg(0, "First beat.", [1]), _seg(1, "Second beat.", [1])]
    overlay = _page(tmp_path / "ov.jpg")
    out = ground_page(adapter, "m", overlay, 2, 28, 3, segs, "Kenja", 1.0)
    assert out == {0: [0], 1: [1]}
    assert len(adapter.calls) == 1  # one call for the whole page
    prompt = adapter.calls[0]["prompt"]
    assert "page 2 of 28" in prompt and 'chapter 1 of the manga "Kenja"' in prompt
    assert "First beat." in prompt and "Second beat." in prompt
    assert "retells the story" in prompt  # match by content, not text
    assert adapter.calls[0]["images"] == [overlay]


def test_ground_page_retries_once_then_gives_up(tmp_path):
    adapter = RouterAdapter(ground=["garbage", "still garbage"])
    out = ground_page(adapter, "m", _page(tmp_path / "ov.jpg"),
                      1, 28, 3, [_seg(0, "t", [1])], "T", 1.0)
    assert out == {}
    assert len(adapter.calls) == 2


# --- judge_grounding ---


def test_judge_grounding_prompt_is_about_story_content_not_text(tmp_path):
    adapter = RouterAdapter(judge=['{"pass": true}'])
    seg = _seg(4, "Shin slays the boar.", [2])
    verdict = judge_grounding(adapter, "m", _page(tmp_path / "ov.jpg"),
                              seg, [0, 2], 2, "Kenja", 1.0)
    assert verdict.passed
    prompt = adapter.calls[0]["prompt"]
    assert '"Shin slays the boar."' in prompt
    assert "Chosen region(s): 1, 3" in prompt  # 0-based back to 1-based labels
    assert "does NOT need to match text" in prompt
    assert "FAIL" in prompt  # unverifiable must not ship


# --- ground_segments (stage driver) ---


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    return tmp_path


def test_ground_segments_happy_path_and_cache(env):
    pages = [_page(env / f"page-{i:03d}.jpg") for i in (1, 2)]
    segs = [_seg(0, "The valley wakes.", [1]), _seg(1, "Shin looks up.", [1, 2])]
    adapter = RouterAdapter(
        panels=[PANELS, PANELS],
        # page-1 batch grounds S0 only (S1 re-grounds focused), page-2 batch
        # grounds S1.
        ground=['{"0": [2]}', '{"1": [1, 3]}', '{"1": [1]}'],
        judge=['{"pass": true}', '{"pass": true}', '{"pass": true}'],
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert stats == {"judged": 3, "ok": 3, "pruned": 0, "vacuous": 0,
                     "segment_fallbacks": 0, "verified": 4, "unverified": 0,
                     "slots": 4, "assignment_rate": 1.0, "rate": 1.0}
    # Anchors carry the resolved (padded) region boxes in reading order.
    assert segs[0].regions == [
        {"page": 1, "kind": "hold", "box": list(REGIONS[1].box)}
    ]
    assert segs[1].regions == [
        {"page": 1, "kind": "hold", "box": list(REGIONS[0].box)},
        {"page": 1, "kind": "hold", "box": list(REGIONS[2].box)},
        {"page": 2, "kind": "hold", "box": list(REGIONS[0].box)},
    ]
    assert grounding_path("s1", "ch-1").exists()
    calls = len(adapter.calls)

    # Second run: everything cached — zero model calls.
    segs2 = [_seg(0, "The valley wakes.", [1]), _seg(1, "Shin looks up.", [1, 2])]
    stats2 = ground_segments(adapter, "m", segs2, pages, "s1", "ch-1", "T", 1.0)
    assert len(adapter.calls) == calls
    assert stats2["rate"] == 1.0
    assert segs2[1].regions == segs[1].regions


def test_ground_segments_reground_with_feedback_after_rejection(env):
    pages = [_page(env / "page-001.jpg")]
    segs = [_seg(0, "The dragon attacks.", [1])]
    adapter = RouterAdapter(
        panels=[PANELS],
        ground=['{"0": [1]}', '{"0": [3]}'],
        judge=['{"pass": false, "issues": ["region 1 is the sky, not the dragon"]}',
               '{"pass": true}'],
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert stats["ok"] == 1 and stats["pruned"] == 0
    assert segs[0].regions == [
        {"page": 1, "kind": "hold", "box": list(REGIONS[2].box)}
    ]
    focused = adapter.prompts("A previous assignment for this beat was rejected")
    assert len(focused) == 1
    assert "region 1 is the sky, not the dragon" in focused[0]
    cached = load_grounding("s1", "ch-1")
    record = next(iter(cached.values()))["judge"]["1"]
    assert record == {"status": "ok", "attempts": 2}


def test_ground_segments_persistent_rejection_prunes_then_segment_falls_back(env):
    pages = [_page(env / "page-001.jpg")]
    segs = [_seg(0, "An abstract reflection.", [1])]
    adapter = RouterAdapter(
        panels=[PANELS],
        ground=['{"0": [1]}', '{"0": [2]}', '{"0": [3]}'],
        judge=['{"pass": false, "issues": ["nope"]}'] * 3,
    )
    logs = []
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0,
                            log=logs.append)
    # The only page was pruned: whole-page pan over the first original page.
    assert segs[0].pages == [1]
    assert segs[0].regions == [{"page": 1, "kind": "pan"}]
    assert stats["pruned"] == 1 and stats["ok"] == 0
    assert stats["segment_fallbacks"] == 1
    assert stats["verified"] == 0 and stats["unverified"] == 1
    assert stats["slots"] == 1 and stats["rate"] == 0.0
    assert stats["assignment_rate"] == 0.0
    # batch + MAX_GROUND_ATTEMPTS-1 focused proposals, one judge call each
    assert len(adapter.prompts("Verdict task")) == MAX_GROUND_ATTEMPTS
    assert any("hit rate" in m and "0.0%" in m for m in logs)


def test_ground_segments_unparseable_judge_counts_as_rejection(env):
    pages = [_page(env / "page-001.jpg")]
    segs = [_seg(0, "A quiet moment.", [1])]
    adapter = RouterAdapter(
        panels=[PANELS],
        ground=['{"0": [1]}', '{"0": [1]}', '{"0": [1]}'],
        judge=["I cannot decide", "not json either", '{"pass": false, "issues": []}'],
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert segs[0].regions == [{"page": 1, "kind": "pan"}]
    assert stats["pruned"] == 1 and stats["segment_fallbacks"] == 1


def test_ground_segments_prunes_rejected_page_from_segment(env):
    """A judge-rejected page is removed from the segment and never rendered;
    surviving pages keep their verified holds (safe over smooth, scoped to
    the page level)."""
    pages = [_page(env / f"page-{i:03d}.jpg") for i in (1, 2)]
    segs = [_seg(0, "Two-page beat.", [1, 2])]
    adapter = RouterAdapter(
        panels=[PANELS, PANELS],
        # page 1: batch + 2 focused retries, all rejected; page 2: batch ok.
        ground=['{"0": [1]}', '{"0": [2]}', '{"0": [3]}', '{"0": [2]}'],
        judge=(['{"pass": false, "issues": ["wrong scene"]}'] * 3
               + ['{"pass": true}']),
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert segs[0].pages == [2]  # page 1 pruned
    assert segs[0].regions == [
        {"page": 2, "kind": "hold", "box": list(REGIONS[1].box)}
    ]
    assert stats["ok"] == 1 and stats["pruned"] == 1
    assert stats["assignment_rate"] == 0.5
    assert stats["segment_fallbacks"] == 0 and stats["unverified"] == 0
    assert stats["verified"] == 1 and stats["slots"] == 1
    assert stats["rate"] == 1.0

    # Cache round-trip: fresh segments with the ORIGINAL pages hit the
    # original-page key and re-apply the pruned pages + anchors, zero calls.
    calls = len(adapter.calls)
    segs2 = [_seg(0, "Two-page beat.", [1, 2])]
    stats2 = ground_segments(adapter, "m", segs2, pages, "s1", "ch-1", "T", 1.0)
    assert len(adapter.calls) == calls
    assert segs2[0].pages == [2]
    assert segs2[0].regions == segs[0].regions
    assert stats2["rate"] == 1.0 and stats2["pruned"] == 1


def test_ground_segments_fully_pruned_segment_pans_first_original_page(env):
    """Every assigned page rejected -> one whole-page pan over the first
    original page, counted as the only unverified slot kind."""
    pages = [_page(env / f"page-{i:03d}.jpg") for i in (1, 2)]
    segs = [_seg(0, "Ungroundable beat.", [1, 2])]
    adapter = RouterAdapter(
        panels=[PANELS, PANELS],
        ground=['{"0": [1]}', '{"0": [1]}', '{"0": [1]}',   # page 1 attempts
                '{"0": [1]}', '{"0": [1]}', '{"0": [1]}'],  # page 2 attempts
        judge=['{"pass": false, "issues": ["different moment"]}'] * 6,
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert segs[0].pages == [1]  # first original page
    assert segs[0].regions == [{"page": 1, "kind": "pan"}]
    assert stats["pruned"] == 2 and stats["ok"] == 0
    assert stats["segment_fallbacks"] == 1
    assert stats["verified"] == 0 and stats["unverified"] == 1
    assert stats["slots"] == 1 and stats["rate"] == 0.0
    assert stats["assignment_rate"] == 0.0


def test_ground_segments_vacuous_page_survives_pruning_and_skips_metric(env):
    """A vacuous page (no sub-page regions) is not a rejection: it survives
    with a pan anchor and stays out of both rate denominators."""
    pages = [_page(env / f"page-{i:03d}.jpg") for i in (1, 2)]
    segs = [_seg(0, "Beat.", [1, 2])]
    adapter = RouterAdapter(
        panels=[PANELS, "[]"],  # page 2 yields no panels -> vacuous
        ground=['{"0": [1]}', '{"0": [1]}', '{"0": [1]}'],
        judge=['{"pass": false, "issues": ["no"]}'] * 3,
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert segs[0].pages == [2]  # page 1 pruned, vacuous page survives
    assert segs[0].regions == [{"page": 2, "kind": "pan"}]
    assert stats["pruned"] == 1 and stats["vacuous"] == 1
    assert stats["segment_fallbacks"] == 0 and stats["unverified"] == 0
    assert stats["slots"] == 0 and stats["rate"] is None
    assert stats["assignment_rate"] == 0.0


def test_ground_segments_omitted_from_batch_gets_focused_call(env):
    pages = [_page(env / "page-001.jpg")]
    segs = [_seg(0, "Skipped by the batch.", [1])]
    adapter = RouterAdapter(
        panels=[PANELS],
        ground=["{}", '{"0": [2]}'],  # batch omits it; focused assigns
    )
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert segs[0].regions == [
        {"page": 1, "kind": "hold", "box": list(REGIONS[1].box)}
    ]
    assert stats["ok"] == 1
    ground_calls = [c for c in adapter.calls if "narration beats" in c["prompt"]]
    assert len(ground_calls) == 2  # batch + one focused


def test_ground_segments_whole_page_only_regions_is_vacuous_pan(env):
    pages = [_page(env / "page-001.jpg")]
    segs = [_seg(0, "Anything.", [1])]
    adapter = RouterAdapter(panels=["[]"])  # extraction finds no panels
    stats = ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    assert segs[0].regions == [{"page": 1, "kind": "pan"}]
    assert stats == {"judged": 0, "ok": 0, "pruned": 0, "vacuous": 1,
                     "segment_fallbacks": 0, "verified": 0, "unverified": 0,
                     "slots": 0, "assignment_rate": None, "rate": None}
    # No grounding or judge calls — only the panel extraction plus its
    # whole-page describe fallback (which also yields nothing here).
    assert len(adapter.calls) == 2
    assert "Describe what happens on this page" in adapter.calls[1]["prompt"]


def test_ground_segments_content_key_regrounds_only_changed_segments(env):
    pages = [_page(env / "page-001.jpg")]
    adapter = RouterAdapter(
        panels=[PANELS], ground=['{"0": [1], "1": [2]}'],
    )
    segs = [_seg(0, "Beat one.", [1]), _seg(1, "Beat two.", [1])]
    ground_segments(adapter, "m", segs, pages, "s1", "ch-1", "T", 1.0)
    calls = len(adapter.calls)

    # Script regeneration changed segment 0's text: only it recomputes
    # (panels cached, so no extraction call either).
    adapter.ground.append('{"0": [3]}')
    segs2 = [_seg(0, "Beat one, reworded.", [1]), _seg(1, "Beat two.", [1])]
    ground_segments(adapter, "m", segs2, pages, "s1", "ch-1", "T", 1.0)
    new = adapter.calls[calls:]
    assert len(adapter.prompts("Divide the page")) == 1  # no re-extraction
    assert [c for c in new if "narration beats" in c["prompt"]
            and "Beat one, reworded." in c["prompt"]]
    assert not [c for c in new if "Beat two." in c["prompt"]
                and "reworded" not in c["prompt"]]
    assert segs2[0].regions == [
        {"page": 1, "kind": "hold", "box": list(REGIONS[2].box)}
    ]
    assert segs2[1].regions == segs[1].regions  # served from cache


def test_load_grounding_missing_corrupt_foreign(env):
    assert load_grounding("s1", "ch-1") is None
    path = grounding_path("s1", "ch-1")
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    assert load_grounding("s1", "ch-1") is None
    path.write_text('{"version": 999, "segments": {}}')
    assert load_grounding("s1", "ch-1") is None


def test_load_grounding_drops_malformed_entries(env):
    store_grounding("s1", "ch-1", {
        "good": {
            "anchors": [{"page": 2, "kind": "hold", "box": [0, 0, 0.5, 0.5]},
                        {"page": 3, "kind": "pan"}],
            "judge": {"2": {"status": "ok", "attempts": 1}},
        },
        "bad-anchors": {"anchors": [{"page": "x", "kind": "hold"},
                                    {"page": 1, "kind": "hold"}],  # no box
                        "judge": {}},
        "not-a-dict": "junk",
    })
    out = load_grounding("s1", "ch-1")
    assert list(out) == ["good"]
    assert out["good"]["anchors"] == [
        {"page": 2, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]},
        {"page": 3, "kind": "pan"},
    ]


def test_anchor_from_dict_validation():
    assert Anchor.from_dict({"page": 1, "kind": "pan"}) == Anchor(1, "pan")
    assert Anchor.from_dict({"page": 1, "kind": "hold",
                             "box": [0, 0, 1, 1]}) == Anchor(1, "hold", [0, 0, 1, 1])
    assert Anchor.from_dict({"page": 1, "kind": "hold"}) is None
    assert Anchor.from_dict({"page": 1, "kind": "weird"}) is None
    assert Anchor.from_dict("junk") is None
