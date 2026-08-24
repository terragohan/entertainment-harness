"""Panel extraction: prompt/parse/validation/retry, cache, regions wiring."""

import json

import pytest
from PIL import Image

from entertainment_harness.video import regions
from entertainment_harness.video.panels import (
    MAX_PANELS,
    Panel,
    PanelError,
    extract_chapter_panels,
    extract_panels,
    load_panels,
    panels_path,
    parse_panels,
    store_panels,
)

GOOD = json.dumps([
    {"box": [0.05, 0.04, 0.95, 0.30], "description": "A valley at dawn."},
    {"box": [0.55, 0.35, 0.95, 0.60], "description": "Shin looks up."},
    {"box": [0.05, 0.35, 0.50, 0.60], "description": "A dragon appears."},
])


class FakeAdapter:
    """Returns queued raw responses, then "[]"; records calls."""

    name = "fake"

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    def generate(self, model, prompt, images=None):
        self.calls.append({"model": model, "prompt": prompt, "images": images or []})
        return self.responses.pop(0) if self.responses else "[]"


def _page(path, size=(800, 1200)):
    Image.new("RGB", size, "white").save(path)
    return path


# --- parse_panels ---


def test_parse_panels_valid_array():
    panels = parse_panels(GOOD)
    assert len(panels) == 3
    assert panels[0].box == pytest.approx([0.05, 0.04, 0.95, 0.30])
    assert panels[0].description == "A valley at dawn."


def test_parse_panels_tolerates_fences_and_preamble():
    raw = 'Here are the panels:\n```json\n' + GOOD + '\n```'
    assert len(parse_panels(raw)) == 3


def test_parse_panels_pixel_coords_normalized():
    raw = json.dumps([{"box": [50, 100, 500, 400], "description": "Panel."}])
    panels = parse_panels(raw, image_size=(800, 1200))
    assert panels[0].box == pytest.approx([0.0625, 100 / 1200, 0.625, 400 / 1200])


def test_parse_panels_qwen_relative_1000_detected_on_large_page():
    # qwen3-vl flips to a 0-1000 relative coordinate system per response
    # (observed live 2026-09-16): all coords <= 1000 on a page whose dims
    # both exceed 1000. Divide by 1000, not by the true pixel dims.
    raw = json.dumps([
        {"box": [0, 0, 930, 712], "description": "Big top panel."},
        {"box": [0, 712, 520, 1000], "description": "Bottom left."},
        {"box": [520, 745, 825, 1000], "description": "Bottom right."},
    ])
    panels = parse_panels(raw, image_size=(1128, 1600))
    assert panels[0].box == pytest.approx([0.0, 0.0, 0.93, 0.712])
    assert panels[1].box == pytest.approx([0.0, 0.712, 0.52, 1.0])
    assert panels[2].box == pytest.approx([0.52, 0.745, 0.825, 1.0])


def test_parse_panels_relative_1000_detected_when_exceeding_dims():
    # A coordinate beyond the image dims can only be relative-1000.
    raw = json.dumps([{"box": [0, 0, 930, 712], "description": "Panel."}])
    panels = parse_panels(raw, image_size=(700, 1000))
    assert panels[0].box == pytest.approx([0.0, 0.0, 0.93, 0.712])


def test_parse_panels_true_pixels_on_large_page_unchanged():
    # Coordinates beyond 1000 (but within the dims) are true pixels.
    raw = json.dumps([{"box": [50, 100, 900, 1500], "description": "Panel."}])
    panels = parse_panels(raw, image_size=(1128, 1600))
    assert panels[0].box == pytest.approx(
        [50 / 1128, 100 / 1600, 900 / 1128, 1500 / 1600]
    )


def test_parse_panels_drops_malformed_entries():
    raw = json.dumps([
        {"box": [0.05, 0.05, 0.5, 0.4], "description": "Kept."},
        {"box": [0.1, 0.1], "description": "short box"},
        {"box": ["a", "b", "c", "d"], "description": "strings"},
        {"box": [0.5, 0.1, 0.2, 0.3], "description": "inverted"},
        {"box": [0.1, 0.1, 0.12, 0.12], "description": "speck"},
        {"box": [0.6, 0.6, 0.9, 0.9], "description": "  "},
        {"box": [0.6, 0.1, 0.9, 0.4]},  # no description
        "not a dict",
    ])
    panels = parse_panels(raw)
    assert [p.description for p in panels] == ["Kept."]


def test_parse_panels_salvages_from_broken_array():
    raw = ('[{"box": [0.1, 0.1, 0.5, 0.5], "description": "ok"},'
           ' {"box": [broken], "description": ]')
    panels = parse_panels(raw)
    assert [p.description for p in panels] == ["ok"]


def test_parse_panels_total_garbage_raises():
    with pytest.raises(PanelError, match="JSON array"):
        parse_panels("I cannot help with that.")


def test_parse_panels_caps_count():
    raw = json.dumps([
        {"box": [0.0, i / 100, 1.0, (i + 1) / 100], "description": f"P{i}"}
        for i in range(0, 40, 2)  # 20 panels, each 1% of the page
    ])
    assert len(parse_panels(raw)) == MAX_PANELS


# --- extract_panels (retry) ---


def _args():
    return {"page": 2, "total": 28, "chapter": "1", "title": "Kenja no Mago"}


def test_extract_panels_prompt_has_context_and_reading_order(tmp_path):
    adapter = FakeAdapter([GOOD])
    page = _page(tmp_path / "page-002.jpg")
    panels = extract_panels(adapter, "fake-vl", page, _args())
    assert len(panels) == 3
    prompt = adapter.calls[0]["prompt"]
    assert "page 2 of 28" in prompt
    assert 'chapter 1 of the manga "Kenja no Mago"' in prompt
    assert "reading order" in prompt
    assert adapter.calls[0]["images"] == [page]


def test_extract_panels_retries_once_then_succeeds(tmp_path):
    adapter = FakeAdapter(["not json at all", GOOD])
    panels = extract_panels(adapter, "m", _page(tmp_path / "p.jpg"), _args())
    assert len(panels) == 3
    assert len(adapter.calls) == 2


def test_extract_panels_falls_back_to_whole_page_describe(tmp_path):
    logs = []
    adapter = FakeAdapter(["garbage", "still garbage",
                           "Shin dives for cover as the cart overturns."])
    panels = extract_panels(
        adapter, "m", _page(tmp_path / "p.jpg"), _args(), log=logs.append
    )
    assert len(adapter.calls) == 3
    assert panels == [Panel(box=[0.0, 0.0, 1.0, 1.0],
                            description="Shin dives for cover as the cart overturns.")]
    assert "Describe what happens on this page" in adapter.calls[2]["prompt"]
    assert any("describing the whole page" in m for m in logs)


def test_extract_panels_persistent_failure_returns_empty(tmp_path):
    logs = []
    adapter = FakeAdapter(["garbage", "still garbage", ""])
    panels = extract_panels(
        adapter, "m", _page(tmp_path / "p.jpg"), _args(), log=logs.append
    )
    assert panels == []
    assert len(adapter.calls) == 3
    assert any("whole-page fallback" in m for m in logs)


def test_extract_panels_jsonish_describe_is_rejected(tmp_path):
    adapter = FakeAdapter(["garbage", "still garbage", "[]"])
    panels = extract_panels(adapter, "m", _page(tmp_path / "p.jpg"), _args())
    assert panels == []


def test_extract_panels_valid_empty_uses_describe_not_retry(tmp_path):
    adapter = FakeAdapter(["[]", "A quiet establishing shot of the valley."])
    panels = extract_panels(adapter, "m", _page(tmp_path / "p.jpg"), _args())
    assert len(adapter.calls) == 2  # second call is the describe prompt
    assert "Describe what happens" in adapter.calls[1]["prompt"]
    assert len(panels) == 1
    assert panels[0].box == [0.0, 0.0, 1.0, 1.0]


# --- cache ---


def _seed_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))


def test_panels_cache_round_trip(tmp_path, monkeypatch):
    _seed_env(tmp_path, monkeypatch)
    pages = {
        "page-001.jpg": [Panel([0.05, 0.04, 0.95, 0.3], "A valley at dawn.")],
        "page-002.jpg": parse_panels(GOOD),
    }
    path = store_panels("s1", "ch-1", pages)
    assert path == panels_path("s1", "ch-1")
    assert path.parent == tmp_path / "works" / "s1" / "chapters" / "ch-1"
    assert load_panels("s1", "ch-1") == pages


def test_load_panels_missing_corrupt_foreign(tmp_path, monkeypatch):
    _seed_env(tmp_path, monkeypatch)
    assert load_panels("s1", "ch-1") is None
    path = panels_path("s1", "ch-1")
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    assert load_panels("s1", "ch-1") is None
    path.write_text('{"version": 999, "pages": {}}')
    assert load_panels("s1", "ch-1") is None


def test_load_panels_drops_malformed_entries(tmp_path, monkeypatch):
    _seed_env(tmp_path, monkeypatch)
    path = panels_path("s1", "ch-1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": 1,
        "pages": {
            "page-001.jpg": [
                {"box": [0.1, 0.1, 0.5, 0.5], "description": "ok"},
                {"box": "bad", "description": "x"},
                {"box": [0.1, 0.1, 0.5, 0.5], "description": ""},
                "junk",
            ],
            "page-002.jpg": [{"box": None}],
        },
    }))
    assert load_panels("s1", "ch-1") == {
        "page-001.jpg": [Panel([0.1, 0.1, 0.5, 0.5], "ok")]
    }


def test_extract_chapter_panels_uses_cache(tmp_path, monkeypatch):
    _seed_env(tmp_path, monkeypatch)
    pages = [_page(tmp_path / f"page-{i:03d}.jpg") for i in (1, 2)]
    store_panels("s1", "ch-1", {
        "page-001.jpg": [Panel([0.0, 0.0, 1.0, 1.0], "cached")]
    })
    adapter = FakeAdapter([GOOD])
    out = extract_chapter_panels(
        adapter, "m", "T", 1.0, "s1", "ch-1", pages
    )
    # Only page-002 hit the model; page-001 came from the cache.
    assert [c["images"][0].name for c in adapter.calls] == ["page-002.jpg"]
    assert len(out["page-001.jpg"]) == 1
    assert len(out["page-002.jpg"]) == 3
    # Fully cached second run: no model calls at all.
    out2 = extract_chapter_panels(
        adapter, "m", "T", 1.0, "s1", "ch-1", pages
    )
    assert len(adapter.calls) == 1
    assert out2 == out


def test_extract_chapter_panels_subset_call_preserves_other_cached_pages(
    tmp_path, monkeypatch
):
    _seed_env(tmp_path, monkeypatch)
    store_panels("s1", "ch-1", {
        "page-001.jpg": [Panel([0.0, 0.0, 1.0, 1.0], "cached one")],
        "page-009.jpg": [Panel([0.0, 0.0, 1.0, 1.0], "unrelated")],
    })
    pages = [_page(tmp_path / f"page-{i:03d}.jpg") for i in (1, 2)]
    adapter = FakeAdapter([GOOD])
    extract_chapter_panels(adapter, "m", "T", 1.0, "s1", "ch-1", pages)
    cached = load_panels("s1", "ch-1")
    assert set(cached) == {"page-001.jpg", "page-002.jpg", "page-009.jpg"}


# --- regions wiring ---


def test_regions_for_panels_orders_without_merging():
    panels = parse_panels(GOOD)
    regs = regions.regions_for_panels(panels)
    assert len(regs) == 3
    assert [r.bubbles for r in regs] == [1, 1, 1]
    top, right, left = regs
    assert top.box[1] < right.box[1]      # the wide top panel leads
    assert right.box[0] > left.box[0]     # RTL within the second band


def test_regions_for_panels_never_merges_adjacent_panels():
    # 0.03 apart: bubble clustering merges these into one region, but
    # panel-level boxes must stay separate panels.
    adjacent = [
        {"box": [0.53, 0.35, 0.95, 0.60], "description": "right"},
        {"box": [0.05, 0.35, 0.50, 0.60], "description": "left"},
    ]
    assert len(regions.cluster_bubbles(adjacent)) == 1
    regs = regions.regions_for_panels(adjacent)
    assert len(regs) == 2
    assert regs[0].box[0] > regs[1].box[0]  # RTL order holds


def test_regions_for_panels_accepts_dicts_and_validates_order():
    # Deliberately scrambled input order comes out in reading order.
    scrambled = [
        {"box": [0.05, 0.35, 0.50, 0.60]},
        {"box": [0.05, 0.04, 0.95, 0.30]},
        {"box": [0.55, 0.35, 0.95, 0.60]},
    ]
    regs = regions.regions_for_panels(scrambled)
    tops = [r.box[1] for r in regs]
    assert tops == sorted(tops) or tops[0] < tops[1]
    assert regs[0].box == pytest.approx((0.03, 0.02, 0.97, 0.32))  # padded
    assert regs[1].box[0] > regs[2].box[0]


def test_regions_for_panels_empty_and_malformed_fall_back():
    whole = [regions.Region((0.0, 0.0, 1.0, 1.0), bubbles=0)]
    assert regions.regions_for_panels([]) == whole
    assert regions.regions_for_panels([{"box": "bad"}, {"no_box": 1}]) == whole
