"""Regions module: bubble clustering, reading order, fallback, cache."""

import math

import pytest

from entertainment_harness.video.regions import (
    Region,
    cluster_bubbles,
    load_regions,
    reading_order,
    regions_for_page,
    regions_path,
    sanitize_box,
    store_regions,
)


def _bubble(x0, y0, x1, y1):
    return {"box": [x0, y0, x1, y1], "original": "…", "translation": "…"}


# --- sanitize_box ---


def test_sanitize_box_clamps_out_of_range():
    assert sanitize_box([-0.5, 0.1, 0.5, 1.5]) == (0.0, 0.1, 0.5, 1.0)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "0.1,0.1,0.2,0.2",
        [0.1, 0.1, 0.2],               # wrong length
        [0.1, 0.1, 0.2, 0.2, 0.3],     # wrong length
        ["a", "b", "c", "d"],          # non-numeric
        [0.1, None, 0.2, 0.2],
        [0.1, math.nan, 0.2, 0.2],     # non-finite
        [0.1, 0.1, math.inf, 0.2],
        [0.5, 0.1, 0.2, 0.3],          # inverted x
        [0.1, 0.5, 0.3, 0.2],          # inverted y
        [0.2, 0.2, 0.2, 0.5],          # zero width
        [1.5, 0.1, 0.5, 0.3],          # inverted after clamping
    ],
)
def test_sanitize_box_drops_malformed(raw):
    assert sanitize_box(raw) is None


# --- clustering ---


def test_adjacent_bubbles_merge_into_one_padded_region():
    # 0.02 x-gap: the padding-expanded boxes overlap, so one panel.
    regions = cluster_bubbles([_bubble(0.1, 0.1, 0.3, 0.2), _bubble(0.32, 0.12, 0.5, 0.22)])
    assert len(regions) == 1
    assert regions[0].bubbles == 2
    assert regions[0].box == pytest.approx((0.08, 0.08, 0.52, 0.24))


def test_distant_bubbles_stay_separate():
    regions = cluster_bubbles([_bubble(0.05, 0.05, 0.2, 0.15), _bubble(0.6, 0.6, 0.8, 0.7)])
    assert len(regions) == 2
    assert all(r.bubbles == 1 for r in regions)


def test_merge_is_transitive():
    # A near B, B near C, but A and C far apart: one cluster.
    regions = cluster_bubbles([
        _bubble(0.10, 0.10, 0.20, 0.15),
        _bubble(0.23, 0.10, 0.33, 0.15),
        _bubble(0.36, 0.10, 0.46, 0.15),
    ])
    assert len(regions) == 1
    assert regions[0].bubbles == 3


def test_region_padding_clamps_at_page_edges():
    regions = cluster_bubbles([_bubble(0.0, 0.0, 0.1, 0.1)])
    assert regions[0].box == pytest.approx((0.0, 0.0, 0.12, 0.12))


def test_cluster_drops_malformed_keeps_valid():
    regions = cluster_bubbles([
        {"box": [0.1, 0.1, 0.2, 0.2]},
        {"box": "garbage"},
        {"box": [0.9, 0.9]},          # wrong length
        {"no_box": True},
        "not a mapping",
    ])
    assert len(regions) == 1
    assert regions[0].bubbles == 1


# --- reading order ---


def test_reading_order_top_bands_first_rtl_within_band():
    top_left = Region((0.05, 0.05, 0.45, 0.40))
    top_right = Region((0.55, 0.05, 0.95, 0.40))
    bottom = Region((0.20, 0.50, 0.80, 0.90))
    ordered = reading_order([top_left, bottom, top_right])
    assert ordered == [top_right, top_left, bottom]


def test_reading_order_tall_right_panel_joins_top_band():
    # A page-spanning right column is read before the left rows it spans.
    tall_right = Region((0.55, 0.05, 0.95, 0.90))
    left_top = Region((0.05, 0.05, 0.45, 0.40))
    left_bottom = Region((0.05, 0.50, 0.45, 0.90))
    ordered = reading_order([left_bottom, left_top, tall_right])
    assert ordered == [tall_right, left_top, left_bottom]


def test_reading_order_vertically_separate_regions_never_share_a_band():
    top = Region((0.05, 0.05, 0.45, 0.20))
    mid = Region((0.50, 0.30, 0.95, 0.45))
    bottom = Region((0.05, 0.55, 0.45, 0.70))
    assert reading_order([bottom, mid, top]) == [top, mid, bottom]


def test_reading_order_full_height_right_column_reads_first_then_rows():
    # kenja ch-001 p1 geometry: a page-tall right panel must not collapse
    # every row into one band (which scrambled the order live).
    establishing = Region((0.45, 0.0, 0.93, 1.0))
    forest = Region((0.0, 0.0, 0.43, 0.27))
    eye = Region((0.0, 0.27, 0.23, 0.52))
    bird = Region((0.24, 0.27, 0.43, 0.52))
    hand = Region((0.0, 0.52, 0.43, 1.0))
    assert reading_order([forest, eye, bird, hand, establishing]) == [
        establishing, forest, bird, eye, hand,
    ]


def test_reading_order_left_tall_panel_reads_after_top_row():
    # kenja ch-001 p5 geometry: a left panel spanning a 2x2 grid's rows is
    # read after the top row of the grid, not after the whole block.
    left = Region((0.0, 0.0, 0.45, 0.45))
    tr = Region((0.62, 0.06, 0.83, 0.24))
    tm = Region((0.46, 0.06, 0.62, 0.24))
    br = Region((0.62, 0.27, 0.83, 0.45))
    bm = Region((0.46, 0.27, 0.62, 0.45))
    bottom = Region((0.0, 0.48, 0.83, 0.63))
    assert reading_order([left, bm, br, tm, tr, bottom]) == [
        tr, tm, left, br, bm, bottom,
    ]


# --- page-level entry point ---


def test_regions_for_page_orders_clusters():
    bubbles = [
        _bubble(0.60, 0.05, 0.90, 0.10),  # top-right
        _bubble(0.10, 0.05, 0.40, 0.10),  # top-left
        _bubble(0.30, 0.60, 0.70, 0.65),  # bottom
    ]
    boxes = [r.box for r in regions_for_page(bubbles)]
    assert boxes[0][0] > boxes[1][0]      # right before left in the top band
    assert boxes[2][1] > boxes[1][1]      # bottom band last


def test_empty_bubbles_fall_back_to_whole_page():
    assert regions_for_page([]) == [Region((0.0, 0.0, 1.0, 1.0), bubbles=0)]


def test_all_malformed_falls_back_to_whole_page():
    assert regions_for_page([{"box": "garbage"}, {"box": [1, 2]}]) == [
        Region((0.0, 0.0, 1.0, 1.0), bubbles=0)
    ]


# --- cache ---


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    pages = {
        "page-001.jpg": [
            Region((0.08, 0.08, 0.52, 0.24), bubbles=2),
            Region((0.2, 0.5, 0.8, 0.9)),
        ],
        "page-002.jpg": [Region((0.0, 0.0, 1.0, 1.0), bubbles=0)],
    }
    path = store_regions("s1", "ch-1", pages)
    assert path == regions_path("s1", "ch-1")
    assert path.parent == tmp_path / "works" / "s1" / "chapters" / "ch-1"
    assert load_regions("s1", "ch-1") == pages


def test_load_regions_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    assert load_regions("s1", "ch-1") is None


def test_load_regions_corrupt_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    path = regions_path("s1", "ch-1")
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    assert load_regions("s1", "ch-1") is None
    path.write_text('{"version": 999, "pages": {}}')  # foreign version
    assert load_regions("s1", "ch-1") is None


def test_load_regions_drops_malformed_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    path = regions_path("s1", "ch-1")
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"version": 1, "pages": {'
        ' "page-001.jpg": [{"box": [0.1, 0.1, 0.5, 0.5], "bubbles": 2},'
        '                  {"box": "bad"}, "junk"],'
        ' "page-002.jpg": [{"box": null}],'
        ' "page-003.jpg": "junk"}}'
    )
    assert load_regions("s1", "ch-1") == {
        "page-001.jpg": [Region((0.1, 0.1, 0.5, 0.5), bubbles=2)]
    }
