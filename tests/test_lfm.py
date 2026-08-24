"""Tests for LFM2.5-VL parsing helpers. No model downloads."""

from __future__ import annotations

import pytest

from entertainment_harness.models.lfm import (
    _extract_json_array,
    _parse_results,
    _repair_json,
    is_available,
)


def test_is_available_matches_import_state():
    # The module either has transformers/torchvision or it doesn't.
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForImageTextToText  # noqa: F401

        expected = True
    except Exception:
        expected = False
    assert is_available() is expected


# --- _extract_json_array ------------------------------------------------------


def test_extract_json_array_finds_outermost_array():
    raw = 'Some prose [{"a": 1}, {"b": 2}] trailing'
    assert _extract_json_array(raw) == '[{"a": 1}, {"b": 2}]'


def test_extract_json_array_prefers_longest_top_level_array():
    raw = '[1] [{"x": [1, 2, 3]}] [2]'
    assert _extract_json_array(raw) == '[{"x": [1, 2, 3]}]'


def test_extract_json_array_ignores_brackets_in_strings():
    raw = '[{"note": "has [brackets] inside"}, 1]'
    assert _extract_json_array(raw) == raw


def test_extract_json_array_returns_none_when_no_array():
    assert _extract_json_array('{"a": 1}') is None


def test_extract_json_array_handles_nested_arrays():
    raw = '[[1, 2], [3, 4]]'
    assert _extract_json_array(raw) == raw


# --- _repair_json -------------------------------------------------------------


def test_repair_json_parses_clean_array():
    raw = '[{"bbox": [0.1, 0.2, 0.3, 0.4], "type": "speech"}]'
    data = _repair_json(raw)
    assert data == [{"bbox": [0.1, 0.2, 0.3, 0.4], "type": "speech"}]


def test_repair_json_repairs_trailing_comma():
    raw = '[{"bbox": [0.1, 0.2, 0.3, 0.4], "type": "speech"},]'
    data = _repair_json(raw)
    assert len(data) == 1


def test_repair_json_raises_when_no_array_present():
    with pytest.raises(ValueError):
        _repair_json('{"a": 1}')


# --- _parse_results -----------------------------------------------------------


def test_parse_results_normalizes_0_1_bbox():
    data = [{"bbox": [0.1, 0.2, 0.3, 0.4], "type": "speech", "text": "hi"}]
    results = _parse_results(data, include_text=True)
    assert len(results) == 1
    assert results[0]["box"] == [0.1, 0.2, 0.3, 0.4]
    assert results[0]["kind"] == "speech"
    assert results[0]["original"] == "hi"


def test_parse_results_normalizes_0_1000_grid():
    data = [{"bbox_2d": [91, 146, 264, 210], "label": "caption"}]
    results = _parse_results(data, include_text=False)
    assert len(results) == 1
    assert results[0]["box"] == pytest.approx([0.091, 0.146, 0.264, 0.21])
    assert results[0]["kind"] == "caption"
    assert results[0]["original"] == ""


def test_parse_results_swaps_inverted_coords():
    data = [{"bbox": [0.9, 0.8, 0.1, 0.2], "type": "speech"}]
    results = _parse_results(data, include_text=False)
    assert results[0]["box"] == [0.1, 0.2, 0.9, 0.8]


def test_parse_results_uses_aliases():
    data = [
        {"box": [0.1, 0.1, 0.2, 0.2], "kind": "bubble", "content": "hello"},
        {"bbox_2d": [100, 100, 200, 200], "label": "caption", "original": "world"},
    ]
    results = _parse_results(data, include_text=True)
    assert len(results) == 2
    assert results[0]["original"] == "hello"
    assert results[1]["original"] == "world"


def test_parse_results_drops_non_dicts_and_bad_bboxes():
    data = [
        "not a dict",
        {"bbox": [0.1, 0.2]},
        {"bbox": ["a", "b", "c", "d"]},
        {"bbox": [0.1, 0.2, 0.3, 0.4]},
    ]
    results = _parse_results(data, include_text=False)
    assert len(results) == 1
    assert results[0]["box"] == [0.1, 0.2, 0.3, 0.4]


# --- LFMAdapter: lfm as a registered model backend (Phase 5) ---------------------


def test_lfm_registered_as_model_backend():
    from entertainment_harness.models.lfm import LFMAdapter
    from entertainment_harness.models.registry import REGISTRY

    assert "lfm" in REGISTRY.names()
    assert REGISTRY.load("lfm") is LFMAdapter
    assert LFMAdapter.name == "lfm"
    assert LFMAdapter.remote is False
    assert LFMAdapter.capabilities == frozenset({"vision"})


def test_lfm_adapter_supports_marks_vision():
    from entertainment_harness.models.lfm import LFMAdapter

    adapter = LFMAdapter()
    info = adapter.supports("LiquidAI/LFM2.5-VL-450M")
    assert info.backend == "lfm"
    assert info.capabilities == frozenset({"vision"})
    assert adapter.list_available() == []
    assert adapter.list_remote("x") is None


def test_lfm_locator_from_stage_resolves_via_registry(monkeypatch):
    import entertainment_harness.models.lfm as lfm
    from entertainment_harness.config import StageConfig
    from entertainment_harness.pipelines import translate

    monkeypatch.setattr(lfm, "_LFM_AVAILABLE", True)
    locator = translate._lfm_locator_from_stage(
        StageConfig(backend="lfm", model="m-x"), log=lambda m: None
    )
    assert isinstance(locator, lfm.LFMLocator)
    assert locator.model_id == "m-x"


def test_lfm_locator_from_stage_warns_without_extra(monkeypatch):
    import entertainment_harness.models.lfm as lfm
    from entertainment_harness.config import StageConfig
    from entertainment_harness.pipelines import translate

    logs: list[str] = []
    monkeypatch.setattr(lfm, "_LFM_AVAILABLE", False)
    locator = translate._lfm_locator_from_stage(
        StageConfig(backend="lfm", model="m-x"), log=logs.append
    )
    assert locator is None
    assert any("dataset extra" in m for m in logs)
