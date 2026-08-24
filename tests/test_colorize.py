"""Colorizer tests: DDColor ONNX session and HF download are faked.
No network, no model weights needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from entertainment_harness.video.colorize import ColorizeError, Colorizer


class FakeSession:
    """Stands in for ort.InferenceSession: zero-chroma output, records runs."""

    def __init__(self, path, providers=None):
        self.runs = 0

    def get_inputs(self):
        class In:
            name = "input"

        return [In()]

    def run(self, outputs, feed):
        self.runs += 1
        x = feed["input"]
        assert x.shape == (1, 3, 512, 512)  # fixed-input contract
        return [np.zeros((1, 2, 512, 512), dtype=np.float32)]  # ab = 0


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(tmp_path / "m.onnx")
    )
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    return tmp_path


def _write_gray_page(path):
    cv2.imwrite(str(path), np.full((200, 100, 3), 128, dtype=np.uint8))


def test_colorize_page_preserves_luminance(fake_ort, tmp_path):
    src = tmp_path / "page-001.png"
    _write_gray_page(src)
    dest = tmp_path / "out" / "page-001.png"
    c = Colorizer(model_dir=tmp_path / "model")
    out = c.colorize_page(src, dest)
    assert out == dest and dest.exists()
    # ab = 0 -> output is the input's grayscale at original resolution
    result = cv2.imread(str(dest))
    assert result.shape == (200, 100, 3)
    assert abs(int(result.mean()) - 128) <= 2


def test_colorize_page_caches(fake_ort, tmp_path):
    src = tmp_path / "page-001.png"
    _write_gray_page(src)
    c = Colorizer(model_dir=tmp_path / "model")
    dest = tmp_path / "out" / "page-001.png"
    c.colorize_page(src, dest)
    c.colorize_page(src, dest)  # cached: no second inference
    assert c._session.runs == 1


def test_colorize_page_unreadable_raises(fake_ort, tmp_path):
    c = Colorizer(model_dir=tmp_path / "model")
    with pytest.raises(ColorizeError, match="Could not read"):
        c.colorize_page(tmp_path / "missing.png", tmp_path / "out.png")


def test_colorize_pages_maps_and_logs(fake_ort, tmp_path, capsys):
    srcs = []
    for i in range(3):
        p = tmp_path / f"page-{i:03d}.jpg"
        _write_gray_page(p)
        srcs.append(p)
    c = Colorizer(model_dir=tmp_path / "model")
    mapping = c.colorize_pages(srcs, tmp_path / "colorized", log=print)
    assert set(mapping) == set(srcs)
    assert all(d.suffix == ".png" and d.exists() for d in mapping.values())


# --- colorize plugin registry (Phase 5) -----------------------------------------


def test_colorize_registry_roundtrip():
    from entertainment_harness.config import Config
    from entertainment_harness.plugins import PluginError
    from entertainment_harness.video.colorize import REGISTRY, get_colorizer

    assert REGISTRY.names() == ["ddcolor"]
    colorizer = get_colorizer("ddcolor", Config())
    assert isinstance(colorizer, Colorizer)
    assert colorizer.name == "ddcolor"
    assert colorizer.capabilities == frozenset({"colorize"})
    with pytest.raises(PluginError):
        get_colorizer("nope")


def test_colorize_provider_config_parsed():
    from entertainment_harness.config import parse_config

    assert parse_config({}).colorize.provider == "ddcolor"
    assert parse_config({"colorize": {"provider": "x"}}).colorize.provider == "x"
