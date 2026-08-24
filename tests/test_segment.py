"""Tests for the optional SAM segmentation module.

These tests do not download the SAM checkpoint; they only exercise the
module's import-time guards and path helpers.
"""

from pathlib import Path

import pytest

from entertainment_harness.models import segment
from entertainment_harness.config import cache_dir


def test_default_checkpoint_path():
    assert segment.default_checkpoint_path() == cache_dir() / "models" / "sam_vit_b_01ec64.pth"


def test_segmenter_refuses_without_dependencies(monkeypatch):
    monkeypatch.setattr(segment, "_SAM_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="SAM dependencies are not installed"):
        segment.SamSegmenter(Path("/fake/checkpoint.pth"))
