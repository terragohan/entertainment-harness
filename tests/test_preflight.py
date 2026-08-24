"""Tests for the pre-flight resource guard module."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from entertainment_harness.config import Config, PreflightConfig
from entertainment_harness.hardware import SystemSnapshot
from entertainment_harness.preflight import (
    GB,
    MEMORY_HEADROOM_BYTES,
    PreflightError,
    ResourcePlan,
    check,
    format_plan,
    plan_for_quantize,
    plan_for_recap,
    plan_for_tiktok,
)


def _snapshot(
    *,
    gpu_backend: str = "metal",
    free_ram_gb: float = 16,
    free_disk_gb: float = 50,
    free_vram_gb: float | None = None,
) -> SystemSnapshot:
    return SystemSnapshot(
        gpu_backend=gpu_backend,
        total_ram_bytes=int(32 * GB),
        free_ram_bytes=int(free_ram_gb * GB),
        budget_bytes=int(16 * GB),
        free_disk_bytes=int(free_disk_gb * GB),
        free_vram_bytes=int(free_vram_gb * GB) if free_vram_gb is not None else None,
    )


def _model_need(role: str, size_gb: float, remote: bool = False):
    from entertainment_harness.preflight import ModelNeed

    return ModelNeed(
        role=role,
        backend="ollama" if not remote else "openai_compat",
        name=f"{role}-model",
        size_bytes=int(size_gb * GB),
        remote=remote,
    )


def test_check_passes_when_resources_are_plentiful():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("vision", 5), _model_need("text", 4)],
        peak_memory_bytes=int(5 * GB) + int(4 * GB) + MEMORY_HEADROOM_BYTES,
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=True,
    )
    issues = check(plan, _snapshot(free_ram_gb=16, free_disk_gb=50), PreflightConfig())
    assert issues == []


def test_check_fails_when_gpu_required_but_cpu_only():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("vision", 5)],
        peak_memory_bytes=int(5 * GB) + MEMORY_HEADROOM_BYTES,
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=True,
    )
    issues = check(plan, _snapshot(gpu_backend="cpu"), PreflightConfig())
    assert any("GPU" in i for i in issues)


def test_check_skips_gpu_when_config_says_so():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("vision", 5)],
        peak_memory_bytes=int(5 * GB) + MEMORY_HEADROOM_BYTES,
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=True,
    )
    issues = check(
        plan, _snapshot(gpu_backend="cpu"), PreflightConfig(skip_gpu_check=True)
    )
    assert not any("GPU" in i for i in issues)


def test_check_fails_when_memory_need_exceeds_free_ram():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("vision", 5)],
        peak_memory_bytes=int(10 * GB),
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=False,
    )
    issues = check(plan, _snapshot(free_ram_gb=8), PreflightConfig())
    assert any("Memory need" in i and "free RAM" in i for i in issues)


def test_check_uses_vram_on_cuda():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("vision", 5)],
        peak_memory_bytes=int(10 * GB),
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=True,
    )
    issues = check(
        plan, _snapshot(gpu_backend="cuda", free_vram_gb=8), PreflightConfig()
    )
    assert any("VRAM" in i for i in issues)


def test_check_warns_on_tight_memory_fit():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("text", 4)],
        peak_memory_bytes=int(4 * GB) + MEMORY_HEADROOM_BYTES,
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=False,
    )
    # free RAM = 7 GB, need = 6 GB -> within 2 GB -> warning
    issues = check(plan, _snapshot(free_ram_gb=7), PreflightConfig())
    assert any("tight" in i.lower() for i in issues)


def test_check_fails_when_disk_need_exceeds_free_space():
    plan = ResourcePlan(
        command="recap --video",
        models=[_model_need("text", 4)],
        peak_memory_bytes=int(4 * GB) + MEMORY_HEADROOM_BYTES,
        min_free_disk_bytes=int(20 * GB),
        requires_gpu=False,
    )
    issues = check(plan, _snapshot(free_disk_gb=10), PreflightConfig())
    assert any("Disk need" in i for i in issues)


def test_format_plan_includes_models_and_resources():
    plan = ResourcePlan(
        command="recap",
        models=[_model_need("vision", 5)],
        peak_memory_bytes=int(5 * GB) + MEMORY_HEADROOM_BYTES,
        min_free_disk_bytes=int(1 * GB),
        requires_gpu=True,
    )
    snap = _snapshot()
    text = format_plan(plan, snap)
    assert "Command: recap" in text
    assert "vision-model" in text
    assert "Peak memory" in text
    assert "Min free disk" in text
    assert "Requires GPU: yes" in text
    assert "free RAM" in text


class FakeAdapter:
    name = "ollama"
    remote = False

    def __init__(self, available=None):
        self._available = available or []

    def list_available(self):
        return self._available


def _fake_selection(name: str, size_gb: float, adapter=None):
    from entertainment_harness.models.base import ModelInfo
    from entertainment_harness.models.registry import Selection

    info = ModelInfo(name=name, backend="ollama", size_bytes=int(size_gb * GB))
    return Selection(adapter=adapter or FakeAdapter(), info=info)


def test_plan_for_recap_includes_vision_text_judge():
    config = Config()
    config.pipeline.thinking = "medium"
    profile = MagicMock()
    profile.budget_bytes = int(16 * GB)

    series = {"id": "s1", "kind": "manga"}
    chapter = {"id": "c1", "chapter_num": 1.0, "pages": 10, "lang": "en"}

    with patch(
        "entertainment_harness.preflight.get_vision_model",
        return_value=_fake_selection("qwen3-vl:8b", 5),
    ), patch(
        "entertainment_harness.preflight.get_text_model",
        return_value=_fake_selection("qwen3:4b", 4),
    ), patch(
        "entertainment_harness.preflight.get_judge_model",
        return_value=_fake_selection("qwen3:4b", 4),
    ):
        plan = plan_for_recap(
            config, profile, series, [chapter],
            video=False, translated=False,
        )

    roles = {m.role for m in plan.models}
    assert roles == {"vision", "text", "judge"}
    assert plan.requires_gpu
    assert plan.peak_memory_bytes is not None
    assert plan.min_free_disk_bytes is not None


def test_plan_for_book_does_not_include_vision():
    config = Config()
    profile = MagicMock()
    profile.budget_bytes = int(16 * GB)

    series = {"id": "s1", "kind": "book"}
    chapter = {"id": "c1", "chapter_num": 1.0, "pages": 0, "lang": "en"}

    with patch(
        "entertainment_harness.preflight.get_text_model",
        return_value=_fake_selection("qwen3:4b", 4),
    ), patch(
        "entertainment_harness.preflight.get_judge_model",
        return_value=_fake_selection("qwen3:4b", 4),
    ):
        plan = plan_for_recap(
            config, profile, series, [chapter],
            video=False, translated=False,
        )

    roles = {m.role for m in plan.models}
    assert "vision" not in roles
    assert plan.requires_gpu  # text model is still local


def test_plan_for_tiktok_uses_text_model_only():
    config = Config()
    profile = MagicMock()
    profile.budget_bytes = int(16 * GB)

    with patch(
        "entertainment_harness.preflight.get_text_model",
        return_value=_fake_selection("qwen3:4b", 4),
    ):
        plan = plan_for_tiktok(config, profile)

    assert len(plan.models) == 1
    assert plan.models[0].role == "text"
    assert not plan.requires_gpu


def test_plan_for_quantize_estimates_disk_from_source_size():
    config = Config()
    profile = MagicMock()
    profile.budget_bytes = int(16 * GB)

    class FakeHFAdapter:
        name = "huggingface"
        remote = False

        def supports(self, repo: str):
            from entertainment_harness.models.base import ModelInfo

            return ModelInfo(name=repo, backend="huggingface", size_bytes=int(10 * GB))

    with patch(
        "entertainment_harness.models.registry.get_adapter",
        return_value=FakeHFAdapter(),
    ):
        plan = plan_for_quantize(config, profile, "owner/repo", "Q4_K_M")

    assert plan.min_free_disk_bytes is not None
    assert plan.min_free_disk_bytes >= int(10 * GB)
    assert not plan.requires_gpu
