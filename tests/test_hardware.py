"""Hardware probe tests: budget math + probe parsing (mocked sysctl/psutil)."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from entertainment_harness import hardware
from entertainment_harness.hardware import BUDGET_FRACTION, probe

GB = 10**9


def _sysctl_mock(memsize: int, brand: str = "Apple M4 Pro"):
    def fake_run(cmd, capture_output, text, check):
        key = cmd[-1]
        value = str(memsize) if key == "hw.memsize" else brand
        return subprocess.CompletedProcess(cmd, 0, stdout=value + "\n", stderr="")

    return fake_run


@pytest.fixture
def apple_silicon():
    with (
        patch.object(hardware.platform, "system", return_value="Darwin"),
        patch.object(hardware.platform, "machine", return_value="arm64"),
    ):
        yield


def test_budget_is_65_percent_of_total_ram(apple_silicon):
    total = 24 * GB
    with patch.object(hardware.subprocess, "run", side_effect=_sysctl_mock(total)):
        profile = probe()
    assert profile.total_ram_bytes == total
    assert profile.budget_bytes == int(total * BUDGET_FRACTION)
    assert profile.gpu_backend == "metal"
    assert profile.chip == "Apple M4 Pro"
    assert profile.budget_gb == pytest.approx(15.6, abs=0.1)


def test_budget_override(apple_silicon):
    with patch.object(hardware.subprocess, "run", side_effect=_sysctl_mock(24 * GB)):
        profile = probe(budget_gb=10)
    assert profile.budget_bytes == 10 * GB


def test_sysctl_parses_bytes(apple_silicon):
    with patch.object(
        hardware.subprocess, "run", side_effect=_sysctl_mock(25769803776)
    ):
        profile = probe()
    assert profile.total_ram_bytes == 25769803776


def test_psutil_fallback_when_sysctl_fails():
    with (
        patch.object(hardware.platform, "system", return_value="Linux"),
        patch.object(hardware.platform, "machine", return_value="x86_64"),
        patch.object(hardware.shutil, "which", return_value=None),
        patch.object(hardware.psutil, "virtual_memory") as vm,
    ):
        vm.return_value.total = 32 * GB
        profile = probe()
    assert profile.total_ram_bytes == 32 * GB
    assert profile.budget_bytes == int(32 * GB * BUDGET_FRACTION)
    assert profile.gpu_backend == "cpu"


def test_cuda_backend_detected():
    with (
        patch.object(hardware.platform, "system", return_value="Linux"),
        patch.object(hardware.platform, "machine", return_value="x86_64"),
        patch.object(hardware.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(hardware.psutil, "virtual_memory") as vm,
        patch.object(hardware, "_cuda_vram_bytes", return_value=16 * 1024**3),
    ):
        vm.return_value.total = 64 * GB
        profile = probe()
    assert profile.gpu_backend == "cuda"
    assert profile.budget_bytes == 16 * 1024**3  # VRAM, not RAM fraction
