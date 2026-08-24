"""Machine probe and safe-budget calculation.

Safe inference budget: ~65% of total unified memory (OS + harness need
headroom; on Apple Silicon memory is shared with the GPU). On CUDA systems
the budget is derived from VRAM instead. Overridable via [hardware]
budget_gb in config.toml.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

BUDGET_FRACTION = 0.65


@dataclass
class HardwareProfile:
    chip: str
    total_ram_bytes: int
    budget_bytes: int
    gpu_backend: str  # "metal" | "cuda" | "cpu"

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / 1e9

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / 1e9


@dataclass
class SystemSnapshot:
    """Current machine state for pre-flight checks."""

    gpu_backend: str  # "metal" | "cuda" | "cpu"
    total_ram_bytes: int
    free_ram_bytes: int
    budget_bytes: int
    free_disk_bytes: int
    free_vram_bytes: int | None  # None when not CUDA

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / 1e9

    @property
    def free_ram_gb(self) -> float:
        return self.free_ram_bytes / 1e9

    @property
    def budget_gb(self) -> float:
        return self.budget_bytes / 1e9

    @property
    def free_disk_gb(self) -> float:
        return self.free_disk_bytes / 1e9

    @property
    def free_vram_gb(self) -> float | None:
        return self.free_vram_bytes / 1e9 if self.free_vram_bytes is not None else None


def _sysctl_int(key: str) -> int:
    out = subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


def _sysctl_str(key: str) -> str:
    out = subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _probe_total_ram() -> int:
    if platform.system() == "Darwin":
        try:
            return _sysctl_int("hw.memsize")
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass  # fall through to psutil
    return psutil.virtual_memory().total


def _probe_chip() -> str:
    if platform.system() == "Darwin":
        try:
            return _sysctl_str("machdep.cpu.brand_string")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def _has_cuda() -> bool:
    return shutil.which("nvidia-smi") is not None


def _cuda_vram_bytes() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        mib = int(out.stdout.strip().splitlines()[0])
        return mib * 1024 * 1024
    except (subprocess.CalledProcessError, ValueError, IndexError, FileNotFoundError):
        return None


def _cuda_vram_free_bytes() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        mib = int(out.stdout.strip().splitlines()[0])
        return mib * 1024 * 1024
    except (subprocess.CalledProcessError, ValueError, IndexError, FileNotFoundError):
        return None


def _free_ram_bytes() -> int:
    return psutil.virtual_memory().available


def _free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def snapshot(data_dir: Path, budget_gb: float | None = None) -> SystemSnapshot:
    """Capture current machine state for pre-flight checks."""
    total_ram = _probe_total_ram()

    if _is_apple_silicon():
        gpu_backend = "metal"
    elif _has_cuda():
        gpu_backend = "cuda"
    else:
        gpu_backend = "cpu"

    if budget_gb is not None:
        budget = int(budget_gb * 1e9)
    elif gpu_backend == "cuda":
        vram = _cuda_vram_bytes()
        budget = vram if vram is not None else int(total_ram * BUDGET_FRACTION)
    else:
        budget = int(total_ram * BUDGET_FRACTION)

    free_vram = _cuda_vram_free_bytes() if gpu_backend == "cuda" else None

    return SystemSnapshot(
        gpu_backend=gpu_backend,
        total_ram_bytes=total_ram,
        free_ram_bytes=_free_ram_bytes(),
        budget_bytes=budget,
        free_disk_bytes=_free_disk_bytes(data_dir),
        free_vram_bytes=free_vram,
    )


def probe(budget_gb: float | None = None) -> HardwareProfile:
    """Probe the machine. budget_gb overrides the derived safe budget."""
    chip = _probe_chip()
    total_ram = _probe_total_ram()

    if _is_apple_silicon():
        gpu_backend = "metal"
    elif _has_cuda():
        gpu_backend = "cuda"
    else:
        gpu_backend = "cpu"

    if budget_gb is not None:
        budget = int(budget_gb * 1e9)
    elif gpu_backend == "cuda":
        vram = _cuda_vram_bytes()
        budget = vram if vram is not None else int(total_ram * BUDGET_FRACTION)
    else:
        budget = int(total_ram * BUDGET_FRACTION)

    return HardwareProfile(
        chip=chip,
        total_ram_bytes=total_ram,
        budget_bytes=budget,
        gpu_backend=gpu_backend,
    )
