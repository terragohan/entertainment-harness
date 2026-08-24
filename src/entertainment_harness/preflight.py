"""Pre-flight resource guards for heavy pipelines.

A pipeline builds an auto-derived ResourcePlan (models, peak memory, disk
needs) and checks it against a live SystemSnapshot before starting work.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.config import Config, PreflightConfig, cache_dir
from entertainment_harness.hardware import HardwareProfile, SystemSnapshot
from entertainment_harness.models.registry import (
    get_judge_model,
    get_text_model,
    get_vision_model,
)

GB = 10**9
# Extra headroom beyond model weights for activations / KV cache / OS jitter.
MEMORY_HEADROOM_BYTES = int(2 * GB)
# Per-page heuristics for disk budgeting.
PAGE_DOWNLOAD_BYTES = 200_000
TRANSLATED_PAGE_BYTES = 500_000
VIDEO_PER_PAGE_BYTES = 2_000_000
VIDEO_PER_SEGMENT_BYTES = 1_000_000
TIKTOK_OUTPUT_BYTES = 200_000_000
QUANTIZE_SCRATCH_BYTES = int(1 * GB)


class PreflightError(Exception):
    """A resource check failed before the pipeline started."""


@dataclass
class ModelNeed:
    role: str
    backend: str
    name: str
    size_bytes: int | None
    remote: bool


@dataclass
class ResourcePlan:
    command: str
    models: list[ModelNeed]
    peak_memory_bytes: int | None
    min_free_disk_bytes: int | None
    requires_gpu: bool


def _model_is_local(adapter, name: str) -> bool:
    """Return True if the named model appears in the adapter's local list."""
    try:
        available = adapter.list_available()
    except Exception:
        return False
    return any(a.name == name for a in available)


def _model_need_from_selection(selection, role: str) -> ModelNeed:
    info = selection.info
    size = info.size_bytes
    remote = getattr(selection.adapter, "remote", False)
    return ModelNeed(
        role=role,
        backend=selection.adapter.name,
        name=info.name,
        size_bytes=size,
        remote=remote,
    )


def _model_size(need: ModelNeed) -> int:
    return need.size_bytes or 0


def _local_models(models: list[ModelNeed]) -> list[ModelNeed]:
    return [m for m in models if not m.remote]


def _peak_memory_bytes(models: list[ModelNeed]) -> int | None:
    """Sum of local model sizes plus working headroom."""
    local = _local_models(models)
    if not local:
        return None
    total = sum(_model_size(m) for m in local)
    if total == 0:
        return None
    return total + MEMORY_HEADROOM_BYTES


def _count_uncached_pages(series_id: str, chapters: list[sqlite3.Row]) -> tuple[int, int]:
    """Return (total_pages, uncached_pages)."""
    total_pages = 0
    uncached_pages = 0
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    for chapter in chapters:
        pages = chapter["pages"] or 0
        total_pages += pages
        source_dir = works.source_dir(series_id, chapter["id"])
        if pages and source_dir.is_dir():
            cached = len(
                [
                    p
                    for p in source_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in image_suffixes
                ]
            )
            uncached_pages += max(0, pages - cached)
        else:
            uncached_pages += pages
    return total_pages, uncached_pages


def plan_for_recap(
    config: Config,
    profile: HardwareProfile,
    series: sqlite3.Row,
    chapters: list[sqlite3.Row],
    *,
    video: bool,
    translated: bool,
) -> ResourcePlan:
    """Build a resource plan for eh recap."""
    command = "recap"
    is_book = series["kind"] == "book"
    models: list[ModelNeed] = []
    selections: dict[str, object] = {}

    text = get_text_model(config, profile)
    selections["text"] = text
    models.append(_model_need_from_selection(text, "text"))

    vision = None
    if not is_book:
        vision = get_vision_model(config, profile)
        selections["vision"] = vision
        models.append(_model_need_from_selection(vision, "vision"))

    # Judge is loaded only at medium/high thinking.
    if config.pipeline.thinking in ("medium", "high"):
        judge = get_judge_model(config, profile)
        selections["judge"] = judge
        models.append(_model_need_from_selection(judge, "judge"))

    # Disk budget.
    total_pages, uncached_pages = _count_uncached_pages(series["id"], chapters)
    disk = uncached_pages * PAGE_DOWNLOAD_BYTES

    if translated and not is_book:
        disk += total_pages * TRANSLATED_PAGE_BYTES

    if video:
        # A script is ~1 segment per 2 pages on average.
        estimated_segments = max(1, total_pages // 2)
        disk += estimated_segments * VIDEO_PER_SEGMENT_BYTES
        disk += total_pages * VIDEO_PER_PAGE_BYTES

    # Models that still need to be pulled/downloaded cost disk too.
    for need in models:
        selection = selections.get(need.role)
        if selection is None:
            continue
        if not need.remote and need.size_bytes and not _model_is_local(selection.adapter, need.name):
            disk += need.size_bytes

    # Config overrides.
    if config.preflight.min_free_disk_gb is not None:
        disk = max(disk, int(config.preflight.min_free_disk_gb * GB))

    peak = _peak_memory_bytes(models)
    if config.preflight.min_free_memory_gb is not None:
        peak = max(peak or 0, int(config.preflight.min_free_memory_gb * GB)) or None

    requires_gpu = any(not m.remote for m in models)

    return ResourcePlan(
        command=f"{command} --video" if video else command,
        models=models,
        peak_memory_bytes=peak,
        min_free_disk_bytes=disk or None,
        requires_gpu=requires_gpu,
    )


def plan_for_tiktok(config: Config, profile: HardwareProfile) -> ResourcePlan:
    """Build a resource plan for eh tiktok."""
    text = get_text_model(config, profile)
    need = _model_need_from_selection(text, "text")
    disk = TIKTOK_OUTPUT_BYTES

    if config.preflight.min_free_disk_gb is not None:
        disk = max(disk, int(config.preflight.min_free_disk_gb * GB))

    peak = _peak_memory_bytes([need])
    if config.preflight.min_free_memory_gb is not None:
        peak = max(peak or 0, int(config.preflight.min_free_memory_gb * GB)) or None

    return ResourcePlan(
        command="tiktok",
        models=[need],
        peak_memory_bytes=peak,
        min_free_disk_bytes=disk,
        requires_gpu=False,
    )


def plan_for_quantize(
    config: Config,
    profile: HardwareProfile,
    repo: str,
    target_quant: str,
) -> ResourcePlan:
    """Build a resource plan for eh quantize.

    llama.cpp quantization is CPU-only; memory needs are modest. Disk is the
    dominant requirement: source FP16/BF16 weights + output quant file.
    """
    from entertainment_harness.models.registry import get_adapter

    adapter = get_adapter("huggingface", config)
    source_size: int | None = None
    try:
        info = adapter.supports(repo)
        source_size = info.size_bytes
    except Exception:
        source_size = None

    # Output quant is typically ~30-60% of source size; use 60% as conservative.
    output_size = int((source_size or 0) * 0.6)
    disk = (source_size or 0) + output_size + QUANTIZE_SCRATCH_BYTES

    if config.preflight.min_free_disk_gb is not None:
        disk = max(disk, int(config.preflight.min_free_disk_gb * GB))

    return ResourcePlan(
        command="quantize",
        models=[],
        peak_memory_bytes=int(4 * GB),
        min_free_disk_bytes=disk or None,
        requires_gpu=False,
    )


def _model_cache_dir() -> Path:
    """Directory where the Hugging Face adapter caches GGUF weights."""
    return cache_dir() / "models" / "hf"


def _free_disk_bytes(path: Path) -> int:
    """Free space on the drive hosting ``path`` (nearest existing ancestor)."""
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def check(plan: ResourcePlan, snapshot: SystemSnapshot, config: PreflightConfig) -> list[str]:
    """Check a plan against current machine state.

    Returns a list of human-readable issues. Empty list means the plan passes.
    Issues are ordered errors first, then warnings.
    """
    issues: list[str] = []

    if plan.requires_gpu and not config.skip_gpu_check:
        if snapshot.gpu_backend == "cpu":
            issues.append(
                "This pipeline requires a local GPU, but only CPU was detected."
            )

    if plan.peak_memory_bytes:
        if snapshot.gpu_backend == "cuda":
            free = snapshot.free_vram_bytes
            label = "free VRAM"
        else:
            free = snapshot.free_ram_bytes
            label = "free RAM"

        if free is not None:
            if plan.peak_memory_bytes > free:
                issues.append(
                    f"Memory need {plan.peak_memory_bytes / GB:.1f} GB exceeds "
                    f"available {label} {free / GB:.1f} GB."
                )
            elif free - plan.peak_memory_bytes < int(2 * GB):
                issues.append(
                    f"Memory need {plan.peak_memory_bytes / GB:.1f} GB is within "
                    f"2 GB of available {label} {free / GB:.1f} GB — tight fit."
                )

        # On Apple Silicon unified memory is shared with the GPU; free RAM can
        # shrink quickly under pressure. Warn when the plan eats most of the
        # safe budget even if current free RAM looks okay.
        if snapshot.gpu_backend == "metal" and snapshot.budget_bytes:
            if plan.peak_memory_bytes > snapshot.budget_bytes * 0.8:
                issues.append(
                    f"Memory need {plan.peak_memory_bytes / GB:.1f} GB is above "
                    f"80% of the safe budget {snapshot.budget_bytes / GB:.1f} GB "
                    "on Apple Silicon."
                )

    if plan.min_free_disk_bytes:
        # Model downloads live in cache_dir, but check it explicitly in case it
        # is on a different drive than the data dir.
        free_disk = min(
            snapshot.free_disk_bytes,
            _free_disk_bytes(_model_cache_dir()),
        )
        if plan.min_free_disk_bytes > free_disk:
            issues.append(
                f"Disk need {plan.min_free_disk_bytes / GB:.1f} GB exceeds "
                f"free space {free_disk / GB:.1f} GB."
            )

    return issues


def format_plan(plan: ResourcePlan, snapshot: SystemSnapshot | None = None) -> str:
    """Human-readable summary of a resource plan."""
    lines = [f"Command: {plan.command}", "Models:"]
    if plan.models:
        for m in plan.models:
            size = f"{m.size_bytes / GB:.1f} GB" if m.size_bytes else "unknown"
            remote = " (remote)" if m.remote else ""
            lines.append(f"  - {m.role}: {m.name} ({m.backend}, {size}){remote}")
    else:
        lines.append("  - none")

    if plan.peak_memory_bytes:
        lines.append(
            f"Peak memory: {plan.peak_memory_bytes / GB:.1f} GB "
            "(models + working headroom)"
        )
    if plan.min_free_disk_bytes:
        lines.append(f"Min free disk: {plan.min_free_disk_bytes / GB:.1f} GB")
    lines.append(f"Requires GPU: {'yes' if plan.requires_gpu else 'no'}")

    if snapshot is not None:
        lines.append(
            f"Current: {snapshot.free_ram_gb:.1f} GB free RAM, "
            f"{snapshot.free_disk_gb:.1f} GB free disk"
            + (
                f", {snapshot.free_vram_gb:.1f} GB free VRAM"
                if snapshot.free_vram_gb is not None
                else ""
            )
        )

    return "\n".join(lines)
