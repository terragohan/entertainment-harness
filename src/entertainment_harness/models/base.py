"""Model adapter protocol, ModelInfo, and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ModelInfo:
    name: str  # full tag, e.g. "qwen3-vl:8b-instruct"
    backend: str  # adapter name, e.g. "ollama"
    params: float | None = None  # billions of parameters, if known
    quant: str | None = None  # e.g. "Q4_K_M", if known
    size_bytes: int | None = None  # on-disk weight size, if known
    # What this artifact can do, e.g. {"vision"} — adapters derive it from
    # registry metadata (HF: the repo ships an mmproj projector file).
    capabilities: frozenset[str] = field(default_factory=frozenset)

    @property
    def size_gb(self) -> float | None:
        return self.size_bytes / 1e9 if self.size_bytes is not None else None


class ModelError(Exception):
    """Base error for the model adapter layer."""


class ModelNotFoundError(ModelError):
    """The requested model is unknown to the adapter's registry."""


class ModelTooLargeError(ModelError):
    """No quant of the requested model fits the budget."""

    def __init__(self, model: str, alternatives: list[ModelInfo]) -> None:
        self.model = model
        self.alternatives = alternatives
        lines = [f"Model {model!r} does not fit the memory budget at any known quant."]
        if alternatives:
            lines.append("Models that would fit:")
            lines.extend(f"  - {alt.name} ({alt.size_gb:.1f} GB)" for alt in alternatives)
        else:
            lines.append("No smaller alternative is locally available.")
        super().__init__("\n".join(lines))


class ModelAdapter(Protocol):
    name: str
    remote: bool  # True = hosted endpoint; registry skips local quant/budget selection
    capabilities: frozenset[str]  # what the backend itself offers (model-level
    # capabilities like "vision" live on ModelInfo)

    def supports(self, model: str) -> ModelInfo:
        """Resolve a model ID, incl. per-quant sizes where the registry reports them."""
        ...

    def ensure(self, model: str, quant: str | None = None) -> None:
        """Pull/download the model (at the given quant) if missing locally."""
        ...

    def remove(self, model: str) -> None:
        """Delete the local copy of a model; it stays in the remote registry."""
        ...

    def list_available(self) -> list[ModelInfo]:
        """Query the local registry for installed models."""
        ...

    def list_remote(self, model: str) -> list[ModelInfo] | None:
        """Per-quant remote registry metadata (sizes without downloading).
        None = the backend has no remote per-quant view (the registry falls
        back to supports()+tag heuristics for those)."""
        ...

    def generate(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        num_ctx: int = 16384,
    ) -> str:
        """Run inference; images are file paths for vision-capable models."""
        ...
