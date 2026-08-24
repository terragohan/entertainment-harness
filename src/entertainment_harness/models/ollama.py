"""OllamaAdapter — Ollama local HTTP API (default backend).

Uses GET /api/tags for local models, POST /api/show for model metadata,
POST /api/pull for ensure(), DELETE /api/delete for remove(), POST /api/chat
for generate (base64 images).
Quant and size come from real API data: /api/tags reports on-disk size and
details.quantization_level (e.g. "Q4_K_M").
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import httpx

from entertainment_harness.config import Config, load_config
from entertainment_harness.models.base import ModelInfo, ModelNotFoundError

_PARAMS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([BM])", re.IGNORECASE)


def _parse_params(value: str | None) -> float | None:
    """Parse a parameter count like "8.2B" or "400M" into billions."""
    if not value:
        return None
    match = _PARAMS_RE.search(value)
    if not match:
        return None
    number = float(match.group(1))
    return number / 1000 if match.group(2).upper() == "M" else number


class OllamaAdapter:
    name = "ollama"
    remote = False

    def __init__(
        self,
        config: Config | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if base_url is None:
            base_url = (config or load_config()).models.ollama.base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def _info_from_tag(self, entry: dict) -> ModelInfo:
        details = entry.get("details", {})
        return ModelInfo(
            name=entry["name"],
            backend=self.name,
            params=_parse_params(details.get("parameter_size")),
            quant=details.get("quantization_level") or None,
            size_bytes=entry.get("size"),
        )

    def list_available(self) -> list[ModelInfo]:
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return [self._info_from_tag(m) for m in resp.json().get("models", [])]

    def list_remote(self, model: str) -> list[ModelInfo] | None:
        # Ollama has no per-quant remote size view; the registry falls back
        # to supports() + tag heuristics for it.
        return None

    def supports(self, model: str) -> ModelInfo:
        # Local install is authoritative for size; fall back to /api/show
        # metadata (works for models known to the Ollama library remotely).
        for info in self.list_available():
            if info.name == model or info.name == f"{model}:latest":
                return info
        resp = self._client.post("/api/show", json={"model": model})
        if resp.status_code == 404:
            raise ModelNotFoundError(f"Ollama does not know model {model!r}")
        resp.raise_for_status()
        details = resp.json().get("details", {})
        return ModelInfo(
            name=model,
            backend=self.name,
            params=_parse_params(details.get("parameter_size")),
            quant=details.get("quantization_level") or None,
            size_bytes=None,  # not installed locally; size unknown until pulled
        )

    def ensure(self, model: str, quant: str | None = None) -> None:
        tag = model
        if quant and quant.lower() not in model.lower():
            tag = f"{model}-{quant.lower()}"
        try:
            self.supports(tag)
            return  # already local
        except ModelNotFoundError:
            pass
        resp = self._client.post(
            "/api/pull", json={"model": tag, "stream": False}, timeout=None
        )
        if resp.status_code == 404:
            raise ModelNotFoundError(f"Ollama cannot pull {tag!r}")
        resp.raise_for_status()

    def remove(self, model: str) -> None:
        resp = self._client.request("DELETE", "/api/delete", json={"model": model})
        if resp.status_code == 404:
            raise ModelNotFoundError(f"Ollama has no local model {model!r}")
        resp.raise_for_status()

    def generate(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        num_ctx: int = 16384,
    ) -> str:
        message: dict = {"role": "user", "content": prompt}
        if images:
            message["images"] = [
                base64.b64encode(p.read_bytes()).decode("ascii") for p in images
            ]
        resp = self._client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [message],
                "stream": False,
                # Ollama defaults num_ctx to 4096 — a few manga pages already
                # exceed that in vision tokens.
                "options": {"num_ctx": num_ctx},
            },
            timeout=None,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
