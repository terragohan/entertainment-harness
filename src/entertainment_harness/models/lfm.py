"""LFM2.5-VL bubble localization and text extraction.

Requires the ``dataset`` extra:
    uv sync --extra dataset --group dev
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _LFM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _LFM_AVAILABLE = False

try:
    import json_repair

    _JSON_REPAIR_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _JSON_REPAIR_AVAILABLE = False

from entertainment_harness.models.base import ModelInfo

DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
LARGE_MODEL_ID = "LiquidAI/LFM2.5-VL-3B"

LOCATE_PROMPT = """Find every speech bubble and caption box that contains story text on this manga page.

Return a JSON array of objects. Each object must have:
- "bbox": [x1, y1, x2, y2] as normalized floats 0.0-1.0 (left, top, right, bottom)
- "type": "speech_bubble" or "caption"
- "confidence": 0.0-1.0
{extra_fields}

Rules:
- One object per text area. Do not merge separate bubbles or captions.
- Ignore sound effects (SFX), watermarks, scanlation credits, and page numbers.
- The bbox should be tight around the text, not the whole panel.
- Output ONLY the JSON array."""

TEXT_EXTRA_FIELDS = '- "text": the exact printed text'


def is_available() -> bool:
    return _LFM_AVAILABLE


def _extract_json_array(raw: str) -> str | None:
    """Return the longest top-level JSON array substring from model output."""
    candidates: list[tuple[int, int]] = []
    start: int | None = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append((start, i + 1))
                    start = None
    if not candidates:
        return None
    start, end = max(candidates, key=lambda se: se[1] - se[0])
    return raw[start:end]


def _repair_json(raw: str) -> Any:
    import json

    body = _extract_json_array(raw)
    if body is None:
        raise ValueError("No JSON array found in model output")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        if _JSON_REPAIR_AVAILABLE:
            return json_repair.loads(body)
        raise


def _select_dtype_device() -> tuple[torch.dtype, str]:
    # Default to CPU for stability with these models; MPS has crashed during
    # model load in this environment. Users with a working CUDA/MPS setup can
    # override via the constructor.
    if torch.cuda.is_available():
        return torch.bfloat16, "cuda"
    return torch.float32, "cpu"


def _parse_results(
    data: list[Any],
    *,
    include_text: bool,
) -> list[dict[str, Any]]:
    """Normalize raw LFM JSON entries into bubble records.

    Accepts several bbox and kind aliases that different LFM2.5-VL checkpoints
    use, and normalizes both 0-1 floats and 0-1000 integer grids.
    """
    results: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        bbox = (
            entry.get("bbox")
            or entry.get("bbox_2d")
            or entry.get("box")
        )
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (ValueError, TypeError):
            continue
        # Support both 0-1 normalized floats and 0-1000 integer grids.
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        x1, x2 = min(x1, x2) / scale, max(x1, x2) / scale
        y1, y2 = min(y1, y2) / scale, max(y1, y2) / scale
        kind = str(
            entry.get("kind")
            or entry.get("type")
            or entry.get("label")
            or "speech"
        )
        text = str(
            entry.get("text")
            or entry.get("original")
            or entry.get("content")
            or ""
        ).strip()
        results.append(
            {
                "box": [x1, y1, x2, y2],
                "kind": kind,
                "confidence": float(entry.get("confidence", 0.0)),
                "original": text if include_text else "",
            }
        )
    return results


class LFMLocator:
    """Load LFM2.5-VL once and run bubble localization / OCR on images."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | str | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        if not _LFM_AVAILABLE:
            raise RuntimeError(
                "LFM dependencies are not installed. Run: uv sync --extra dataset --group dev"
            )
        self.model_id = model_id
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device = device
        self._dtype = dtype
        self._log = log

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        self._log(f"Loading LFM model {self.model_id}...")
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            clean_up_tokenization_spaces=False,
        )

        if self._dtype is None or self._device is None:
            dtype, device = _select_dtype_device()
        else:
            dtype = self._dtype
            device = self._device

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device if device in ("auto", "cuda") else None,
        )
        if device not in ("auto", "cuda"):
            model = model.to(device)

        self._processor = processor
        self._model = model
        self._log("LFM model loaded.")
        return processor, model

    def locate(
        self,
        image: Image.Image | np.ndarray | Path | str,
        include_text: bool = False,
        max_new_tokens: int = 2048,
    ) -> list[dict[str, Any]]:
        """Return bubble detections with normalized bboxes (0-1).

        Each entry contains ``bbox`` [x1, y1, x2, y2], ``kind``,
        ``confidence``, and ``text`` when ``include_text=True``.
        """
        processor, model = self._load()
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(np.asarray(image)).convert("RGB")

        w, h = pil_image.size
        extra = TEXT_EXTRA_FIELDS if include_text else ""
        prompt = LOCATE_PROMPT.format(extra_fields=extra)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
        )
        raw = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        try:
            data = _repair_json(raw)
        except Exception as exc:
            raise RuntimeError(f"LFM returned unparseable JSON: {exc}\nRaw: {raw[:500]}") from exc

        if not isinstance(data, list):
            raise RuntimeError(f"LFM did not return a JSON array. Raw: {raw[:500]}")

        return _parse_results(data, include_text=include_text)


class LFMAdapter:
    """Model-backend plugin wrapping the local LFM2.5-VL runtime.

    Registered as "lfm" so scanlation stages (backend = "lfm") and role
    bindings resolve through the same registry as every other backend.
    Models download from HF Hub on first use (transformers cache); ensure()
    is a no-op because the download happens lazily at load.
    """

    name = "lfm"
    remote = False
    capabilities = frozenset({"vision"})

    def __init__(
        self,
        config: Any | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        log: Callable[[str], None] = print,
    ) -> None:
        self.model_id = model_id
        self._log = log
        self._locator: LFMLocator | None = None

    def locator(self) -> LFMLocator:
        """The shared bubble locator (loads the model on first use)."""
        if self._locator is None:
            self._locator = LFMLocator(self.model_id, log=self._log)
        return self._locator

    def supports(self, model: str) -> ModelInfo:
        return ModelInfo(
            name=model, backend=self.name, capabilities=frozenset({"vision"})
        )

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass  # weights download from HF Hub lazily at first load

    def remove(self, model: str) -> None:
        import shutil

        cache = (
            Path.home() / ".cache" / "huggingface" / "hub"
            / f"models--{model.replace('/', '--')}"
        )
        if cache.is_dir():
            shutil.rmtree(cache)

    def list_available(self) -> list[ModelInfo]:
        return []  # the transformers cache is not a queryable registry

    def list_remote(self, model: str) -> list[ModelInfo] | None:
        return None

    def generate(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        num_ctx: int = 16384,
    ) -> str:
        """One vision chat call; returns the raw decoded completion."""
        from PIL import Image as _Image

        locator = self.locator()
        processor, lfm_model = locator._load()
        content: list[dict[str, Any]] = [
            {"type": "image", "image": _Image.open(p).convert("RGB")}
            for p in (images or [])
        ]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(lfm_model.device)
        outputs = lfm_model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=min(num_ctx, 4096),
        )
        return processor.batch_decode(outputs, skip_special_tokens=True)[0]
