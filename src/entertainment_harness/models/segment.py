"""Optional SAM-based speech-bubble segmentation.

Requires the ``sam`` extra:
    uv sync --extra sam

The first call will download the SAM ViT-B checkpoint to the data directory
unless a custom checkpoint path is provided.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

from entertainment_harness.config import cache_dir

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    _SAM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SAM_AVAILABLE = False

DEFAULT_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
DEFAULT_MODEL_TYPE = "vit_b"

_T = TypeVar("_T")


def is_available() -> bool:
    return _SAM_AVAILABLE


def default_checkpoint_path() -> Path:
    return cache_dir() / "models" / "sam_vit_b_01ec64.pth"


def ensure_checkpoint(
    path: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path | None:
    """Return an existing checkpoint path, downloading the default if needed."""
    path = path or default_checkpoint_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading SAM checkpoint to {path} ...")
    try:
        urllib.request.urlretrieve(DEFAULT_CHECKPOINT_URL, path)
    except Exception as exc:  # pragma: no cover - network failure
        log(f"Failed to download SAM checkpoint: {exc}")
        return None
    log("SAM checkpoint ready.")
    return path


def _to_rgb_array(image: Image.Image | np.ndarray | Path | str) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
        return np.array(img)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    return arr


class SamSegmenter:
    """Thin wrapper around SAM for one-image, many-box batch segmentation."""

    def __init__(self, checkpoint_path: Path) -> None:
        if not _SAM_AVAILABLE:
            raise RuntimeError(
                "SAM dependencies are not installed. Run: uv sync --extra sam"
            )
        self.checkpoint_path = checkpoint_path
        self._predictor: SamPredictor | None = None
        self._image_shape: tuple[int, int] | None = None

    def _load(self) -> SamPredictor | None:
        if self._predictor is not None:
            return self._predictor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        sam = sam_model_registry[DEFAULT_MODEL_TYPE](checkpoint=str(self.checkpoint_path))
        sam.to(device=device)
        self._predictor = SamPredictor(sam)
        return self._predictor

    def set_image(self, image: Image.Image | np.ndarray | Path | str) -> bool:
        predictor = self._load()
        if predictor is None:
            return False
        img_np = _to_rgb_array(image)
        predictor.set_image(img_np)
        self._image_shape = img_np.shape[:2]
        return True

    def predict(self, boxes: list[list[float]]) -> list[np.ndarray | None]:
        """Return boolean masks for each normalized [x1,y1,x2,y2] box."""
        predictor = self._load()
        if predictor is None or self._image_shape is None:
            return [None] * len(boxes)

        h, w = self._image_shape
        masks: list[np.ndarray | None] = []
        for box in boxes:
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            input_box = np.array([x1, y1, x2, y2], dtype=np.float32)
            try:
                sam_masks, scores, _ = predictor.predict(
                    box=input_box[None, :],
                    multimask_output=True,
                )
            except Exception:  # pragma: no cover - runtime failures
                masks.append(None)
                continue

            mask = sam_masks[int(np.argmax(scores))]
            # Keep only the portion inside the prompt box so tails don't leak
            # into attached characters or neighboring art.
            clipped = np.zeros_like(mask)
            clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            masks.append(clipped)
        return masks
