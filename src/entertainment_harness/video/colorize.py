"""Optional page colorization for video visuals (DDColor-tiny, ONNX).

DDColor (ICCV 2023) predicts ab chroma at 512x512 from a grayscale image;
we recombine the predicted chroma with the *original-resolution* L channel,
so line art stays crisp and only color is inferred. Model: ~130 MB ONNX
(edgetools/ddcolor), downloaded from HF Hub on first use into
data/colorize/. Runs on CPU via onnxruntime (~1 s/page on Apple Silicon).

Honest expectation: this is automatic photo colorization applied to manga —
palettes are plausible but not canon, and dense screentones can pick up
chroma speckle. It is opt-in ([video] colorize = true or eh recap
--colorize), never on silently.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from entertainment_harness.config import Config, cache_dir
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginRegistry

MODEL_REPO = "edgetools/ddcolor"
MODEL_FILE = "ddcolor-tiny-fp16.onnx"
INPUT_SIZE = 512


class ColorizeError(Exception):
    pass


class ColorizeProvider(Protocol):
    name: str
    capabilities: frozenset[str]  # e.g. {"colorize"}

    def colorize_page(self, src: Path, dest: Path) -> Path:
        """Colorize one page image to dest (reused when it exists)."""
        ...

    def colorize_pages(
        self, pages: list[Path], dest_dir: Path, log: Callable[[str], None] = print
    ) -> dict[Path, Path]:
        """Colorize a set of pages into dest_dir; returns src -> dest."""
        ...


class Colorizer:
    """Lazy-loaded DDColor ONNX session with per-page disk caching."""

    name = "ddcolor"
    capabilities = frozenset({"colorize"})

    def __init__(self, config: Config | None = None, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or cache_dir() / "colorize"
        self._session = None
        self._input_name: str = ""

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        try:
            model_path = hf_hub_download(
                MODEL_REPO, MODEL_FILE, local_dir=str(self._model_dir)
            )
        except Exception as exc:
            raise ColorizeError(f"Could not download {MODEL_REPO}: {exc}") from exc
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def colorize_page(self, src: Path, dest: Path) -> Path:
        """Colorize one page; dest is reused when it already exists."""
        if dest.exists():
            return dest
        self._ensure_session()
        img = cv2.imread(str(src))
        if img is None:
            raise ColorizeError(f"Could not read image {src}")
        h, w = img.shape[:2]

        # L at original resolution (keeps line art); gray RGB at 512 for input
        lab_full = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        L_orig = lab_full[:, :, 0]
        small = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        L = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 0]
        gray_lab = np.stack([L, np.zeros_like(L), np.zeros_like(L)], axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_Lab2RGB).astype(np.float32)
        x = gray_rgb.transpose(2, 0, 1)[None]

        ab = self._session.run(None, {self._input_name: x})[0][0]  # (2, 512, 512)
        ab_full = np.stack([cv2.resize(ab[i], (w, h)) for i in range(2)], axis=-1)
        out_lab = np.concatenate([L_orig[:, :, None], ab_full], axis=-1)
        out = cv2.cvtColor(out_lab.astype(np.float32), cv2.COLOR_Lab2RGB)
        out_bgr = cv2.cvtColor((out * 255).clip(0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), out_bgr)
        return dest

    def colorize_pages(
        self,
        pages: list[Path],
        dest_dir: Path,
        log: Callable[[str], None] = print,
    ) -> dict[Path, Path]:
        """Colorize a set of pages into dest_dir (PNG). Returns src -> dest."""
        mapping: dict[Path, Path] = {}
        for i, src in enumerate(pages, start=1):
            dest = dest_dir / f"{src.stem}.png"
            if not dest.exists():
                log(f"  colorizing page {i}/{len(pages)}: {src.name}...")
            mapping[src] = self.colorize_page(src, dest)
        return mapping


REGISTRY = PluginRegistry("colorize", ENTRY_POINT_GROUPS["colorize"])
REGISTRY.register("ddcolor", "entertainment_harness.video.colorize:Colorizer")


def get_colorizer(name: str, config: Config | None = None) -> ColorizeProvider:
    return REGISTRY.create(name, config)
