# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the frozen `eh-serve` backend (onedir).

Build from the repo root:
    uv run pyinstaller packaging/eh-serve.spec --distpath ui/resources/backend \
        --workpath packaging/build --noconfirm

Output: ui/resources/backend/eh-serve/eh-serve (staged for the Electrobun
build, which copies it into the .app's Resources — see ui/electrobun.config.ts).

What must be in the bundle and why:
- collect_submodules("entertainment_harness"): the serve path imports lazily
  (cli/serve.py -> uvicorn/server.app; server/runs.py -> pipelines, video,
  store), so list the whole package explicitly rather than relying on
  bytecode scanning alone.
- kokoro_onnx + phonemizer + espeakng_loader + onnxruntime + cv2: the default
  TTS/video generation path. kokoro_onnx needs its config.json data file and
  its dist metadata (it calls importlib.metadata.version("kokoro-onnx"));
  espeakng_loader ships the libespeak-ng dylib and espeak-ng-data that it
  loads via ctypes from its package directory.
- Model weights are NOT bundled: kokoro-v1.0.onnx / voices-v1.0.bin download
  to the cache dir on first use (video/tts.py), and LLM/vision models are
  external (ollama / HF CLI / OpenAI-compatible endpoints).
- Excluded: the optional `dataset`/`sam`/`qwen3` extras (torch, transformers,
  segment-anything, mlx-audio) — not installed in the default env and not on
  the default recap path; the frozen CLI errors clearly if a command needs
  them.
"""

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

datas = []
datas += collect_data_files("kokoro_onnx")  # config.json
datas += collect_data_files("espeakng_loader")  # espeak-ng-data/
datas += copy_metadata("kokoro-onnx")

binaries = []
binaries += collect_dynamic_libs("espeakng_loader")  # libespeak-ng*.dylib

hiddenimports = [
    # lazy imports inside functions (video/tts.py imports kokoro_onnx on
    # first synthesis; uvicorn's contrib hook covers uvicorn itself)
    "kokoro_onnx",
    "phonemizer",
    "espeakng_loader",
    "onnxruntime",
    "cv2",
]
hiddenimports += collect_submodules("entertainment_harness")

excludes = [
    # optional extras, not on the default serve/recap path
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "segment_anything",
    "mlx_audio",
    "json_repair",
    # dev-only
    "pytest",
    "respx",
]

a = Analysis(
    [os.path.join(SPECPATH, "eh_entry.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="eh-serve",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="eh-serve",
)
