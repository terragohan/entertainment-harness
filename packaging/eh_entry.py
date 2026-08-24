"""PyInstaller entry point for the frozen `eh-serve` backend binary.

Exposes the full `eh` CLI (Typer app) with argv passthrough — the desktop
app only invokes `eh-serve serve --port 0`, but the other commands keep
working when the binary is run by hand.

`--check-imports` (packaging smoke test, not part of the CLI): imports every
`entertainment_harness` submodule plus the third-party serve/generation
dependencies, prints one `ok`/`FAIL` line per module, and exits non-zero on
any failure. Used by the packaging gate to prove the frozen binary can
import everything a recap run needs.

`--check-tts` (packaging smoke test, not part of the CLI): phonemizes and
synthesizes one short line through the default Kokoro TTS engine, proving
the frozen binary resolves the espeak-ng data dir (a build-machine path is
compiled into the bundled dylib; the fix in video/tts.py must be present).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

_THIRD_PARTY = [
    "fastapi",
    "uvicorn",
    "starlette",
    "tomlkit",
    "httpx",
    "typer",
    "rich",
    "psutil",
    "numpy",
    "PIL",
    "cv2",
    "onnxruntime",
    "kokoro_onnx",
    "phonemizer",
    "espeakng_loader",
    "boto3",
    "ddgs",
    "huggingface_hub",
    "youtube_transcript_api",
]


def _check_imports() -> int:
    import entertainment_harness

    walk_errors: list[str] = []

    def _onerror(name: str) -> None:
        walk_errors.append(name)

    names = sorted(
        module.name
        for module in pkgutil.walk_packages(
            entertainment_harness.__path__,
            prefix="entertainment_harness.",
            onerror=_onerror,
        )
    )
    failures = 0
    for name in walk_errors:
        failures += 1
        print(f"FAIL {name}: raised during package walk")
    for name in [*_THIRD_PARTY, *names]:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"check-imports: {failures} failure(s)")
    return 1 if failures else 0


def _check_tts() -> int:
    import tempfile
    from pathlib import Path

    from entertainment_harness.video.tts import KokoroEngine

    try:
        engine = KokoroEngine()
        with tempfile.TemporaryDirectory() as tmp:
            engine.synthesize(
                "Packaging check.", engine.default_voice, Path(tmp) / "check.wav"
            )
    except Exception as exc:
        print(f"check-tts: FAIL {type(exc).__name__}: {exc}")
        return 1
    print("check-tts: ok")
    return 0


def main() -> None:
    if "--check-imports" in sys.argv[1:]:
        raise SystemExit(_check_imports())
    if "--check-tts" in sys.argv[1:]:
        raise SystemExit(_check_tts())
    from entertainment_harness.cli import app

    app()


if __name__ == "__main__":
    main()
