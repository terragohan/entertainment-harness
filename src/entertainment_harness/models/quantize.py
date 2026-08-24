"""Local quantization fallback (opt-in, user-confirmed).

For when a repo has no published quant that fits: download the FP16/BF16
source GGUF, run llama.cpp's llama-quantize. Costs the user must accept
first (docs/design.md): the source is huge on disk plus the output file, and
conversion takes minutes to hours. Policy: always prefer a published quant;
this module refuses to run when the target quant is already published.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from entertainment_harness.models.huggingface import HFAdapter, quant_from_filename

# Preferred local-quantization sources, best first.
SOURCE_QUANTS = ("F16", "BF16", "F32")


class QuantizeError(Exception):
    pass


def quantize(
    repo: str,
    target_quant: str,
    adapter: HFAdapter | None = None,
    confirm: Callable[[str], bool] = lambda msg: False,
    log: Callable[[str], None] = print,
) -> Path:
    """Quantize a repo's full-precision GGUF to target_quant.

    `confirm` is called with the cost summary; returning False aborts.
    Returns the output path.
    """
    if shutil.which("llama-quantize") is None:
        raise QuantizeError("llama-quantize not found — install llama.cpp")
    adapter = adapter or HFAdapter()
    target = target_quant.upper()

    files, _ = adapter._gguf_files(repo)
    published = {quant_from_filename(f["rfilename"]) for f in files}
    if target in published:
        raise QuantizeError(
            f"{repo} already publishes {target} — prefer the published quant"
            f" ({repo}:{target}); local quantization is a last resort."
        )

    source_entry = next(
        (f for q in SOURCE_QUANTS for f in files
         if quant_from_filename(f["rfilename"]) == q),
        None,
    )
    if source_entry is None:
        available = ", ".join(sorted(p for p in published if p))
        raise QuantizeError(
            f"{repo} has no FP16/BF16 source to quantize from."
            f" Published quants: {available or 'none'}."
        )

    source_name = source_entry["rfilename"]
    source_size = source_entry.get("size")
    source_quant = quant_from_filename(source_name)
    out_name = source_name.replace(f"-{source_quant}.gguf", f"-{target}.gguf")
    if out_name == source_name:
        out_name = source_name[: -len(".gguf")] + f"-{target}.gguf"

    size_gb = f"{source_size / 1e9:.1f} GB" if source_size else "unknown size"
    if not confirm(
        f"Local quantization of {repo}: download {source_name} ({size_gb}),"
        f" then convert to {target} (another file on disk; minutes-to-hours)."
        " A published quant always beats this on quality. Proceed?"
    ):
        raise QuantizeError("Aborted by user.")

    log(f"Downloading {source_name} ({size_gb})...")
    source_path = adapter._download(repo, source_name, source_size)
    out_path = source_path.parent / out_name
    log(f"Quantizing to {target} (this can take a while)...")
    result = subprocess.run(
        ["llama-quantize", str(source_path), str(out_path), target],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out_path.exists():
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-5:])
        raise QuantizeError(f"llama-quantize failed:\n{tail}")
    log(f"Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    return out_path
