"""Optional output compression: re-encode the master out.mp4 to a quality
preset (CRF ladder).

The master out.mp4 is never touched; the compressed copy lives next to it
as out-<preset>.mp4, so switching presets re-compresses from the master
instead of degrading an already-compressed file.

Presets target quality, not bitrate: masters are libx264 CRF 23 (~4.7 Mbps
at 1080p30), so bitrate presets sized for "good" uploads could end up larger
than the master. CRF 21-25 H.264 typically lands at 0.5-2 Mbps for manga
panel videos with slow zoompan motion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from entertainment_harness.video.script import VideoError

# name -> (max width, max height, codec, crf). Measured on real narration
# masters (1080p30 libx264 CRF 23, 1-5 Mbps): CRF-only re-encodes at 1080p
# save ~25% — the wins come from resolution (720p ≈ 2.2x) and HEVC (~1.6x on
# top of that).
PRESETS: dict[str, tuple[int, int, str, int]] = {
    "hd": (1920, 1080, "libx264", 22),
    "balanced": (1280, 720, "libx264", 25),
    "small": (1280, 720, "libx265", 28),
}

AUDIO_BITRATE = "128k"


def compress(src: Path, preset: str, workdir: Path) -> Path:
    """Re-encode src to the given preset. Returns out-<preset>.mp4."""
    if preset not in PRESETS:
        raise VideoError(
            f"Unknown compression preset {preset!r}; expected one of:"
            f" {', '.join(PRESETS)}"
        )
    width, height, codec, crf = PRESETS[preset]
    dest = workdir / f"out-{preset}.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease"
                   ":force_divisible_by=2",
            "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1" if codec == "libx265" else "avc1",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-c:s", "mov_text",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise VideoError(f"ffmpeg compression failed ({preset}):\n{tail}")
    return dest
