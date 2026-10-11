"""Synthetic animation-conductor demo: no network, no paid APIs.

Draws a tiny cast of Pillow assets (background + one character with blink /
mouth / pose variants), synthesizes a tone "dialogue" WAV, writes a sparse
5 s / 24 fps / 120-frame timeline.json exercising every Phase-1 mechanism —
static hold, camera push, blink, crude mouth cycling, pose swap, audio at
an exact frame — and renders it with the real Sequencing Renderer.

Usage:
    uv run python scripts/anime_conductor_demo.py [--out /tmp/eh-animation-demo]

Requires ffmpeg on PATH. Visual quality is not the point; sequencing is.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from entertainment_harness.anime.render import render_timeline
from entertainment_harness.anime.timeline import (
    AssetEvent,
    AudioClip,
    CameraKey,
    Layer,
    MouthEvent,
    Timeline,
    save_timeline,
)

WIDTH, HEIGHT = 640, 360
FPS = 24
DURATION_FRAMES = 120  # exactly 5 seconds

SKIN = (240, 205, 170, 255)
SHIRT = (80, 110, 200, 255)
SHIRT_ALT = (190, 80, 90, 255)
INK = (30, 30, 40, 255)


def draw_background(path: Path) -> None:
    """A simple landscape with landmarks, so camera motion is visible."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    d = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        sky = (round(120 + 60 * t), round(170 + 40 * t), round(230 - 40 * t))
        d.line([(0, y), (WIDTH, y)], fill=sky)
    d.ellipse([520, 30, 590, 100], fill=(250, 230, 140))  # sun
    d.polygon([(60, 260), (180, 120), (300, 260)], fill=(90, 110, 130))
    d.polygon([(200, 260), (340, 150), (480, 260)], fill=(110, 130, 150))
    d.rectangle([0, 250, WIDTH, HEIGHT], fill=(95, 150, 90))  # ground
    d.rectangle([0, 250, WIDTH, 258], fill=(75, 125, 75))
    img.save(path)


def draw_character(path: Path, *, eyes: str = "open", mouth: str = "closed",
                   shirt=SHIRT, wave_arm: bool = False) -> None:
    """A 300x400 transparent sprite; variants differ only in the named part."""
    img = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # body
    d.rounded_rectangle([80, 190, 220, 390], radius=40, fill=shirt)
    # arms; the pose-2 variant waves one
    if wave_arm:
        d.line([(90, 220), (30, 120)], fill=SKIN, width=26)
        d.ellipse([12, 96, 48, 132], fill=SKIN)
    else:
        d.line([(90, 220), (40, 320)], fill=SKIN, width=26)
    d.line([(210, 220), (260, 320)], fill=SKIN, width=26)
    # head
    d.ellipse([70, 20, 230, 190], fill=SKIN)
    # eyes
    if eyes == "open":
        d.ellipse([105, 85, 130, 115], fill=INK)
        d.ellipse([170, 85, 195, 115], fill=INK)
    else:  # blink: closed lids
        d.line([(105, 100), (130, 100)], fill=INK, width=6)
        d.line([(170, 100), (195, 100)], fill=INK, width=6)
    # mouth
    if mouth == "small":
        d.ellipse([130, 145, 170, 165], fill=INK)
    elif mouth == "open":
        d.ellipse([120, 138, 180, 180], fill=(120, 40, 45, 255))
    else:
        d.line([(130, 155), (170, 155)], fill=INK, width=6)
    img.save(path)


def synthesize_dialogue(path: Path) -> None:
    """A 1.5 s sine 'melody' standing in for dialogue — deterministic,
    synthesized in-process, no TTS or network involved."""
    rate = 22050
    notes = [440.0, 523.25, 659.25, 523.25, 440.0]
    note_s = 0.3
    frames = bytearray()
    for freq in notes:
        for i in range(int(rate * note_s)):
            envelope = min(1.0, i / 200.0, (rate * note_s - i) / 200.0)
            sample = 0.4 * envelope * math.sin(2 * math.pi * freq * i / rate)
            frames += int(sample * 32767).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(frames))


def write_demo(out: Path) -> Path:
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    draw_background(assets / "background.png")
    draw_character(assets / "character-base.png")
    draw_character(assets / "character-blink.png", eyes="closed")
    draw_character(assets / "character-mouth-small.png", mouth="small")
    draw_character(assets / "character-mouth-open.png", mouth="open")
    draw_character(
        assets / "character-pose-2.png", shirt=SHIRT_ALT, wave_arm=True
    )
    synthesize_dialogue(out / "dialogue.wav")

    timeline = Timeline(
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        duration_frames=DURATION_FRAMES,
        background="#0b0e14",
        assets={
            "background": "assets/background.png",
            "character-base": "assets/character-base.png",
            "character-blink": "assets/character-blink.png",
            "character-mouth-small": "assets/character-mouth-small.png",
            "character-mouth-open": "assets/character-mouth-open.png",
            "character-pose-2": "assets/character-pose-2.png",
        },
        layers=[
            Layer(id="background", asset="background", scale=1.0),
            Layer(id="character", asset="character-base", x=0.5, y=0.62,
                  scale=0.55),
        ],
        camera=[
            CameraKey(frame=0, x=0.5, y=0.5, scale=1.0),
            CameraKey(frame=DURATION_FRAMES - 1, x=0.56, y=0.46, scale=1.15),
        ],
        events=[
            # blink: eyes closed for exactly frames 36-38
            AssetEvent(frame=36, target="character", asset="character-blink"),
            AssetEvent(frame=39, target="character", asset="character-base"),
            # crude mouth cycling while the dialogue tone plays
            MouthEvent(
                start_frame=60, end_frame=95, target="character",
                states=[
                    "character-base",
                    "character-mouth-small",
                    "character-mouth-open",
                    "character-mouth-small",
                ],
                period_frames=2,
            ),
            # pose/expression swap at an exact frame
            AssetEvent(frame=100, target="character",
                       asset="character-pose-2"),
        ],
        audio=[AudioClip(start_frame=60, path="dialogue.wav")],
    )
    timeline_path = out / "timeline.json"
    save_timeline(timeline, timeline_path)
    return timeline_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/eh-animation-demo"),
        help="demo directory (assets, timeline.json, render/)",
    )
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    timeline_path = write_demo(out)
    print(f"Wrote {timeline_path}")
    mp4 = render_timeline(timeline_path, log=print)
    print(f"Demo rendered: {mp4}")


if __name__ == "__main__":
    main()
