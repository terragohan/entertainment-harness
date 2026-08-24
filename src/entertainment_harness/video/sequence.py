"""Frame-by-frame sequence animator for [video] mode = "sequence".

Where the `animate` mode's frame animator generates a few independent
variations of one panel crop, the sequence animator builds real temporal
animation:

1. The grounded panel crop is **expanded to video size** — gen4_image
   outpaints it to the configured resolution's ratio (frame 0).
2. Each following frame is **chained**: generated with the previous frame
   as `@prev` reference (plus the original `@panel` as an art anchor), so
   motion actually advances across the sequence.
3. An optional **drift critic** (the vision-role model) checks each chained
   frame against its predecessor; feedback triggers one regeneration,
   persistent drift truncates the chain at the last good frame — the
   truncated sequence still fills the slot at assembly.

Generated frames are content-addressed (sha256 of source bytes + prompt +
model + ratio + chain index) under workdir/"sequence": they survive clip
invalidation and can never be stale. Cached frames are re-critiqued when
the critic is on, so a re-run converges on the same accepted chain instead
of trusting a frame that failed last time.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from entertainment_harness.config import Config
from entertainment_harness.video.frames import SUPPORTED_RATIOS, _own_size
from entertainment_harness.video.script import Segment, VideoError

EXPAND_PROMPT = (
    "Outpaint the manga panel @panel to a full {ratio} video frame: keep the"
    " panel's exact composition, characters, and art style, and imagine the"
    " scene beyond its borders — background, effects, and atmosphere"
    " continuing naturally. Scene context: {text}"
)

CHAIN_PROMPT = (
    "Draw frame {phase} of {count} of a subtle frame-by-frame animation of"
    " the scene in @prev: keep the exact same composition, characters, and"
    " art style as @prev and the original panel @panel, and advance natural"
    " motion one small step — hair, cloth, smoke and effects drifting,"
    " expressions and impact lines shifting slightly. Scene context: {text}"
)

CRITIC_FEEDBACK = "\nThe previous attempt drifted — fix this: {feedback}"

DRIFT_PROMPT = (
    "These are two consecutive frames of a manga panel animation (first"
    " image: previous frame; second image: the new frame). The new frame"
    " must keep the same characters, composition, and art style — only"
    " motion may advance. If it does, answer exactly: OK. If characters,"
    " composition, or style changed, answer: DRIFT: <one sentence describing"
    " what changed>"
)

# critic(prev_frame, new_frame) -> None when the frame is faithful, else a
# feedback string for the regeneration prompt.
DriftCritic = Callable[[Path, Path], "str | None"]


def video_ratio(resolution: str) -> str:
    """The supported gen4_image ratio closest to the video resolution."""
    width, _, height = resolution.partition("x")
    try:
        aspect = int(width) / int(height)
    except (ValueError, ZeroDivisionError):
        aspect = 16 / 9
    return min(
        SUPPORTED_RATIOS,
        key=lambda r: abs(
            math.log(aspect) - math.log(int(r.split(":")[0]) / int(r.split(":")[1]))
        ),
    )


def frame_count(share_s: float, fps: float = 1.5, max_frames: int = 12) -> int:
    """Generated frames per anchor: fps per second of slot share, at least
    two (expansion + one chained frame), capped for cost."""
    return max(2, min(max_frames, round(share_s * fps)))


def _sequence_key(image: Path, prompt: str, model: str, ratio: str, index: int) -> str:
    h = hashlib.sha256(image.read_bytes())
    h.update(prompt.encode())
    h.update(model.encode())
    h.update(ratio.encode())
    h.update(str(index).encode())
    return h.hexdigest()[:16]


def drift_critic(adapter, model: str) -> DriftCritic:
    """Build a drift critic over a vision adapter (the vision-role model).

    Parses the DRIFT_PROMPT verdict: OK -> accept; anything else -> the
    feedback text for the regeneration prompt.
    """

    def criticize(prev: Path, new: Path) -> str | None:
        reply = adapter.generate(model, DRIFT_PROMPT, images=[prev, new]).strip()
        if reply.upper().startswith("OK"):
            return None
        verdict, _, feedback = reply.partition(":")
        if verdict.strip().upper() == "DRIFT" and feedback.strip():
            return feedback.strip()
        return "the frame changed characters, composition, or art style"

    return criticize


class RunwaySequenceAnimator:
    """gen4_image expansion + chained frames (the default [sequence] provider)."""

    name = "sequence"
    animated = True
    capabilities = frozenset({"frame-sequence", "image-gen"})

    def __init__(self, config: Config) -> None:
        from entertainment_harness.video.gen.runway import RunwayProvider

        self.config = config
        try:
            self.client = RunwayProvider(config)  # raises without an API key
        except VideoError as exc:
            raise VideoError(
                f"{exc} Alternatively, set [sequence] provider ="
                " \"openrouter-sequence\" to generate frames via the"
                " OpenAI-compatible endpoint ([models.openai_compat])."
            ) from exc
        self.model = config.sequence.model or "gen4_image"

    def generate_frames(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
        log=lambda m: None,
        critic: DriftCritic | None = None,
    ) -> list[Path]:
        return sequence_frames_with(
            self.client, self.model, self.config,
            image, segment, duration, workdir, log, critic,
        )


class OpenRouterSequenceAnimator:
    """Expansion + chained frames via the OpenAI-compatible endpoint."""

    name = "openrouter-sequence"
    animated = True
    capabilities = frozenset({"frame-sequence", "image-gen"})

    def __init__(self, config: Config) -> None:
        from entertainment_harness.video.gen.openrouter import (
            DEFAULT_IMAGE_MODEL,
            OpenRouterImageProvider,
        )

        self.config = config
        self.client = OpenRouterImageProvider(config)  # raises without a key
        self.model = config.sequence.model or DEFAULT_IMAGE_MODEL

    def generate_frames(
        self,
        image: Path,
        segment: Segment,
        duration: float,
        workdir: Path,
        log=lambda m: None,
        critic: DriftCritic | None = None,
    ) -> list[Path]:
        return sequence_frames_with(
            self.client, self.model, self.config,
            image, segment, duration, workdir, log, critic,
        )


def sequence_frames_with(
    client,
    model: str,
    config: Config,
    image: Path,
    segment: Segment,
    duration: float,
    workdir: Path,
    log=lambda m: None,
    critic: DriftCritic | None = None,
) -> list[Path]:
    """The shared sequence loop: frame 0 outpaints the panel to the video
    ratio, later frames chain off their predecessor, and the optional drift
    critic gets one regeneration per frame before truncating the chain.
    `client` is any image provider with Runway's text_to_image interface."""
    cfg = config.sequence
    count = frame_count(duration, cfg.fps, cfg.max_frames)
    ratio = video_ratio(config.video.resolution)
    frames_dir = workdir / "sequence"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Path] = []
    generated = 0
    for i in range(count):
        if i == 0:
            prompt = EXPAND_PROMPT.format(ratio=ratio, text=segment.text[:300])
            refs = [("panel", image)]
            ref_ratio = _own_size(image)  # never crop the panel
        else:
            prompt = CHAIN_PROMPT.format(
                phase=i + 1, count=count, text=segment.text[:300]
            )
            refs = [("prev", frames[-1]), ("panel", image)]
            ref_ratio = _own_size(frames[-1])

        dest = frames_dir / f"s-{_sequence_key(image, prompt, model, ratio, i)}-{i}.png"
        if not dest.exists():
            client.text_to_image(
                prompt, refs, model=model, ratio=ratio,
                ref_ratio=ref_ratio, dest=dest,
            )
            generated += 1

        if i > 0 and critic is not None:
            feedback = critic(frames[-1], dest)
            if feedback is not None:
                retry_prompt = prompt + CRITIC_FEEDBACK.format(feedback=feedback)
                retry = frames_dir / (
                    f"s-{_sequence_key(image, retry_prompt, model, ratio, i)}-{i}.png"
                )
                if not retry.exists():
                    client.text_to_image(
                        retry_prompt, refs, model=model, ratio=ratio,
                        ref_ratio=ref_ratio, dest=retry,
                    )
                    generated += 1
                if critic(frames[-1], retry) is not None:
                    log(f"    segment {segment.index:02d}: frame {i} keeps"
                        f" drifting — truncating the chain at {len(frames)}"
                        " frame(s)")
                    break
                dest = retry
        frames.append(dest)

    log(f"    segment {segment.index:02d}: {len(frames)} sequence frame(s)"
        f" ({generated} generated, {len(frames) - generated} cached)")
    return frames
