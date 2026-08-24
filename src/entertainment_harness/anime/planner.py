"""Anime-scene stage 1: guided storyboard planning.

The configured vision-role model reads ALL selected manga pages plus the
user's direction and produces one validated SceneSpec (scene.json). The plan
is human-editable: rerun with --stop-after plan, edit scene.json, resume.
"""

from __future__ import annotations

import re
from pathlib import Path

from entertainment_harness.anime.scene import (
    MAX_SCENE_SECONDS,
    MAX_SHOTS,
    MIN_SHOTS,
    SHOT_SECONDS,
    SceneError,
    SceneSpec,
    parse_scene,
)
from entertainment_harness.models.base import ModelAdapter

# Bump when the planner prompt changes in a way cached scenes shouldn't keep.
PLANNER_PROMPT_VERSION = 1

PLANNER_PROMPT = """You are adapting selected pages from the manga "{title}" into ONE short anime scene.

You may ONLY use facts visibly supported by the attached manga pages.

Do not use outside knowledge of this manga or its anime adaptation.

USER DIRECTION:
---
{instruction}
---

Create a storyboard consisting of {min_shots} to {max_shots} sequential shots.

Each shot will become exactly {shot_seconds} seconds of generated video.

Requirements:

- Preserve the event order shown in the manga.
- Preserve visible character identity, hair, clothing, props and setting.
- Do not invent characters, attacks, dialogue, objects, locations or plot events.
- Keep this as one continuous scene, not a recap or montage.
- Prefer clear cinematic coverage: establishing/action/reaction/impact as
  appropriate to the source.
- Choose the best source page for every shot.
- `keyframe_prompt` describes ONE static anime production frame.
- `animation_prompt` describes ONLY what should move during the {shot_seconds}-second shot:
  character action, environmental motion and camera motion.
- Do not put dialogue in the animation prompt.
- Do not ask the video model to redesign characters.

Return ONLY valid JSON matching the required schema:

{{
  "scene_summary": "one paragraph describing the scene",
  "continuity": "the visual threads that must stay consistent across shots",
  "shots": [
    {{
      "index": 0,
      "duration_s": {shot_seconds},
      "source_page": <one of {pages}>,
      "source_crop": null,
      "shot_type": "establishing | action | reaction | impact | close-up | wide | ...",
      "action": "what happens in this shot",
      "composition": "framing and layout of the frame",
      "camera": "camera angle and any camera motion",
      "continuity": "what must carry over from adjacent shots",
      "keyframe_prompt": "the static anime frame description",
      "animation_prompt": "what moves during the shot"
    }}
  ]
}}"""


def plan_scene(
    adapter: ModelAdapter,
    model: str,
    page_paths: list[Path],
    source_pages: list[int],
    title: str,
    chapter_num: float,
    instruction: str,
    max_attempts: int = 2,
) -> SceneSpec:
    """Plan one scene from the selected pages. Every selected page is sent to
    the vision model as an image, in page order."""
    prompt = PLANNER_PROMPT.format(
        title=title,
        instruction=instruction.strip() or "(no extra direction — stay faithful)",
        min_shots=MIN_SHOTS,
        max_shots=MAX_SHOTS,
        shot_seconds=SHOT_SECONDS,
        pages=", ".join(str(p) for p in source_pages),
    )
    last_error: SceneError | None = None
    for _ in range(max_attempts):
        raw = adapter.generate(model, prompt, images=page_paths)
        if len(source_pages) == 1:
            # With a single attached page the model often numbers it "1"
            # (image position) instead of the real page number; there is only
            # one valid choice, so coerce instead of failing the whole plan.
            raw = re.sub(
                r'("source_page"\s*:\s*)\d+',
                lambda m: f"{m.group(1)}{source_pages[0]}",
                raw,
            )
        try:
            spec = parse_scene(raw, source_pages=source_pages)
        except SceneError as exc:
            last_error = exc
            continue
        # The planner speaks for the storyboard, not the request metadata:
        # stamp identity ourselves so a chatty model can't misreport it.
        spec.title = title
        spec.chapter = chapter_num
        spec.instruction = instruction.strip()
        return spec
    raise SceneError(
        f"Planner failed after {max_attempts} attempts: {last_error}"
    )
