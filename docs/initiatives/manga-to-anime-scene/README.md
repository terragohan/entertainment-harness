# Manga → anime scene (guided vertical slice)

Status: **done** (all five phases gated; Phase 5 paid gate passed 2026-09-14 — see Evidence)

## Goal

Prove that one manually selected, contiguous manga scene can be adapted into
one coherent ≤30 s anime scene under explicit user guidance:

    manga pages → guided storyboard (scene.json) → consistent anime keyframes
    → 3–5 generated 5 s video clips → one 16:9 MP4

Proof-of-concept target: Kenja no Mago. Not a general manga-to-anime system.

## Non-goals

Full-chapter adaptation, automatic scene/panel detection, lip sync, dialogue,
TTS/music/SFX, LoRA training, ComfyUI, local diffusion, character sheets, DB
schema changes, recap/TikTok behavior changes. Silent video is acceptable in
v0. Do not reuse the recap `Segment` class — an anime shot is not a narration
segment.

## Command concept

    uv run eh anime-scene kenja --chapter 0 --pages 8-10 \
        --instruction "Stay faithful to the manga. Serious fantasy anime tone."

Control flags: `--stop-after plan|keyframes`, `--regenerate-shot N`.

## Stages

1. **plan** — vision-role model (`get_vision_model()`) reads ALL selected
   pages + user instruction → validated SceneSpec (3–5 shots × exactly 5 s,
   manga event order, page bounds checked) → `scene.json` (human-editable;
   resume from edits).
2. **keyframes** — Runway `gen4_image` via `POST /v1/text_to_image`, 1280x720
   16:9, tagged `referenceImages`: `@Source` (manga page/crop), plus `@Anchor`
   (shot-00 keyframe) and `@Previous` (prior keyframe) for shot 1+.
3. **shots** — Runway image-to-video (default model `gen4.5`), keyframe as
   first frame, 1280:720, exactly 5 s, animation prompt = motion only.
4. **assembly** — ffmpeg concat (NOT the recap narration muxer), normalize to
   1280x720 h264, verify `ffprobe` duration ≤ 30 s.

Artifacts under `works/<series>/chapters/<ch>/anime-scene/`:
`scene.json`, `state.json`, `sources/shot-NN.png`, `keyframes/shot-NN.png`,
`clips/shot-NN.mp4`, `out.mp4`. Filesystem is source of truth; no DB rows.

Caching: sha256 fingerprints in `state.json` (plan: page contents +
instruction + planner model + prompt version; keyframe: shot fields + crop
content + anchor/prev content + instruction + image model; clip: keyframe
content + animation prompt + video model + duration + ratio; final: ordered
clip contents). Fingerprint change invalidates only that artifact +
downstream. Never timestamps.

Runway client: modernize to `https://api.dev.runwayml.com`,
`X-Runway-Version: 2024-11-06`, httpx, reusable submit/poll/download/data-URI
helpers; TikTok path (gen3a_turbo, 768:1280) behavior preserved.
Model names configurable (`[anime]` config: `keyframe_model`, `video_model`).
seedance2_5 as alternate video model if trivial.

## Phases & gates

- [x] Phase 1: initiative + scene domain (SceneSpec/Shot, validation, crops)
  with unit tests. Gate: focused tests pass.
- [x] Phase 2: Runway client refactor, TikTok tests green. Gate:
  `uv run pytest tests/test_video_gen.py` output.
- [x] Phase 3: planner + CLI `--stop-after plan`. Gate: live Kenja ch0 plan
  command output + resulting scene.json pasted here.
- [x] Phase 4: keyframes/shots/assembly stages, fingerprints, per-shot
  regeneration; mocked tests for all request shapes. Gate: full suite output.
- [x] Phase 5: manual paid gate (human-run, not in tests): full Kenja scene
  generation, inspect keyframes + out.mp4. Gate: command output + duration.
  Done 2026-09-14 on a single page (ch0 p9, 3 shots × 5 s).

## Evidence

Phase 1/2/4 — tests (2026-09-13):

    $ uv run pytest tests/test_anime_scene.py -q
    ................................                                         [100%]
    32 passed in 1.13s

    $ uv run pytest tests/test_video_gen.py -q
    9 passed

    $ uv run pytest -q
    1 failed, 459 passed in 4.00s
    (the one failure, tests/test_lfm.py::test_repair_json_repairs_trailing_comma,
    is pre-existing: neither tests/test_lfm.py nor src/.../models/lfm.py is
    touched by this initiative — Python 3.13's json decoder rejects the
    trailing-comma case the old repair helper expected to parse)

Phase 3 — live Kenja ch0 plan (local free vision role, no Runway spend):

    $ uv run eh anime-scene kenja --chapter 0 --pages 8-10 \
        --instruction "Stay faithful to the manga. Serious fantasy anime tone. Preserve the characters and setting. Emphasize the attack. Do not add dialogue or new story events." \
        --stop-after plan
    Stage 1/4 plan: storyboarding with qwen/qwen3-vl-32b-instruct...
    Stage 1/4 plan: wrote .../works/kenja-no-mago/chapters/ch-000/anime-scene/scene.json (4 shots, 20s)
    Stopped after plan — edit scene.json, then rerun to continue.

scene.json: 4 shots x 5s = 20s — establishing (p8, ruined caravan in rain) ->
reaction (p9, Merlin's horrified face) -> action (p9, rushes to wagon, hears
the baby) -> close-up (p10, healing magic on the infant). Event order follows
the manga, all pages in range, no invented events; validated on write and on
every reload.

Phase 5 — full single-page scene, Kenja ch0 p9 (2026-09-14, real Runway
spend):

    $ uv run eh anime-scene kenja --chapter 0 --pages 9 \
        --instruction "Stay faithful to the manga. Serious fantasy anime tone. Preserve the characters and setting. Emphasize the action. Do not add dialogue or new story events."
    Stage 1/4 plan: cached (3 shots)
      keyframe shot-00: generating (gen4_image, refs: @Source)...
      keyframe shot-01: generating (gen4_image, refs: @Anchor)...
      keyframe shot-02: generating (gen4_image, refs: @Anchor, @Previous)...
      clip shot-00: generating (gen4.5, 5s)...
      clip shot-01: generating (gen4.5, 5s)...
      clip shot-02: generating (gen4.5, 5s)...
    Stage 4/4 assembly: cutting clips together...
    Done: .../works/kenja-no-mago/chapters/ch-000/anime-scene/out.mp4 (15.1s)

    ffprobe duration: 15.099674 s, 1280x720, 5.3 MB, silent.

Keyframes + mid-clip frames inspected: Merlin's identity (gray hair, full
beard, white robe) is consistent across all three shots via the @Anchor chain,
rain and ruined-village setting hold, and the event order follows the manga
(mourning → hears the cry → kneels by the newborn). The 3-shot scene.json was
hand-edited from the planner's first draft (see moderation notes below); the
pipeline resumed from the edited file and regenerated only downstream stages.

Findings from the live run (baked into code / worth knowing):

- Single-page selections: VLMs number the one attached image "page 1" instead
  of the real page number, failing page-bounds validation every time. Fixed in
  `anime/planner.py` — when exactly one page is selected, `source_page` is
  coerced to it pre-validation (test:
  test_planner_single_page_coerces_misnumbered_shots).
- Runway `gen4_image` caps `promptText` at 1000 chars; the composed keyframe
  prompt can exceed that and the API rejects it with a 400. Fixed in
  `video/gen/runway.py` — clamp to 1000, mirroring the existing 512 clamp on
  image_to_video (test: test_runway_keyframe_prompt_capped_at_api_limit).
- Runway moderation is the big practical limit for manga sources. On p9,
  keyframe tasks FAILED with "Content did not pass content moderation" whenever
  the reference crop included a downed/injured character, and the planner's
  gore vocabulary ("mangled bodies", "corpses", "blood") tripped it even
  earlier. A fully benign crop of the same page passed every probe. Mitigation
  that worked: per-shot `source_crop`s that exclude flagged panels + reworded
  keyframe_prompts. Moderation verdicts are also marginal/flaky — the same
  woman-panel crop passed one probe and failed three pipeline runs — so expect
  to iterate on scene.json (the designed workflow) rather than trusting the
  first plan for violent pages.

