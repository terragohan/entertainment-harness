# Panel animation — `animate` video mode (AI-generated frames per beat)

Status: **done** (2026-09-27 — all three phases gated; live Runway frame
render deliberately excluded, same policy as motion mode)

> Cross-initiative note: the `frames` registry defined here is the contract;
> the frame-sequence initiative later added the hosted `openrouter` /
> `openrouter-sequence` animators (frame-sequence Phase 4, 2026-09-27) —
> see `docs/initiatives/frame-sequence/README.md`.

## Goal

A seventh video mode, `animate`: panel-by-panel like `panels`, but each
beat's panel crop becomes a short animation stitched from **frames generated
by an image generator** (the panel crop as the reference image) instead of a
Ken Burns zoom. Selectable via `[video] mode = "animate"` / `--video-mode
animate` / the app dropdown.

## Non-goals

- No video-generation models (that's `motion` mode's image-to-video seam).
- No new image-generator integrations beyond the existing Runway client.
- No live paid Runway renders as gate evidence — gate is code + mocked
  provider tests (same policy as video-styles Phase D).
- No changes to script/TTS/page-assignment/grounding stages.

## Rationale

`panels` guides the eye but the panel itself is still a frozen crop; `motion`
animates but replaces the panel with a video model's reinterpretation of a
whole page. Frame generation sits between: the art stays the chapter's own
panel (supplied as a reference image), and the motion comes from a small
sequence of generated variations dissolved together — an "animated
illustration" look at image-generation cost and latency instead of
video-generation cost and latency.

## Design

- Stages 1–3.5 are exactly `panels` (grounding runs for `animate`;
  panel-first segments use their born-with spans).
- Stage 4 assembly: the `panels` anchor loop generalizes. Per anchor, a
  **frame animator** generates N frames from the panel crop (pan anchors:
  the full page), N = clamp(round(share / `[frames] seconds_per_frame`),
  2, `[frames] max_frames`); the frames stitch through the slideshow
  mode's slot-exact `xfade` chain into one clip per anchor. A non-animated
  animator (`local`) renders the panels-mode Ken Burns crop clip instead —
  `animate` degrades to `panels` per anchor.
- Provider seam (`video/frames.py`, new `frames` plugin category):
  - `local` — `animated = False`; no generation (fallback path).
  - `runway` — gen4_image text-to-image (`video/gen/runway.py`'s existing
    client) with the crop as a `@panel` reference image and a motion-phase
    prompt ("same composition/characters/style, motion phase i/N"); output
    ratio = nearest supported gen4 ratio to the crop's aspect; the reference
    is sent uncropped (aspect-preserving prep).
  - Frame cache is **content-addressed**: `frames/f-<sha256(image bytes +
    prompt + model + ratio)>-<i>.png`, so frames survive clip invalidation
    and can never be stale.
- Config: `[frames] provider = "local" | "runway"`, `model = "gen4_image"`,
  `seconds_per_frame = 2.0`, `max_frames = 6`. The Runway API key keeps
  coming from `[video_gen.runway] api_key` / `RUNWAY_API_KEY`.
- `render_state.json` records the frames provider for animate renders;
  `videos` metadata attributes the provider when frames were generated.
- Cost: 2–6 image generations per grounded beat — a full chapter is
  typically 100+ gen4_image calls. Opt-in per run, off by default.

## Phases

### Phase A — frames provider seam + config (gate: tests green)

- `plugins.py`: `frames` entry-point group. `video/frames.py`: protocol,
  registry, `local`, `runway` (mocked client tests). `config.py`:
  `FramesConfig` + `[frames]` load.

### Phase B — assembly + pipeline wiring (gate: tests green)

- `assemble.py`: merge the panels/animate anchor loop, `frame_animator`
  param, xfade stitching. `pipeline.py`: `VIDEO_MODES`, grounding/restamp
  conditions, stage-4 dispatch, `wanted["frames"]`, provider attribution.
- Tests: fake animated animator drives frames per anchor; local falls back
  to panels behavior; grounding runs for animate.

### Phase C — surface + release (gate: full suite, typecheck, smoke, package)

- UI `VIDEO_MODES` + tooltip, CLI help, config sample, `docs/design.md`,
  smoke option list; full pytest + typecheck + smoke + package + frozen
  checks.

## Evidence

### Phase A — frames provider seam + config (done 2026-09-27)

- `uv run pytest -q` → **793 passed**, 1 known pre-existing failure. 7 new
  tests (`tests/test_frames.py`): frame-count bounds, nearest-ratio picking,
  local animator (not animated, raises with guidance), runway animator
  motion-phase prompts + panel-as-reference + uncropped ref prep,
  content-addressed caching (repeat = all cached, new text = new frames),
  `[frames]` config parsing, plugin-group registration.
- The animator owns the frame-count decision (`generate_frames(image,
  segment, duration, ...)`), so cost knobs live in `[frames]`.

### Phase B — assembly + pipeline wiring (done 2026-09-27)

- `uv run pytest -q` → **797 passed**, 1 known pre-existing failure. 4 new
  tests: animate assembly stitches generated frames per anchor (crop for
  holds, page for pans, slot shares), stills animator falls back to the
  panels crop clip, pipeline grounds + passes the animator + records
  `render_state {mode, grounded, frames}` + provider attribution, local
  provider logs the panels-mode fallback.
- The `panels` anchor loop now covers both modes: `animate` with no
  animated frames provider *is* `panels` per anchor.

### Phase C — surface + release (done 2026-09-27)

- UI dropdown lists all seven styles (tooltip describes animate + its
  `[frames]` dependency); CLI `--video-mode` help, `[video] mode` config
  comment, smoke option list, and `docs/design.md` (config sample incl. the
  new `[frames]` section, per-mode sub-bullet, grounding bullet) updated.
- Gates: `uv run pytest -q` → **797 passed**, 1 known pre-existing failure;
  `bun run typecheck` clean; `bun run smoke` → **49 checks pass**;
  `bun run package` ok; frozen `eh-serve --check-imports` → **0 failures**
  (`entertainment_harness.video.frames` bundled and importable);
  `--check-tts` → **ok**.
