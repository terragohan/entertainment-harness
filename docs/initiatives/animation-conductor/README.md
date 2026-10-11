# Animation conductor — deterministic timeline-driven animation runtime

Status: **done** (Phase 1 complete; Conductor phase deferred — see status note below)

## Goal

An AI Animation Conductor: a component that understands a manga's
characters, actions, emotions, dialogue, scene continuity, camera intent,
and timing — and controls an **explicit animation timeline** that a
deterministic runtime executes frame by frame:

    manga + character/context information
                |
                v
        semantic scene/storyboard (SceneSpec / scene.json)
                |
                v
             Conductor  (future: produces/edits timelines)
                |
                v
          timeline.json  (sparse, human-editable, versioned)
                |
                v
       Sequencing Renderer
          |            |
          |            +--> audio scheduling
          |
          +--> per-frame scene state
                |
                v
            Compositor
                |
                v
          PNG frame sequence
                |
                v
              ffmpeg
                |
                v
             out.mp4

Phase 1 builds only the **Sequencing Renderer** (timeline model, sequencer,
compositor, frame/ffmpeg assembly) — the deterministic execution substrate
the future Conductor will control. Timelines are written by hand in this
phase.

## Concepts (the durable contracts)

- **SceneSpec / storyboard** (`anime/scene.py`, manga-to-anime-scene
  initiative): describes the *semantic/cinematic* scene — shots, actions,
  prompts. It says nothing about exact frames.
- **Timeline** (`anime/timeline.py`, `timeline.json`): describes *exactly
  when* animation state changes — sparse keyframes and discrete events on
  an integer-frame clock. Executable, validated, human-editable.
- **Conductor** (future): produces/edits timelines from a SceneSpec and
  character/context information. Not built in Phase 1.
- **Sequencer** (`anime/sequencer.py`): resolves the sparse timeline into
  the complete scene state at an arbitrary frame (`state_at(timeline, n)`).
  Decides *which* state applies; never draws pixels.
- **Compositor** (`anime/compositor.py`): converts one resolved frame state
  plus visual assets into pixels. Understands no story intent.
- **Sequencing Renderer** (`anime/render.py`): the combined deterministic
  runtime — sequencer + compositor + frame output + ffmpeg assembly +
  audio scheduling.
- **Asset generator** (existing image-gen seams: `video/frames.py`,
  `video/sequence.py`, `video/gen/`): creates visual assets when new visual
  information is required (poses, expressions, mouth/eye states,
  backgrounds, difficult keyframes). Image generation is NOT the animator.
- **ffmpeg** (`video/assemble.py` helpers): encodes/muxes the completed
  frame sequence and audio.

`video/sequence.py` is NOT this initiative's sequencer: it is a *generative*
chained-frame animator (an image model invents each frame from the previous
one). `anime/sequencer.py` is *deterministic*: it resolves an explicit
timeline into per-frame state with no model involvement.

## Non-goals (Phase 1)

- No LLM timeline generation (the Conductor is a later phase).
- No generative video, optical flow, skeletons/rigs, phoneme lip sync.
- No visual-quality optimization — synthetic shapes suffice.
- No redesign of the existing harness; `anime-scene`, `SceneSpec`,
  `scene.json`, `anime/planner.py`, and `anime/pipeline.py` stay intact.
- No per-frame composite caching: composite frames are cheap, AI
  generations are expensive — re-rendering all PNGs and re-running ffmpeg
  after a timeline edit is acceptable. Asset identity stays addressable so
  future selective regeneration is straightforward (a camera edit never
  implies regenerating `character-pose-1.png`).

## Design principles

1. The Conductor controls time; the renderer never makes creative timing
   decisions.
2. The timeline is sparse — no per-frame repetition.
3. The sequencer resolves state ("what should the scene look like at
   frame N?").
4. The compositor draws state ("given this state and these assets, what
   are the pixels?").
5. Image generation creates missing visual information, nothing more.
6. Deterministic operations (timing, holds, cuts, pans, zooms, blinks,
   mouth scheduling, asset switching, audio placement, compositing) stay
   deterministic.
7. Frames are cheap; generations are expensive.
8. The timeline stays editable by humans, the Conductor, repair agents,
   and eventually a GUI editor.

## Phase 1 — deterministic Sequencing Renderer

Scope: static holds, exact cuts, camera pan/zoom (linear interpolation),
asset/pose/expression swaps, blinking (as exact asset swaps), crude
deterministic mouth cycling, audio placement at exact frame offsets, exact
frame timing, deterministic re-rendering.

- [x] **1a — initiative + timeline domain model** (`anime/timeline.py`).
  `load_timeline` / `save_timeline` / `validate_timeline`; integer frames
  canonical (`duration_frames`, default fps 24); sparse: assets, layers,
  camera keyframes, events (asset swap, mouth interval), audio clips;
  schema version 1. Validation: version, fps/dimensions/duration > 0, frame
  indices in range, event ranges, asset/layer references, ordered camera
  keyframes, audio start frames; malformed files fail before rendering.
  Gate: `uv run pytest tests/test_timeline.py -q` green.
- [x] **1b — sequencer** (`anime/sequencer.py`). `state_at(timeline, n)`:
  persistent layer state, exact event boundaries, linear camera
  interpolation (normalized x/y + scale), blink via asset events,
  deterministic mouth cycling (`states[(offset // period) % len]`).
  Gate: sequencer tests in `tests/test_timeline.py` green.
- [x] **1c — compositor** (`anime/compositor.py`). Frame state + assets ->
  Pillow image: ordered alpha-composited layers (normalized placement,
  height-relative scale), camera crop/pan/zoom, exact output size.
  Gate: pixel-exact tests in `tests/test_anime_render.py` green.
- [x] **1d — sequencing renderer + CLI + demo** (`anime/render.py`,
  `eh anime-render`, `scripts/anime_conductor_demo.py`). Frame loop over
  `range(duration_frames)` writing `frame-000000.png` …, ffmpeg assembly at
  the timeline's fps, audio placed at `start_frame / fps` seconds, always
  `duration_frames / fps` long regardless of audio length. No manga, DB,
  Runway, OpenRouter, or Ollama required to render a timeline.
  Gate: `uv run pytest tests/test_anime_render.py -q` green (real ffmpeg,
  guarded), full `uv run pytest -q` at the accepted baseline, demo renders
  a 120-frame 5 s MP4 with no network.

## Definition of done (Phase 1)

1. `timeline.json` is a validated, human-editable format.
2. Integer frames are the canonical animation clock.
3. Deterministic sequencer resolves complete scene state at frame N.
4. Compositor turns resolved scene state into pixels.
5. Sequencing Renderer produces the complete ordered frame sequence.
6. Static holds work.
7. Camera pan/zoom works.
8. Blink asset sequencing works.
9. Crude mouth-state sequencing works.
10. Pose/expression asset swaps work at exact frames.
11. Audio can begin at an exact frame offset.
12. A 5 s 24 fps timeline generates exactly 120 visual frames.
13. Frames + audio assemble into a valid MP4.
14. Tests verify exact timing/state behavior, not merely file existence.
15. No paid API or network access in the tests.
16. Existing tests stay at the repository's accepted baseline.
17. This README contains real gate evidence.

## Eval harness

The end-to-end gate is a standalone script, separate from the unit tests —
the same split as the desktop UI's `bun run smoke`:

    uv run python scripts/eval_animation_conductor.py [--out DIR]

It is self-contained (no network, models, or paid APIs; needs only
ffmpeg/ffprobe), builds its own synthetic fixtures, renders real MP4s
through the real Sequencing Renderer, checks exact behavior, prints a
PASS/FAIL report, and exits 1 on any failure. Two scenarios: `exact`
(160x90, 96 frames — frame counts/names, pixel-exact swaps and camera pans,
byte-identical re-render, fps/duration, audio offset via RMS) and `demo`
(the `scripts/anime_conductor_demo.py` timeline — 120 frames / 5.0 s,
blink and pose-swap visibility over camera drift, dialogue offset, the
`eh anime-render` CLI). The fast unit gate stays `uv run pytest`.

Run 2026-10-07:

    == scenario: exact (160x90, 96 frames @ 24fps, tone at frame 48) ==
    PASS  invalid timeline rejected before rendering
    PASS  no output produced for the invalid timeline
    PASS  exactly duration_frames files with deterministic names
    PASS  frame dimensions are exact
    PASS  camera pan: first frame shows the red half
    PASS  camera pan: last frame shows the blue half
    PASS  asset swap holds before its frame
    PASS  asset swap applies at its exact frame
    PASS  re-render is byte-identical (deterministic)
    PASS  mp4 is h264 at the timeline's exact geometry and fps
    PASS  mp4 duration is duration_frames / fps (audio does not stretch it)
    PASS  second render's mp4 matches too
    PASS  audio starts at frame 48 / 24fps = 2.0s (silent before)
    PASS  audio is audible after 2.0s
    PASS  1s tone ends but the video continues
    == scenario: demo (640x360, 120 frames @ 24fps = 5s) ==
    PASS  5s at 24fps produces exactly 120 frames (0..119, no frame 120)
    PASS  mp4 geometry/fps match the timeline
    PASS  mp4 duration is exactly 5.0s
    PASS  blink at frame 36 is visible over camera drift
    PASS  pose swap at frame 100 is visible over camera drift
    PASS  dialogue tone starts at frame 60 / 24fps = 2.5s
    PASS  tone ends at 4.0s while the video runs to 5.0s
    PASS  `eh anime-render` exits 0 and prints the output path
    == summary: 23/23 checks passed ==
    EVAL PASS

## Evidence

Phase 1 (2026-10-04) — all gates green.

Focused tests (28 timeline/sequencer + 15 compositor/renderer/CLI):

    $ uv run pytest tests/test_timeline.py -q
    28 passed in 0.03s

    $ uv run pytest tests/test_anime_render.py -q
    15 passed in 0.66s        # includes real-ffmpeg end-to-end: exact frame
                              # count, fps 24/1, 4.0s duration, audio RMS
                              # silent before frame 48 (=2.0s) and loud after

Full suite (baseline kept; the one failure is the known pre-existing
Python-3.13 json-strictness failure in tests/test_lfm.py, untouched by this
initiative and deselected in CI):

    $ uv run pytest -q
    FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma
    1 failed, 990 passed, 2 warnings in 17.38s

Synthetic demo (no network, no paid APIs — Pillow-drawn assets + a
synthesized tone WAV, 5 s / 24 fps / 120 frames: static hold, camera push,
blink at frames 36-38, mouth cycling 60-95, pose swap at frame 100, audio
at frame 60 = 2.5 s):

    $ uv run python scripts/anime_conductor_demo.py --out /tmp/eh-animation-demo
    Wrote /tmp/eh-animation-demo/timeline.json
    Timeline: 120 frames at 24 fps (5s), 640x360, 2 layer(s), 4 event(s), 1 audio clip(s)
      rendered 120 frame(s) at 640x360 -> /tmp/eh-animation-demo/render/frames
      assembled /tmp/eh-animation-demo/render/out.mp4 (5.000s nominal 5.000s, 24 fps)
    Demo rendered: /tmp/eh-animation-demo/render/out.mp4

    $ ffprobe -v error -show_entries format=duration -of csv=p=0 out.mp4
    5.000000
    $ ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames -of csv=p=0 out.mp4
    h264,640,360,24/1,120          # exactly 120 frames; no frame 120

    Audio offset (extracted WAV, RMS windows):
    RMS 1.0-2.2s (pre-dialogue):  0.0
    RMS 2.7-3.5s (dialogue):      6401.0     # tone starts at frame 60/24 = 2.5s
    RMS 4.3-4.9s (post-dialogue): 0.0        # 1.5s tone ends; video continues

    Inspected frames 0/37/64/110: base hold, blink (eyes closed), mouth open,
    red-shirt wave pose; the camera push-in is visible across the scene.

`eh anime-render --help` renders, and the command rejects an invalid
timeline before rendering anything (tests/test_anime_render.py CLI cases).

Status note: Phase 1 complete (all 17 definition-of-done items met). Marked
**done** 2026-10-07: the next phase (Conductor-produced timelines) was never
started here and is deferred — when picked up it should be its own
initiative. The single active slot passes to
[frame-gen-evals](../frame-gen-evals/).
