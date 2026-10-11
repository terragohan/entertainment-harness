# Frame-gen evals — experiment harness for animated-frame image generation

Status: **done** (all four phases gated; evidence below)

## Goal

A repeatable, comparable experiment harness over the animated-frame image
generation seams (`video/frames.py` animate mode, `video/sequence.py`
sequence mode). Experiments are declared as data (TOML), sweep dimensions
like image model and frame cadence, and each run produces a self-contained
directory with the generated frames, a JSON manifest of configs + metrics +
vision-judge scores, and an HTML gallery for human review.

    experiment.toml (arms = config overrides)
                |
                v
            runner  (drives generate_frames_with / sequence_frames_with)
                |
                v
      runs/<timestamp>--<name>/
        arms/<arm-slug>/frames|sequence/*.png
        manifest.json   (configs, metrics, judge scores)
        index.html      (gallery + scores, human review)

The question this answers: **for animated frames, which model × cadence ×
critic settings give faithful frames with real motion at acceptable cost?**
Cadence is the interesting trade-off — higher frame rates risk stagnation
(motion never advances), lower ones risk drift (each frame invents too much)
— and today there is no way to measure it.

## Rationale

- The generation loops already exist and are duck-typed on the client
  (`text_to_image`); the harness drives them directly — no product-pipeline
  changes.
- Content-addressed caching in the loops keys on image+prompt+model+ratio,
  and prompts embed "frame {phase} of {count}", so cadence changes naturally
  produce distinct cache keys. Arm isolation = per-arm workdirs.
- Judge semantics mirror the pipeline's drift critic exactly where they
  overlap (`DRIFT_PROMPT` reused verbatim, OK-prefix parsing), so eval
  scores predict pipeline behavior. Two eval-only dimensions are added
  because the cadence question needs them: per-frame faithfulness vs the
  source panel, and pairwise motion progress (stagnant vs advancing).
- Gallery + manifest keep results reviewable and comparable across runs and
  machines; runs are artifacts, never committed (`evals/runs/` gitignored).

## Concepts (the durable contracts)

- **Experiment** (`evals/experiments/*.toml`): name, mode
  (`"animate"` | `"sequence"`), panel list, slot duration, optional scene
  `context` (embedded in the frame prompts), judge flag, and arms. Declared
  as data; validated strictly (unknown keys, bad values, duplicate or
  path-unsafe arm names rejected).
- **Arm**: one config variation. `provider` selects the image client:
  `"openrouter"` | `"runway"` (hosted, reference-aware) | `"diffusers"`
  (local open-weights, eval-scoped — see below). The experiment's mode
  selects the loop. `model` may be empty (provider default via
  `resolve_image_model`). Cadence knobs: animate — `seconds_per_frame`,
  `max_frames`; sequence — `fps`, `max_frames`, `interp_fps`, `critic`,
  `resolution` (the outpainting ratio derives from
  `config.video.resolution`, so it is a generation-affecting knob here).
- **The diffusers client** (`evals/imagegen/clients.py`): runs open-weights
  image models locally (FLUX, SDXL, …) through the same `text_to_image`
  duck type, lazily imported — never a product dependency; run with
  `uv run --with diffusers --with torch python -m evals.imagegen …` and it errors
  clearly when diffusers is absent. Open T2I models cannot consume the
  `@panel`/`@prev` reference images (no server-side tagging, no IP-Adapter
  in scope): the client accepts-and-ignores references and the manifest
  records `references_used: false`. Their faithfulness scores therefore
  measure blind generation — the low numbers are a real finding (this is
  the gap reference conditioning closes), and the gallery labels these
  arms so nobody misreads them.
- **Run**: one execution of an experiment → `evals/runs/<timestamp>--<name>/`
  with per-arm workdirs (frames land inside them), `manifest.json`,
  `index.html`.
- **Manifest** (`manifest.json`, versioned): base config snapshot, per-arm
  effective config + effective ratio + `references_used` + frame counts
  (requested / produced / cached / truncated — truncation = produced <
  `frame_count(...)` result), timing, provider errors, judge scores.
  Renderable offline (`--render-only`).
- **Judge scores** (`evals/imagegen/judge.py`): three dimensions per arm —
  faithfulness (each frame vs the source panel, `FAITHFULNESS_PROMPT`),
  drift (consecutive pairs, `DRIFT_PROMPT` verbatim), motion (consecutive
  pairs, `MOTION_PROMPT`: advancing vs stagnant). Vision-role adapter
  resolved like the pipeline does (`get_vision_model` + `probe` + `ensure`,
  `video/pipeline.py:742-756`); unavailable → warn-and-skip, recorded in the
  manifest. The judge model is whatever the vision role resolves to — bind a
  cheap vision model (e.g. `openai/gpt-6-luna`, $0.5/M out tokens) via
  `[roles.vision]` for large scoring sweeps. Scoring runs after generation
  (and after any caching), so a re-run of a fully cached arm still gets
  scored.
- **Assembly-only knobs**: `interp_fps` never changes generated frames; the
  manifest labels it, and it only affects preview-clip playback fps.

## Non-goals

- No changes to the product pipeline (`video/frames.py`,
  `video/sequence.py`, `video/assemble.py` behavior untouched — the harness
  only calls into them).
- No new plugin seams, no new *product* provider clients, no new product
  dependencies. Eval-scoped clients inside `evals/` (the diffusers client)
  are in scope; the product package never imports them.
- Not the animation-conductor's Conductor; generative frames remain the
  asset generator, not the animator.
- No automatic "winner" selection — the harness measures; humans decide.
- Live API runs are user-initiated (keys required); tests stay offline.

## Design principles

1. Experiments are data; arms are config overrides, nothing else.
2. Drive the shared loops (`generate_frames_with` / `sequence_frames_with`)
   — eval what the pipeline actually runs, not a reimplementation.
3. A failing arm (missing API key, missing diffusers, provider error,
   OOM) never kills a run; it is recorded and shown in the gallery.
4. Runs are self-contained, comparable artifacts: frames + manifest +
   gallery in one directory. Comparability caveats (e.g.
   `references_used: false`) are data in the manifest, not exclusions.
5. Judge semantics match the pipeline critic byte-for-byte where they
   overlap, so eval scores predict pipeline behavior.
6. Tests are offline: fake image client, scripted vision adapter, no paid
   APIs, no DB, no network, no diffusers.

## Phase 1 — harness core (experiment files, runner, CLI skeleton, manifest)

- [x] `experiment.py`: TOML load/validate (`Experiment`/`Arm` dataclasses);
  strict schema; per-mode knob sets; provider→client map (`openrouter` /
  `runway` / `diffusers`).
- [x] `clients.py`: client construction for the three providers; the
  diffusers client lazily imports diffusers, honors `ratio` via the
  pipeline's accepted sizes where possible, writes normalized PNGs, and
  raises a clear "run with `uv run --with diffusers --with torch`" error otherwise.
- [x] `runner.py`: per-arm config copy + overrides, workdir per arm under
  the run dir, drive both loops, metrics (counts incl. truncation via
  `frame_count` vs produced, timing, cache hits from dir snapshots, provider
  errors), `manifest.json`.
- [x] `__main__.py` (Typer): `uv run python -m evals.imagegen run
  <experiment.toml> [--out DIR] [--limit N]`, and `list` — experiments with
  estimated generation count per arm (from `frame_count` math — a cost
  preview).
- [x] `tests/test_evals_imagegen.py`: validation cases; fake client
  (deterministic Pillow PNGs, call recording, failure injection) for both
  modes; arm isolation; manifest shape. (`sys.path` insert for `evals/` —
  product packaging untouched.)

Gate: `uv run pytest tests/test_evals_imagegen.py -q` green.

## Phase 2 — judge scoring

- [x] `judge.py`: `FAITHFULNESS_PROMPT` (frame vs panel), `DRIFT_PROMPT`
  reuse (pairwise), `MOTION_PROMPT` (pairwise: advancing vs stagnant);
  verdict parsing mirroring `drift_critic`; per-arm aggregates into the
  manifest.
- [x] Adapter resolution mirroring `video/pipeline.py:742-756`, injectable;
  warn-and-skip recorded in the manifest.
- [x] Tests: scripted adapter (OK / `DRIFT: …` / garbage / stagnant),
  garbage tolerated per existing semantics, scoring-after-caching.

Gate: focused tests green; manifest contains the three score dimensions.

## Phase 3 — gallery (+ preview clips)

- [x] `gallery.py`: self-contained `index.html` (inline CSS, relative img
  refs, arm sections with knob deltas vs base highlighted, metrics, judge
  summary, verdict badges, drift/stagnation highlights, provider-error arms
  shown, `references_used: false` arms labeled).
- [x] `--render-only` regenerates HTML from a manifest; `--clips` builds
  per-arm preview MP4s (sequence: `build_sequence_clip` at `interp_fps`;
  animate: ffmpeg slideshow at `seconds_per_frame` per frame). ffmpeg
  absence degrades to no clips with the gallery intact.
- [x] Tests: HTML from a fixture manifest; ffmpeg-guarded clip test (like
  `tests/test_anime_render.py`).

Gate: focused tests green.

## Phase 4 — seed experiments + docs

- [x] `evals/experiments/cadence-animate.toml`
  (`seconds_per_frame` × `max_frames`).
- [x] `evals/experiments/cadence-sequence.toml` (`fps` × `critic`;
  `interp_fps` labeled assembly-only).
- [x] `evals/experiments/models-frames.toml` — the model sweep (cadence
  fixed at provider defaults). Arms: **all** hosted image-output models
  from the live OpenRouter list (`/api/v1/models`, image in
  `output_modalities`, 2026-10-07):
  `google/gemini-3.1-flash-lite-image`, `openai/gpt-5-image-mini`,
  `google/gemini-2.5-flash-image` (default),
  `google/gemini-3.1-flash-image`, `google/gemini-3.1-flash-image-preview`,
  `google/gemini-nano-banana-2.1`, `openai/gpt-5-image`,
  `google/gemini-3-pro-image`, `google/gemini-3-pro-image-preview`,
  `openai/gpt-5.4-image-2`
  (`openrouter/auto` / `auto-beta` excluded — routing aliases, not models).
  **Hosted-only decision (2026-10-08): the sweep runs everything through
  OpenRouter/OpenAI — the Runway arms and the local diffusers arm were
  dropped from `models-frames.toml` after the first live run.** The
  diffusers client stays in `evals/imagegen/clients.py` (proven live:
  `stabilityai/sdxl-turbo` generated its 4 frames in the first sweep), and
  open-weights arms can come back as their own experiment file — see
  Future / deferred.
  Model ids rot upstream (precedent: `gemini-2.5-flash-image-preview`
  retired) — re-check the live list before a big run; the manifest records
  the resolved id either way.
  **Model notes**: `openai/gpt-6-luna` is a text-OUTPUT model (accepts image
  input but cannot draw — `output_modalities: ["text"]`, verified
  2026-10-07); it cannot be an image-gen arm, but at $0.5/M output tokens
  it's a good cheap **judge** — eval runs pass `--judge-model
  openai/gpt-6-luna` (no config edit, no ollama needed). The `gpt-6-luna`
  example under `[frames]`/`[sequence]` model in `docs/design.md` was wrong
  the same way and was corrected in this initiative (4 places).
  OpenRouter hosts no open-weights image models (verified 2026-10-07),
  which is why open arms had to run locally via diffusers rather than as
  more OpenRouter ids.
- [x] `evals/README.md`: writing experiments, running (incl. live-API usage,
  `--limit` cost probes, `--gen-timeout`, `--with diffusers --with torch`
  for open arms), where runs land, how to read the gallery.
- [x] `.gitignore`: `evals/runs/`.

Gate: full `uv run pytest -q` at baseline (the one known pre-existing
`tests/test_lfm.py` failure); `list` CLI output pasted; README accurate.

## Definition of done

1. Experiments are TOML data; adding an arm requires no code.
2. Both loops drivable per arm with per-arm config overrides.
3. Runs produce frames + manifest + gallery in one self-contained dir.
4. Judge scores (faithfulness / drift / motion) recorded per arm.
5. A broken arm never aborts a run.
6. `list` previews estimated generation counts (cost).
7. Full test suite at baseline; tests offline.
8. This README holds real gate evidence.

## Future / deferred (not in the phases)

- Shared cross-run generation cache (dedup identical
  image+prompt+model+ratio across runs to save API cost).
- Parallel arm execution (rate-limit aware).
- More judge dimensions (e.g. per-frame detail / text legibility).
- Scene-keyframe evals (`anime/pipeline.py` seam) as another experiment mode.
- Open-weights experiment file (diffusers arms — `stabilityai/sdxl-turbo`
  proven, `black-forest-labs/FLUX.1-schnell`, `Qwen/Qwen-Image` candidates)
  as its own TOML, out of `models-frames.toml` per the hosted-only decision.
- Hosted open-weights image APIs (fal.ai, Replicate, HF Inference) as
  additional eval-scoped clients, if a key becomes available — same client
  map, one more entry.
- Reference conditioning for open models (IP-Adapter / img2img with the
  panel as init) — would make the diffusers arms reference-aware.
- Cross-run cache + a `run --into <existing run dir>` / arm filter, so a
  partially failed run can be topped up without regenerating every arm.

## Evidence

Phases 1–4 gates (2026-10-08) — all green.

Focused tests (experiment/clients/runner/judge/gallery/CLI, fully offline —
fake image client, scripted judge adapter; incl. per-panel error
granularity and `--gen-timeout` coverage added after the first live run):

    $ uv run pytest tests/test_evals_imagegen.py -q
    49 passed in 0.56s

Full suite at the accepted baseline (the one failure is the known
pre-existing Python-3.13 json-strictness failure in tests/test_lfm.py,
untouched by this initiative and deselected in CI):

    $ uv run pytest -q
    FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma
    1 failed, 1039 passed, 2 warnings

    $ uv run python -m evals.imagegen list
    ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ experiment       ┃ mode     ┃ panels ┃ arms ┃ est. generations ┃
    ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ cadence-animate  │ animate  │ 2      │ 3    │ 22               │
    │ cadence-sequence │ sequence │ 1      │ 6    │ 42               │
    │ models-frames    │ animate  │ 2      │ 10   │ 40               │
    └──────────────────┴──────────┴────────┴──────┴──────────────────┘

Live runs (real API spend, judge = `openai/gpt-6-luna`):

1. Smoke (`--limit 1`, 2026-10-08): end-to-end OK in 34 s — 2 frames
   generated, judge scored all three dimensions, clip + gallery written.
   It exposed that `samples/kenja-ch0/page-03.jpg` is a contents page, not
   a story panel — the sweep panels are now page-07 (action) + page-09
   (rain scene).
2. First full sweep (13 arms incl. Runway ×2 + local sdxl-turbo,
   `evals/runs/20261008T054253Z-models-frames/`): every arm produced
   frames; judge table (faithfulness pass rate / drift incidents /
   stagnation):

       nano-banana-2.1          100% / 0 / 0      <- best fidelity
       gemini-3.1-flash-image    75% / 0 / 0      <- best cheap tier
       gemini-3-pro-image        75% / 1 / 0
       gemini-3-pro-image-preview 25% / 1 / 0
       gpt-5.4-image-2            0% / 1 / 100%   <- frames frozen (stagnant)
       everything else            0% (runway gen4/turbo: 2 drift each;
                                    sdxl-turbo: blind generation, as labeled)

   Two arms lost their second panel to *transient* network failures
   ("read operation timed out", "incomplete chunked read" after a
   hours-long trickling stream — httpx's per-chunk read timeout cannot
   catch a slow-trickle response). That run took 8.9 h wall for ~35 min of
   real work, which produced two harness fixes, both tested:
   - errors are now recorded per arm×**panel** (a transient failure no
     longer aborts the arm's remaining panels), and
   - `--gen-timeout` (default 900 s, signal-based hard deadline) bounds
     any single generation call.
3. OpenRouter-only re-sweep (10 arms, per the hosted-only decision;
   `evals/runs/20261008T154718Z-models-frames/`): **40/40 frames, zero
   errors, 35.5 min wall** (vs 8.9 h for the first run — the two hard fixes
   work). Judge table (faithfulness % / drift incidents / stagnation %):

       nano-banana-2.1                50 / 0 / 0     <- most consistent (top both runs)
       gemini-3.1-flash-lite          25 / 2 / 0
       gemini-2.5-flash-image         25 / 0 / 0
       gemini-3.1-flash-image         25 / 0 / 0
       gemini-3.1-flash-image-preview 25 / 0 / 0
       gpt-5.4-image-2                25 / 2 / 50    <- slow (122 s/frame) + frozen frames
       gpt-5-image-mini                0 / 0 / 0
       gpt-5-image                     0 / 2 / 0     <- slow (94 s/frame)
       gemini-3-pro-image              0 / 2 / 0
       gemini-3-pro-image-preview      0 / 1 / 0

   Run-to-run variance is real (nano-banana 100→50, gemini-3.1-flash-image
   75→25 on identical configs): 4 frames per arm and one strict judge are
   directional, not final. nano-banana-2.1 topped BOTH runs with zero
   drift; visual spot-check confirms it reproduces the page layout,
   characters, and text most conservatively. OpenAI image models are
   4–10× slower per frame than the gemini tier and drift/stagnation-prone
   here. gpt-6-luna judging is strict (a changed sign text = drift).

Initiative complete: all four phases gated with evidence above; the two
cadence experiment files are ready to run (`uv run python -m
evals.imagegen run evals/experiments/cadence-*.toml`) but live runs are
user-initiated by design.

