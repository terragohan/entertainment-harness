# Evals

Experiment harnesses for the Entertainment Harness — declared as data,
executed into self-contained run directories, reviewed in HTML. The
product package (`src/entertainment_harness/`) never imports from here.

## imagegen — frame-generation experiments

Sweeps over the animated-frame image-generation seams: the `video-animation`
loop (`video/frames.py`, independent motion-phase frames per panel) and the
`frame-animation` loop (`video/sequence.py`, outpaint + chained frames with an
optional drift critic). The question it answers: **which model × cadence ×
critic settings give faithful frames with real motion at acceptable
cost?** The harness drives the product's own loops — it evals what the
pipeline actually runs, not a reimplementation.

### Writing an experiment

Experiments are TOML in `evals/experiments/`:

```toml
name = "my-sweep"             # slug; part of the run dir name
mode = "video-animation"      # "video-animation" | "frame-animation"
duration = 4.0                # seconds of slot share per panel
panels = ["samples/kenja-ch0/page-03.jpg"]
context = "A sentence of scene context, embedded in the frame prompts."
judge = true                  # default true; vision scoring after generation

[[arms]]
name = "gemini-flash-lite"    # slug; the arm's subdir name
provider = "openrouter"       # "openrouter" | "runway" | "diffusers" | "fal" | "replicate"
model = "google/gemini-3.1-flash-lite-image"  # optional; empty = provider default
                                              # fal: the endpoint id ("fal-ai/flux-pro/kontext");
                                              # replicate: "owner/name" (official) or
                                              # "owner/name:version" (community)
seconds_per_frame = 2.0       # video-animation knobs: seconds_per_frame, max_frames
max_frames = 6

[[arms]]
name = "gen4"
provider = "runway"           # model empty -> gen4_image
```

Arms are config overrides, nothing else — adding one never requires code.
Frame-animation-mode arms take `fps`, `max_frames`, `interp_fps`, `critic`,
`director`, and `resolution` instead (`resolution` is frame-animation-only
because the outpainting ratio derives from it; `interp_fps` is
assembly-only and never changes generated frames — it only affects
`--clips` playback; `director = true` swaps the plain loop for the
instruct/assess/revise one in `evals/imagegen/director.py` — per-model
instructions, judge-verdict revisions, trajectories in the manifest — and
conflicts with `critic`, which it subsumes). Validation
is strict: unknown keys, bad modes, duplicate/unsafe arm names,
non-positive durations, out-of-mode knobs, and missing panels are all
rejected. Panel paths resolve from the cwd first, then relative to the
experiment file — run from the repo root.

### Running

```bash
uv run python -m evals.imagegen list                        # cost preview
uv run python -m evals.imagegen run evals/experiments/cadence-video-animation.toml
uv run python -m evals.imagegen run ... --limit 2           # cheap probe
uv run python -m evals.imagegen run ... --no-judge          # skip scoring
uv run python -m evals.imagegen run ... --clips             # + preview MP4s
uv run python -m evals.imagegen render evals/runs/<run>     # re-render HTML
```

- Hosted arms need their usual keys (`OPENROUTER_API_KEY` /
  `[models.openai_compat]`, `RUNWAY_API_KEY` / `[video_gen.runway]`,
  `FAL_KEY` for fal arms, `REPLICATE_API_TOKEN` for replicate arms).
  fal/replicate are instruction-editing i2i providers — the first reference
  image is the edit source, so they condition on the panel for real
  (unlike diffusers' blind generation).
- Open-weights arms are local and need the eval-only extra:
  `uv run --with diffusers --with torch python -m evals.imagegen run ...`
- The judge (and the frame-animation drift critic) uses the configured vision
  role, resolved like the pipeline does. When ollama is down, bind a cheap
  hosted vision model without touching config:
  `--judge-model openai/gpt-6-luna`. Judge failure never kills a run —
  the warning is recorded in the manifest.
- A failing arm (missing key, missing diffusers, provider error, OOM) is
  recorded with its error and the run continues.
- Each panel generation call has a hard deadline (default 900 s,
  `--gen-timeout SECONDS`; `0` disables). It is **per loop call**, not per
  image — a call produces up to `max_frames` images, so dense cadence arms
  (e.g. `fps = 3.0`, `seconds_per_frame = 1.0`) may need a larger value.
  A timed-out panel is recorded with its error like any provider failure
  and the run continues.

### Where runs land

`evals/runs/<UTC timestamp>-<experiment name>/` (gitignored):

```
manifest.json        configs, metrics, judge scores — the comparable artifact
index.html           the review gallery (open in a browser)
panels/              copies of the source panels (keeps the dir portable)
arms/<arm-slug>/frames|sequence/*.png
arms/<arm-slug>/clip-NN.mp4      with --clips
```

### Reading the results

The gallery shows, per arm: the knob table (values differing from your
base config highlighted), metrics (produced/requested frames, generated
vs cached, wall time, truncation), the judge summary, and the frame strip
with verdict badges. Three judge dimensions, parsed exactly like the
pipeline's drift critic:

- **faithfulness** — each frame vs the source panel (ok / drift)
- **drift** — consecutive pairs, `DRIFT_PROMPT` reused verbatim, so eval
  scores predict pipeline-critic behavior
- **motion** — consecutive pairs (advancing / stagnant): the stagnation
  detector the cadence question needs

`references_used: false` arms (local diffusers models can't consume the
`@panel`/`@prev` references) are labeled in the gallery and recorded in
the manifest — their low faithfulness scores measure blind generation,
which is the finding, not a bug.

`manifest.json` (`manifest_version: 1`) carries the base config snapshot,
per-arm effective config + resolved model id + `references_used`, per-panel
frame lists (paths relative to the run dir) + metrics, judge verdicts and
aggregates, and per-arm errors. Everything the gallery shows comes from
it, so results are comparable across runs and machines.

Tests are offline (fake image client, scripted vision adapter):
`uv run pytest tests/test_evals_imagegen.py`.
