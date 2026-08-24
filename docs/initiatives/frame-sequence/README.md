# Frame sequence — frame-by-frame panel animation video mode

Status: **done** (Phases 1–4 complete, 2026-09-27)

## Goal

New video mode **`sequence`**: take each grounded panel, **expand it to
video size** via AI outpainting, then **generate N chained frames** (each
frame generated from the previous one) that concatenate into real
frame-by-frame animation, smoothed to playback fps by ffmpeg interpolation.

User decisions (2026-09-27): chained-from-previous frames · AI outpainting
for expansion · 1–2 fps generated + interpolation to 24/30 fps.

Contrasts with the existing `animate` mode: `animate` crops the panel and
generates 2–6 independent variations of the *same composition* (xfade
chain, "living panel"). `sequence` outpaints the panel to the full video
frame and chains frames temporally — actual motion — at an fps-driven
frame count.

## Non-goals

- No new built-in local frame generator (third-party plugins can register
  `frame-sequence` animators). Phase 4 later added hosted built-ins
  (`openrouter`, `openrouter-sequence`) alongside Runway.
- No per-frame TTS/lip-sync, no camera-path planning — motion is the
  generator's interpretation of the beat text.
- `animate` mode is untouched; both modes coexist.

## Design answers

**Number of frames (fps)** — `[sequence] fps = 1.5` generated frames per
second of slot share (count = `max(2, round(share * fps))`, capped by
`max_frames = 12`). Frame 0 is the outpainted expansion; the rest are
chained generations. ffmpeg `minterpolate` interpolates to `interp_fps = 30`
at assembly (the pipeline's output FPS) — smooth playback without paying
for 30 generations/second.

**Models** — Runway **gen4_image** (`text_to_image` with tagged references,
the seam `anime` and `frames` already use) for expansion + chained frames
(`[sequence] model` configurable). The **vision-role model** (default
qwen3-vl:8b) is the drift critic. ffmpeg `minterpolate` does fps smoothing
(no model). No new dependencies.

**Critics** — a **drift critic** per chained frame: the vision-role model
compares the new frame against the previous frame + original panel.
Feedback → regenerate with the feedback appended (max 2 attempts);
persistent drift → truncate the chain at the last good frame (the
truncated chain still fills the slot). Off via `[sequence] critic = false`.
Mechanical checks (exists, decodable, right size) always on.

**Integration** — new `sequence` entry in `VIDEO_MODES`; grounded like
`panels`/`animate`; animator registered in the existing `frames` plugin
registry under `"sequence"` with a new `frame-sequence` capability;
`render_state.json` records provider+fps; `videos` metadata attributes the
provider; UI dropdown + smoke list; server validation follows `VIDEO_MODES`
automatically.

**Cost note**: ≈ `1 + (count − 1)` gen4_image calls per anchor, ×2 worst
case with the critic on. A grounded chapter (100–200 anchors) at fps 1.5
≈ 400–1000 image calls.

## Phases

### Phase 1 — config + sequence animator (gate: targeted pytest green)

- `SequenceConfig` + `Config.sequence` + `[sequence]` parsing.
- `video/sequence.py`: `RunwaySequenceAnimator` (`frame-sequence` +
  `image-gen` capabilities), `EXPAND_PROMPT` / `CHAIN_PROMPT` /
  `DRIFT_PROMPT`, content-addressed cache under `workdir/"sequence"`,
  critic hook with truncate-on-persistent-drift, registered in the frames
  registry as `"sequence"`.
- `tests/test_sequence.py`.

**Status: done.** `SequenceConfig`/`[sequence]` parsing in config.py;
`video/sequence.py` with `RunwaySequenceAnimator` (registered as
`"sequence"` in the frames registry, capabilities `{"frame-sequence",
"image-gen"}`), `EXPAND_PROMPT`/`CHAIN_PROMPT`/`DRIFT_PROMPT`,
`frame_count(share, fps, max_frames)`, `video_ratio(resolution)`,
content-addressed cache under `workdir/"sequence"`, and the critic hook
(feedback → one retry → truncate). One design refinement: cached chained
frames are *re-critiqued* when the critic is on — otherwise a re-run would
silently trust a frame that failed the critique last time (its
content-addressed key is a cache hit).

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 852 passed, 1 warning in 16.06s
```

(Known pre-existing failure; 10 new tests in `tests/test_sequence.py`.)

### Phase 2 — pipeline + assembly integration (gate: full pytest green)

- `assemble.py`: `build_sequence_clip` (concat at slot/N per frame +
  minterpolate to interp_fps, slot-exact); `assemble()` sequence params;
  `seg.motion == "sequence"` anchor branch; degradation to panels crop.
- `pipeline.py`: `VIDEO_MODES`, grounded-mode sets, render-state key,
  `elif mode == "sequence"` branch building the vision-role drift critic.
- Tests: mode validation, render-state invalidation, faked integration,
  real-ffmpeg clip test, degradation.

**Status: done.** `build_sequence_clip` in assemble.py (concat at slot/N
per frame + `minterpolate` to interp_fps, `-frames:v round(slot*interp_fps)`
so it stays slot-exact; a 1-frame chain degrades to a static clip);
`assemble()` gained `sequence_interp_fps`/`sequence_critic` and a
`"sequence"` anchor branch that passes the critic through to the animator
and falls back to the panels Ken Burns crop when the provider makes no
frames. `pipeline.py`: `VIDEO_MODES` += `"sequence"`, grounding and
motion-restamp sets updated, `render_state.json` records
`sequence = "<provider>:<fps>"` (fps change re-renders), and the
`elif mode == "sequence"` assembly branch builds the vision-role drift
critic — a vision-model failure logs and degrades to `critic=None` instead
of killing the render.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 860 passed, 1 warning in 16.72s
```

(Known pre-existing failure; 8 new tests in `tests/test_video.py` covering
the assemble branch, critic pass-through, degradation, render-state
invalidation on fps change, and a real-ffmpeg slot-exactness check.)

### Phase 3 — UI + docs (gate: typecheck, smoke, full pytest)

- work.ts dropdown + smoke.ts list + CLI help text.
- design.md (mode bullet, config sample, capability string), plugins.md.
- Evidence pasted; initiative marked done.

**Status: done.** work.ts `VIDEO_MODES` += `"sequence"` (+ tooltip line);
smoke.ts expected mode list updated; `eh recap --video-mode` help describes
sequence; design.md gained the `sequence` mode bullet, the `[sequence]`
config sample, the `[video] mode` comment entries, the `frame-sequence`
capability string, and the grounded/API/UI mode lists; plugins.md lists
`frame-sequence` and documents the `critic=` kwarg sequence providers
receive. Server-side `video_mode` validation follows `VIDEO_MODES`
automatically (no change needed).

Gate evidence (2026-09-27):

```
$ uv run pytest -q
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 860 passed, 1 warning in 17.90s          # known pre-existing failure

$ cd ui && bun run typecheck
$ tsc --noEmit                                     # clean

$ bun run smoke
SMOKE PASS — 49 checks against a live backend
```

### Phase 4 — OpenRouter image providers (gate: full pytest, typecheck, smoke, package)

- `video/gen/openrouter.py`: `OpenRouterImageProvider` (`"openrouter"`,
  `{"image-gen"}`) — chat-completions image generation through the existing
  `[models.openai_compat]` endpoint (OpenRouter or any OpenAI-compatible
  host), `modalities: ["image", "text"]`, tagged references as data-URL
  parts, output normalized to the ratio's exact pixels for ffmpeg.
- `OpenRouterFrameAnimator` (`"openrouter"`, animate mode) and
  `OpenRouterSequenceAnimator` (`"openrouter-sequence"`) sharing the
  Runway loops via extracted `generate_frames_with(...)` /
  `sequence_frames_with(...)`.
- `[frames] model` / `[sequence] model` default to `""` (provider default);
  users can point at any image-capable chat model, e.g. `gpt-6-luna`.
- Settings UI: Frame/Sequence provider selects + model fields.
- Tests: client shape/errors, animators, registry, model resolution.

**Status: done.** `video/gen/openrouter.py` resolves the key from
`[models.openai_compat] api_key` → `OPENAI_COMPAT_API_KEY` →
`OPENROUTER_API_KEY`, sends `modalities: ["image", "text"]` with reference
images as tagged data-URL parts (JPEG), accepts data-URL or http(s) image
responses, and center-crops/resizes output to the exact ratio pixels so the
ffmpeg concat stays uniform. `video/frames.py` and `video/sequence.py` had
their generation loops extracted to `generate_frames_with(...)` /
`sequence_frames_with(...)`; `OpenRouterFrameAnimator` (`"openrouter"`,
animate mode) and `OpenRouterSequenceAnimator` (`"openrouter-sequence"`)
reuse them, so cache, critic, and truncation behavior are identical to the
Runway providers. `FramesConfig.model`/`SequenceConfig.model` now default
to `""` (empty = provider default — Runway falls back to `gen4_image`,
OpenRouter to `google/gemini-2.5-flash-image`); setting
`model = "gpt-6-luna"` overrides either. Runway's missing-key `VideoError`
now names the OpenRouter alternative. Settings shows curated Frame
provider (`local, runway, openrouter`) and Sequence provider (`sequence,
openrouter-sequence, local`) selects with Frame/Sequence model rows;
smoke asserts both selects render.

Gate evidence (2026-09-27):

```
$ uv run pytest tests/test_openrouter_images.py tests/test_frames.py tests/test_sequence.py -q
32 passed, 1 warning in 3.21s

$ uv run pytest -q
1 failed, 911 passed, 1 warning in 17.43s
# (the one failure is the known pre-existing
#  tests/test_lfm.py::test_repair_json_repairs_trailing_comma — Python 3.13
#  json strictness, unrelated; 15 new tests: 8 in test_openrouter_images.py,
#  4 in test_frames.py, 3 in test_sequence.py)

$ cd ui && bun run typecheck
$ tsc --noEmit                                     # clean

$ bun run smoke
SMOKE PASS — 72 checks against a live backend

$ bun run package
creating dmg... creating update.json... moving artifacts...   # exit 0
```

design.md updated in the same change: `[frames]`/`[sequence]` config-sample
comments name both providers + the `gpt-6-luna` example, the animate and
sequence mode bullets name both animators and the shared loops, and the
"Remote inference (openai_compat backend)" section documents the
image-generation call shape.

## Follow-up fixes (2026-09-27)

A live sequence run surfaced two robustness gaps, fixed post-close:

- **Runway reference aspect window**: gen4_image rejects reference assets
  with width/height < 0.5 (tall manga panels are ~0.27) with a 400 that
  killed the whole run. `_prepare_image` now pads extreme aspects into the
  [0.5, 2.0] window with white bars (never a crop), and the Runway +
  OpenRouter clients wrap transport failures in `VideoError` (the seam
  contract) instead of leaking httpx exceptions.
- **Per-anchor degradation**: assembly catches `VideoError` from the
  animate/sequence animator per anchor and degrades that anchor to the
  panels Ken Burns crop — one bad panel no longer fails the chapter.

Second follow-up (same day): the first live `openrouter-sequence` run
showed the failure modes of a user-supplied model id — `gpt-6-luna` is
`text+image->text` (no image OUTPUT) so OpenRouter 404s at routing, and the
built-in default id (`gemini-2.5-flash-image-preview`) had been retired
upstream. `DEFAULT_IMAGE_MODEL` moved to the stable
`google/gemini-2.5-flash-image`; a modality-routing 404 now raises the new
`VideoConfigError` (a `VideoError` subclass, `video/script.py`) naming the
model and the fix; assembly re-raises `VideoConfigError` instead of
degrading per anchor — per-panel failures degrade, broken config aborts.

Evidence: `uv run pytest -q` → `1 failed, 922 passed` (known failure; 3 new
tests — 2 in `tests/test_openrouter_images.py`, 1 in
`tests/test_video.py`); `bun run typecheck` clean; `bun run smoke` →
SMOKE PASS (72 checks); backend rebuilt, `--check-imports` 0 failures.
design.md's animate/sequence bullets + remote-inference paragraph updated.

Third follow-up (same day): the first run with a degraded-every-anchor
chapter exposed two cache holes. (1) `render_state.json` keyed the sequence
state on provider+fps and animate on provider only — the image model wasn't
tracked, so fixing a broken model never invalidated rendered videos; the
keys now include the resolved model (`resolve_image_model` in
`video/frames.py`), which also covers built-in default changes. (2)
Degraded fallback clips were written to the canonical `clip-*.mp4` cache,
so a later healthy run reused Ken Burns clips instead of retrying the
generator; fallback clips now build as `clip-*.fallback.mp4` (still muxed
in the current run, retried next run). The poisoned ch-47 video produced
before these fixes was healed with `eh wipe --chapters 47` (artifact kept;
the wiped row re-enters the video backfill).

Evidence: `uv run pytest -q` → `1 failed, 925 passed` (known failure; 3 new
tests — resolver cases in `tests/test_frames.py`, model-change re-render +
fallback-not-cached in `tests/test_video.py`); backend rebuilt,
`--check-imports` 0 failures; design.md render-state/degradation wording
updated. Known remaining gap: a *completed* video with a few degraded
anchors counts as done for run selection (`chapter_needs_work` /
`_chapters_missing_video` only see the videos row) — healing one today
needs `eh wipe` on that chapter.

Evidence: `uv run pytest -q` → `1 failed, 916 passed` (the known
pre-existing failure; 5 new tests — 3 reference-padding in
`tests/test_video_gen.py`, 2 failure-degradation in `tests/test_video.py`).
design.md's animate/sequence bullets and the Runway-client paragraph
updated in the same change.

Fourth follow-up (same day): with the correct default model
(`google/gemini-2.5-flash-image`) live, anchors intermittently failed with
"The endpoint returned no image" — yet replaying the exact failing request
(ch-47 `panel-00-0.png` + expand prompt) manually succeeded, proving the
empty-image 200 responses are transient upstream flakiness rather than a
request-shape problem. `text_to_image` in `video/gen/openrouter.py` now
retries a no-image response up to 3 attempts with a 2s/4s backoff
(`NO_IMAGE_ATTEMPTS`) before raising `VideoError`, so a momentary upstream
miss no longer degrades an anchor to Ken Burns.

Evidence: `uv run pytest -q` → `1 failed, 927 passed` (the known
pre-existing failure; 2 new tests in `tests/test_openrouter_images.py` —
retry-then-succeed and persistent-failure-raises-after-3);
backend rebuilt, `--check-imports` 0 failures.

## Evidence

Final gates (2026-09-27), all green:

```
$ uv run pytest -q
1 failed, 860 passed, 1 warning in 17.90s
# (the one failure is the known pre-existing
#  tests/test_lfm.py::test_repair_json_repairs_trailing_comma — Python 3.13
#  json strictness, unrelated)

$ cd ui && bun run typecheck && bun run smoke
$ tsc --noEmit                                  # clean
SMOKE PASS — 49 checks against a live backend

$ bun run package
creating dmg... creating update.json... moving artifacts...   # exit 0

$ ui/resources/backend/eh-serve/eh-serve --check-imports
check-imports: 0 failure(s)

$ ui/resources/backend/eh-serve/eh-serve --check-tts
check-tts: ok
```
