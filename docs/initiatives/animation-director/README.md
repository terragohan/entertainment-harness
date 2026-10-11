# Animation director — instruct + assess loop for i2i animation models

Status: **active**

## Goal

A "director" component that **instructs** image-to-image animation models
(per-model prompt templates built from panel + scene context) and
**assesses** their output (faithfulness / drift / stagnation verdicts),
closing the loop: assess → revise instruction → regenerate, keep the best
chain. Built on the frame-gen-evals machinery — the experiment harness is
the training environment, the judge is the reward signal.

Today every model gets the same fixed template (`FRAME_PROMPT`,
`video/frames.py:38`) and no assessment feedback reaches generation. The
first i2i sweeps (`evals/runs/20261010T234628Z-models-i2i/`) showed why a
director is worth it: faithfulness vs motion is a per-model trade-off
(seedream-v5-pro 75% faithful but stagnant; klein-9b moves but drifts),
and each provider has its own quirks a director must know (BFL endpoints
run a flaky content checker that needs sanitized prompts + retry; ideogram
takes a singular `image_url`; nano-banana-21 balances both).

```
panel + context
      |
      v
  instruct  (per-model template from the instruction bank)
      |
      v
  generate (existing loops, content-addressed frames)
      |
      v
  assess   (judge: faithfulness / drift / stagnation)
      |
      +-- drift   -> "preserve composition" revision, regenerate
      +-- stagnant -> "increase motion" revision, regenerate
      v
  keep best chain (judge-score ordered), record trajectory
```

## Training-spectrum plan (what "training" means here)

- **Level 0 — heuristic director** (Phases 1–3): hand-tuned per-model
  templates + the assess→revise loop. No weight training; validates the
  concept and generates labeled trajectories as a byproduct.
- **Level 1 — searched prompts** (Phase 2): instruction phrasings swept as
  eval arms; winners per model get baked into the bank.
- **Level 2 — router/bandit** (deferred, Phase 4 accumulates the data):
  pick which model animates a panel from panel features + score history.
- **Level 3 — fine-tuned judge** (deferred): LoRA a local VLM to predict
  rubric scores, once call cost or judge deprecation bites. A sibling repo
  at that point — training toolchain and artifacts don't belong here; the
  judge interface stays stable so the swap is a config change.

## Concepts (the durable contracts)

- **Instruction bank**: named prompt templates, each tagged with the models
  it applies to (or `default`). The bank records *why* a template exists
  (e.g. `bfl-safe`: no character-preservation phrasing, no series titles —
  BFL content checker, verified 2026-10-11) so rewords are deliberate.
- **Revision**: a bounded retry driven by judge verdicts: drift →
  composition-preserving phrasing; stagnant → motion-push phrasing; max 2
  revisions per frame, best-score wins. Bounded so cost stays predictable.
- **Trajectory**: one (panel, model, instruction id, scores) record per
  assess-revise round — the dataset Level 1–3 consume.
- **Director loop**: instruct → generate → assess → revise, wrapped around
  the existing `generate_frames_with` / `sequence_frames_with` loops. Eval
  arms opt in with a knob (`director = true`); the pipeline wiring comes
  only after eval evidence (Phase 3).
- **Reward = judge scores** (`evals/imagegen/judge.py`), unchanged. The
  judge stays the single source of truth; the director optimizes for it.

## Non-goals

- No weight training (no LoRA, no RL) anywhere in the phases.
- No judge-semantics changes — the director optimizes for the judge, it
  doesn't redefine it.
- No new product dependencies; provider clients stay as they are.
- No UI changes in the phases (a director summary per run is gallery data,
  not UI).
- Not the animation-conductor's Conductor; generative frames remain the
  asset generator.
- Live API runs stay user-initiated; tests offline.

## Design principles

1. The judge is the reward; instruction changes are data (templates),
   never per-arm code.
2. Drive the shared loops — eval what the pipeline actually runs.
3. Revisions are bounded and logged; every trajectory lands in the
   manifest so "the director improved things" is verifiable from artifacts.
4. Provider quirks are knowledge in the bank/client, not silent hacks —
   each entry carries its verification date and reason.
5. Eval-first: nothing enters the product pipeline without judge-score
   deltas from a sweep.

## Phase 1 — eval director loop

- [ ] `evals/imagegen/director.py`: instruction bank (template + model
  tags + provenance note), revision logic (drift/stagnant verdict →
  revision template), best-chain selection by judge aggregates.
- [ ] Runner knob `director = true` on an arm: the arm's panel loop runs
  the director loop instead of the plain loop; trajectories recorded in
  the manifest per panel (instruction id per frame, verdicts, revisions,
  final pick).
- [ ] Seed the bank from live findings: `default` (current FRAME_PROMPT),
  `bfl-safe` (verified pass on flux-2-pro/flux-3 2026-10-11),
  `motion-push`, `preserve-composition`.
- [ ] Tests: scripted judge + fake image client — drift verdict triggers a
  preserve-composition revision; stagnant triggers motion-push; bounded
  retries; manifest trajectory shape.

Gate: focused tests green; manifest contains trajectories.

## Phase 2 — instruction sweep experiment

- [ ] `evals/experiments/instructions-i2i.toml`: instruction variants × top
  models from the 2026-10-10 sweep (nano-banana-21-edit, flux-2-pro-edit,
  ideogram-v45-edit, seedream-v5-pro-edit), directed arms vs baseline.
- [ ] Live run, user-initiated; per-model winner + judge-score delta
  recorded in Evidence below.

Gate: sweep results pasted here; per-model winners identified.

## Phase 3 — pipeline wiring (config-gated, default off)

- [ ] `video/director.py` in the product package: the Phase-1 loop behind
  `[frames] director = true` / `[frame_animation] director = true`;
  instruction bank shared with evals (single source, one module imported
  by both).
- [ ] Product tests: director on/off paths, budget guard (revision cap
  enforced), default behavior byte-identical when off.
- [ ] `docs/design.md` updated in the same change.

Gate: full suite at baseline; a live directed run on a real chapter
produces a video with better judge aggregates than its baseline arm.

## Phase 4 — director memory

- [ ] Trajectory table (additive `ALTER TABLE` in `db.py`): panel features
  hash, model, instruction id, judge scores, timestamp.
- [ ] Sweep runs and pipeline runs both write trajectories; `evals` gains a
  `memory` report command (win rates per model × instruction).
- [ ] Tests: table migration, write/read round-trip.

Gate: report command output pasted here from real accumulated data.

## Definition of done

1. A directed arm measurably beats its baseline twin on judge aggregates.
2. Per-model instruction winners recorded and wired into the pipeline
   behind a default-off flag.
3. Trajectories accumulate across eval and pipeline runs.
4. All phases gated with evidence pasted below.
5. Full test suite at baseline; tests offline.

## Future / deferred

- **Level 2 router** (bandit over models given panel features) — needs the
  Phase 4 table to have real volume.
- **Level 3 fine-tuned judge** (LoRA a local VLM on accumulated
  trajectories + human spot-checks) — **sibling repo** when it happens:
  training toolchain, datasets, and checkpoints don't belong in the
  runtime harness. Contract: artifact + scoring API behind the existing
  judge interface.
- Learned instruction generation (fine-tuned director LLM on trajectories)
  — only if Level 1 plateaus.
- Human-labeling tooling to audit judge bias before Level 3.

## Evidence

_(nothing yet)_
