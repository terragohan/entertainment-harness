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

## Jobs we train models for (the small-model fleet)

The long-run coordinator is **not one giant model but a fleet of small
specialists**, one per job, each behind the interface the harness already
calls. Shared rule: a job earns a trained model only after its
heuristic/frontier version has accumulated enough trajectories to train on
— the fleet is distilled from the harness's own logs, never trained cold.
This section changes no phase gates; it is the map the deferred levels
refer to.

Owned by this initiative:

1. **Frame judge** — faithfulness / drift / motion verdicts on frames and
   frame pairs. Today: a frontier VLM per call (`evals/imagegen/judge.py`,
   the drift critic). Fleet model: a local VLM LoRA'd on accumulated
   verdicts plus teacher labels (an Astra re-judging audit as the
   pseudo-ground-truth). Interface unchanged: same prompts, same verdict
   parsing. Trigger: judge call cost or deprecation bites (Level 3;
   sibling repo for the training toolchain). Deployment shape: a **cascade**
   — the small judge scores everything, and low-confidence/inconclusive
   pairs escalate to the frontier teacher (the verdict parser's
   inconclusive channel is the escalation hook).
2. **Instruction writer** — the per-model generation/revision prompt
   itself. Today: the static bank + two revision templates. Fleet model:
   a small LM fine-tuned to imitate winning instructions from
   trajectories, then preference-optimized against judge scores. Trigger:
   Level 1 search plateaus.
3. **Model router** — which i2i model animates a given panel. Today: the
   user picks arms; sweeps try every model equally. Fleet model: a
   contextual bandit / tiny classifier over panel features × historical
   win rates (the Phase 4 table). Trigger: trajectory volume (Level 2).
4. **Revision triage** — whether a frame is worth revising at all, and how
   much of the ≤3-attempt budget to spend on it. Today: the cap is uniform
   and every frame gets assessed. Fleet model: a cheap classifier
   predicting P(drift) and P(a revision fixes it) from the panel + the
   first verdict. Training data: the trajectories Phase 1 already records
   (attempt counts, revision outcomes). Payoff: judge calls are the
   recurring cost — triage stops spending them where they don't pay.
5. **Motion tagger** — what motion a panel wants (ambient: rain, hair,
   smoke; action: impact lines; camera move), as input to the instruction
   writer. Today: baked into template prose. Fleet model: a tagger from
   panel content; labels mined from successful trajectories. Feeds job 2.
6. **Beat selector** — which anchors get generated frames at all, and at
   what fps. Today: every grounded anchor animates (`docs/design.md`'s
   cost note: ≈400–1000 image calls per grounded chapter). Fleet model: a
   per-beat animate-vs-hold + budget classifier over panel features and
   beat position — climactic beats animate, connective tissue holds.
   Training signal: judge deltas between animated and held renditions of
   the same beat (sweeps can manufacture these on purpose). Payoff: the
   single biggest cost lever in the pipeline.

Jobs 3–6 all consume "panel features" — the fleet's shared infrastructure
is **one small panel encoder** (trained once on the strip corpus, then
frozen) with tiny per-job heads, not four separate vision models.

Adjacent jobs — other parts of the pipeline, same distillation pattern,
**not owned here**, listed so the fleet is designed once:

- **Grounding judge / region picker / page picker** (`video/grounding.py`):
  "which pages illustrate this beat" and "does this region illustrate it"
  — vision-role calls; the same small-VLM candidate behind a different
  rubric. Panel/region segmentation itself stays heuristic CV until its
  errors show up downstream.
- **Recap text judge** (the script judge loop): text-only, the cheapest to
  distill, and the least animation-specific.
- **Final QC gate** (end of the video pipeline): an artifact detector over
  the finished video — sampled frames + audio — catching gibberish text,
  melted frames, dead pixels, and sync failures before a video ships.
  Judge-family (same verdict interface, new rubric); extends the
  grounding stage's "safe over smooth" philosophy to the last mile.
- **Colorizer** (the `colorize` path): manga colorization is a solved-ish
  specialty with dedicated small models; the fleet version is judged on
  colored frames like any other i2i output.
- **Narrator audio director** (the TTS stage; user's name: *narrator
  audio*): a prosody DSL for narration — intonation, crescendo, pacing,
  emphasis, dialogue-vs-narration register per beat — compiled to whatever
  the active TTS engine supports (SSML / audio tags / word timings), plus
  an **audio judge** to score expressiveness. The only fleet job with no
  reward signal today: the bootstrap order is frontier LLM proposes
  markup → audio judge scores → accumulate → only then distill a small LM
  (narration + story context → markup).
- **Scanlation repair** (the strip itself): detect text regions, clean
  them (inpaint the art behind speech-bubble text), retypeset translated
  or reworded text in a matched style. Sub-jobs: detector/cleaner, domain
  translator (manga MT fine-tuned on accumulated translations — SFX,
  honorifics, rolling-context-aware), style-matched typesetter, quality
  judge (OCR match rate is a free objective signal; aesthetics reuse the
  judge interface behind a new rubric). Doubles as an **animation
  preprocessor**: baked-in bubble text warps across chained frames, so a
  textless plate + static retypeset overlay at assembly is the correct
  animation input even for users who never translate. Feasibility and
  training data are proven in the wild (manga-image-translator et al.).

Deliberately **not** fleet jobs: narration script writing (open-ended
prose stays frontier — no honest reward signal, no small-model parity),
music/SFX direction (no such product surface today), taste/personalization
models (vague reward, no data), camera-path planning (hold-and-glide
heuristics suffice until sweep evidence says otherwise).

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
- Track B (cloud execution) adds its own non-goals — see that section.

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
6. Diversity over monolith: the coordinator is a fleet of small
   specialists behind stable interfaces (judge verdicts, instruction bank,
   model choice), each distilled from accumulated trajectories — never one
   giant model doing every job, never a model trained cold.

## Track B — Cloud execution (added 2026-10-15)

Scope amendment: the initiative absorbs **cloud deployment of the harness's
job execution**. Motivation, in this initiative's terms: directed sweeps and
frame-animation runs are the heavy, bursty jobs (≈400–1000 image calls per
grounded chapter); executing them on Cloudflare Containers scales trajectory
collection and directed runs off the Mac, while the local machine only
**submits** jobs ("run locally to deploy jobs"). Track A (Phases 1–4) is
unchanged; Track B phases are labelled C0–C4 and gate independently.

> **2026-10-17:** the wrangler project moved from `cloud/` in this repo to
> the sibling infra repo (`../infra/entertainment-harness/`); the container
> image build context is still this repo's root. Paths below are updated;
> verbatim gate evidence is left as originally pasted.

Target architecture:

```
local: eh cloud submit / desktop app (eh serve remote mode proxies run endpoints)
        |  HTTPS (bearer EH_CLOUD_TOKEN)
        v
infra/entertainment-harness/ Worker (TS) -- D1: harness schema + jobs ledger (metadata + current cursor)
        | enqueue    -- R2: works/... (sources, pages, recaps, videos),
        v                     jobs/<id>/events.jsonl
Queue pipeline-jobs --> Container (python:3.13-slim + ffmpeg + espeak-ng + kokoro)
                        stage-in (R2 works dir + D1 snapshot) -> run -> stage-out
```

Hard constraints (verified 2026-10-15):

- Workers can't run this codebase (Python + ffmpeg subprocess + onnxruntime
  + OpenCV); **Containers** can — CPU-only, no persistent disk, Workers Paid
  + per-second usage.
- D1 bindings are Workers-only → the container reaches D1 via REST.
  Per-statement REST from a minutes-long sync pipeline is slow and burns the
  rows-written budget → the container stages a **local SQLite snapshot per
  job** and flushes batched row diffs at chapter boundaries (the repo's own
  "SQLite is a derived cache" philosophy).
- No GPU → model roles default to `openai_compat` (OpenRouter; adapter
  exists at `models/openai_compat.py`). The default narrator is the
  hosted Fish Audio API (`drama-3`; the Worker passes
  FISH_AUDIO_API_KEY through when set — until the secret exists, the
  cloud profile falls back to in-container Kokoro) — no in-container
  synthesis;
  Kokoro ONNX TTS also runs CPU in-container as the local engine;
  mlx-audio/qwen3 voice cloning stays local-only.

Reused seams (no pipeline changes): `db.py connect()` (single entry point,
`db_path` param), `store/r2.py` id-keyed works layout, `server/runs.py`
`RunOptions` + `_run_flow()` + `ServerProgress`, Typer sub-app pattern for
`eh cloud`, `test_store_r2.py` fake-client fixtures (respx for D1 REST).

Track B non-goals: no D1 read replicas; no multi-user auth (single bearer
token); no mlx-audio/qwen3 or SAM/torch extras in the container; no UI
redesign (desktop app unchanged); sources/search stay HTTP from wherever
the job runs.

### Phase C0 — Spike: containerized headless run (local Docker)

- [x] `../infra/entertainment-harness/Dockerfile`: python:3.13-slim +
  ffmpeg + espeak-ng; package pip-installed from source, runtime deps only
  (no sam/qwen3/dataset extras, no torch). Root `.dockerignore` added; build
  context is this repo's root (`docker build -f
  ../infra/entertainment-harness/Dockerfile -t eh-job .`).
- [x] `src/entertainment_harness/jobs.py` + `cli/jobs.py`: `eh job-run
  --spec <json>` headless entrypoint reusing `server.runs._run_flow`;
  events stream as JSONL on stdout and to EH_JOB_EVENTS (default
  <data_dir>/jobs/<id>/events.jsonl). Exit 0 done/cancelled, 1 error,
  2 bad spec.
- [x] `src/entertainment_harness/store/staging.py`: `pull_work`/`push_work`
  whole-work staging over R2 (id-keyed layout, chapter slug dirs rewritten
  to chapter ids; clips/, seg-*.wav, OS noise excluded). (Named staging.py:
  store/__init__.py's `works` import would shadow a store/works.py.)
- [x] Env-driven cloud profile: EH_CLOUD=1 → `config.apply_cloud_profile`
  forces openai_compat vision/text roles (EH_CLOUD_VISION_MODEL /
  EH_CLOUD_TEXT_MODEL overrides; defaults qwen/qwen3-vl-235b-a22b-instruct,
  qwen/qwen3-32b), R2 store, preflight off. R2 bucket/account fall back to
  R2_BUCKET / R2_ACCOUNT_ID env vars (store/r2.py), keys already did.

Gate (MET 2026-10-16 — evidence below): `docker run … eh job-run` recapped +
rendered Paper Moons ch1 (10 synthetic pages) with OpenRouter + Kokoro in
296s wall; mp4 (76s, 6.56 MB) and recap.json verified in the R2 bucket.
Findings: Kokoro CPU is fine (12 segs ≈ 60s incl. first-use weight
download); ~5 min/chapter end-to-end → per-chapter chunking stays deferred
to C4; wrangler containers config/local-dev questions remain for C2 (this
spike was Docker-only).

### Phase C1 — DB seam + D1 snapshot sync

- [x] `config.py`: `[cloud]` dataclass section (enabled, worker_url,
  account_id, database_id, max_job_seconds, token). The submit token comes
  from `[cloud] token` (Settings → API keys) with `EH_CLOUD_TOKEN` env as
  fallback; `eh serve` re-reads it per request so settings changes apply
  live.
- [x] `src/entertainment_harness/store/d1.py`: D1 REST client (batched
  `/query`); `snapshot_pull` (build local sqlite from rows) and
  `snapshot_flush` (per-table row diff vs. staged baseline → batched
  INSERT OR REPLACE / DELETE).
- [x] `../infra/entertainment-harness/migrations/0001_init.sql`: db.py
  schema + ALTER history + `jobs` ledger table; applied via `wrangler d1
  execute --remote --file` (a single hand-ordered file rather than a
  wrangler migrations dir).
- [x] Flush hooks at chapter boundaries via `ServerProgress` callbacks
  (`_FlushOnChapterDone` in jobs.py; pipelines commit per chapter, so a
  second read connection sees boundary state). When the snapshot staged
  rows, `index_works()` is skipped — it clears all tables and would turn
  remote rows into DELETEs.
- [x] Tests: `test_store_d1.py` (respx client tests + offline diff
  round-trip incl. deletes and param-budget chunking), D1 wiring tests in
  `test_jobs.py`, `[cloud]` parse tests in `test_config.py`.

Gate (MET 2026-10-15 — evidence below): tests green; full suite at
baseline (1 known pre-existing test_lfm failure); live snapshot
round-trip against the real D1 pasted (insert → upsert → no-op →
delete → pristine).

### Phase C2 — Cloudflare control plane (`../infra/entertainment-harness/` wrangler project)

- [x] `../infra/entertainment-harness/wrangler.jsonc` +
  `../infra/entertainment-harness/src/worker.ts`: D1 + R2 bindings,
  Queue `pipeline-jobs`, containers binding; secrets EH_CLOUD_TOKEN,
  OPENROUTER_API_KEY, R2 keys (passed to container env at start).
- [x] Endpoints: POST /api/jobs (RunOptions-shaped; one active job per
  work), GET /api/jobs{,/{id}} (ledger + cursor; events from R2 JSONL),
  POST /api/jobs/{id}/stop (flag; container polls at chapter boundaries —
  mirrors cooperative `Run.request_stop()`).
- [x] Queue consumer starts a container per message; v1 = one message per
  run (per-chapter chunking deferred to C4 if duration limits bite).
- [x] Container runner cloud sink: buffered PUT of `jobs/<id>/events.jsonl`
  + ledger PATCH at each chapter boundary.

Gate (MET 2026-10-17 — evidence below): deployed; POST /api/jobs completes
one chapter; D1 ledger row + R2 events.jsonl + mp4 verified; wrangler tail
excerpt pasted.

### Phase C3 — Local submit path

- [x] `cli/cloud.py`: `eh cloud submit|status|stop|logs` (sub-typer beside
  `models`/`voices`).
- [x] `eh serve` remote mode: `[cloud] enabled` → run-control endpoints
  (`/api/runs*`, auto-toggle supervisor) proxy to the Worker instead of
  RunManager; library/stream unchanged; desktop app untouched.
- [x] Tests: proxy endpoints + CLI with mocked Worker.

Gate (MET 2026-10-17 — evidence below): desktop app auto toggle → job
executes on Cloudflare → video in R2 → `eh play` pulls and plays locally.
Evidence pasted.

### Phase C4 — Parity + hardening

- [ ] Scanlation jobs (`pipelines/translate.py`) and exports
  (`server/exports.py`) as job kinds on the same queue/ledger.
- [ ] Retries with backoff, per-chapter chunking (if needed),
  `max_job_seconds` cost cap, per-account concurrency limit.
- [x] `docs/design.md` updated (cloud execution, generalized remote store,
  `[cloud]` config) in the same change.

Gate: one scanlation job + one recap run green remotely; suite at baseline.

### Track B evidence

#### Frame fallback + snapshot-wipe fixes (2026-10-19) — cloud run unblocked

Job `8f6b05faaec1` (kenja-no-mago ch-0, full + frame-animation) recapped
successfully, then died at assembly: `Runway provider requires an API key`
(the container carries `OPENROUTER_API_KEY`, no Runway key) — and its only
D1 flush failed with `FOREIGN KEY constraint failed (sql: 'DELETE FROM
series WHERE id = ?')`. Two root causes, three fixes:

- **Frame-animation fallback** (`config.apply_cloud_profile`): a default
  `frame-animation` provider with no resolvable Runway key now falls back
  to `openrouter-frame-animation` (the container's OpenAI-compatible
  endpoint), mirroring the fish→Kokoro narration fallback; an explicit
  `[frame_animation] provider` is respected. Deployed in worker `920c8bf7`.
- **Container auto-migrate wiped the snapshot** — the FK failure's root.
  Stage-in writes id-keyed dirs by design; `db.connect()`'s auto-migration
  saw them, renamed them to slugs, and `index_works()` then cleared every
  table — including the demo work's rows restored from the D1 snapshot.
  The flush diff therefore ended in `DELETE FROM series` for the demo,
  which D1 rejected (its children were still present), aborting every
  insert queued behind it (kenja's chapters never landed). Fixed by baking
  `EH_NO_AUTO_MIGRATE=1` into the job-runner image
  (`infra/entertainment-harness/Dockerfile`) — a fresh snapshot plus
  id-keyed stage-in has no legacy layout to migrate.
- **Flush op ordering** (`store/d1.py diff_statements`): upserts first in
  schema order (parents before children), DELETEs last in reverse schema
  order (children before parents) — a rejected delete can no longer abort
  the inserts behind it. And `jobs.py` now **always** re-indexes the staged
  work additively (`index_work` upserts) instead of skipping when the
  series row is present, so a partial snapshot (series row but no
  chapters — exactly kenja's state after the aborted flush) self-heals.

Tests — 2 new (FK-ordered diff replayed against an FK-enforcing SQLite
copy; partial-snapshot reindex through run_job), 1 rewritten
(`keeps_known_work` → `reindexes_known_work`):

    $ uv run pytest tests/test_jobs.py tests/test_store_d1.py -q
    50 passed

    $ uv run pytest -q
    1 failed, 1188 passed   # known pre-existing test_lfm trailing-comma failure

Redeployed: worker version 7c01a73f (new container image). End-to-end
verification, two jobs:

- `8fcccb86291e` (full spec: ch-0, detail full, frame-animation) —
  **video built and staged out** (`video-narration/out.mp4`, 107 s), with
  the narration reused from R2 ("Chapter 0 narration already stored; video
  only"). Two caveats: it ran the pre-fix image (submitted before the new
  image propagated — flush FK-failed as before), and every frame-generation
  call got OpenRouter `402` (account balance < $1) so anchors degraded to
  the panels fallback — the designed per-anchor path, billing-blocked, not
  a code bug.
- `4387f2546bf5` (same scope, skip_done — nothing to rebuild) on the fixed
  image: `Indexed staged work` fired unconditionally, 2 store-held videos
  registered, ch-0 correctly skipped as done, and the flush landed clean —
  `chapters +100, progress +1, recaps +2, series_context +1, videos +2,
  deleted 0`. D1 now holds kenja-no-mago's full index (verified: 100
  chapters). Frame-animation itself remains to be exercised once the
  OpenRouter balance is topped up; a forced re-run (skip_done off)
  regenerates ch-0's video with frames.

#### Scoped stage-in fix (2026-10-18) — big works fit the container disk

Follow-up failure to the stage-up fix: `Could not pull from r2 bucket
'entertainment-harness': [Errno 28] No space left on device` — the
container's whole-work stage-in pulled all 11 GB of kenja-no-mago (100
chapters: 1.8 GB sources, 742 MB chapter videos, plus an 8.5 GB
hand-assembled `recap-full.mp4` at the work root that should never have
crossed the store at all). Fixes:

- **Two-phase stage-in** (`jobs.py run_job` + `staging.pull_work_matching`):
  metadata first (`work.json`, the cover, every JSON — the indexable
  state), index, then just the `source/`/`translated/` pages of the
  chapters the run will touch. The scope is computed with `_run_flow`'s
  own selection logic, extracted into `server/runs.py`
  `config_for_run`/`select_run_chapters` so the two can't drift; the
  video backfill set (`_chapters_missing_video`) is included when the job
  has video on. mp4s never cross at stage-in.
- **Store-held videos registered without the files**
  (`jobs._register_store_videos`): the skip checks (`chapter_needs_work`,
  `_chapters_missing_video`) are pure DB queries, so the runner
  synthesizes `videos` rows from the R2 listing (metadata from the pulled
  `video.json`). Without this, a locally-rendered work's cloud re-run
  would try to rebuild every chapter's video (the backfill is scope-less).
- **Work-root `*.mp4` excluded from staging** (`staging._excluded`): not
  a works-layout artifact. The 8.5 GB object uploaded by the initial
  whole-work stage-up was deleted from the bucket.

Tests — 5 new (root-mp4 exclusion, metadata/scope key filters, selective
pull, store-video synthesis + idempotency, scoped stage-in through
run_job incl. the backfill set); `test_run_job_stages_around_the_flow`
updated for the `pull_work_matching` seam:

    $ uv run pytest tests/test_store_works.py tests/test_jobs.py \
        tests/test_server.py tests/test_server_remote.py tests/test_cli_cloud.py -q
    148 passed

    $ uv run pytest -q
    1 failed, 1184 passed   # known pre-existing test_lfm trailing-comma failure

    $ cd ui && bun run typecheck && bun run smoke
    # tsc clean; SMOKE PASS — 100 checks against a live backend

Redeployed: worker version c9532439 (new container image). Retry path:
flip the work's toggle (or `eh cloud submit 01J76XYB… --all --video`) —
stage-up is a no-op (everything already in R2 from the earlier push), and
the container now stages in ~100 MB for a 4-chapter scope instead of
11 GB.

#### Stage-up fix (2026-10-18) — local-only works run in the cloud

User report: `No series matches '01J76XYBTCD15FW69W889WKHB0'` from cloud
job 3bab846dd52c. Root cause: the work existed only locally — R2 held
just the demo work and the D1 snapshot only its rows, so the container's
`resolve_series` failed before the pipeline started. Three gaps closed:

- **Stage-up on submit**: both submit paths push the work's files to R2
  before queueing (`staging.stage_up` — forces the r2 backend regardless
  of the local provider, since containers always stage from R2;
  incremental, same-size keys skipped via the new `r2.list_sizes`).
  `eh serve` remote mode skips with a stderr note when the work has no
  local dir or R2 is unconfigured (the cloud may already hold it from an
  earlier job); a failed push aborts the submit with 502.
  `eh cloud submit` behaves the same (yellow note / exit 1) and also
  resolves the work against the local index first — a typo exits 1
  before any HTTP.
- **Additive index in the container** (`jobs.py run_job`): when the D1
  snapshot is populated but lacks the submitted work and the staged-in
  work dir exists, `index_work` indexes just that work into the snapshot
  (`index_works` stays skipped — it clears every table, and the flush
  would turn remote rows into DELETEs). The new rows reach D1 via the
  job's normal boundary flush.
- **R2 creds in the user config** (`[store.r2]` in config.toml), so
  app-launched `eh serve` (no sourced env) can stage up; the R2_* env
  vars remain the fallback. `store.provider` unchanged.

Tests — 12 new across the four touched files (incremental skip,
forced-r2 stage-up, unconfigured → StoreError, stage-up wire/skip/abort
on both submit paths, submit-side resolve fail-fast, additive vs skipped
index):

    $ uv run pytest tests/test_store_works.py tests/test_server_remote.py \
        tests/test_cli_cloud.py tests/test_jobs.py -q
    74 passed

    $ uv run pytest -q
    1 failed, 1178 passed
    # the single failure is the known pre-existing
    # tests/test_lfm.py::test_repair_json_repairs_trailing_comma

    $ cd ui && bun run typecheck && bun run smoke
    # tsc clean; SMOKE PASS — 100 checks against a live backend

Redeployed (container image sha256:87385ed4…, worker version 1e2f5fa5).
Live stage-up of the failing work against the real bucket (11 GB, first
push — incremental resume picked up a killed first attempt, and the
immediate re-push was a true no-op):

    Pushed 4055 file(s) for 01J76XYB… (1088 unchanged skipped)
    re-push: Pushed 0 file(s) (5143 unchanged skipped)

Known limitation: submit-time stage-up is synchronous inside
POST /api/runs, so a first-time submit of a huge work blocks the request
for the upload's duration (pre-paid for this work by the push above; a
background/queued stage-up is a C4 candidate if it ever bites).

#### Phase C3 gate (2026-10-17) — MET

Isolated data dir (`EH_DATA_DIR=/tmp/eh-c3`, `[cloud] enabled` +
`worker_url`, R2 bucket via env, token via `EH_CLOUD_TOKEN`) so the real
library stayed untouched; the work's local video state was wiped so the
chapter read as unfinished. The desktop app is a pure HTTP client of
`eh serve`, so the gate drives the app's exact endpoints.

`eh serve` came up in remote mode — `/api/runs` proxies the worker ledger
(title resolved from the local index, options normalized to the full
RunOptions shape):

    $ curl 127.0.0.1:8471/api/runs
    {"runs": [{"id": "1a790e6f9648", "title": "Paper Moons", "status": "done",
               "options": {"work": ..., "chapter": 1, "video": true, ...}}]}

Auto toggle (the desktop app's run control) submitted a cloud job:

    $ curl -X PUT -d '{"enabled": true, "detail": "standard", ...}' \
        127.0.0.1:8471/api/works/5865f5fd…/auto
    {"work": "5865f5fd…", "enabled": true,
     "run": {"id": "e914d0de4d2a", "status": "queued", "options": {…, "video": true}}}

Job `e914d0de4d2a` ran on a Cloudflare container: queued 17:55:52Z → done
(~4 min — the recap was cached so only the video re-rendered). Verified:
R2 mp4 mtime 17:56:10; the SSE bridge (`GET /api/runs/{id}/events`)
replayed the job's 12 events ending in `run-done`; `eh cloud
status`/`logs` read the live ledger; the auto toggle one-shot disarmed
after the terminal job (`GET /api/library` → `"auto": false`).

The pull half of the gate first failed and exposed a real bug:
`store.pull_video` restores into the id-named remote layout, but
`_reconcile_landing` no-oped whenever the id-named dir existed (which the
download itself creates), so on the normal slug-named local layout
`eh play` checked the pre-pull slug path and reported "Video file is
missing" after a successful restore — and the next connect() would have
renamed the duplicate tree to a `…-2` slug. Fixed: the reconcile now
moves the pulled chapter/tiktok content into the canonical metadata-
matching dir and prunes the landing tree (regression test
`test_pull_video_reconciles_into_slug_named_work`). After the fix, from a
fresh stage-in + D1 index restore:

    $ eh play 5865f5fd… --chapter 1
    Playing chapter 1 (76s):
    /tmp/eh-c3/works/paper-moons/chapters/ch-001/video-narration/out.mp4

Findings: (1) the round trip needs an explicit "pull the cloud index back
to local" step — `d1.snapshot_pull` does it (4 rows restored), but there
is no user-facing command yet (C4 candidate: `eh cloud sync`). (2) Until
then, on a machine that never indexed the work locally, `eh play` has no
`videos` row to key the pull off — the C3 flow assumes the work is known
locally, which holds for the desktop app (library view) but not for a
bare second machine.

#### Phase C2 gate (2026-10-17) — MET

Worker deployed (`entertainment-harness.personal-6e3.workers.dev`, version
5085bc7a; container application on the freshly built image). Secrets via
`wrangler secret put` (EH_CLOUD_TOKEN, OPENROUTER_API_KEY, R2 S3 keypair,
D1_API_TOKEN); migrations 0001+0002 already applied to the remote D1.

    $ curl -X POST -H "Authorization: Bearer $EH_CLOUD_TOKEN" \
        -d '{"work": "5865f5fd…", "kind": "recap",
             "options": {"chapter": 1, "video": true}}' \
        https://entertainment-harness.personal-6e3.workers.dev/api/jobs
    {"job": {"id": "1a790e6f9648", "status": "queued", ...}}

Ledger lifecycle: queued (17:11:32Z) → running (17:11:36Z, queue consumer
started the container) → done (17:23:59Z), `error: null`.

Event stream (66 events, `GET /api/jobs/1a790e6f9648/events`), abridged —
note the boundary flush landing this time (see the pre-fix failure below):

    17:11:41  D1 snapshot staged (4 row(s)).
    17:11:48  Staged in 22 file(s) … / Index restored from D1 snapshot.
    17:11:50  run-start
    17:13:47  Stage 1/4 script: panel-first grouping (qwen/qwen3-32b) …
    17:17:23  Stage 3/4 visuals / 4/4 assembly …
    17:23:52  chapter-done
    17:23:53  D1 flush: {'recaps': {'upserted': 1, 'deleted': 0},
                         'videos': {'upserted': 1, 'deleted': 0}}
    17:23:53  run-done
    17:23:59  Staged out 22 file(s)

Artifacts verified after the run: `jobs/1a790e6f9648/events.jsonl` (13,024
B) in R2; chapter mp4 `works/5865f5fd…/…-0001/video-narration/out.mp4`
(5,124,057 B, mtime 17:23:58); D1 rows series/chapters/recaps/videos = 1
each + 4 ledger rows.

wrangler tail excerpt (live request log):

    GET https://entertainment-harness.personal-6e3.workers.dev/api/jobs/1a790e6f9648 - Ok @ 10/17/2026, 10:28:43 AM
    GET https://entertainment-harness.personal-6e3.workers.dev/api/jobs - Ok @ 10/17/2026, 10:28:44 AM

Findings: the first two cloud jobs (4302da4b4c18, d0e001d44629, earlier on
2026-10-17) completed the full pipeline but every D1 flush died with
"SELECTs to the left and right of EXCEPT do not have the same number of
result columns" — the cloud container was still running a pre-fix image
from before the `_schema_conn` migration fix (regression test
`test_diff_live_db_against_staged_baseline` covers it). After rebuilding
the image + redeploying, the flush landed at the chapter boundary and at
terminal. Two hardening bits shipped in the same change: `diff_statements`
now raises a drift error naming the table and both column lists instead of
the raw EXCEPT message, and `snapshot_pull` deletes stale snapshot files
before staging (insurance against a persisted volume carrying an older
schema). Also observed: the very first queue message once failed with
"there is no container instance that can be provided to this durable
object" (7ecc6c158f32, ledger-visible error, acked) — transient while the
container app settled; subsequent starts are clean.

#### C0 code-complete (2026-10-15) — live Docker gate still pending

New tests (staging + jobs), then the full suite:

    $ uv run pytest tests/test_jobs.py tests/test_store_works.py -q
    23 passed in 0.06s

    $ uv run pytest -q
    1 failed, 1093 passed in 23.66s
    # the single failure is the known pre-existing
    # tests/test_lfm.py::test_repair_json_repairs_trailing_comma
    # (Python 3.13 json strictness; CI deselects it)

Headless runner smoke (isolated EH_DATA_DIR, unknown work — exercises
stage-in skip, index rebuild, JSONL event stream, error exit code):

    $ EH_DATA_DIR=$(mktemp -d) uv run eh job-run --spec \
        '{"work": "no-such-work", "id": "smoke1", "options": {"chapter": 1}}'
    {"event": "log", "run_id": "smoke1", ..., "message": "Store not configured; running against local data only."}
    {"event": "log", "run_id": "smoke1", ..., "message": "Indexed 0 work(s)."}
    {"event": "run-error", "run_id": "smoke1", ..., "message": "No series matches 'no-such-work'"}
    exit=1

Blocked for the live gate: Docker daemon not running locally (CLI present);
the local config.toml has no [store.r2] bucket/keys (two api_key entries
exist); OpenRouter key + an R2 bucket with credentials are needed in env
for the container run.

Update (2026-10-15): daemon started, image built, in-container smoke green.
Still blocked on Cloudflare credentials only — no CF token exists anywhere
on the machine (no .env in repo/home/Downloads/sibling projects; the two
config.toml api_keys probe as OpenRouter + Runway, both rejected by
Cloudflare's token-verify; the saved wrangler OAuth login is expired).
OpenRouter is confirmed in config.toml [models.openai_compat] and can be
mounted into the container read-only.

    $ docker build -f cloud/Dockerfile -t eh-job .
    ... DONE  # eh-job:latest, 1.41 GB

    $ docker run --rm eh-job job-run --spec \
        '{"work": "no-such-work", "id": "csmoke", "options": {"chapter": 1}}'
    {"event": "log", "run_id": "csmoke", ..., "message": "Store not configured; running against local data only."}
    {"event": "log", "run_id": "csmoke", ..., "message": "Indexed 0 work(s)."}
    {"event": "run-error", "run_id": "csmoke", ..., "message": "No series matches 'no-such-work'"}
    exit=1

    # demo library seeded for the live run:
    # Paper Moons (5865f5fdbf7f40b8b4a176183f2c1ad0, comic, 1 ch, 10 synthetic pages)
    # at /tmp/eh-demo-data

Cloudflare account resources (2026-10-16; token at
~/.config/terragohan/cloudflare_token — Workers/Queues/D1/R2-bucket OK,
account-token minting 403):

    account:   Personal  6e3ad204dd01e40310a1d2c2791e2d9c
    D1:        entertainment-harness  f78cb32b-1041-4eaa-ba50-62452a0f9046  (created)
    R2 bucket: entertainment-harness  (created; account had to enable R2 in
               the dashboard first — API error 10042 otherwise)
    R2 S3 creds: PENDING — permanent R2 tokens are dashboard-only and the
               API token lacks "API Tokens" scope; ids/env scaffolding at
               ~/.config/terragohan/r2_s3_credentials (add the two keys)

#### Phase C0 gate (2026-10-16) — MET

User added the R2 S3 keypair to ~/.config/terragohan/cloudflare_token
(CLOUDFLARE_R2_S3_ACCESS_KEY_ID / CLOUDFLARE_R2_S3_SECRET_ACCESS_KEY);
r2_s3_credentials maps them to the canonical names. Object-level auth
verified (put/list/delete round-trip). Demo work pushed (12 objects), then
the live container run:

    $ docker run --rm -e R2_BUCKET -e R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID \
        -e R2_SECRET_ACCESS_KEY -e OPENROUTER_API_KEY eh-job job-run \
        --spec '{"work": "5865f5fdbf7f40b8b4a176183f2c1ad0", "id": "spike1", "options": {"chapter": 1, "video": true}}'
    ...74 JSONL events... exit 0

Timing from the event stream (296s wall, apple-silicon Docker):

    stage-in + index          2s    (12 files pulled, 1 work indexed)
    recap (vision, 10 pages)  44s   qwen/qwen3-vl-235b-a22b-instruct
    script (panel-first)      163s  CV panel extract ~50s + grouping ~112s (qwen3-32b)
    narration (kokoro CPU)    60s   12 segs / ~153 words, incl. first-use download
    visuals + assembly        17s   12 clips, ffmpeg
    per-chapter + stage-out   9s    22 objects total in bucket

Artifacts verified in R2 (22 objects), incl.:

    6.56 MB  works/5865f5fd…/chapters/5865f5fd…-0001/video-narration/out.mp4  (76s video)
             works/5865f5fd…/chapters/5865f5fd…-0001/recap.json
             works/5865f5fd…/chapters/5865f5fd…-0001/panels.json
             …/video-narration/{script.json, subs.srt, render_state.json, video.json}

Gotchas found: Docker Desktop file-sharing denies ~/.local mounts (pass the
OpenRouter key as -e OPENROUTER_API_KEY instead — must be *exported*; an
unexported shell var silently no-ops docker's -e passthrough); docker
daemon had quit once between build and run.

#### Phase C1 gate (2026-10-15) — MET

New tests (D1 client + diff round-trip + jobs wiring + [cloud] config),
then the full suite:

    $ uv run pytest tests/test_store_d1.py tests/test_jobs.py tests/test_config.py -q
    52 passed in 0.23s

    $ uv run pytest -q
    1 failed, 1108 passed in 24.76s
    # the single failure is the known pre-existing
    # tests/test_lfm.py::test_repair_json_repairs_trailing_comma
    # (Python 3.13 json strictness; CI deselects it)

Live snapshot round-trip against the real D1 (migration 0001_init.sql
already applied; empty DB → 0 rows staged; test row inserted, upserted,
no-op flushed, deleted, verified after each step):

    $ source ~/.config/terragohan/r2_s3_credentials && uv run python /tmp/eh_d1_roundtrip.py
    D1 configured: account=6e3ad2... db=f78cb32b...
    1. snapshot_pull: 0 rows staged (expect 0)
    2. flush after insert: {'series': {'upserted': 1, 'deleted': 0}}
       remote: [{'id': 'd1test-roundtrip', 'title': 'D1 Roundtrip Test'}]
    3. flush after update: {'series': {'upserted': 1, 'deleted': 0}}
       remote: [{'title': 'D1 Roundtrip UPDATED'}]
    4. no-op flush: {} (expect empty)
    5. flush after delete: {'series': {'upserted': 0, 'deleted': 1}}
       remote: [] (expect empty)
    LIVE D1 ROUND-TRIP: PASS

    # post-check: SELECT COUNT(*) FROM series WHERE id LIKE 'd1test-%' → 0;
    # series total 0 — D1 left pristine.

Design note: `snapshot_flush` advances the baseline through SQL
(DELETE+INSERT via ATTACH), not file copy, because the pipeline's own
connection may still be open on the local DB; on any D1 error the
baseline is untouched and the next flush retries the full delta.

## Phase 1 — eval director loop

Scope note (amended 2026-10-11): the director wraps the **frame-animation**
chain loop (`sequence_frames_with`) only — video-animation mode is out of
scope (the `director` knob is rejected on video-animation arms). Shipped
alongside in the same plan: a per-run `fps` override along the existing
`video_mode` path (pipeline `build_video(fps=…)` on a deep-copied config,
`eh recap --fps`, `fps` fields on `POST /api/runs` + the auto toggle, and
an fps input in the UI run panel, enabled only for the frame-animation
style).

- [x] `evals/imagegen/director.py`: instruction bank (template + model
  tags + provenance note), revision logic (drift/stagnant verdict →
  revision template), best-chain selection by judge aggregates.
- [x] Runner knob `director = true` on an arm: the arm's panel loop runs
  the director loop instead of the plain loop; trajectories recorded in
  the manifest per panel (instruction id per frame, verdicts, revisions,
  final pick).
- [x] Seed the bank from live findings: `default` (EXPAND_PROMPT /
  CHAIN_PROMPT verbatim from `video/sequence.py`), `bfl-safe` (chain
  phrasing verified pass on flux-2-pro/flux-3 2026-10-11; the expand
  variant was never verified and stays default), `motion-push`,
  `preserve-composition`.
- [x] Tests: scripted judge + fake image client — drift verdict triggers a
  preserve-composition revision; stagnant triggers motion-push; bounded
  retries (≤3 attempts/frame, identical revision prompts dedup via
  content addressing); truncation on persistent drift; tie → earliest
  pick; manifest trajectory shape; judge-unavailable → arm error; loader
  validation (animate-mode/conflict/non-bool rejection); cached rerun
  reproduces the trajectory.

Gate: focused tests green; manifest contains trajectories. **Met** —
evidence below.

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

(The fleet inventory above names these as jobs; the levels are the
sequencing.)

- **Level 2 router** (fleet job 3: bandit over models given panel
  features) — needs the Phase 4 table to have real volume.
- **Level 3 fine-tuned judge** (fleet job 1: LoRA a local VLM on
  accumulated trajectories + human spot-checks) — **sibling repo** when it
  happens: training toolchain, datasets, and checkpoints don't belong in
  the runtime harness. Contract: artifact + scoring API behind the
  existing judge interface.
- Learned instruction generation (fleet job 2: fine-tuned director LLM on
  trajectories) — only if Level 1 plateaus.
- Revision triage + motion tagger (fleet jobs 4–5) — opportunistic; their
  labels fall out of the trajectories Phase 1 already records, so they
  train cheap whenever someone picks them up.
- Human-labeling tooling to audit judge bias before Level 3 — an Astra
  re-judging pass over an existing sweep's manifest is the cheap first
  cut (pseudo-ground-truth labels double as job-1 training data).

## Evidence

### Phase 1 gate (2026-10-11)

Focused director tests + full suite + UI gates, all run with an isolated
`EH_DATA_DIR`:

    $ uv run pytest tests/test_evals_imagegen.py -q
    69 passed in 4.93s                     # incl. 11 new director tests

    $ uv run pytest -q
    1 failed, 1070 passed in 23.73s
    # the single failure is the known pre-existing
    # tests/test_lfm.py::test_repair_json_repairs_trailing_comma
    # (Python 3.13 json strictness; CI deselects it)

    $ cd ui && bun run typecheck && bun run smoke
    $ tsc --noEmit                         # clean
    SMOKE PASS — 80 checks against a live backend

Manifest trajectory excerpt (fake client + scripted judge: frame 1 drifts
once, the preserve-composition revision is picked; metrics carry
`revisions`/`judge_calls`):

    "metrics": { "requested": 2, "produced": 2, "generated": 3,
                 "revisions": 1, "judge_calls": 4, ... },
    "trajectory": [
      { "frame": 0,
        "attempts": [ { "instruction": "default",
                        "verdicts": { "faithfulness": { "verdict": "ok" } } } ],
        "picked": 0 },
      { "frame": 1,
        "attempts": [
          { "instruction": "default",
            "verdicts": { "drift": { "verdict": "drift",
                                     "note": "background moved" } } },
          { "instruction": "preserve-composition",
            "verdicts": { "drift": { "verdict": "ok" },
                          "motion": { "verdict": "advancing" } } } ],
        "picked": 1 }
    ]
