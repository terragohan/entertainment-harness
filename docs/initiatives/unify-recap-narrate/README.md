# Unify recap & narrate (detail levels)

Status: **done**

Amendment (2026-09-27): the `eh narrate` deprecated alias was removed for the
open-source release — `eh recap --detail full` is the only path
(`pipelines/narrate.py` deleted; `preflight.plan_for_recap_or_narrate`
renamed `plan_for_recap`). The `narrations` table, `video-narration/` workdir,
and the `--narration` flags on `eh show`/`eh play` remain as decided here.

## Goal

One chapter-text pipeline with a detail knob: `eh recap --detail
gist|brief|standard|detailed|full`. At `full` detail the artifact *is* a
narration (full in-order retelling) — the separate narrate pipeline, table,
and progress semantics fold into recaps. One video per chapter, rendered from
that chapter's artifact at whatever detail it has.

User decisions (2026-09-15): one video per chapter (best detail wins);
narrations fold into recaps as `detail = 'full'`; the 100-chapter scroll
re-render waits until this merge lands.

## Non-goals

- Changing recap/narration *prompts*, judge axes, or thinking levels — the
  text generation itself is proven; this is a structural merge.
- Dropping the `narrations` table or renaming video workdirs (migrations are
  additive-only; `video-recap`/`video-narration` dirs stay as the two kind
  containers).
- Touching the video-mode scroll work, store layout, or the tiktok/short
  pipelines.

## Design

- **Detail levels (5 grains)**: `gist | brief | standard | detailed | full`
  (config `[pipeline] detail`, CLI `--detail`; default `standard`).
  - `gist` — 2–4 sentence "what happened" at a glance.
  - `brief` — short summary, one tight paragraph.
  - `standard` — today's recap (default; existing recaps map here).
  - `detailed` — expanded summary covering every notable beat, still
    compressed prose (between standard and full).
  - `full` — today's narration: complete in-order retelling, ~1000+ words
    (existing narrations map here).
  Prompt selection by detail (gist/brief/detailed are new prompt variants of
  the recap prompt; full keeps the narration prompt). Judge: `full` keeps the
  narration judge with the completeness axis; the other four grains use the
  recap judge. Thinking levels behave identically across grains.
- **Storage**: `recaps` gains `detail TEXT NOT NULL DEFAULT 'standard'`
  (additive ALTER). One-time migration folds `narrations` rows in: a chapter
  with only a narration gets a recaps row (detail='full', summary=text,
  model, created_at); a chapter with both keeps the **narration** text as
  the artifact (detail='full' — full supersedes; the standard summary is
  discarded, rolling context is unaffected since it already incorporated the
  chapter). `narrations` table left behind, unused. `eh narrate` becomes a
  deprecated alias for `eh recap --detail full`.
- **Selection/progress**: grains are ranked gist < brief < standard < detailed
  < full. A chapter whose artifact is already at the requested grain is
  skipped; a chapter with a *higher*-grain artifact is never downgraded by
  default. Pending per run has two buckets (mirrors today's two pipelines):
  (a) chapters after `last_read` with **no** artifact at any grain — these
  advance `progress` and incorporate into `series_context` as recaps do
  today; (b) chapters with a lower-grain artifact — upgrade candidates,
  **ungated** on `last_read` (today's narrate semantics), stored at the
  requested grain without touching progress, and context is re-incorporated
  only for chapters not already covered by `series_context.through_chapter`.
  `--all`/`--chapter` force re-runs at the requested grain (overwriting),
  leaving context/progress untouched, as today.
- **Videos**: one per chapter. `videos.kind` and the two workdirs keep their
  meanings but are now derived from the artifact's detail (`full` →
  narration kind/workdir, else recap). Building a video after a detail
  upgrade supersedes the other kind's video (files pruned, row replaced).
  `render_state.json` gains a `detail` key so a detail change re-renders
  (same pattern as mode/colorize). Script stage by detail: `full` → split
  the retelling verbatim (no model call, current narration behavior); below
  → text model writes beats (current recap behavior). `eh play` opens the
  chapter's one video regardless of kind; `eh concat`/`eh wipe` unchanged.
- **CLI surfaces**: `eh recap --detail`; `eh narrate` (deprecated alias,
  prints a pointer); `eh show` prints the artifact (drops `--narration`);
  `eh list` shows artifact detail.
- **Steering instructions (Phase 5)**: user direction for recap generation,
  orthogonal to grain (grain = length/completeness; instruction = content and
  voice). `[pipeline] instructions` config default + `eh recap --instruction`
  per run, injected as a USER DIRECTION block into the batch/combine prompts
  of every grain (precedent: `[video] steering_prompt` for tiktok; the
  anime-scene planner's `--instruction`). Covers content filtering ("skip
  chapter-opening recaps / author notes / next-chapter previews"), visual
  grounding ("describe the specific images"), and voice ("Gen Z slang").
  **The judge sees the instruction too**: omissions and style mandated by
  the user direction are not issues; faithfulness to the pages still applies
  where the instruction is silent (otherwise the completeness axis would
  fight instructed omissions, and the form axis would fight slang). The
  instruction is recorded on the artifact (recap.json + recaps row) for
  attribution; changing it does not auto-invalidate — re-runs stay explicit
  via --all/--chapter.

## Phases

1. ✅ **Schema + migration + pipeline merge.** — DONE 2026-09-15. `recaps.detail`
   ALTER; narrations fold-in migration; `recap_series` gains `detail`;
   `narrate_series` becomes a thin wrapper; prompts/judge/context rules
   selected by detail; selection semantics above. Tests.
   - Gate: suite green; migration on a copy of the kenja DB yields 100 recaps
     rows = 64 full + 36 standard, `series_context` untouched. **PASSED** —
     see Evidence below.
2. ✅ **CLI unify.** — DONE 2026-09-16. `--detail` on recap; narrate →
   deprecated alias; show/list/play handle the single artifact + single
   video. Tests.
   - Gate: suite green; `eh list` / `eh show kenja --chapter 1` sane on the
     migrated DB. **PASSED** — see Evidence below.
3. ✅ **Video merge.** — DONE 2026-09-16. Script stage by detail;
   render_state `detail` key; supersede-other-kind on build. Tests.
   - Gate: suite green; render one cached kenja chapter's video from its
     existing `full` artifact with zero model calls, then from a standard
     chapter; `ffprobe` durations recorded; `eh play` opens both.
     **PASSED** — see Evidence below. (The standard-detail half was verified
     via the mocked suite instead of a second live render — Kokoro
     re-synthesis of a wiped chapter is slow, and the mocked tests cover the
     script-model path, render_state re-render, and supersede.)
4. ✅ **Docs + cleanup.** — DONE 2026-09-16. design.md updated (config sample,
   data layout, store caveat, schema, recap-flow grain + steering paragraphs,
   narration section shrunk to a pointer, video one-per-chapter/kind, judge
   cross-ref, CLI list); narrate deprecation verified in `eh --help` /
   `eh narrate --help` (the Phase-2 docstring already surfaces it — no code
   change needed); AGENTS.md test count.
   - Gate: docs consistent; suite green. **PASSED** — see Evidence below.
5. ✅ **Steering instructions.** — DONE 2026-09-16. `[pipeline] instructions`
   config + `--instruction`/`-i` on `eh recap` and the narrate alias (flag
   wins); USER DIRECTION block appended to the rules of every grain's batch +
   combine prompts (book prompts included; empty instruction leaves prompts
   byte-identical); `judge_recap`/`judge_narration` (text + vision) receive
   it with the mandated-omission rule — the context judge deliberately does
   not (it checks fact preservation against the artifact as produced, where
   a mandated omission is correct by definition); attribution via additive
   `recaps.instruction` ALTER + `RecapMetadata.instruction` (old recap.json
   parses via the default); `eh show` header shows a truncated instruction;
   no auto-invalidation. Tests: config parse + flag precedence, injection
   present/absent on vision+text grains batch+combine, judge rule on both
   artifact judges, attribution row + recap.json, show header, no-reselect.
   - Gate: suite green; gate demo below drives `recap_series` with a fake
     adapter + instruction and shows it in the batch prompt, judge prompt,
     stored row, and recap.json. **PASSED** — see Evidence below. (The live
     kenja re-run stays optional — it spends OpenRouter tokens; the mocked
     demo covers the same code path with zero spend.)

## Dependencies

Blocks the 100-chapter scroll re-render (user decision). None otherwise.

## Evidence

### Phase 1 — 2026-09-15

1. Full suite (`uv run pytest -q`): **495 passed, 1 failed** — the one failure
   is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`
   (Python 3.13 json strictness; not to be fixed here):

   ```
   FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
   1 failed, 495 passed in 4.32s
   ```

2. Migration on a copy of the real DB
   (`cp ~/.local/share/entertainment-harness/harness.db /tmp/eh-migrate-test.db`,
   then `db.connect(Path('/tmp/eh-migrate-test.db'))` with
   `EH_NO_AUTO_MIGRATE=1` — the real code path: additive ALTERs +
   `db.fold_narrations_into_recaps` run on open; only the legacy file-layout
   migration is skipped):

   BEFORE (kenja = `01J76XYBTCD15FW69W889WKHB0`; `PRAGMA table_info(recaps)`
   shows no `detail` column):

   ```
   kenja_recaps | kenja_narrations = 100 | 64
   series_context kenja: through_chapter=94.0, LENGTH(rolling_summary)=1745
   ```

   AFTER:

   ```
   kenja_recaps_total | full | standard | other = 100 | 64 | 36 | 0
   kenja_narrations = 64                      -- table left fully populated
   series_context kenja: through_chapter=94.0, LENGTH(rolling_summary)=1745
   diff ctx-before.txt ctx-after.txt -> series_context IDENTICAL (both series)
   ```

   Idempotency + correctness spot-checks on the migrated copy (second
   `db.connect`, then SQL):

   ```
   re-open: total | full | standard = 100 | 64 | 36   -- unchanged (no-op)
   SELECT COUNT(*) FROM narrations n JOIN recaps r ON r.chapter_id=n.chapter_id
    WHERE r.summary != n.text;                          -> 0  (full supersedes)
   SELECT COUNT(*) FROM narrations n WHERE NOT EXISTS
    (SELECT 1 FROM recaps r WHERE r.chapter_id=n.chapter_id); -> 0  (none stranded)
   ```

3. AGENTS.md test count updated 474 -> 495.

### Phase 2 — 2026-09-16

1. Full suite (`uv run pytest -q`): **504 passed, 1 failed** — again only the
   known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`:

   ```
   FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
   1 failed, 504 passed in 4.44s
   ```

2. Live read-only sanity on the real library (the Phase-1 ALTER + fold-in ran
   on the real DB at first open of the new code — additive, idempotent,
   already proven on the copy):

   ```
   $ uv run eh list
   │ 01J76XYB │ Kenja no Mago │ ? │ 100 │ 58 │ 100 │ standard 36, full 64 │
   ```

   ```
   $ uv run eh show kenja --chapter 1
   Chapter 1 (full, qwen/qwen3-vl-32b-instruct, 2026-08-30T23:53:07+00:00)
   The scene opens with a wide, peaceful landscape: a forested valley nestled
   between rocky mountains under a bright sky dotted with clouds. …

   $ uv run eh show kenja --chapter 59
   Chapter 59 (standard, qwen/qwen3-vl-32b-instruct, 2026-08-28T08:03:36+00:00)
   Shin Walford presents a newly developed wireless communication device at the
   Bean Workshop, …

   $ uv run eh show kenja --chapter 1 --narration
   Note: '--narration' is deprecated — 'eh show' prints the chapter's artifact
   at its current detail (full = the old narration).
   Chapter 1 (full, qwen/qwen3-vl-32b-instruct, 2026-08-30T23:53:07+00:00)
   …
   ```

   Deprecation/validation smoke (no pipeline run, no model calls):

   ```
   $ uv run eh narrate kenja --max-chapters 0
   Note: 'eh narrate' is deprecated — use 'eh recap --detail full' (identical
   behavior).
   --max-chapters must be at least 1            (exit=1)
   $ uv run eh recap kenja --detail bogus
   --detail must be one of: gist, brief, standard, detailed, full   (exit=1)
   ```

3. AGENTS.md test count updated 495 -> 504.

### Phase 3 — 2026-09-16

1. Full suite (`uv run pytest -q`): **506 passed, 1 failed** — again only the
   known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`:

   ```
   FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
   1 failed, 506 passed in 4.38s
   ```

2. Live render of one full-detail chapter (kenja ch 1, artifact detail='full',
   17794 chars) via `build_video` directly (`uv run python`; no CLI, so no
   re-recap spend). The chapter's two videos rows were both wiped beforehand
   (narration 2026-09-05, recap 2026-09-15) and both workdirs held stale
   caches — a real supersede case. Zero model calls: script/TTS/visuals/clips
   all served from cache; only the final mux ran (14 s total):

   ```
   Stage 1/4 script: cached (47 segments)
   Stage 2/4 narration (kokoro): cached
   Stage 3/4 visuals: cached
     pacing: 47 beats, ~3016 words, 101 cuts, 22.9s avg beat, 10.6s per visual
   Stage 4/4 assembly: rendering...   (all clips cached)
     pruned clips/
     pruned 48 segment WAV(s)
     superseded the recap video (one video per chapter)
   Done: .../chapters/ch-001/video-narration/out.mp4 (1085s)
   ELAPSED 14s
   ```

   ffprobe + on-disk state after the build:

   ```
   $ ffprobe .../video-narration/out.mp4
   duration=1085.200000
   size=307661143

   $ ls .../chapters/ch-001/
   chapter.json  narration.json  recap.json  source  video-narration
     # video-recap/ pruned by the supersede

   render_state.json: {"colorize": false, "translated": false,
                       "mode": "kenburns", "detail": "full", "pacing": 1}
   ```

   videos rows for ch 1 — exactly one row, kind follows the artifact's
   detail, wiped mark cleared by the re-render:

   ```
   {'from_chapter': 1.0, 'to_chapter': 1.0, 'kind': 'narration',
    'path': '.../video-narration/out.mp4', 'duration_s': 1085.118,
    'wiped_at': None}
   rowcount: 1
   ```

   `eh play` lookup against the real library (subprocess.run monkeypatched —
   the player was NOT opened):

   ```
   ['play', 'kenja', '--chapter', '1']              -> exit 0  Playing chapter 1 (1085s)
   ['play', 'kenja', '--chapter', '1', '--narration'] -> exit 0  Playing chapter 1 (1085s)
   WOULD OPEN: ['open', '.../video-narration/out.mp4']   (both)
   ```

   concat sanity with one-video-per-chapter (no writes; fails fast):

   ```
   $ uv run eh concat kenja --kind narration
   Need at least 2 narration videos to concatenate (found 1).   exit=1
   $ uv run eh concat kenja --kind recap
   Need at least 2 recap videos to concatenate (found 0).       exit=1
   ```

   The standard-detail half of the gate (script model writes beats; kind =
   recap) is covered by mocked tests rather than a second live render:
   `test_build_video_end_to_end_and_cached_second_run` (script model invoked,
   recap workdir/kind), `test_build_video_render_state_detail_change_rerenders`
   (standard -> detailed re-renders via render_state), and
   `test_build_video_supersedes_other_kind_on_detail_change` (files pruned +
   single videos row).

3. AGENTS.md test count updated 504 -> 506.

### Phase 4 — 2026-09-16

1. Full suite (`uv run pytest -q`): **520 passed, 1 failed** — again only the
   known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`:

   ```
   FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
   1 failed, 520 passed in 6.79s
   ```

2. Docs consistent: `docs/design.md` now documents the grain system
   (`[pipeline] detail`, `recaps.detail`, pending-selection buckets), the
   steering instructions (`[pipeline] instructions` + `--instruction`), the
   narration merge (section shrunk to a pointer; `narrations`/`narration.json`
   marked legacy), one-video-per-chapter with detail-derived kind and
   supersede (incl. the store's no-remote-delete caveat), `render_state`'s
   `detail` key, the verbatim script split for full-detail artifacts, and the
   updated CLI list (`eh narrate` deprecated alias; `eh show` detail +
   instruction header; `eh play` one video per chapter, `--narration`
   deprecated).

   The "deprecate narrate in help text" item needed no code change — the
   Phase-2 docstring already surfaces in both help outputs:

   ```
   $ uv run eh --help
   │ narrate         Deprecated alias for 'eh recap --detail full': narrate all   │

   $ uv run eh narrate --help
    Usage: eh narrate [OPTIONS] {series}
    Deprecated alias for 'eh recap --detail full': narrate all pending chapters
    (default), EVERY synced chapter with --all, or one chapter with --chapter — a
    full, in-order retelling of the chapter, stored as the chapter's artifact at
    detail='full'.
   ```

3. AGENTS.md test count updated 506 -> 520.

### Phase 5 — 2026-09-16

1. Full suite: same run as Phase 4 above — **520 passed, 1 failed** (known
   pre-existing only). The 14 new tests (520 - 506) cover: config parse +
   flag-over-config precedence, injection present/absent on the vision grain
   and the full (narration) grain — batch and combine prompts — plus the book
   path, the judge mandated-omission rule on `judge_recap` and
   `judge_narration`, byte-identical prompts when empty, attribution on the
   recaps row + recap.json (incl. pre-`instruction` recap.json parsing), the
   `eh show` header (shown truncated; absent when empty), and no
   auto-reselection of completed chapters.

2. Gate demo (`uv run python`; tmp EH_DATA_DIR; fake adapters recording
   prompts; `recap_series(..., chapter_num=1, client=FakeClient(),
   instruction="write in Gen Z slang")`):

   ```
   chapters completed: ['ch-1']

   === batch prompt (vision, pages 1-…) ===
   USER DIRECTION — the reader steering this run asked:
   write in Gen Z slang
   Follow it: content it says to skip stays out of the artifact, and a voice or emphasis it asks for applies. It never overrides the rules above.

   === combine prompt (vision model, last call) ===
   USER DIRECTION — the reader steering this run asked:
   write in Gen Z slang

   === judge prompt (artifact judge) ===
   USER DIRECTION — the reader steering this run asked:
   write in Gen Z slang
   Omissions and style choices this direction mandates are NOT issues: content it says to skip is correctly absent, and a voice it asks for is not a form problem. Where the direction is silent, every criterion above applies as written — faithfulness to the source material (or pages) still applies in full.

   recaps row: detail='standard' instruction='write in Gen Z slang'
   recap.json: detail='standard' instruction='write in Gen Z slang'

   ALL GATE ASSERTIONS PASSED
   ```

   The optional live kenja re-run (`--instruction "skip chapter-opening recap
   pages and author notes"`) is skipped: it spends OpenRouter tokens, and the
   demo above exercises the identical code path (prompt assembly, judge call,
   INSERT, works.write_recap) with zero spend.

3. AGENTS.md test count updated 506 -> 520 (same edit as Phase 4).

### Post-close amendment — 2026-09-16

`--instruction`/`[pipeline] instructions` accept `@file` (curl-style):
`cli.load_instruction` resolves the file contents (stripped; relative paths
from the cwd; empty file = no instruction) for both the flag and the config
value, in `_run_chapter_pipeline` so other commands never fail on an
unreadable `@path`; missing file → error naming the path, exit 1. Suite:
**524 passed, 1 failed** (known pre-existing only); AGENTS.md 520 -> 524.
