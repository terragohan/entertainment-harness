# Character bible — per-work cast registry injected into recap prompts

Status: **done** (2026-09-27 — Phases 1–5 gated)

## Goal

Narrations read badly because the pipeline cannot know character names ahead
of time — and this is structural: `STRICT_RULES`
(`src/entertainment_harness/pipelines/recap.py:87-91`) forbids any name not
printed on the current pages (an anti-hallucination guard), the 300-word
rolling story-so-far compresses names out, standalone chapters get no context
at all, and the judges would reject externally-sourced names as outside
knowledge.

Fix: a per-work **character registry** (canonical name, aliases, one-line
role, chapter span), learned incrementally from each chapter's judged
artifact, injected as a compact CAST block into every prompt that writes or
judges prose, and user-viewable/editable in the work view so one correction
fixes every future chapter.

## Non-goals

- No external seed (AniList/MAL cast APIs) — possible later via
  `origin="seeded"`; the observed registry already carries the scanlation's
  own romanizations.
- No per-chapter cast scoping (the whole registry is injected, capped) and no
  character–panel visual linking.
- No `work.json` mirror of the registry (DB is the source of truth).

## Design

**Storage** — additive `characters` table in `db.py` SCHEMA:

```sql
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(id),
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',        -- JSON array
    role TEXT NOT NULL DEFAULT '',
    first_seen REAL, last_seen REAL,           -- chapter_nums
    origin TEXT NOT NULL DEFAULT 'observed',   -- observed | user
    edited INTEGER NOT NULL DEFAULT 0,         -- user-locked
    UNIQUE(series_id, name)
);
```

**Extraction & merge** (`pipelines/characters.py`) — after each chapter
artifact is stored (`recap_series`, all buckets — upserts are idempotent, a
registry is a set not a chronology), the text model lists the chapter's
characters as JSON `[{"name", "aliases", "role"}]`; `judge_cast` verifies
against the artifact via `judge_loop`; persistent failure or garbage JSON
keeps the old registry (the never-destroy-accumulated-state guard, same as
`_suspicious_context`). The merge is deterministic Python: case-insensitive
name-or-alias match → union aliases, latest non-empty role,
`last_seen = max(chapter_num)`; no match → insert; `edited=1` rows keep
name/aliases/role; canonical names never flip-flop (a differing incoming
name becomes an alias). Opt-out: `[pipeline] characters = false`.
`eh cast <series>` lists; `eh cast <series> --rebuild` re-extracts
sequentially from existing artifacts (backfill / chapter-1 cold start).

**Injection** — `cast_block(cast)` renders "" for an empty registry (prompts
stay byte-identical, the `user_direction_block` discipline) and otherwise a
capped list (~15 by `last_seen` desc) carrying its own scoped rule
amendment, appended to the `{rules}` slot every batch/combine prompt already
has:

```
CAST — characters known from previous chapters:
- Shin (also "Shinu") — young hunter protagonist
You MAY use these names when that character appears (they override "only
printed names" for these characters only); never apply them to others.
```

Judges receive the same list via a `cast=""` pass-through (precedent:
`instruction=` in `judge_artifact`) — "names from the CAST list are
pre-approved" — in `judge_artifact`, `judge_context`, and
`judge_translation`. `TRANSLATE_PROMPT` gets it as canonical-spelling
guidance; `SCRIPT_PROMPT` (video beats) gets it so beats keep the names.
`CONTEXT_PROMPT` is unchanged (names flow in via the recap).

**UI** — `GET/PUT /api/works/{work}/characters` (PUT replaces the list;
UI-touched rows are written `origin="user", edited=1`) and a "Characters"
disclosure in the work view between header and run panel: name/aliases/role
rows with add/edit/delete and one save.

## Phases

### Phase 1 — storage + extraction (gate: full pytest green)

- `characters` table + db helpers (`get_characters`, `upsert_character`,
  `replace_characters`), `pipelines/characters.py` (prompt, parse, merge,
  block renderer), `judge_cast` in `judge.py`, the `recap_series` hook,
  `PipelineConfig.characters`, `eh cast` (+ `--rebuild`).
- Tests: merge rules (alias union, edited-lock, no flip-flop), collapse
  guard, fold hook populates the table across two chapters with fake
  adapters, CLI.

**Status: done.** All pieces landed as designed; the hook runs on every
stored artifact (a cast update never fails a chapter — parse/judge failure
keeps the old registry, and any unexpected error is caught and logged).
Two pre-existing recap assertions were made prompt-content-filtered instead
of position/count-based (the per-chapter cast call shifted them).

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 875 passed, 1 warning in 17.83s
```

(Known pre-existing failure; 15 new tests in `tests/test_characters.py`.)

### Phase 2 — recap + judge injection (gate: full pytest green)

- `cast_block` into every batch/combine prompt path (manga 5 grains + book)
  and into `judge_artifact` / `judge_context` / `judge_translation`.
- Tests: byte-identical prompts with an empty registry; CAST present and
  judge blocks approve listed names with a non-empty one.

**Status: done.** `manga_batches`, `book_chapter`, `_recap_book_chapter`,
and `judged_artifact` take `cast: str = ""` and compose it into the rules
slot; `recap_series` builds the block once per run via
`load_cast_block(conn, series_id)` (after `context_block`, so standalone
chapters get it too) and passes it to all three call sites. `judge.py` got
`_judge_cast_block` (extracts the `- ` lines, adds "pre-approved … NOT an
outside-knowledge or faithfulness issue"), composed into the direction slot
of `judge_artifact` and threaded through `judge_recap`/`judge_narration`.
Deliberate deviation: **`judge_context` gets no cast param** — its inputs
(previous context + recap) already carry the cast names, so no name is ever
"absent from both inputs"; `judge_translation` pairs with Phase 3's
`TRANSLATE_PROMPT` injection. 4 new tests in `tests/test_characters.py`.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 879 passed, 1 warning in 21.61s
```

(Known pre-existing failure only.)

### Phase 3 — translation + video script (gate: full pytest green)

- Blocks into `TRANSLATE_PROMPT` and `SCRIPT_PROMPT`.
- Tests: prompts include the cast when present, unchanged when absent.

**Status: done.** `cast_block` gained an `amendment` parameter (the
co-located rule line, worded per prompt family) with two new variants:
`cast_script_block` ("may use these names … even where the recap leaves
them unnamed") and `cast_translation_block` (canonical spellings under any
listed alias); `load_cast_block(conn, series_id, block=…)` selects the
wording. `TRANSLATE_PROMPT` got a `{cast}` slot after the "Use names as
printed" rule (empty cast renders byte-identically), threaded
`translate_chapters` → `_translate_page` → `_translate_bubbles`;
`judge_translation` takes `cast=""` and composes `_judge_cast_block` into
its criteria list. `SCRIPT_PROMPT` got the slot after "no new events,
names, or details", threaded through `generate_script` from the video
pipeline's recap→script branch (the panel-first and narration-split stage-1
paths don't call `generate_script` — out of scope). 7 new tests in
`tests/test_characters.py`.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 886 passed, 1 warning in 17.17s
```

(Known pre-existing failure only.)

### Phase 4 — UI + docs (gate: typecheck, smoke, full pytest, package)

- Server endpoints, work-view Characters section, smoke checks.
- `docs/design.md`: recap flow, DB schema list, server endpoints, CLI list,
  work-view description.

**Status: done.** `GET/PUT /api/works/{work}/characters` in
`server/app.py` (GET lists most-recently-seen-first with origin/edited
flags; PUT validates then `db.replace_characters` — blank aliases stripped,
empty names 422, unknown work 404). Work view: a collapsible
`.characters-panel` `<details>` between header and run panel — count in the
summary, one editable row per character (name / comma-separated aliases /
role), add/remove row buttons, single Save PUTting the whole list and
re-rendering from the response; blank rows dropped silently client-side.
5 server tests + 6 smoke checks (panel renders, add → save round-trips
through the live backend, GET returns the user-edited row). `docs/design.md`
gained the Character-registry paragraph under Recap flow, the `characters`
table in the schema list, the cast bullets under Judge, the
canonical-spelling note in Translation, the CAST note in Video script, the
`eh cast` CLI entry, the endpoint bullet, the work-view section
description, and `[pipeline] characters` in the config sample.

Gate evidence (2026-09-27):

```
$ cd ui && bun run typecheck
$ tsc --noEmit                       # clean

$ cd ui && bun run smoke
SMOKE PASS — 67 checks against a live backend

$ uv run pytest -q
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 891 passed, 1 warning in 17.92s   # known pre-existing failure only

$ cd ui && bun run package           # exit 0 (dmg created)
```

### Phase 5 — UI-triggered rebuild (gate: pytest, typecheck, smoke, package)

- Share the `eh cast --rebuild` fold as `characters.rebuild_cast` (CLI stays
  a thin wrapper), expose it to the work view as a background job
  (`server/castbuilds.py`, the exports pattern: daemon thread, one active
  per work, polled via `GET /api/cast-builds`), and add a two-step Rebuild
  button to the Characters panel (it replaces the whole registry, manual
  edits included — the first click only arms).
- Tests: endpoint 202/409/422/404 + done/error paths with a stubbed
  rebuild; smoke: button arming + the no-recaps 422 (the dev fixture has 26
  recaps — a confirmed rebuild would fire real model calls, so smoke never
  confirms for Kenja).

**Status: done.** `characters.rebuild_cast` holds the fold (CLI `_rebuild_cast`
is a thin wrapper mapping ValueError to exit 1); `server/castbuilds.py`
mirrors the exports manager (daemon thread, `CastBuildJob` with an 8-line
log tail, one active per work, in-memory); endpoints
`POST /api/works/{work}/characters/rebuild` (202; 404 unknown work, 422 no
recaps — pre-checked before any model is touched, 409 conflict) and
`GET /api/cast-builds`. The job resolves models at the configured
`[pipeline] thinking` level. The Characters panel's Rebuild button is
two-step (4 s arm timeout), disables Save while the job runs, tails the job
log in the status line, and re-renders rows from the rebuilt registry on
done. 5 server tests + 3 smoke checks. `docs/design.md`: endpoint bullet,
work-view description, recap-flow mention.

Gate evidence (2026-09-27):

```
$ uv run pytest -q
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 896 passed, 1 warning in 24.63s   # known pre-existing failure only

$ cd ui && bun run typecheck        # clean
$ cd ui && bun run smoke
SMOKE PASS — 70 checks against a live backend

$ cd ui && bun run package          # exit 0 (dmg created)
```

## Evidence

(phases paste gate output here as they close)
