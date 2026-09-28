# AGENTS.md

Guidance for AI agents working in this repo.

## Docs

- `docs/design.md` — architecture, schema, rationale. The canonical reference for how things work.
- `docs/plan.md` — the original build plan; fully executed, kept for history and the phase/gate format.
- `docs/initiatives/` — tracked multi-phase work. **Read this first** when the task touches planned or ongoing features.

## How initiatives work

- One directory per initiative under `docs/initiatives/<name>/`, holding a `README.md` with: goal, non-goals, rationale, phases, and status. Supporting artifacts (specs, spikes, measurements) live in the same directory.
- Statuses: `proposed` → `active` → `done` | `dropped`. **Exactly one initiative is `active` at a time.** Everything else waits unless the user says otherwise.
- A phase is complete only when its gate evidence (command output, measurements) is pasted into the initiative file. If there's no evidence, the phase isn't done — verify before relying on it.
- At the start of a session involving initiative work: read the `active` initiative's `README.md`, implement against its current phase, and update the file as phases close.
- Dropping is fine; record the reason in the file, don't delete the directory.
- Cross-initiative contracts (e.g. a shared time-map format) are owned by whichever initiative defines them first; the other initiative links to it rather than restating it.

## Code conventions

- Python, `uv` for everything (`uv run pytest`, `uv run eh …`). 947 of 948 tests must stay green (one known pre-existing failure: `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`, Python 3.13 json strictness; CI deselects it via `.github/workflows/test.yml`).
- Desktop app: `ui/` is a self-contained Electrobun (Bun/TypeScript) project — use `cd ui && bun run typecheck` and `bun run smoke` (headless UI test against a real `eh serve`). Run only one app/`eh serve` instance per data dir (sqlite write-lock).
- Typer CLI (`src/entertainment_harness/cli/`), works layout helpers in `library/works.py`, DB migrations are additive-only `ALTER TABLE`s in `db.py`.
- When you change behavior documented in `docs/design.md`, update the doc in the same change.
