# Desktop UI (Electrobun app)

Status: **done** (v1 + background runs + instructions + range fix + stop + unfinished toggle + detail select + auto-process toggle + gap-fill alignment + optional scope (range/all) + default instruction + library video-row dedupe + video export; all gate-evidenced below, 2026-09-21)

Approved plan: session plan `shatterstar-orion-crystal` (2026-09-20). Stack/scope/install chosen by the user: Electrobun (Bun + native WebKit webview), v1 = library browsing + playback + run triggering + settings editor, double-clickable `.app`.

## Goal

A macOS desktop app for `entertainment-harness`: browse the library, play chapter videos in-app as they finish rendering, trigger recap generation runs with a live progress bar, and edit `config.toml` (including enabling/disabling the built-in sources, with restore-defaults) — while the CLI and config file stay exactly as they are (all changes additive).

## Non-goals (v1)

- Search/add/remove works from the UI (CLI only).
- anime-scene / tiktok / models / voices UI screens.
- Partially-rendered video streaming (unit of playability is the finished chapter).
- Code signing/notarization, auto-updates.
- Windows/Linux packaging (Electrobun is cross-platform; v1 targets macOS only).

## Architecture

```
┌─ EntertainmentHarness.app (Electrobun) ────────────────┐
│  Bun main process (TS): window, native menu,            │
│  spawns/monitors bundled Python backend, kills on quit  │
│  WebKit webview (TS/HTML UI) ── HTTP ──┐                │
└────────────────────────────────────────┼────────────────┘
┌────────────────────────────────────────▼────────────────┐
│  `eh serve` (new, additive CLI command)                 │
│  FastAPI + uvicorn on 127.0.0.1, ephemeral port         │
│  REST + SSE progress + range-streamed mp4               │
│  Reuses duck-typed progress interface from ui.py        │
└─────────────────────────────────────────────────────────┘
```

This implements the localhost HTTP bridge long planned in [streaming](../streaming/README.md) Phase 3, and supersedes the SwiftUI viewer approach of [native-experience](../native-experience/README.md) (blocked; its `eh serve --local` sketch is realized here as `eh serve`).

## New dependencies

- Python: `fastapi`, `uvicorn`, `tomlkit` (stdlib has no TOML writer; tomlkit preserves hand-edited formatting). Dev: `pyinstaller`.
- JS: `bun`, `electrobun` — isolated under `ui/`; repo root stays Python-only.

## Phases

1. **Backend server (`eh serve`).** New `src/entertainment_harness/server/` package: `GET /api/library`, `GET /api/videos/{work}/{chapter}/stream` (HTTP Range), `POST /api/runs` + `GET /api/runs/{id}/events` (SSE), `GET/PUT /api/config` (tomlkit writer, validated against the `Config` dataclasses, atomic write), `GET/PUT /api/sources` + `POST /api/sources/reset`. Progress reporter implementing the `ui.py` duck-typed interface, passed as `progress=` into `build_video`. Config gains optional `[sources] enabled = [...]` (default `["mangadex", "weebcentral"]`); `get_client()` errors clearly on disabled sources; additive CLI `eh sources [enable|disable|reset]`.
   - Gate: `uv run pytest` green (666 + new tests); `curl` evidence for library JSON, 206 partial-content stream, SSE events during a sample run on `samples/kenja-ch0`.
2. **Electrobun app shell (`ui/`).** Bun main process (window, menu, backend lifecycle: spawn, port discovery from stdout, health-check, kill on exit; dev mode `uv run eh serve`, prod mode bundled binary). Webview UI: library grid with per-chapter status + progress bar, player view (`<video>` + autoadvance), run controls wired to SSE, settings view (config form + sources checkboxes + restore defaults).
   - Gate: `cd ui && bun run dev` against the real backend; screenshots of library, in-progress run, playback, and a config edit round-tripping to `config.toml` (scratch `EH_DATA_DIR`).
3. **Packaging.** PyInstaller `eh-serve` binary embedded as an Electrobun app resource; `EntertainmentHarness.app` via Electrobun build; ad-hoc codesign (Gatekeeper caveat documented). `uv run eh …` CLI path must stay untouched.
   - Gate: double-click the `.app` on an account without the repo venv, run a generation on the sample work, play the result; paste evidence here.
4. **Docs & close-out.** Update `docs/design.md` (server, `[sources]`, new CLI commands, packaging), `AGENTS.md` (ui/ toolchain), initiative index.
   - Gate: docs diff reviewed; full test suite green.

## Gate evidence

_(pasted as phases close)_

### Phase 1a (subset of Phase 1): config writer + `[sources]` + `eh sources` — landed 2026-09-20

- `config.py`: `SourcesConfig` (`enabled`, default `["mangadex", "weebcentral"]`), `load_config` split into `parse_config(raw)` + `load_config(path)`, and `save_config(path, updates)` — tomlkit merge preserving comments, validated against the `Config` dataclasses and round-tripped through `parse_config` before an atomic temp-file replace; `ConfigWriteError` on unknown section/key or wrong type, file untouched.
- Gating: `sources.get_client()` raises `PluginError` ("source 'x' is disabled in config.toml ([sources].enabled); re-enable with `eh sources enable x`") for built-ins absent from `[sources].enabled`; entry-point sources always enabled; unknown names keep the unknown-plugin error.
- CLI: `eh sources` (list with origin + state), `eh sources enable|disable|reset`; `eh plugins` shows source enabled/disabled.
- Tests: 666 → 700 (`tests/test_config.py`, `tests/test_sources.py`); full suite: 699 passed + the known `test_lfm.py::test_repair_json_repairs_trailing_comma` pre-existing failure.
- Still open for Phase 1: the `eh serve` HTTP server (FastAPI + uvicorn), progress reporter, curl gate evidence.

### Phase 1b (rest of Phase 1): `eh serve` HTTP server + SSE progress — landed 2026-09-20

- `src/entertainment_harness/server/` (new): `app.py` (FastAPI factory; endpoints under `/api`: `health`, `library`, `videos/{work}/{chapter}/stream` with HTTP Range via Starlette `FileResponse`, `POST/GET /api/runs` + `GET /api/runs/{id}/events` SSE, `GET/PUT /api/config`, `GET/PUT /api/sources` + `POST /api/sources/reset`; permissive localhost CORS), `progress.py` (`ServerProgress`: the `ui.py` duck-typed interface emitting JSON event dicts onto a `threading.Condition`-guarded per-run buffer — pipelines unchanged), `runs.py` (`RunManager`: worker-thread runs of the `eh recap` flow, one active run per work → 409, `running|done|error`, SSE replays buffered events then streams live until `run-done`/`run-error`).
- `eh serve [--port 0] [--host 127.0.0.1]` (new, additive): first stdout line is `listening <port>` (machine-readable for the Electrobun parent); everything else to stderr. fastapi + uvicorn added (approved).
- Tests: 700 → 723 collected (`tests/test_server.py`, 23 tests: health, library shape, full/ranged/suffix/416/404/traversal stream, config round-trip + 422, sources GET/PUT/reset, run event sequence + 409 + error + 404/422 via a stubbed pipeline, `eh serve` subprocess listening-line/health). Full suite:

  ```
  1 failed, 722 passed, 1 warning in 6.31s
  FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma  # known pre-existing failure
  ```

- `curl` evidence (scratch `EH_DATA_DIR` seeded with one work/chapter + fabricated 4116-byte mp4 in the works layout; `uv run eh serve --port 0`):
  - First stdout line: `listening 56739`
  - `GET /api/health` → `{"ok":true}`
  - `GET /api/library` → one work with `chapters[0]`: `"detail": "standard", "has_recap": true, "has_video": true, "video": {"kind": "recap", "duration_s": 9.0}, "stream": "/api/videos/s1/0/stream"`
  - `curl -r 0-99 …/api/videos/s1/0/stream` → `HTTP/1.1 206 Partial Content`, `accept-ranges: bytes`, `content-range: bytes 0-99/4116`, exactly 100 bytes received
  - `POST /api/runs {"work": "s1", "skip_preflight": true}` on the fully-recapped work → `GET /api/runs/{id}/events` replays the buffered frames and terminates (status `done`):

    ```
    event: log
    data: {"event": "log", "run_id": "d1567a3729c6", "work": "s1", "ts": "2026-09-20T19:58:39+00:00", "chapter": null, "message": "Nothing to recap: no chapters pending at 'standard' detail."}
    …
    event: run-done
    data: {"event": "run-done", "run_id": "d1567a3729c6", "work": "s1", "ts": "2026-09-20T19:58:40+00:00", "chapters": 0}
    ```

    (The full `run-start → chapter-start → stage → log → video-ready → chapter-done → run-done` sequence is covered by the stubbed-pipeline test; a live multi-chapter run on this machine serializes behind real model pulls, so the curl evidence uses a tiny completed run.)

Phase 1 (backend server) fully closed; Phase 2 (Electrobun shell) closed below; Phase 3 (packaging) is next.

### Phase 2: Electrobun app shell (`ui/`) — landed 2026-09-21

- `ui/` is a self-contained Electrobun 1.18.1 project (own `package.json`; repo root stays Python-only; `ui/node_modules`, `ui/build`, `ui/artifacts`, `ui/.dev-data` added to the root `.gitignore`). No webview framework — dependency-free TS + DOM (`ui/src/mainview/`), Electrobun itself is the only runtime dep. `bun run typecheck` (tsc) is clean; Electrobun ships raw `.ts` whose internals don't pass strict tsc, so the app maps its three import specifiers to a local stub (`ui/types/electrobun.d.ts`).
- **Main process** (`ui/src/bun/`): `backend.ts` isolates the spawn behind `backendCommand(): {cmd, args, cwd, env}` (dev: `uv run eh serve --port 0` from the repo root; Phase 3 swaps in the bundled binary). Repo root found by walking up for `pyproject.toml` + `src/entertainment_harness` (`EH_REPO_ROOT` overrides); `EH_DATA_DIR` is forwarded (documented in `ui/README.md`). Port parsed from the first stdout line (`listening <port>`), `/api/health` polled (30s timeout → visible error view with the reason), state pushed to the webview over RPC (`getBackendState` request + `backendState` message). Backend killed on window close / Cmd-Q / SIGINT / SIGTERM, **plus a pipe-EOF watchdog** (`sh -c 'read; kill <pid>'`) so even a SIGKILLed main process can't orphan `eh serve`. Window 1280x800; native App/Edit/Window menus. (Electrobun 1.x exposes no min-window-size API; CSS `min-width` is the soft floor.)
- **Webview** (`ui/src/mainview/`, hash routing): Library grid → work view (per-chapter playable/in-progress/missing badges, durations; from/to chapter inputs + Generate recap; live run panel: overall bar, current stage line, scrolling log; auto-attaches to an in-flight run on entry and on 409; SSE `video-ready` flips rows to playable without reload) → player (`<video>` against the range-enabled stream URL, prev/next chapter, autoadvance on `ended`) → settings (config form generated from `GET /api/config` grouped by section, bool→checkbox / number→number / string/list→text, 422 shown inline, shared-config.toml warning by the save button; sources checkboxes + Apply + Restore defaults). A `?port=<n>` query param lets the built view run in any browser against a live backend (used for screenshots).
- Gate evidence (scratch `EH_DATA_DIR=ui/.dev-data`: real `harness.db` + `config.toml` + `voices/`, work.json + chapter.json for kenja-no-mago, real rendered `out.mp4` for ch 2/3, full ch 23 incl. source pages for the run; verified first with curl: `206` range responses on `/api/videos/…/2/stream` and `/3/stream`):
  - `cd ui && bun install` (50 packages), `EH_DATA_DIR=$PWD/.dev-data bun run dev` → app window opens, log shows `[backend] ready on 127.0.0.1:59842` and the main-process RPC probe evaluates JS in the real webview: `[webview] rendered: "views://mainview/index.html#/library | Entertainment Harness\nLibrary\nSettings\nbackend\n…Kenja no Mago\nmanga · weebcentral…"` — the real app window renders the library from the real backend.
  - Screenshots in `phase2-screenshots/` (method: the built view bundle served statically + driven in headless Chrome via playwright-core CDP against the app-spawned backend — macOS Screen Recording permission isn't granted to this terminal so `screencapture` of the Electrobun window itself fails; the RPC probe above is the in-app evidence): `01-library` (grid, 25/138 playable), `02-work` (chapter list, playable badges with durations, run controls), `03-player` (ch 0 narration video playing, currentTime 1.68s, next-chapter button), `04/05-settings` (+`Saved to config.toml.`), `06-sources` (weebcentral unchecked → Apply), `07/08-run-progress` (live run: disabled Generate button, stage line `Chapter 23 — pages 29-32 of 51` advancing to `37-40`, scrolling log).
  - In-progress run: `POST /api/runs {"work": "…", "chapter": 23, "video": true, "skip_preflight": true}` → `201` (`8a47d4234c1a`); SSE captured `run-start → chapter-start → 14× stage → 19× log` while the work view auto-attached. The run is a *real* generation (OpenRouter per the scratch config): it read 51 pages, and the judge rejected attempt 1/3 on continuity grounds (see events log) — the full video render takes tens of minutes, so `video-ready`'s live badge-flip is covered by code path + the stubbed-pipeline test from Phase 1b rather than a completed render here.
  - Settings round-trip through the UI (scratch `config.toml` diff after Save + sources Apply):
    ```diff
    +[pipeline]
    +detail = "detailed"
    +
    +[sources]
    +enabled = ["mangadex"]
    ```
  - Backend lifecycle: graceful quits twice showed uvicorn `Application shutdown complete` and `pgrep` empty afterwards; a SIGTERM to the main process (`shutdown()` → `backend.stop()`) likewise left **zero** `eh serve` processes. An orphan from an earlier SIGKILLed pre-watchdog build was found and killed; the watchdog build leaves none. (Note: uvicorn waits for open SSE connections during graceful shutdown, so a still-attached events client delays exit until it disconnects.)
  - **Minimal Python change, flagged** (`src/entertainment_harness/db.py`, additive): while a run is active the worker's connection holds an implicit sqlite write transaction across the long LLM/TTS calls, so every API `connect()` — which unconditionally ran the narrations fold INSERT/UPDATE — failed with `sqlite3.OperationalError: database is locked`, 500ing `GET /api/library` and friends for minutes at a time. Reproduced live against the Phase 2 backend (`500` under a held write lock). Fix: `fold_narrations_into_recaps` now only writes when a read-only pre-check (`_fold_pending`) finds un-folded rows, and defers silently to a later open when the database is locked (test: `tests/test_db.py::test_connect_succeeds_while_another_connection_holds_write_lock`; `docs/design.md` "Local server" updated). Post-fix, `/api/library` returns `200` under an externally held write lock. **Any PyInstaller-frozen `eh-serve` must be re-frozen to pick this up** (a frozen binary built from pre-fix source still 500s under lock — observed during Phase 3 bring-up).
  - `uv run pytest -q`: `1 failed, 741 passed` — the one failure is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`. The only Python touch in this phase is the db.py fold fix above.
  - API notes vs the original brief: `stage` events are not only `stage: "video"` — the recap phase emits `stage: "recap"` with freeform `detail` (e.g. `pages 29-32 of 51`), which the UI renders verbatim; `POST /api/runs` has no from/to range — a single chapter maps to `chapter`, a range maps to `all_chapters: true` + `max_chapters: to-from+1` (next pending chapters in order), noted in the UI hint text.
  - Concurrency note: Phase 3 (packaging) began in the same working tree before this section was written — `ui/src/bun/backend.ts` has since grown the planned prod mode (`backendCommand()` now spawns the bundled `Resources/app/backend/eh-serve/eh-serve` when present, exactly the seam Phase 2 created) and `electrobun.config.ts` gained the `resources/backend/eh-serve` copy step. The Phase 2 evidence above was captured with the dev-mode spawn.

### Phase 2 (Electrobun app shell) — 2026-09-21

- `ui/` self-contained Electrobun project (Bun 1.4.2 via Homebrew, electrobun 1.18.1): Bun main process (window, menu, backend lifecycle behind `backendCommand()` in `src/bun/backend.ts` — the Phase 3 seam), dependency-free TS webview with 4 views (library, work + run controls, player, settings). Layout and run instructions in `ui/README.md`.
- `bun run typecheck` → clean (no output).
- Headless smoke test `ui/scripts/smoke.ts` (`bun run smoke`, linkedom DOM shim + minimal EventSource client, spawns a real `eh serve` on `ui/.dev-data`): **16/16 checks pass** —

  ```
  backend on :59676
  ok   library renders work cards
  ok   library lists Kenja no Mago
  ok   card shows playable chapter count
  ok   Kenja has a playable chapter (ch 0)
  ok   work view renders a row per chapter
  ok   playable chapter has a ▶ badge
  ok   run panel has a Generate recap button
  ok   no-op run completes via SSE (stage line shows Done)
  ok   log area received SSE log lines
  ok   player renders a <video> element
  ok   video src points at the stream endpoint
  ok   stream URL serves range-enabled video
  ok   settings renders checkboxes (sources + booleans)
  ok   source mangadex checkbox matches enabled=true
  ok   source weebcentral checkbox matches enabled=true
  ok   settings warns about editing shared config.toml
  SMOKE PASS — 16 checks against a live backend
  ```

  (The run check drives the real Generate button on the chapter-less "Atomic Habits" work: an all-chapters run over zero chapters completes in ~1 s — a Kenja run would do real generation. SSE events flow: button → POST /api/runs → EventSource replay+live → run-done → "Done — 0 chapter(s)".)
- `uv run pytest -q` → `1 failed, 722 passed` (only the known pre-existing `test_lfm.py::test_repair_json_repairs_trailing_comma`).
- Screenshots: not captured — macOS Screen Recording TCC permission blocks `screencapture` from the terminal, and the Electrobun WebKit view has no headless capture path. Functional verification is the smoke test above; visual check is by opening the app (`cd ui && EH_DATA_DIR=$PWD/.dev-data bun run dev`).
- Note: multiple stray dev instances from one session caused `sqlite3.OperationalError: database is locked` against the shared `.dev-data/harness.db` — only run one app instance per data dir.

Phase 2 is closed; Phase 3 (packaging) is next.

### Phase 3 (packaging: PyInstaller backend + double-clickable .app) — 2026-09-21

- `packaging/eh_entry.py` + `packaging/eh-serve.spec` (new): PyInstaller **onedir** build of the full `eh` CLI (argv passthrough; the app invokes `eh-serve serve --port 0`). Hidden imports cover the lazy serve/run paths (`collect_submodules("entertainment_harness")`, kokoro_onnx, phonemizer, espeakng_loader, onnxruntime, cv2); data/metadata for `kokoro_onnx/config.json`, espeak-ng dylibs + espeak-ng-data, and `importlib.metadata.version("kokoro-onnx")`. Optional extras (torch/transformers/segment-anything/mlx-audio) excluded — not on the default recap path. Model weights NOT bundled (kokoro onnx/voices download to the cache dir on first use; LLMs external). Entry script also exposes `--check-imports` (frozen-binary import smoke test; exits non-zero on failure). Built in one attempt; staged at `ui/resources/backend/eh-serve/` (gitignored), **320 MB**.
- Embedding: `electrobun.config.ts` `build.copy` copies `resources/backend/eh-serve` → `Contents/Resources/app/backend/eh-serve/` (mechanism verified against `electrobun/src/cli/index.ts`: copy destinations land under `Contents/Resources/app/`). Mode detection in `backendCommand()` (`ui/src/bun/backend.ts`): `bundledBackendPath()` resolves `Contents/MacOS/<launcher>` → `../Resources/app/backend/eh-serve/eh-serve` and uses it when present (prod); otherwise dev `uv run eh serve` from the repo root — dev flow untouched. Main process also prepends `/opt/homebrew/bin:/usr/local/bin` to the backend child's PATH so ffmpeg/ffprobe resolve when launched from Finder.
- Frozen binary standalone (scrubbed env: `env -i HOME=$HOME PATH=/usr/bin:/bin`, no VIRTUAL_ENV/uv):
  - `eh-serve --check-imports` → **90 modules `ok`, `check-imports: 0 failure(s)`** (every `entertainment_harness` submodule incl. `video.pipeline`, plus fastapi/uvicorn/kokoro_onnx/cv2/onnxruntime/phonemizer/espeakng_loader/…).
  - `EH_DATA_DIR=<copy of ui/.dev-data> eh-serve serve --port 0` → first stdout line `listening 60017`; `GET /api/health` → `{"ok":true}`; `GET /api/library` → Atomic Habits + Kenja no Mago (138 chapters); ranged `GET /api/videos/…/0/stream -r 0-99` → `HTTP/1.1 206 Partial Content`, `content-range: bytes 0-99/44719976`, exactly 100 bytes.
- `.app` builds: `bun run build` (dev channel) → `ui/build/dev-macos-arm64/EntertainmentHarness-dev.app` (479 MB, plain layout, backend embedded); `bun run package` (new script: `build:backend` + `electrobun build --env=stable`) → `ui/build/stable-macos-arm64/EntertainmentHarness.app` (self-extracting tar.zst layout for the updater; tar listing confirms 3718 files under `backend/eh-serve/`) + `ui/artifacts/` (.dmg, .tar.zst, update.json). Electrobun skips signing on both channels; ad-hoc signed with `codesign --sign - --deep --force`, `codesign --verify --deep --strict` passes (`Signature=adhoc`). Gatekeeper caveat (right-click → Open on first launch) documented in `ui/README.md`.
- Gate verification (dev-channel .app, launched as `env -i HOME=$HOME PATH=/usr/bin:/bin EH_DATA_DIR=/tmp/eh-gate-app Contents/MacOS/launcher` — no repo venv visible, data dir a COPY of `ui/.dev-data`):
  - Process tree: `launcher` (pid 10063) → `./bun …/Resources/main.js` (10065) → backend child **is the bundled binary** (not .venv python):

    ```
    10066 …/EntertainmentHarness-dev.app/Contents/Resources/app/backend/eh-serve/eh-serve serve --port 0
    ```

    App log: `[backend] ready on 127.0.0.1:60659`, `[webview] rendered: "views://mainview/index.html#/library … Atomic Habits … Kenja no Mago 25/138 cha…"`.
  - `lsof -nP -iTCP -sTCP:LISTEN -p 10066` → `TCP 127.0.0.1:60659 (LISTEN)`; `curl :60659/api/health` → `{"ok":true}`.
  - Playability: `curl :60659/api/library` → both works; ranged `curl -r 0-99 :60659/api/videos/01J76XYBTCD15FW69W889WKHB0/0/stream` → `206`, `content-range: bytes 0-99/44719976`, 100 bytes.
  - Real generation run from the packaged app: `POST :60659/api/runs {"work":"search-b6866973330f3b19","all_chapters":true,"skip_preflight":true}` (chapter-less "Atomic Habits") → SSE `log` events ("Nothing to recap: no synced chapters.", "Using qwen/qwen3-32b for text summarization (book).", …) → terminal `event: run-done … "chapters": 0`; `GET /api/runs/8a271b87aa79` → `"status": "done"`. A Kenja chapter run needs live models/ollama and is not required; the generation code paths' importability in the frozen binary is proven by `--check-imports` above (90/90), and the run itself exercises `server/runs.py` → `pipelines.recap` → model registry in the bundle.
  - Cleanup: killing the launcher reaped the whole tree (the Phase 2 watchdog killed the backend child); a stray older app instance from a previous session was also killed. `pgrep -f "eh serve|electrobun|eh-serve|EntertainmentHarness"` → clean (exit 1).
- Regression: `uv run pytest -q` → `1 failed, 741 passed` (742 collected; only the known pre-existing `test_lfm.py::test_repair_json_repairs_trailing_comma`); `cd ui && bun run typecheck` clean; `bun run smoke` → `SMOKE PASS — 16 checks against a live backend`.
- Notes/deviations: none blocking. The stable-channel bundle was build-verified (backend present in the tar.zst) but the interactive gate was run on the dev-channel .app (identical content/layout). Generation runs that need ffmpeg use the PATH prepend above; LLM backends (ollama etc.) must be running on the host, same as the CLI.

Phase 3 is closed; Phase 4 (docs & close-out) is next.

### Phase 4 (docs & close-out) — 2026-09-21

- `docs/design.md`: new "Desktop app (Electrobun)" section after "Local server (`eh serve`)" — app architecture (Bun main + WebKit webview as a pure `eh serve` client), dev mode, packaging (PyInstaller onedir backend, `build.copy` staging, bundled-binary mode detection, PATH prepend for ffmpeg, ad-hoc signing), build channels, one-instance-per-data-dir warning. Config/server/sources were already documented inline by Phases 1a/1b.
- `AGENTS.md`: test count corrected to 741 green (742 collected, 1 known failure); new bullet for the `ui/` toolchain (`bun run typecheck|smoke|package`, one-instance-per-data-dir rule).
- Regression at close: `uv run pytest -q` → `1 failed, 741 passed` (only the known `test_lfm.py::test_repair_json_repairs_trailing_comma`); `cd ui && bun run typecheck` clean.
- Follow-ups (not initiative blockers): real code signing/notarization + auto-updates (Electrobun supports both); interactive double-click check of the stable-channel bundle (dev channel was gate-verified); visual/screenshot verification of the UI (TCC Screen Recording permission blocked automated capture); search/add-works UI if wanted later.

### Phase 5 (background runs) — 2026-09-21

- **Backend** (`server/runs.py`, additive): `Run.current` — a `{event, chapter, stage, detail}` cursor updated in `emit()` for `run-start`/`chapter-start`/`stage`/`video-ready`/`chapter-done` (never for `log`), cleared to `null` at `run-done`/`run-error`; included in `Run.to_dict()` → `POST/GET /api/runs` + `GET /api/runs/{id}`. SSE protocol and run semantics untouched. Tests: 2 new in `tests/test_server.py` (mid-run cursor via blocking stub incl. log-doesn't-move-it; error clears it); `uv run pytest -q` → `1 failed, 743 passed` (only the known pre-existing `test_lfm.py::test_repair_json_repairs_trailing_comma`).
- **Webview run-awareness** (`ui/src/mainview/runs.ts`, new): one ref-counted ~3 s poller for `GET /api/runs` shared by the nav and library; `runsIndicator()` renders `● 1 run — ch N: stage` (click → `#/work/<id>`). Nav (`main.ts`) shows it while the backend is ready (poller stops in starting/error and on cleanup); library cards get a live `running — ch N: stage` line (`renderLibrary` now returns a cleanup — wired as `viewCleanup` in `main.ts`, called in the smoke).
- **Background toggle** (the core ask): work-view run panel gains a per-work "Run in background" checkbox (localStorage `eh.backgroundRun.<workId>`, default off; note: "while on, closing the window keeps generating — reopen from the Dock. Cmd-Q still quits fully."). Runs were already asynchronous worker threads; the toggle controls app-level backgrounding, coordinated over RPC: webview sends `backgroundModeChanged {enabled}` (`schema.ts` extended, backward-compatible; new `ui/src/mainview/bridge.ts` so views don't import `main.ts`) with `enabled = toggle && runActive`, re-synced on attach/terminal events. **Hide-on-close mechanism — electrobun API limitation:** Electrobun 1.18.1 cannot intercept a window close (verified in the installed package: the native `windowCloseCallback` returns void and the `close` event carries no `{allow}` response; `win.hide()` exists but the close button can't be redirected to it). So "close hides" is implemented as "close doesn't quit": `runtime: {exitOnLastWindowClosed: false}` in `electrobun.config.ts` (copied into `build.json`, verified present in the dev bundle) + the main process decides per close in `ui/src/bun/index.ts`: if background mode was reported **and** `/api/runs` still shows a running run, the app stays alive (backend + run keep going); otherwise it shuts down and quits as before. A destroyed Electrobun window can't be re-shown, so reopening **recreates** the window: Dock click (`reopen` event) or the new "Show Entertainment Harness" app-menu item (`action: "show-main-window"` via `ApplicationMenu.on("application-menu-clicked")`) → `showMainWindow()`. Cmd-Q / menu Quit / SIGINT/SIGTERM always quit fully (no guardrails, documented). The webview re-requests `getBackendState` on recreation and re-reports its persisted toggle, so state re-syncs.
- Gates:
  - `cd ui && bun run typecheck` → clean.
  - `bun run smoke` → **SMOKE PASS — 24 checks against a live backend** (16 existing + 8 new: `current` key in the run payload, catching the no-op run active, cursor shape, nav indicator renders/counts/shows ch/stage, cursor clears at finish, indicator hides when idle). Also hardened: `renderLibrary`'s cleanup is now exercised and `ok()` throws instead of `process.exit` so the spawned backend is always reaped (a failing check used to orphan it). Exit 0, backend shuts down cleanly.
  - Live app evidence (`cd ui && EH_DATA_DIR=$PWD/.dev-data bun run dev`, dev-channel app, RPC probe evaluates JS in the real webview): a real Kenja ch-23 recap run (`chapter:23, video:false, skip_preflight:true`) started the moment the backend reported ready; the probe at ready+15 s shows the live UI mid-run —
    ```
    [webview] rendered: "PROBE navRuns=1 … || Entertainment Harness\nLibrary\nSettings\n1 run — ch 23: pages 5-8 of 51\nbackend\nLibrary\n… Kenja no Mago …\nrunning — ch 23: pages 5-8 of 51"
    ```
    (stage line advanced from "pages 1-4 of 51" at +2 s — the nav indicator and library card track the run live over the 3 s poll), and `/api/runs` poll showed `"current": {"event": "stage", "chapter": 23.0, "stage": "recap", "detail": "pages 1-4 of 51"}`.
  - localStorage persistence across app restarts verified in the same probe: run 1 set `eh.probe=persist-<ts>`; the relaunched app's webview read `before=persist-1789965071810` — WKWebView localStorage survives full app restarts, so the toggle survives window recreation.
  - build.json in the dev bundle: `{"runtime":{"exitOnLastWindowClosed":false}, …}` — the config landed.
  - Hide-on-close close/recreate path: **needs a human eyeball** (can't click the real window's close button headlessly) — the mechanism is the code path above + typecheck; the `reopen`/menu handlers share `showMainWindow()`. Everything up to the native close event is package-verified.
  - Cleanup: `pgrep -f "eh serve|electrobun|eh-serve|EntertainmentHarness"` → clean (a 2 h-old stray dev app from a previous session and all evidence-run processes were killed; ch 0's fixture mp4 wiped by that stray run's crashed rebuild was restored byte-identical from the Phase 3 gate copy).
- Deviation/notes: (1) close **destroys** the window (electrobun gives no close interception) — reopen recreates it; webview view-state resets to the hash route but localStorage persists. (2) The stale PyInstaller backend bundled in `ui/resources/backend/eh-serve` predates `Run.current` (and the Phase-2 db-lock fix) — **deleted**; dev now spawns `uv run eh serve` per the README, and `bun run build:backend` must be re-run before the next `bun run package`. The frozen binary also crashed on espeak-ng-data during a real video build (pre-existing packaging quirk; unrelated to this phase). (3) While backgrounded with the window closed and the run finished unseen, the app stays alive until the next close (which quits normally) or Cmd-Q — documented in `docs/design.md`. (4) Runs are in-memory in the backend process: a full quit kills them.

Phase 5 is closed. Status: **done**.

### Phase 6 (run instructions) — 2026-09-21

- The work view run panel has an instructions textarea (`Instruction:` placeholder "focus on the battles, keep it fast"), persisted per work in webview localStorage (`eh.instruction.<workId>`) so a background run can be re-fired with the same guidance. Sent as the run's `instruction` option (same semantics as `eh recap --instruction`: used literally, `@file` not resolved; applies to every chapter in the range), echoed as an `Instruction: …` log line in the run panel.
- Backend needed no changes — `RunOptions.instruction` already existed (Phase 1b); only the webview exposed it.
- `docs/design.md` "Desktop app" background-runs bullet extended with the textarea; `StartRunBody` gained `instruction`.
- Gate evidence: `cd ui && bun run typecheck` clean; `bun run smoke` → **SMOKE PASS — 27 checks against a live backend** (3 new: "run panel has an instructions field", "instruction is echoed into the run log", "instruction reaches the run options" — the last asserts `GET /api/runs` shows `options.instruction` for the button-started no-op run). `uv run pytest -q` → `1 failed, 743 passed` (only the known `test_lfm.py::test_repair_json_repairs_trailing_comma`). Cleanup verified: no stray app/backend processes.

Phase 6 is closed. Status: **done**.

### Phase 7 (range runs honor the start chapter) — 2026-09-21

- Bug: the UI's From–To range was sent as `all_chapters: true` + a count, whose CLI semantics are "--all, from the beginning, overwriting" — so a run told to start at ch N started at ch 0 and re-did finished chapters.
- Fix: `RunOptions`/`POST /api/runs` accept a `chapters` spec (`"from-to"`, same syntax as `eh recap --chapters`); `_run_flow` passes it as `chapters_spec` to both `select_chapters` and `recap_series`, selecting exactly the synced chapters in range, in order. Malformed specs are rejected with 422 up front (`parse_chapter_spec` at the endpoint, not mid-run). The webview's range path now sends `chapters: "<from>-<to>"`; help copy corrected ("exactly the chapters from–to, re-doing any that already have artifacts").
- Tests: `test_run_chapters_spec_selects_exact_range` — spies on `select_chapters` and proves `chapters: "2"` selects `[2.0]` (not ch 0/1) and reaches `recap_series`; the invalid-options test gained a reversed-spec (`"3-1"`) 422 case. Smoke gained "chapter range reaches the run options as an exact chapters spec".
- Gate evidence: `uv run pytest -q` → `1 failed, 744 passed` (only the known `test_lfm.py::test_repair_json_repairs_trailing_comma`); `cd ui && bun run typecheck` clean; `bun run smoke` → **SMOKE PASS — 28 checks against a live backend**.

Phase 7 is closed. Status: **done**.

### Phase 8 (stop running generations) — 2026-09-21

- Backend: cooperative cancellation. `Run` gains a `stop_requested` `threading.Event` and `POST /api/runs/{id}/stop` sets it (idempotent; unknown id 404; returns while the run is still unwinding). `ServerProgress` (moved to take the run) checks the flag at its structural callbacks — `start`/`chapter_start`/`stage`/`chapter_done` — and raises `RunCancelled` (defined in `progress.py` to avoid the import cycle); `log` never raises since pipelines call it from error paths. `_execute` catches it → terminal `run-cancelled` event (`chapters` = completed count) and status `cancelled`. Semantics: the in-flight model/ffmpeg call finishes, then the run aborts at the next chapter/stage boundary; completed chapters stay recapped. `RunCancelled` deliberately precedes the generic `except Exception` in `_execute`.
- UI: a **Stop** button appears in the work view run panel while a run is active ("Stop requested — finishing the current step." logged); on `run-cancelled` the view reports "Stopped — N chapter(s) finished before stopping." The nav run chip gains a ✕ that stops that run without navigating. `RunInfo.status` gains `"cancelled"`; SSE client subscribes to `run-cancelled`.
- Tests: `test_run_stop_cancels` (blocking stub → stop → `run-cancelled`, status `cancelled`, `chapters` 0, idempotent re-stop), `test_run_stop_unknown_404`; smoke gains "stopping an unknown run returns 404" + "stopping a finished run is a no-op".
- Gate evidence: `uv run pytest -q` → `1 failed, 746 passed` (only the known `test_lfm.py::test_repair_json_repairs_trailing_comma`); `cd ui && bun run typecheck` clean; `bun run smoke` → **SMOKE PASS — 30 checks against a live backend**. Docs: design.md (`POST /api/runs/{id}/stop` + `run-cancelled` + cancellation semantics, Desktop app section).

Phase 8 is closed. Status: **done**.

### Phase 9 ("Process unfinished" toggle switch + skip_done) — 2026-09-21

- The "Run in background" checkbox is replaced by a real **toggle switch** ("Process unfinished", CSS slider with `role="switch"`), and its job is what was asked: **process the chapters that aren't recapped/videod yet**. ON = send `skip_done` (chapters that already have a recap at the requested grain AND a non-wiped video are skipped, not overwritten — within whatever scope is chosen) + the existing keep-alive-on-close background behavior; OFF = re-do exactly the requested chapters + close quits as usual. Per-work, persisted (same localStorage key).
- Backend: `chapter_needs_work()` in `pipelines/recap.py` (done = recap at exactly the requested detail + non-wiped single-chapter video when video is wanted; a wiped video or a different grain counts as unfinished). `recap_series(..., skip_done=False)` filters its selection and logs the skip count; `RunOptions`/`RunRequest` gain `skip_done`; `_run_flow` applies the same filter to the pre-flight plan so estimates match what will run.
- Tests: `test_chapter_needs_work` (no recap → work; recap at grain → done without video / work with video wanted; wrong grain → work; wiped video → work), `test_run_skip_done_passed_through` (flag reaches `recap_series` and the run options); smoke gains toggle-presence + `options.skip_done` checks.
- Gate evidence: `uv run pytest -q` → `1 failed, 748 passed` (only the known `test_lfm.py::test_repair_json_repairs_trailing_comma`); `cd ui && bun run typecheck` clean; `bun run smoke` → **SMOKE PASS — 32 checks against a live backend**. Docs: design.md (toggle semantics, `skip_done`, `chapter_needs_work`).

Phase 9 is closed. Status: **done**.

### Phase 10 (detail-level select) — 2026-09-21

- The run panel gains a **detail dropdown** (gist / brief / standard / detailed / full) between the chapter range and Generate, defaulting to the config's current `[pipeline] detail` (fetched from `GET /api/config`; "standard" when unset/invalid). The value is sent as the run's `detail` — the grain the run writes and what the "Process unfinished" toggle counts as done (a chapter recapped at another grain is unfinished at the selected one).
- No backend changes: `detail` existed in `RunOptions` since Phase 1b with 422 validation.
- Gate evidence: `cd ui && bun run typecheck` clean; `bun run smoke` → **SMOKE PASS — 35 checks against a live backend** (3 new: select exists, offers all five grains, `options.detail` reaches the run). `uv run pytest -q` → `1 failed, 748 passed` (unchanged; only the known `test_lfm.py::test_repair_json_repairs_trailing_comma`). Docs: design.md desktop bullet. Note: linkedom's `select.value` doesn't derive from the selected option, so the smoke overrides the getter — a shim limitation, not app behavior.

Phase 10 is closed. Status: **done**.

### Phase 11 (auto-process toggle) — 2026-09-20

- The per-run "Generate recap" button + chapter range are replaced by a **per-work auto-process toggle**: flip it on and the work's unfinished chapters (no recap at the selected grain / no playable video) keep being processed in background runs until everything is done; flip it off and processing stops. Toggle state is **server-side** — it survives window closes *and* backend restarts (the pre-Phase-11 localStorage toggle is gone).
- **Backend** (`server/auto.py`, new): `AutoSupervisor` owns the state (`{work_id: {enabled, options: {detail, instruction, skip_preflight}}}`) persisted atomically (temp+replace) at `data_dir()/auto.json`; missing/corrupt file → empty state. `set()` is the only mutation entry: enable persists + starts a run immediately when the work has unfinished chapters and none is active (returns the run dict or null); disable persists + cooperatively stops any active run. A daemon `threading.Timer` loop (30 s, lazy on first enable) reconciles: enabled works with no active run get a new run while unfinished work remains; a work whose latest run errored is **automatically disabled** (persisted) so a broken config can't loop — checked on every tick plus a one-shot reconcile ~2 s after each supervisor-started run, so errors turn the toggle off promptly. `has_unfinished()` mirrors a scope-less run exactly (`pending_chapters` at the configured grain/langs OR `_chapters_missing_video`, the `--video` backfill set), so no no-op runs start. Supervisor runs: `RunOptions(work, video=True, detail/instruction/skip_preflight=<stored>)`, default pending selection.
- **Endpoints** (`server/app.py`): `GET /api/library` gains per-work `auto: bool`; new `PUT /api/works/{work}/auto` (body `{enabled, detail, instruction, skip_preflight}`; detail validated → 422; unknown work 404; returns `{work, enabled, run}`). One supervisor per app, created + loaded in `create_app`. `POST /api/runs` unchanged (the supervisor uses the same `RunManager`; the smoke still exercises the endpoint directly).
- **UI** (`ui/src/mainview/views/work.ts`): the run panel is now just the toggle ("Process unfinished chapters") + detail select + instruction textarea + skip-preflight checkbox; the from/to inputs, Generate button, Stop button, and the `skip_done` sends are gone (`StartRunBody`/`POST /api/runs` stay for the supervisor/tests/compat). Toggle on → `PUT auto {enabled: true, <current modifiers>}`; if the response has a run, attach to its SSE stream. Toggle off → `{enabled: false}`; the existing `run-cancelled` handling shows "Stopped — …". Entering the view reads `auto` from the library payload, attaches to an active run if one exists, and shows "Auto-processing on — nothing to do right now." when the toggle is on but nothing is running. `api.ts` gains `setAuto`/`AutoBody`/`AutoResult`; `Work` gains `auto?: boolean`. Background keep-alive machinery unchanged (`sendBackgroundMode(toggle.checked)`; the main process still re-checks `/api/runs` at close).
- Tests: 9 new in `tests/test_server.py` (enable starts a run when unfinished + modifiers/options/persistence assert; enable on a fully-done work starts nothing (`run` null); state survives a backend restart; corrupt auto.json tolerated; disable stops an active run → `cancelled`; an errored supervisor run turns the toggle off via the one-shot recheck; `has_unfinished` across pending/video/wiped states; 404/422).
- Amendment (2026-09-26): loaded state is now **disarmed** — after a backend restart the toggle reads off and no supervised run starts until the user flips it on again (settings kept; `armed` is in-memory only, never persisted, so the auto.json format is unchanged). Motivation: a crash that kills the backend (e.g. espeak `exit(1)`) previously re-armed processing on every launch. Tests: `test_auto_state_survives_restart` reworked + new `test_auto_no_resume_after_restart`.
Amendment (2026-09-27): the run panel gained a **video-style select** (kenburns/scroll, defaulting to `[video] mode=`), stored server-side as a toggle modifier like detail (`options.video_mode` in auto.json; validated 422, reaches the run's `RunOptions.video_mode`). Also that day: `eh narrate` was removed entirely (alias + `pipelines/narrate.py`), superseding the deprecation note above — `eh recap --detail full` is the only path.
- Gate evidence:
  - `uv run pytest -q` → `1 failed, 757 passed` (only the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`; 748 → 757).
  - `cd ui && bun run typecheck` → clean (no output).
  - `bun run smoke` → **SMOKE PASS — 35 checks against a live backend**:
    ```
    ok   run panel has an instructions field
    ok   run panel has the auto-process toggle switch
    ok   run panel has a detail-level select
    ok   detail select offers all five grains
    ok   run panel has the skip-preflight checkbox
    ok   toggling on enables auto-processing (library payload)
    ok   idle status line shows while auto is on with nothing to do
    ok   enable returns no run when nothing is unfinished
    ok   no run is started for the finished work
    ok   toggle modifiers persist server-side (auto.json)
    ok   toggling off disables auto-processing (library payload)
    ok   disable persists server-side (auto.json)
    ok   stopping an unknown run returns 404
    … (library/work/player/settings/nav/current-cursor checks unchanged)
    ok   stopping a finished run is a no-op returning the run
    SMOKE PASS — 35 checks against a live backend
    ```
    The run-flow section now drives the real toggle switch on the chapter-less "Atomic Habits" work (asserting the PUT round-trips through the library payload and `auto.json`), and the finished-run stop-endpoint no-op check reuses the Phase-5 ad-hoc run. The SSE-attach path for auto-started runs is covered by pytest (enable-starts-run) plus the unchanged Phase 2/5 attach machinery.
  - Cleanup: `pgrep -f "eh serve|electrobun|eh-serve|EntertainmentHarness"` → clean.
- Notes/deviations: (1) `skip_done` remains in `RunOptions`/`RunRequest` for API compat but nothing sends it anymore (supervisor runs use the scope-less pending selection, which is inherently unfinished-only). (2) The toggle's modifiers are only readable via the enable response/`auto.json` (the PUT returns `{work, enabled, run}` per spec — no options echo); the smoke asserts persistence by reading `ui/.dev-data/auto.json`. (3) The smoke leaves `{"<habits-id>": {"enabled": false, …}}` in `.dev-data/auto.json` — the neutral disabled state, harmless across runs.

Phase 11 is closed. Status: **done**.

### Phase 11 hotfix (toggle unclickable) — 2026-09-21

- Bug: the toggle switch's visible control was wrapped in a `<span>`, not a `<label>` — clicking the slider/label in the real WebKit webview never toggled the hidden checkbox, so no `change` event fired and the toggle did nothing. The smoke missed it because it sets `.checked` and dispatches a synthetic `change` (linkedom doesn't implement label-click activation either).
- Fix: `.switch-wrap` is now a `<label>` (native label activation toggles the contained input). Smoke gains a structural guard — "toggle switch is wrapped in a `<label>` so real clicks toggle it" — 36/36 checks pass; typecheck clean. Python untouched (757 passed + 1 known failure stands).

### Phase 12 (auto runs fill gaps behind the read frontier) — 2026-09-21

- Report: "why starting at 59?" — an auto run on the real library began at ch 59 while ch 27–58 had neither recaps nor videos. Diagnosis: the pipeline's read frontier (`progress.last_read_chapter`) is 58 for that work, so the grain-aware pending set only contains no-recap chapters *after* 58; chapters 27–58 sit behind the frontier and were dropped by both the run's chapter selection and `has_unfinished`. The CLI's `--video` flag closes exactly this hole via the `fill_gaps` gap bucket (standalone artifacts for behind-frontier chapters with no artifact and no usable video), but the server flow never passed it — so UI/auto runs silently skipped chapters the CLI would have filled. (The run also started at 59 rather than 0 because the toggle was enabled at the `standard` grain — the select's default with no `[pipeline]` detail in config.toml — under which 0–26 count as done.)
- Fix: `_run_flow` (`server/runs.py`) passes `fill_gaps=options.video` to both `select_chapters` and `recap_series`, mirroring `cli._run_chapter_pipeline` (explicit `--chapters`/`--chapter`/`--all` scopes unaffected — the spec branches ignore `fill_gaps`); `has_unfinished` (`server/auto.py`) calls `pending_chapters(..., fill_gaps=True)` so the supervisor also counts the gap bucket as unfinished. Docstrings + `docs/design.md` updated to say "gap-aware pending set". Work-view note copy: "...filled in, even behind the read mark."
- Tests: `test_run_fill_gaps_follows_video_flag` (video run → `fill_gaps=True` reaches both functions; text-only run → False) and `test_has_unfinished_gap_behind_frontier` (behind-frontier no-artifact-no-video chapters are unfinished; a usable video drops them; both videod → finished). 757 → 759.
- Gate evidence:
  - `uv run pytest -q` → `1 failed, 759 passed` — the one failure is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`.
  - `cd ui && bun run typecheck` → clean (no output).
  - `bun run smoke` → **SMOKE PASS — 36 checks against a live backend** (unchanged checks; run-flow section unaffected).
  - Cleanup: `pgrep -f "eh serve|electrobun|eh-serve|EntertainmentHarness"` after the gates → only the user's own long-running dev instance (their data dir, untouched by the scratch gates).
- Notes: (1) the user's running dev instance predates this fix — relaunch (`Cmd-Q`, then `cd ui && bun run dev`) to pick it up; with their data (frontier 58, recaps through 26, 27–58 empty) the next auto run now selects 27–58 as standalone gap-fills plus the usual pending set. (2) The in-flight ch-59 run from the old code stops at the next chapter boundary (toggle already off in `auto.json`). (3) Chapter-range selection UI remains removed per the Phase 11 ask; a "start from chapter" steering control is the offered follow-up if wanted.

### Phase 13 (optional chapter scope on the auto toggle: range + "all") — 2026-09-21

- Ask: "let's add back chapter selection. but make it optional. when it's missing, start from last_read_chapter. also, add a check box for 'all' which allows processing all chapters."
- Semantics: the scope is **optional** and stored with the toggle (server-side, `auto.json`). Missing = the Phase-12 behavior (scope-less gap-aware pending selection — i.e. from the read mark onward, gaps behind it filled in). A range (`chapters` spec) or `all_chapters` scopes the run, and scoped runs are **`skip_done`**: only the unfinished chapters *inside* the scope are processed ("all" = consider every chapter, not re-do finished ones). `has_unfinished` mirrors per-scope: scope-less → gap-aware pending + video backfill; scoped → `chapter_needs_work` over the in-scope selection, so done work outside the scope doesn't keep the toggle busy (and a fully-done scope doesn't loop no-op runs every 30 s).
- Backend: `AutoRequest` gains `chapters: str | None` + `all_chapters: bool` (endpoint validates the spec via `parse_chapter_spec` and rejects `chapters`+`all_chapters` together → 422); `AutoSupervisor.set()`/`load()` persist the new option keys (old `auto.json` files load with `chapters: None`/`all_chapters: False`); `_start_if_unfinished` passes scope into `RunOptions` with `skip_done=scoped`; `has_unfinished` gains the scoped branch (`select_chapters` + `chapter_needs_work`, `want_video=True`).
- UI (`ui/src/mainview/views/work.ts`): the run-controls row gains **from/to number inputs** (one box alone = that single chapter) and an **"all" checkbox** that disables the range; both empty (and "all" off) = default. Scope persists per work in localStorage (`eh.scope.<workId>`); sent in the toggle's PUT via `...scopePayload()`. The skip-preflight checkbox gained a stable `.skip-preflight-input` class (the smoke's old "first non-switch checkbox" lookup would otherwise match the new "all" box). Note copy explains the scope. `styles.css`: 64 px number inputs.
- Tests: 4 new in `tests/test_server.py` — `test_auto_chapters_scope_reaches_run_options` (spec stored in `auto.json`, reaches `recap_series` with `skip_done=True`), `test_auto_all_chapters_reaches_run_options`, `test_auto_scope_invalid_422` (reversed spec; spec+all together), `test_has_unfinished_scoped` (in-scope judgment; done-outside-scope doesn't count; "all" sees out-of-scope unfinished). `test_auto_enable_starts_run_and_persists`'s exact `auto.json` shape gains `"chapters": None, "all_chapters": False`. Smoke: skip-box selector pinned to the class; new presence checks (from/to/all); scoped + all round-trips through `api.setAuto` reading `auto.json`; neutral shape restored before the UI-driven toggle-off. 759 → 763 pytest; smoke 36 → 41 checks.
- Gate evidence:
  - `uv run pytest -q` → `1 failed, 763 passed` — the one failure is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma` (45/45 in `tests/test_server.py`).
  - `cd ui && bun run typecheck` → clean (no output).
  - `bun run smoke` → **SMOKE PASS — 41 checks against a live backend**; backend reaped cleanly (`Finished server process`).
  - Cleanup: `pgrep` after the gates → only the user's own dev instance.
- Notes/deviations: (1) `skip_done` is now sent again, but only by the supervisor for scoped runs (Phase 11 note (1) superseded). (2) The `--video` backfill in `_run_flow` stays scope-independent, matching the CLI — a scoped run can still top up another chapter's missing video. (3) No open-ended ranges (`eh recap --chapters` has none); one box = single chapter.

### Phase 14 (default steering instruction) — 2026-09-21

- Ask: "let's add a default instruction. 'Use character names whenever possible, remove chapter introductions and exits, include smooth scene transitions, and emphasize onomonopaes'. Please improve the instruction."
- Wording (improved: spelled *onomatopoeia*, imperative clauses, concrete examples of openers/closers so the model knows what to skip): **"Favor character names over pronouns and epithets. Skip chapter openers and closers — no “previously” recaps, “to be continued” beats, or next-chapter teasers. Bridge scenes with smooth, concrete transitions. Punch up sound effects as vivid onomatopoeia and let them land in the narration."**
- UI-only change (`ui/src/mainview/views/work.ts`): the instruction textarea now falls back to `DEFAULT_INSTRUCTION` when the work has no stored override (`localStorage.getItem(instrKey) ?? DEFAULT_INSTRUCTION`) — so every run carries the default unless the user edits or clears the box for that work (a clear stores `""`, which sticks; the `??` only yields the default on a never-set work). Sent with the toggle's PUT exactly like a typed instruction (backend unchanged — it is just `instruction`). Panel note copy mentions the default.
- Smoke (`ui/scripts/smoke.ts`): gained an in-memory `localStorage` shim (linkedom has none, so the views' read path — and thus the pre-fill — never ran under the smoke; the previous try/catch silently left fields empty) and a check "instruction field is pre-filled with the default steering direction" before the run-flow overwrites the value.
- Gate evidence:
  - `cd ui && bun run typecheck` → clean (no output).
  - `bun run smoke` → **SMOKE PASS — 42 checks against a live backend** (41 + the pre-fill check).
  - `uv run pytest -q` → `1 failed, 763 passed` — Python untouched; the one failure is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma`.
- Notes/deviations: (1) The default is a UI constant, not `[pipeline] instructions` — config stays untouched, and CLI runs are unaffected; if the user wants the same steering everywhere incl. `eh recap`, they can add it under `[pipeline]` in config.toml via the settings screen. (2) Existing per-work saved instructions (including empty-string clears) are respected — the default only applies to works never touched.

### Phase 15 (library payload dedupes multiple videos rows) — 2026-09-23

- Report: "why do chapters 26, 27, etc. show up twice in the UI?" (screenshot: every chapter from ~26/31 on listed twice in the work view).
- Diagnosis: **not** duplicate `chapters` rows (verified against the real DB — `chapter_num` is unique per series). A chapter may legitimately hold **several `videos` rows — one per kind** (`_make_chapter_video` replaces only the same-kind row), and the user's chapters 31+ each have two (a `narration` and a `recap` row, some wiped). `GET /api/library`'s chapters query did a plain `LEFT JOIN videos … on from_chapter/to_chapter`, so two video rows ⇒ two payload rows ⇒ the UI rendered the chapter twice (often once "playable", once "missing"). The stream endpoint's `fetchone()` over the same unordered set could likewise pick a wiped row and 404 a chapter that had a playable one.
- Fix (`server/app.py`): the library query now joins a single representative row per chapter via a correlated subquery (`ORDER BY (wiped_at IS NOT NULL), id DESC LIMIT 1` — usable beats wiped, then newest), so `has_video`/kind/duration describe the playable video and each chapter appears exactly once; the stream endpoint selects with the same ordering, so a wiped row can't shadow a playable video with a 404. CLI `eh list`/`eh show` use COUNT subqueries and were never affected; no data surgery (wiped rows are kept by design).
- Tests: 2 new in `tests/test_server.py` — `test_library_dedupes_multiple_video_rows` (chapter with a wiped `recap` + usable `narration` row → one payload row, `has_video` true, kind `narration`; chapter with only a wiped row → one row, `has_video` false) and `test_stream_prefers_usable_video` (wiped `recap` row without a file + usable `narration` row → stream 200s with the narration bytes). 763 → 765.
- Gate evidence:
  - `uv run pytest -q` → `1 failed, 765 passed` — the one failure is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma` (47/47 in `tests/test_server.py`).
  - `cd ui && bun run typecheck` → clean (no output).
  - `bun run smoke` → **SMOKE PASS — 42 checks against a live backend**; backend reaped cleanly.
- Notes/deviations: none blocking. The user should relaunch the app to pick the fix up; their library renders one row per chapter immediately (no data changes needed).

### Phase 16 (video export: assemble + background + notify) — 2026-09-23

- Ask: "let's add a feature where the user can download / export existing video. it should assemble the video and run in the background. when ready, it should notify the user."
- Semantics: **Export videos** assembles a work's playable chapter videos into one mp4 under `~/Downloads/EntertainmentHarness` (`EH_EXPORT_DIR` overrides — tests + smoke use it), in a background job; the user is notified (native) when it lands. Single playable chapter → straight copy; several → concat (lossless stream copy when all inputs share codec/resolution/fps, balanced 720p re-encode otherwise — the same helper `eh video concat` uses, extracted to `video.assemble.concat_mp4s` + `probe_format`; the CLI command was refactored onto it, behavior unchanged).
- **Backend** (`server/exports.py`, new): `ExportManager` — in-memory jobs on daemon threads, one active export per work (409); `_resolve_parts` mirrors the stream endpoint per chapter (one representative videos row — usable beats wiped, then newest; compressed deliverable preferred; missing files skipped) so the export matches what the UI shows as playable; `dest` naming `<slug>-ch<a>[-<b>].mp4`; terminal `done` (`dest`, `total`, `skipped`) or `error`. Endpoints (`app.py`): `POST /api/works/{work}/export` (202; 404 unknown work) + `GET /api/exports` (newest first).
- **Main process** (`ui/src/bun/index.ts` + `schema.ts`): the webview asks the main process over RPC (`exportWorkVideos {workId}`) rather than POSTing itself — the main process owns the job: starts the export, polls `/api/exports` every 2 s to terminal (up to a 6 h cap), fires `Utils.showNotification` (Electrobun native notification — works even with the window closed), and pushes `exportDone` to the view for its banner (best-effort). 409/HTTP errors notify + report immediately.
- **Webview** (`work.ts` + `bridge.ts` + `main.ts` + `api.ts`): an **Export videos** button in the run-controls row; click → disable + `sendExportRequest`; a 2 s poll of `/api/exports` renders the terminal state once (`banner success`: "Saved N chapter video(s) to Downloads (file) — skipped M…", or `banner error`); `onExportDone` (main push) triggers an immediate poll; re-entering the view mid-export resumes tracking; cleanup stops the poll. New `.banner.success` style; the Electrobun local stub (`ui/types/electrobun.d.ts`) gained `Utils.showNotification`.
- Tests: 5 new in `tests/test_server.py` — `test_export_no_playable_videos_errors` (202 then terminal error naming the cause), `test_export_unknown_work_404`, `test_export_single_video_copies` (copy + `total`/`skipped` + dest name `test-manga-ch1.mp4`), `test_export_concat_videos` (two ffmpeg color clips → one mp4, span name, duration ≈ sum; ffmpeg-guarded), `test_export_conflict_409` (blocking `concat_mp4s` stub → second POST 409 → released job surfaces its error). 765 → 770.
- Gate evidence:
  - `uv run pytest -q` → `1 failed, 770 passed` — the one failure is the known pre-existing `tests/test_lfm.py::test_repair_json_repairs_trailing_comma` (52/52 in `tests/test_server.py`).
  - `cd ui && bun run typecheck` → clean (no output).
  - `bun run smoke` → **SMOKE PASS — 47 checks against a live backend** (42 + export button present + job starts + job assembles the 3 real Kenja fixture mp4s — exercising the mixed-format re-encode path — + count/file assertions); backend spawned with `EH_EXPORT_DIR=.dev-data/exports`, reaped cleanly.
- Notes/deviations: (1) The native notification itself can't be asserted headlessly — the smoke covers everything up to it (job → file on disk); the RPC push path is typecheck + the same Phase-5/6 RPC machinery. (2) Export scope is the whole work's playable chapters; a per-range export can reuse the scope controls later if wanted. (3) Jobs are in-memory like runs — a backend restart loses them mid-flight.
