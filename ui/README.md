# Entertainment Harness desktop UI

Electrobun app (Bun main process + native WebKit webview) that fronts
`eh serve`: library browsing, in-app video playback, recap-run triggering
with live SSE progress, and a config.toml/sources editor.

**Background runs**: the work view has a per-work "Process unfinished
chapters" toggle (server-side, default off). While on and a run is active,
closing the window keeps the app + backend generating (reopen from the Dock
or the "Show Entertainment Harness" menu item); with the toggle off, closing
quits as usual. Processing never auto-resumes: after an app/backend restart
the toggle reads off until it is flipped on again (the scope/detail/
instruction settings are kept). Cmd-Q always quits fully. The nav and
library cards poll `GET /api/runs` (~3 s) and show active runs
("● 1 run — ch 12: narration") from the run's `current` cursor. Runs are
in-memory in the backend — a full quit kills them.

## Run (dev)

```sh
cd ui
bun install
EH_DATA_DIR=/path/to/scratch-data bun run dev
```

`bun run dev` builds the app (`build/dev-macos-arm64/`) and launches it.
The main process spawns `uv run eh serve --port 0` from the repo root,
reads the port from its first stdout line, waits for `/api/health`, then
hands the port to the webview over RPC. The backend child is killed on
quit (window close, Cmd-Q, SIGINT/SIGTERM).

### Environment variables

- `EH_DATA_DIR` — forwarded to the backend. **Always point this at a
  scratch copy of the data dir in dev**: the Settings view writes the real
  `config.toml` inside it.
- `EH_REPO_ROOT` — override repo-root detection (default: walk up from the
  working directory looking for `pyproject.toml` + `src/entertainment_harness`).

### Building without launching

```sh
bun run build        # electrobun build (dev environment)
bun run typecheck    # tsc --noEmit
bun run smoke        # headless UI smoke test against a real eh serve (uses .dev-data)
```

## Packaging (the double-clickable .app)

The packaged app embeds a PyInstaller-frozen build of the backend and spawns
it instead of `uv run eh serve` (mode detection: `backendCommand()` uses the
bundled binary when it exists inside the app bundle, dev-mode uv otherwise —
see `bundledBackendPath()` in `src/bun/backend.ts`).

```sh
bun run build:backend   # PyInstaller -> ui/resources/backend/eh-serve/ (gitignored)
bun run package         # build:backend + electrobun build --env=stable
```

- `packaging/eh-serve.spec` (repo root) builds an **onedir** binary exposing
  the full `eh` CLI; the app invokes `eh-serve serve --port 0`. It bundles
  the serve + default-generation deps (fastapi/uvicorn, kokoro-onnx +
  phonemizer + espeak-ng, onnxruntime, opencv) — ~320 MB. Model weights are
  NOT bundled: kokoro's onnx/voices files download to the cache dir on
  first use; LLM/vision models stay external (ollama / HF / HTTP endpoints).
  The optional torch/transformers/mlx-audio extras are excluded.
- Smoke-test the frozen binary with `eh-serve --check-imports` (imports every
  `entertainment_harness` submodule + the third-party serve/generation deps;
  exits non-zero on failure).
- `electrobun.config.ts` `build.copy` stages `resources/backend/eh-serve`
  into `Contents/Resources/app/backend/eh-serve/` inside the bundle.
- Artifacts: `build/<channel>-macos-arm64/*.app` (dev channel: plain layout;
  stable channel: self-extracting tar.zst bundle for the updater) and
  `artifacts/` (.dmg, .tar.zst, update.json). Note the stable launcher
  extracts the inner app on first run.
- **Code signing**: electrobun skips signing unless configured. Ad-hoc sign
  after building: `codesign --sign - --deep --force build/<channel>-macos-arm64/*.app`.
  Ad-hoc-signed apps are not notarized, so on first launch Gatekeeper may
  block the app — right-click → Open (or System Settings → Privacy &
  Security → "Open Anyway") the first time.
- Launched from Finder, PATH has no Homebrew — the main process prepends
  `/opt/homebrew/bin:/usr/local/bin` for the backend child so ffmpeg/ffprobe
  resolve during generation.

## Layout

```
ui/
├── electrobun.config.ts
├── src/
│   ├── shared/schema.ts      # RPC contract (bun <-> webview)
│   ├── bun/
│   │   ├── index.ts          # window, menu, lifecycle glue
│   │   └── backend.ts        # backendCommand() + spawn/health/kill
│   └── mainview/             # webview UI (dependency-free TS + DOM)
│       ├── index.html
│       ├── main.ts           # boot + hash router
│       ├── api.ts            # REST/SSE client for eh serve
│       ├── store.ts          # library cache
│       └── views/            # library, work (+run controls), player, settings
├── build/                    # electrobun output (gitignored)
└── resources/backend/        # PyInstaller staging for the .app (gitignored)
```

`backendCommand()` in `src/bun/backend.ts` picks the backend: the bundled
PyInstaller binary when present (packaged .app), `uv run eh serve` from the
repo checkout otherwise (dev).
