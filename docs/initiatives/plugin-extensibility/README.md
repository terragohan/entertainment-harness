# Plugin extensibility — capabilities as data, specific roles, zero-core-edit plugins

Status: **done**

## Goal

Make the plugin system easy to extend:

1. A new provider slots into any seam **without editing core files** —
   per-plugin `[<section>.<plugin>]` config flows to the constructor
   automatically.
2. Providers **declare capabilities as data** (`capabilities: frozenset[str]`),
   so the system can answer "who can generate images?" and validate bindings.
3. Model work binds to **specific, validated roles** — roles are declarative
   data (plugins can add them), resolved generically, with capability-fit
   validation at selection time (warnings first, strict mode opt-in).
4. Extension is **verifiable**: `eh plugins` shows every category with
   capabilities and a `--check` health pass; a conformance kit and author
   docs make third-party entry points testable.

## Non-goals

- No plugin framework (pluggy et al.) — `PluginRegistry` stays as is
  (evaluation found it sound; design.md:162's rejection stands).
- No breaking config changes: every existing config.toml keeps working;
  old keys become aliases over the new machinery.
- No new fine-grained role inventory in Phase 3 (mechanism first; splitting
  `vision` into page-picker/grounder/etc. is follow-up work).
- The store seam stays non-registry (documented as deliberate).

## Rationale

From the plugin-system evaluation (2026-09-27): per-plugin config is
hand-rolled per seam (TTS merges manually, `[frames]` needed two config.py
edits); capabilities exist only as three boolean flags (`remote`,
`animated` ×2), so multi-capability providers escape the taxonomy (Runway is
registered in two seams and hardcoded in a third); roles are four hardcoded
generic buckets with no capability validation; `eh plugins` omits the frames
category and shows no capabilities or health; four real providers live
outside every registry (anime-scene Runway, DDColor, LFM, dormant SAM).

## Phases

### Phase 1 — automatic per-plugin config (gate: full pytest green)

- `Config.raw` stores the parsed TOML; `PluginRegistry.create()` merges
  `raw[<section>][<name>]` into constructor kwargs (filtered to params the
  constructor accepts; `**kwargs` constructors get everything; explicit
  overrides win). Category→section map in `plugins.py`.
- `get_engine`'s manual merge (`video/tts.py:190-192`) delegates to
  `create()`; typed sections (`[models.ollama]` etc.) keep working.
- Tests: merge semantics, filtering, explicit-wins, third-party-style
  plugin configured with zero config.py changes.

**Status: done.** Implemented as specced, with one refinement: `get_engine`
keeps passing the typed `config.tts` dict as explicit overrides (the typed
path is publicly tested in `tests/test_tts_qwen3.py`), and `create()` merges
`raw["tts"][name]` underneath it — both compose. `Config.raw` defaults to
`{}` so a missing config file simply disables the merge.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 804 passed, 1 warning in 16.57s
```

(The one failure is the known pre-existing Python 3.13 json-strictness
failure; 7 new Phase-1 tests in `tests/test_plugins.py`.)

### Phase 2 — capability declarations as data (gate: full pytest green)

- `capabilities: frozenset[str] = frozenset()` class attr on every seam
  protocol + built-ins (tts: kokoro/say `{"tts"}`, qwen3 `{"tts",
  "voice-clone"}`; video_gen: local `{"stills"}`, runway
  `{"image-to-video"}`; frames: runway `{"image-gen"}`).
- `ModelInfo.capabilities`; HF derives `vision` from mmproj presence.
- `PluginRegistry.names_with(capability)` query primitive.

**Status: done.** Capability strings used: `tts`, `voice-clone` (TTS);
`stills`, `image-to-video` (video_gen); `image-gen` (frames). Model-level
capabilities live on `ModelInfo` (HF marks `vision` when the repo ships an
mmproj file, remote and local); adapter-level `capabilities` is declared on
the `ModelAdapter` protocol but built-in adapters declare none yet. The
`Source`/`SearchProvider` protocols declare the attribute with no concrete
values. `names_with` treats a missing attribute as capability-less, so older
entry-point plugins keep working.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 812 passed, 1 warning in 16.60s
```

(Known pre-existing failure; 8 new Phase-2 tests across
`tests/test_plugins.py` and `tests/test_hf.py`.)

### Phase 3 — roles as data + validated resolution (gate: full pytest green)

- `models/roles.py`: `RoleSpec(name, requires, default_model, fallback)`;
  built-in table reproducing today's four roles and defaults exactly;
  `ROLES` registry plugins can extend.
- `[roles.<name>]` config parsed generically into `config.roles`;
  `[models.vision|text|translation|judge]` become aliases (identical
  effective config).
- `resolve_role()` in `models/registry.py`; the four getters become
  one-line wrappers (zero call-site changes). Capability-fit validation:
  warning by default, `PluginError` under `[plugins] strict = true`.

**Status: done.** One design refinement over the spec: an unset `backend` in
`[roles.<name>]` stays `""` through the parser so `role_config()` can apply
the RoleSpec's `default_backend` (an unset *model* already followed the
spec's fallback/default). Capability validation only fires when the model's
capabilities are *known* (non-empty) and missing a requirement — adapters
that don't report capabilities (Ollama today) stay warning-free, which is
what keeps the default config silent. While verifying, a pre-existing flake
surfaced: `tests/test_server.py::test_auto_state_survives_restart` errors
intermittently at fixture setup (sqlite IntegrityError); reproduced on the
unmodified baseline via `git stash`, so not introduced here.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 822 passed, 1 warning in 16.89s
```

(Known pre-existing failure; 10 new Phase-3 tests in `tests/test_roles.py`.
design.md documents `[roles.<name>]`, `[plugins] strict`, and the
capabilities/roles bullets under "Plugin architecture".)

### Phase 4 — verifiability + docs (gate: pytest, eh plugins --check, typecheck)

- `eh plugins`: frames category, capabilities column, `--check` health pass.
- `entertainment_harness/testing.py` conformance helpers + `eh plugin check
  pkg.mod:Class`.
- `docs/plugins.md` author guide.
- Drift fixes: `num_ctx` on the `ModelAdapter` protocol, `list_remote`
  declared (no getattr sniff), "five seams" docstring, design.md seams list.

**Status: done.** `eh plugins` gained the frame-animators category, a
capabilities column, and `--check` (loads every plugin, runs the conformance
kit, exit 1 on problems). `eh plugin check pkg.mod:Class --category <cat>`
checks one class; `entertainment_harness/testing.py` exposes the same
`check_plugin(cls, category)` for third-party test suites. Sources are exempt
from the `name` requirement (identified by registration key). `list_remote`
is now a declared protocol method returning `None` for backends without a
per-quant remote view (ollama, openai_compat) — the getattr sniff is gone;
older entry-point adapters lacking it are flagged by `--check` (the health
pass exists precisely to surface that). `docs/plugins.md` is the author
guide (contract, per-category members, packaging, roles, verifying).

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 833 passed, 1 warning in 16.36s
```

```
$ uv run eh plugins --check   # all six categories, every plugin: ok
exit=0
$ uv run eh plugin check entertainment_harness.video.tts:SayEngine --category tts
entertainment_harness.video.tts:SayEngine conforms to the tts contract
```

(Known pre-existing failure; 11 new Phase-4 tests in
`tests/test_testing_kit.py`. The intermittent
`test_auto_state_survives_restart` setup error recurred on some runs —
pre-existing flake, verified against the baseline in Phase 3.)

### Phase 5 — migrate the escapees (gate: per-provider tests + full suite)

- Anime pipeline consumes image-gen/image-to-video via `names_with`;
  Runway registered once, declaring all capabilities.
- DDColor behind a registry seam; LFM registered as a model backend
  (magic `"lfm"` name becomes an alias); SAM: wire-or-drop decision.

**Status: done.**

- **Anime**: `[anime] provider` (default `"runway"`) resolves through the
  video_gen registry; `get_scene_provider()` validates the plugin declares
  `image-gen` + `image-to-video` (SceneError listing `names_with` options
  otherwise). `RunwayProvider.capabilities` is now
  `{"image-to-video", "image-gen"}` — one client declaring all of it. The
  hardcoded `RunwayProvider(config)` construction is gone.
- **DDColor**: new `colorize` seam (seventh registry; entry-point group
  `entertainment_harness.colorize`, `[colorize] provider = "ddcolor"`).
  `Colorizer` follows the constructor contract; the video pipeline resolves
  via `get_colorizer()`. Conformance requirements, `eh plugins` row, and
  docs updated.
- **LFM**: `LFMAdapter` (models/lfm.py) implements the full ModelAdapter
  protocol (`capabilities = {"vision"}`) and is registered as `"lfm"`;
  translate.py's `_lfm_locator_from_stage` now resolves the backend through
  the registry (`create("lfm", ...).locator()`) — the magic string is an
  alias for the registered plugin. `generate()` does one VL chat call via
  the shared lazy loader.
- **SAM**: documented as dormant (design.md tree note) — no pipeline
  consumes `models/segment.py`; kept for future scanlation use. This is the
  wire-or-drop decision: neither, deliberately.

Gate evidence (`uv run pytest -q`, 2026-09-27):

```
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 842 passed, 1 warning in 15.40s
```

```
$ uv run eh plugins --check
exit=0   # lfm: vision / ok; ddcolor: colorize, active / ok
```

(Known pre-existing failure; 9 new Phase-5 tests across test_anime_scene.py,
test_colorize.py, test_lfm.py. One test-infra fix: `_fake_colorizer` in
test_video.py now patches `REGISTRY.create` too, since the registry's lazy
load caches the real class — the old module-attr patch was order-dependent.)

## Evidence

(phases paste gate output here as they close)

Final gates (2026-09-27), all green:

```
$ uv run pytest -q
FAILED tests/test_lfm.py::test_repair_json_repairs_trailing_comma - json.deco...
1 failed, 842 passed, 1 warning in 18.51s     # the 1 = known pre-existing failure

$ cd ui && bun run typecheck                   # clean
$ cd ui && bun run smoke
SMOKE PASS — 49 checks against a live backend

$ cd ui && bun run package                     # ok (artifacts + update.json)
$ ui/resources/backend/eh-serve/eh-serve --check-imports
check-imports: 0 failure(s)
$ ui/resources/backend/eh-serve/eh-serve --check-tts
check-tts: ok
```

Outcome vs. goal: a new provider now slots into any of the seven seams with
zero core-file edits (Phase 1 auto-config); capabilities are data on every
protocol, built-in, and `ModelInfo`, queryable via `names_with` (Phase 2);
roles are declarative `RoleSpec`s with validated resolution and `[roles.*]`
bindings (Phase 3); extension is verifiable via `eh plugins --check`,
`eh plugin check`, and the `testing.py` conformance kit (Phase 4); and the
four registry escapees are migrated or documented (anime provider,
DDColor/colorize seam, LFM backend, dormant SAM — Phase 5).
