# Writing plugins

The harness has seven pluggable seams. A plugin is a Python class registered
under a name — either a built-in, or a third-party class discovered via
stdlib entry points. No core changes are needed to add one: not in
`plugins.py`, not in `config.py`.

| Seam | Category | Entry-point group | Config section |
|---|---|---|---|
| Model backends | `model_backend` | `entertainment_harness.model_backends` | `[models.<name>]` |
| Content sources | `sources` | `entertainment_harness.sources` | `[sources.<name>]` |
| TTS engines | `tts` | `entertainment_harness.tts` | `[tts.<name>]` |
| Video generators | `video_gen` | `entertainment_harness.video_gen` | `[video_gen.<name>]` |
| Frame animators | `frames` | `entertainment_harness.frames` | `[frames.<name>]` |
| Page colorizers | `colorize` | `entertainment_harness.colorize` | `[colorize.<name>]` |
| Search providers | `search` | `entertainment_harness.search` | `[search.<name>]` |

## The contract

Every plugin shares one constructor shape:

```python
class MyEngine:
    name = "myengine"                      # shown by `eh plugins`
    capabilities = frozenset({"tts"})      # what it can do (data, not code)

    def __init__(self, config: Config | None = None, model: str = "base") -> None:
        ...
```

- `config` is always passed. Every other parameter must have a default.
- The plugin's `[<section>.<name>]` config.toml table flows into the
  constructor automatically: keys matching named parameters become kwargs
  (unknown keys are ignored; a `**kwargs` constructor receives everything).
  Explicit overrides passed to `create()` win over the config table. This is
  how a plugin gets settings with zero `config.py` changes.
- Declare `capabilities` as data so the system can answer "who can generate
  images?" (`PluginRegistry.names_with("image-gen")`) and validate role
  bindings. Strings in use today: `tts`, `voice-clone`, `stills`,
  `image-to-video`, `image-gen`, `frame-sequence`, `vision`.
- Unknown plugin names raise `PluginError` (a `ValueError` subclass) listing
  the available plugins. Raise the seam's own error type for runtime
  failures (`VideoError` for tts/video_gen/frames, `ModelError` for
  model_backend).

Category-specific members (also enforced by the conformance kit):

- `model_backend`: `remote: bool` (True = hosted endpoint; skips local
  quant/budget selection), `supports()`, `ensure()`, `remove()`,
  `list_available()`, `list_remote()` (return `None` when the backend has no
  per-quant remote view), `generate(model, prompt, images=None, num_ctx=16384)`.
- `sources`: `search()`, `get_series()`, `chapters()`, `download_pages()`,
  `looks_like_id()` (identified by registration key, no `name` needed).
- `tts`: `default_voice: str`, `synthesize(text, voice, dest)`.
- `video_gen`: `animated: bool` (False = stills/Ken Burns fallback),
  `generate_segment(image, segment, duration, workdir)`.
- `frames`: `animated: bool`, `generate_frames(image, segment, duration, workdir)`.
  When assembled as the `[sequence] provider` (sequence video mode), the call
  also passes `critic=` — a drift-critic hook the animator may call per
  generated frame — so sequence-capable animators should accept that kwarg.
- `colorize`: `colorize_page(src, dest)`, `colorize_pages(pages, dest_dir, log)`.
- `search`: `search(query, max_results=10)`.

## Packaging

Register the class under the seam's entry-point group:

```toml
[project.entry-points."entertainment_harness.tts"]
myengine = "my_pkg.tts:MyEngine"
```

Discovery happens once per process; a broken entry point is skipped so
built-ins stay usable, and a built-in wins a name collision. `eh plugins`
marks entry-point plugins with their origin.

## Model roles

Model-bound work resolves through named roles (`models/roles.py`), not
hardcoded model names. A plugin can add a role:

```python
from entertainment_harness.models.roles import RoleSpec, register_role

register_role(RoleSpec(
    "page-picker",
    requires=frozenset({"vision"}),   # validated against ModelInfo.capabilities
    default_model="qwen3-vl:8b-instruct",
    default_backend="ollama",         # used when [roles.page-picker] omits backend
    fallback="",                      # role to use when left unconfigured
))
```

Users bind it with `[roles.page-picker]` in config.toml; the built-in
`[models.vision|text|translation|judge]` tables are aliases for the four
built-in roles (`[roles.<name>]` wins when both are set). Resolution goes
through `resolve_role(name, config, profile)`, which validates the pick
against `requires`: a model with *known* capabilities that miss a
requirement warns (or raises `PluginError` under `[plugins] strict = true`);
models whose capabilities are unknown pass silently.

## Verifying

```console
$ eh plugin check my_pkg.tts:MyEngine --category tts
my_pkg.tts:MyEngine conforms to the tts contract

$ eh plugins --check     # every plugin in every category, health column
```

Or from the plugin's own test suite:

```python
from entertainment_harness.testing import check_plugin

def test_conformance():
    assert check_plugin(MyEngine, category="tts") == []
```
