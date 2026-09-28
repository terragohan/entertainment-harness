# Entertainment Harness — Design

> Docs: this file (architecture/schema/rationale) · `plan.md` (executed build phases) · `initiatives/` (tracked multi-phase work).

## Goal

A Python CLI "entertainment harness": media collection manager + content aggregator + content generator.

**v1 scope:**

- Content source: **fetch manga from MangaDex** via its public REST API (`api.mangadex.org`).
- Recap generation: **local inference by default — no fine-tuning.** A locally-run vision model reads chapter pages and summarizes. Models are swappable via config (see "Model adapter architecture"), including opt-in remote backends (OpenRouter, vLLM pods) via `openai_compat` — see "Remote inference".
- Video generation: **automatic chapter-recap videos** — narration over real manga pages with Ken Burns motion (see "Video pipeline").
- Interface: **CLI**.
- Granularity: **chapter-by-chapter** recaps with rolling context ("story so far" per series).
- Data: local files for chapter images; **SQLite** for library state, progress, and recaps.

## Hardware & local models

Target machine: **Apple M4 Pro, 24 GB unified memory** (arm64). Safe inference budget: ~16 GB (OS + harness need headroom; unified memory is shared with the GPU).

### Quantization: what fits

Model file size ≈ RAM needed for weights; add ~10–20% for runtime + KV cache (grows with context length and image count). Approximate sizes:

| Params | FP16 | Q8_0 | Q6_K | Q4_K_M | Verdict on 24 GB |
|---|---|---|---|---|---|
| 4B | 8 GB | 4.5 GB | 3.5 GB | 2.5 GB | any quant |
| 8B | 16 GB | 8.5 GB | 6.5 GB | 5 GB | any quant |
| 12B | 24 GB | 13 GB | 10 GB | 7.5 GB | Q8 fits; Q4 leaves room for a second model |
| 27–32B | — | 30+ GB | — | 16–20 GB | Q4 borderline — swap risk; warn user |
| 72B | — | — | — | ~45 GB | no |

Quantization unlocks the 12B tier comfortably (Q8 for near-lossless quality, Q4 to run vision+text concurrently) and makes 27–32B Q4 *possible but borderline* — the harness should detect this and warn rather than silently swapping.

### Vision models (page-reading role)

| Model | Ollama tag | Default quant size | Architecture notes |
|---|---|---|---|
| **Qwen3-VL 8B** | `qwen3-vl:8b-instruct` | ~5–8.5 GB (Q4–Q8) | **Default.** 2025 arch: dynamic resolution up to 4096×4096 (no downscaling tall manga pages), best-in-class CJK/multilingual OCR |
| Qwen3-VL 8B Thinking | `qwen3-vl:8b-thinking` | ~5–8.5 GB | Chain-of-thought variant — slower, more accurate on hard pages |
| Qwen3-VL 4B | `qwen3-vl:4b` | ~2.5–4.5 GB | Lightest of the new arch; fastest iteration |
| Qwen3-VL 32B | `qwen3-vl:32b` (Q4) | ~19 GB | Borderline on 24 GB — only with quantization warning |
| MiniCPM-V 4.5 | `openbmb/minicpm-v4.5` | ~5–8.5 GB | 8B, Qwen3-8B + SigLIP2 backbone, unified 3D-resampler (¼ the visual tokens of peers); OCRBench above GPT-4o |
| Gemma 4 12B | `gemma4:12b` | ~7.5–13 GB | Strong multimodal all-rounder; Q8 fits on 24 GB |
| Llama 3.2 Vision 11B | `llama3.2-vision:11b` | ~7.5 GB | 2024 arch; still good, superseded by Qwen3-VL for CJK/OCR |
| bakllava / gemma4:e2b | — | 4.7 / 7.2 GB | Already pulled on this machine — fallback/smoke-test models |

Not suitable for the vision role: pure OCR models (LightOnOCR 1B, PaddleOCR-VL) — they extract text but don't understand narrative, panels, or art.

### Text models (rolling-summary / scriptwriting role, optional)

| Model | Ollama tag | Size (Q4–Q8) | Notes |
|---|---|---|---|
| Qwen3 4B | `qwen3:4b` | 2.5–4.5 GB | Cheap, good instruction following |
| Gemma 4 (small) | `gemma4:e2b` | 7.2 GB | Already pulled; usable immediately |
| Qwen3 8B | `qwen3:8b` | 5–8.5 GB | Better quality if RAM allows |

Both roles can be served by one vision model to start; splitting them is a config change, not a code change. When two models are loaded concurrently, their combined footprint must stay under the safe budget — the quantization policy below handles this.

## Hardware detection & quantization policy

The harness probes the machine at startup and picks quants deliberately, never silently.

### `hardware.py`

- Probe total RAM (`sysctl hw.memsize` on macOS, `psutil` elsewhere), GPU backend (Metal on Apple Silicon, CUDA via `nvidia-smi`, else CPU), and currently available memory.
- Derive a **safe budget**: ~65% of total unified memory on macOS (24 GB → ~16 GB); on discrete-GPU systems, use VRAM when the backend is CUDA, else the same RAM rule.
- Expose `HardwareProfile(total_ram, budget, gpu_backend)` consumed by the registry.

### Quantization selection (in `registry.py`)

`ModelInfo` carries `params`, `quant`, and `size_bytes` per tag (Ollama and HF both report file sizes). Given a requested model:

1. List available quants for the model from the adapter's registry.
2. Preference order: `Q8_0 → Q6_K → Q4_K_M → Q4_0` (quality-first) or reversed if config sets `quant_policy = "prefer-speed"`.
3. Pick the highest-preference quant whose size fits `budget - currently_loaded`. Config `quant = "q8_0"` pins an explicit quant and skips auto-selection.
4. **Never silently downgrade**: if the pick falls below Q8, or the model only fits at Q4 with <2 GB headroom, emit a user-facing alert (below). If nothing fits, refuse with a clear message listing what *would* fit.

### Automatic quantization (optional, opt-in)

Quantization here is **post-training quantization (PTQ)** — a weight conversion, not fine-tuning: no training data, no gradient updates. Two paths:

1. **Pre-quantized artifacts (default).** Registries already publish quants: the Ollama library exposes per-quant tags, and HF Hub hosts GGUF repos (often imatrix-calibrated, e.g. bartowski's). The registry picks and `ensure()` downloads one. This covers virtually all popular models and always beats local naive quantization on quality.
2. **Local quantization (fallback, user-confirmed).** When a model has no published quant, the harness *can* quantize automatically: download the FP16/BF16 GGUF, then run llama.cpp's `llama-quantize` (optionally with an imatrix calibration file for better Q4 quality). Costs the user must accept first: the FP16 source is huge on disk (32B ≈ 65 GB) plus the output file; conversion takes minutes-to-hours on a laptop; and for VLMs only the language-model weights are quantized — the vision encoder stays at higher precision. Never triggered silently: the harness proposes it, the user confirms.

Policy: always prefer a published quant; quantize locally only when none exists and the user approves.

### User alerts

- `eh models` shows each model with quant, size, and a fit verdict (`fits`, `tight`, `too large`) against the detected budget.
- `eh recap` prints the resolved model line before generating, e.g.:
  - `Using qwen3-vl:8b-instruct (q8_0, 8.5 GB) — full quality.`
  - `Warning: qwen3-vl:32b resolved to q4_k_m (19 GB). Quantized below 8-bit — OCR accuracy on small text may degrade, and this is tight on 24 GB RAM.`
- The full model tag (including quant) is stored in `recaps.model`, so recap quality is attributable to the exact artifact that produced it.

### Pre-flight resource guards (`preflight.py`)

Heavy commands (`eh recap`, `eh tiktok`, `eh quantize`) build a `ResourcePlan` (models, peak memory, disk needs) and check it against a live `SystemSnapshot` before starting work. Guards cover:

- **GPU**: local-model pipelines fail if no GPU/Metal/CUDA is detected (bypass with `--skip-preflight` or `[preflight] skip_gpu_check = true`).
- **Memory / VRAM**: on CUDA the plan is checked against free VRAM; elsewhere against free RAM. On Apple Silicon a warning is emitted when the plan exceeds 80% of the safe budget because unified memory is shared with the GPU.
- **Disk**: auto-estimated from uncached pages, translated overlays, video output, and model downloads. The check covers the data dir and the HF model cache dir. `[preflight] min_free_disk_gb` and `min_free_memory_gb` override the auto-derived values.
- **Dry-run**: `--dry-run` prints the plan and exits without running.

## Model adapter architecture

Inference is abstracted behind a small **adapter** interface. Each adapter wraps one runtime or model registry; the harness never calls Ollama, llama.cpp, or Hugging Face directly. Models are chosen by config, never hard-coded.

### Adapter interface (`models/base.py`)

```python
class ModelAdapter(Protocol):
    name: str
    remote: bool  # True = hosted endpoint; registry skips local quant/budget selection
    def supports(self, model: str) -> ModelInfo: ...       # resolve a model ID incl. per-quant sizes
    def ensure(self, model: str, quant: str | None = None) -> None: ...  # pull/download if missing
    def remove(self, model: str) -> None: ...              # delete the local copy (stays in the registry)
    def list_available(self) -> list[ModelInfo]: ...       # query local + remote registry
    def generate(self, model: str, prompt: str, images: list[Path] | None = None) -> str: ...
```

`ensure()` is what makes registries pluggable: an adapter knows how to fetch its own models (Ollama library, HF Hub GGUFs), so `eh recap` can auto-pull a missing model — at the quant the policy selected — instead of failing. `remove()` is the counterpart: local copies are disposable caches and can be blown away (`eh models rm`) when not in use.

### Adapters

```
src/entertainment_harness/models/
├── __init__.py
├── base.py            # ModelAdapter protocol, ModelInfo (params/quant/size), errors
├── registry.py        # config + HardwareProfile -> adapter + quant selection
├── ollama.py          # OllamaAdapter — local HTTP API; ensure() = ollama pull (default)
├── openai_compat.py   # OpenAICompatAdapter — any OpenAI-compatible endpoint (OpenRouter, vLLM, LM Studio)
├── huggingface.py     # HFAdapter — resolve/download GGUF from HF Hub, run via llama.cpp server
└── mlx.py             # optional later: MLXAdapter — mlx-vlm, direct Apple Silicon inference
```

### Evaluated options

1. **Hard-code Ollama calls** — simplest, but locks the harness to one runtime and one model family. Rejected.
2. **Adapter + registry (chosen)** — one protocol, one module per runtime/registry, config-driven selection. New runtimes are new modules; `ensure()` gives each adapter control over its own registry (Ollama library, HF Hub).
3. **`pluggy` / full plugin frameworks** — hook fan-out machinery for what is pick-one-per-category dispatch. Rejected; stdlib `importlib.metadata` entry-point discovery (~25 lines in `plugins.py`) covers third-party plugins without a framework.

### Plugin architecture (`plugins.py`)

All seven pluggable seams share one registry idiom (`PluginRegistry`): model backends (`models/registry.py`), sources (`sources/`), TTS engines (`video/tts.py`), video generators (`video/gen/`), frame animators (`video/frames.py`), page colorizers (`video/colorize.py`), and search providers (`search/`). Built-ins register as lazy dotted strings (`"pkg.mod:Class"` — no import until `create()`); third-party packages register via stdlib entry points, discovered once per process (a broken entry point is skipped, built-ins win name collisions):

```toml
[project.entry-points."entertainment_harness.tts"]
elevenlabs = "my_pkg.tts:ElevenLabsEngine"
```

Groups: `entertainment_harness.model_backends`, `.sources`, `.tts`, `.video_gen`, `.frames`, `.colorize`, `.search`. `eh plugins` lists every category with built-in/entry-point origins, declared capabilities, and the active plugin per current config (for sources: enabled/disabled per `[sources].enabled`); `eh plugins --check` loads every plugin and runs the conformance kit on it, and `eh plugin check pkg.mod:Class --category tts` checks one class (see `docs/plugins.md`).

Conventions shared by all categories:

- **Constructor contract**: `__init__(self, config: Config | None = None, **explicit_overrides)` — explicit kwargs win, else the plugin's config section, else `load_config()`.
- **Per-plugin config**: `[<category>.<plugin>]` sections (e.g. `[models.openai_compat]`, `[models.ollama]`, `[video_gen.runway]`).
- **Declared capability flags** on the protocols instead of `getattr` sniffing: `ModelAdapter.remote` (skips local quant/budget selection), `VideoGenProvider.animated` (generated clips vs. Ken-Burns stills). Registry reads still tolerate missing flags for older entry-point plugins.
- **Capabilities as data**: plugins declare `capabilities: frozenset[str]` (TTS: `tts`, `voice-clone`; video_gen: `stills`, `image-to-video`; frames: `image-gen`, `frame-sequence`), and models carry them per artifact on `ModelInfo` (HF marks `vision` when the repo ships an mmproj projector). `PluginRegistry.names_with(cap)` answers "who can generate images?"; missing attribute = capability-less, never an error.
- **Roles as data** (`models/roles.py`): model work binds to named roles — `RoleSpec(name, requires, default_model, fallback)` in the `ROLES` registry, extensible by plugins. `resolve_role()` in `models/registry.py` resolves a role's binding (`[roles.<name>]`, else the legacy `[models.<name>]` alias, else the spec default) and validates the pick against `requires`: known capability mismatches warn, or raise under `[plugins] strict = true` (unknown capabilities pass). The four `get_*_model()` getters are one-line wrappers, so call sites are unchanged.
- **Errors**: unknown names raise `PluginError` (a `ValueError` subclass) listing the available plugins — one taxonomy everywhere.

Deliberately not borrowed from DeepSeek Harness/Cordis: layered profile/patch config composition, dependency-graph plugin loading, event waterfalls/middleware, per-scope contexts — overkill at this scale.

### Config (`config.toml` in data dir, overridable per command)

```toml
[hardware]
budget_gb = 16          # optional override; default = auto-detected

[library]
langs = ["en"]          # preferred reading languages; first is the translation target
# To fetch a source not in your preferred list, use: eh sync <series> --lang pt-br

# [sources]             # content-source gating (managed by `eh sources`)
# enabled = ["mangadex", "weebcentral"]   # the default; a built-in source
#                       # absent from this list is disabled — get_client()
#                       # errors with a re-enable hint. Third-party
#                       # (entry-point) sources are always enabled, listed
#                       # or not, so existing plugin users are unaffected.

[models]
quant_policy = "prefer-quality"   # or "prefer-speed"

[models.vision]
backend = "ollama"
model = "qwen3-vl:8b-instruct"
# quant = "q8_0"        # optional pin; omit for auto-selection

[models.text]
backend = "ollama"
model = "qwen3:4b"

# [models.translation]         # optional; defaults to the text role
# model = "qwen3:8b"           # text model used for translating extracted bubble text

# [models.judge]               # optional; defaults to the text role
# model = "qwen3:8b"

# [roles.<name>]               # generic role bindings (models/roles.py): wins
# backend = "ollama"           # over the [models.vision|text|translation|judge]
# model = "..."                # aliases above, and can bind plugin-registered
#                              # roles. An unset backend inherits the role's
#                              # default; translation/judge fall back to text.

# [plugins]
# strict = false               # true: binding a role to a model that doesn't
#                              # declare the role's required capabilities is a
#                              # PluginError; false (default): warning only.
#                              # Models with unknown capabilities always pass.

# [models.openai_compat]       # remote OpenAI-compatible endpoint (see "Remote inference")
# base_url = "https://openrouter.ai/api/v1"
# api_key = "..."              # or OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY env

# [models.ollama]
# base_url = "http://localhost:11434"

# [pipeline]
# thinking = "medium"   # low (no judge) | medium | high (vision-verified judging)
# detail = "standard"   # artifact grain: gist | brief | standard | detailed | full
# instructions = ""     # steering direction injected into recap prompts + judges;
#                       # "@path/to/file.txt" reads the direction from a file
# characters = true     # per-work character registry: learn the cast from judged
#                       # artifacts, inject it as a CAST block into prose prompts
#                       # (see "Recap flow"); false opts out entirely

# [preflight]
# enabled = true               # set false to disable all guards by default
# min_free_disk_gb = 10        # override auto-derived disk need
# min_free_memory_gb = 8       # override auto-derived memory need
# skip_gpu_check = false       # allow local-model pipelines on CPU

[store]
provider = "hf"               # or "r2" (Cloudflare R2, S3-compatible)
repo = "owner/manga-archive"  # hf: dataset repo for video backup; auth via HF_TOKEN env

# [store.r2]                  # r2: bucket-backed video backup
# bucket = "manga-archive"
# account_id = "..."          # https://<account_id>.r2.cloudflarestorage.com
# access_key_id = "..."       # or R2_ACCESS_KEY_ID env var
# secret_access_key = "..."   # or R2_SECRET_ACCESS_KEY env var

[video]
tts_engine = "kokoro"   # or "say" (macOS fallback), "qwen3" (voice cloning)
voice = "af_heart"      # engine-specific voice id; for qwen3, an `eh voices` name
resolution = "1920x1080"
# mode = "kenburns"   # presentation style: kenburns (per-page pan/zoom),
#                     # scroll (descend a per-segment page strip), slideshow
#                     # (stills dissolving within a segment), cards (caption
#                     # cards — the only mode for books), panels (per-beat
#                     # panel crops), motion (generated clips, [video_gen]),
#                     # animate (AI-generated panel frames, [frames]),
#                     # sequence (panel outpainted to video size, AI frames
#                     # chained into motion, [sequence])
#                     # — see "Video pipeline"
colorize = false        # opt-in: DDColor-colorize B&W pages for videos
translated = false      # opt-in: use translated pages in videos
# steering_prompt = ""  # custom direction for tiktok scripts
# compress = ""         # default quality preset for every render: hd | balanced | small
# keep_master = true    # false: prune out.mp4 once a compressed copy exists
# panel_first = true    # default: script videos from cached panel beats (see
#                       # Video pipeline); false (or --no-panel-first per run)
#                       # opts out into the grounded assign+grounding path

# [tts.qwen3]                       # per-TTS-engine settings (qwen3 extra:
# model = "mlx-community/..."       #   uv sync --extra qwen3)
# language = "Auto"
# x_vector_only = false
# max_new_tokens = 2048

# [search]                # eh online-summary / eh tiktok web search
# provider = "duckduckgo"
# max_sources = 6

# [video_gen]             # tiktok clip generation backend
# provider = "local"      # or "runway" (needs [video_gen.runway] api_key or RUNWAY_API_KEY env)

# [frames]                # panel animation for [video] mode = "animate"
# provider = "local"      # or "runway" (gen4_image; Runway API key) /
#                         # "openrouter" (any image model on the OpenAI-compatible
#                         # endpoint, e.g. gpt-6-luna; [models.openai_compat] key)
# model = ""              # empty = the provider's default image model
# seconds_per_frame = 2.0 # generated-frame spacing within a beat
# max_frames = 6          # cost cap per beat

# [sequence]              # frame-by-frame animation for [video] mode = "sequence"
# provider = "sequence"   # gen4_image (Runway API key) or "openrouter-sequence"
#                         # (OpenAI-compatible endpoint, e.g. gpt-6-luna)
# model = ""              # empty = the provider's default image model
# fps = 1.5               # generated frames per second of a beat's slot
# max_frames = 12         # cost cap per beat
# interp_fps = 30         # minterpolate smoothing to playback fps
# critic = true           # vision-role drift critic on chained frames

# [anime]                 # eh anime-scene generation
# provider = "runway"     # a video_gen plugin declaring image-gen + image-to-video
# keyframe_model = "gen4_image"   # text_to_image
# video_model = "gen4.5"          # image_to_video (alt: seedance2_5)

# [colorize]              # page colorization for [video] colorize = true
# provider = "ddcolor"    # the colorize plugin registry (video/colorize.py)
```

Swapping models = editing config or passing `eh recap <series> --vision-model openbmb/minicpm-v4.5 --quant q8_0`. `eh models` lists configured models plus what each adapter reports locally and in its registry, annotated with fit verdicts.

Programmatic writes go through `save_config(path, updates)` in `config.py`: nested `{section: {key: value}}` updates are merged into the hand-edited file via tomlkit (comments and formatting preserved; nested dataclass tables merge key-by-key), validated against the `Config` dataclasses (unknown section/key or wrong type → `ConfigWriteError` before the file is touched) and round-tripped through the same parsing code `load_config` uses, then written atomically (temp file in the same dir + `os.replace`). `eh sources enable|disable|reset` is the first consumer; the desktop app's settings editor is the next.

### Remote inference (`openai_compat` backend)

Opt-in third backend for hosted open-weight APIs and self-hosted GPU pods — any OpenAI-compatible chat-completions endpoint (`models/openai_compat.py`). Remote models bypass the local quant/budget selection entirely (`remote = True` on the adapter; `quant` is ignored). Credentials come from `[models.openai_compat]` or env (`OPENAI_COMPAT_API_KEY`, `OPENROUTER_API_KEY`); localhost `base_url`s need no key (vLLM ignores the token). Failures surface as `ModelError` (never raw httpx/KeyError/JSONDecodeError), and `generate()` retries transient faults — network/timeout errors, 5xx, 408/409/425/429, and malformed HTTP-200 bodies (a gateway hiccup can return 200 with no `choices`) — with exponential backoff (2 s → 8 s → 30 s, 3 retries, `Retry-After` honored); other 4xx are permanent config/auth errors and fail fast.

**Cost-optimized default (OpenRouter):** at hobby scale (~50 chapters/mo) per-token APIs run **~$1–2/mo** — cheaper than any GPU rental (serverless ~$2–5/mo, rented pods ~$1–4/mo plus ops). Rental only wins past ~50–100M tokens/mo or for custom models.

The same credentials also serve **image generation** (`video/gen/openrouter.py`): chat-completions with `modalities: ["image", "text"]`, reference images as data-URL content parts, and the response normalized (center-crop + resize) to the requested ratio. The `openrouter` / `openrouter-sequence` frame animators (`[frames]` / `[sequence]` providers) use it for the animate and sequence video modes — `[frames] model` / `[sequence] model` pick the image model (any model whose architecture lists `image` in its **output** modalities — accepting image *input* is not enough, e.g. `gpt-6-luna` is `text+image->text` and is rejected; empty = the built-in default), so those modes no longer require a Runway key. An endpoint routing 404 (no image-output route for the model) raises `VideoConfigError` with guidance, aborting the render instead of degrading per anchor.

```toml
[models.openai_compat]
base_url = "https://openrouter.ai/api/v1"
# api_key = "..."   # or export OPENROUTER_API_KEY

[models.vision]
backend = "openai_compat"
model = "qwen/qwen2.5-vl-72b-instruct"   # best open VLM for JP manga OCR+translation; ~$0.25/$0.75 per 1M tok

[models.text]
backend = "openai_compat"
model = "qwen/qwen3-32b"                 # quality/cost sweet spot (~$0.08/$0.28); step up: meta-llama/llama-3.3-70b-instruct, qwen/qwen3-235b-a22b

# judge unset -> inherits the text role (a 32B-class model is plenty)
```

Cheaper vision fallback: `qwen/qwen2.5-vl-32b-instruct`. DeepInfra (`https://api.deepinfra.com/v1/openai`) is usually the price floor for 70B-class text.

**Self-host escape hatch:** rent a 4090 pod on Vast.ai (~$0.20–0.35/hr) or RunPod ($0.74/hr), run `vllm/vllm-openai` serving `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (~20 GB VRAM; AWQ/FP8 are fine for all three roles), and point `base_url` at the pod's `:8000/v1`. Pod lifecycle is manual — the harness only sees an endpoint.


## Architecture

```
entertainment-harness/
├── pyproject.toml            # uv-managed; deps: httpx, typer, rich, psutil, Pillow, kokoro-onnx, huggingface_hub, opencv-python-headless, fastapi, uvicorn
├── config.toml               # model/backend/quant/video selection (see above)
├── src/entertainment_harness/
│   ├── __init__.py
│   ├── cli/                  # Typer app; entry point `eh`
│   │   ├── __init__.py       # app + shared pipeline helpers; recap and narrate commands
│   │   ├── library.py        # search, chapters, add, remove, sync, list, rebuild-index, show, import
│   │   ├── models.py         # models and quantize commands
│   │   ├── plugins.py        # plugins command
│   │   ├── serve.py          # serve command (localhost HTTP API for the desktop UI)
│   │   ├── sources.py        # sources list/enable/disable/reset commands
│   │   ├── video.py          # tiktok, online-summary, play
│   │   └── voices.py         # voices add/list/remove
│   ├── config.py             # paths, config.toml load (tomllib) + save (tomlkit, validated, atomic)
│   ├── plugins.py            # PluginRegistry + PluginError shared by all five seams
│   ├── hardware.py           # machine probe, safe-budget calculation
│   ├── db.py                 # SQLite schema + helpers (stdlib sqlite3)
│   ├── preflight.py          # resource plans and pre-flight checks
│   ├── models/               # model adapter layer (see above)
│   │   ├── quantize.py       # local GGUF quantization via llama.cpp
│   │   ├── lfm.py            # LFM2.5-VL bubble locator + "lfm" model backend (dataset extra)
│   │   └── segment.py        # SAM bubble segmentation (dormant: no pipeline
│   │                         # consumes it yet; kept for future scanlation use)
│   ├── sources/
│   │   ├── __init__.py       # Source protocol + get_client() factory (mangadex, weebcentral)
│   │   ├── mangadex.py       # MangaDex client: search, chapters, at-home page URLs
│   │   └── weebcentral.py    # Weeb Central client: English scanlations via HTML scraping
│   ├── library/
│   │   ├── __init__.py       # collection-manager logic: add series, sync chapters, progress
│   │   ├── works.py          # resource-centric layout: work.json/chapter.json I/O + id -> dir resolution
│   │   ├── importer.py       # eh import: local/URL books (EPUB/TXT/MD) and comics (CBZ/CBR/folder)
│   │   ├── index.py          # rebuild the SQLite index from data/works/
│   │   └── migrate.py        # one-time migrations of legacy data layouts
│   ├── pipelines/
│   │   ├── judge.py          # eval loop: verdict prompts for recap/context/translation outputs (retry with feedback); thinking levels
│   │   ├── recap.py          # recap pipeline (one pipeline, detail knob); calls models via registry only
│   │   └── translate.py      # translation pipeline: grounded bubbles -> overlay pages
│   ├── store/
│   │   ├── __init__.py       # video store dispatch
│   │   ├── hf.py             # HF dataset-repo backend
│   │   └── r2.py             # Cloudflare R2 (S3-compatible) backend
│   ├── ui.py                 # PipelineUI: Rich progress bars + stage reporting for pipeline runs
│   ├── server/               # localhost HTTP bridge for the desktop UI (eh serve; FastAPI + uvicorn)
│   │   ├── app.py            # app factory: /api endpoints (library, video stream, runs, config, sources)
│   │   ├── progress.py       # ServerProgress: the ui.py duck-typed interface, emitting SSE event dicts
│   │   └── runs.py           # RunManager: background `eh recap` runs, one active run per work
│   └── video/
│       ├── __init__.py
│       ├── pipeline.py       # build_video: staged chapter recap/narration videos
│       ├── script.py         # recap -> narration script + segments (text-role model)
│       ├── visuals.py        # key-panel selection (vision model), crops, Ken Burns specs
│       ├── regions.py        # anchored scroll: ordered panel regions from text boxes (cluster, reading order, per-chapter cache)
│       ├── panels.py         # anchored scroll: per-page panel extraction (vision model, whole-page describe fallback), panels.json cache
│       ├── grounding.py      # anchored scroll: segment->region grounding + verifying judge, grounding.json cache
│       ├── panelfirst.py     # anchored scroll: panel beats -> grouped narration segments born with their spans, panelfirst.json cache
│       ├── colorize.py       # optional DDColor page colorization (opt-in)
│       ├── compress.py       # CRF quality re-encode presets (hd/balanced/small)
│       ├── cards.py          # caption-card images for book short-form videos
│       ├── short.py          # whole-work vertical short-form (TikTok) videos
│       ├── gen/              # clip generators (plugin registry): local, runway
│       ├── tts.py            # TTS engines (plugin registry): Kokoro (default), macOS say (fallback), qwen3 (lazy)
│       ├── tts_qwen3.py      #   qwen3 engine: Qwen3-TTS Base voice cloning via mlx-audio (optional `qwen3` extra)
│       ├── voices.py         #   voice library for cloning: named reference WAV + transcript under data/voices/
│       └── assemble.py       # ffmpeg: zoompan render, concat, audio mux, subtitles
└── tests/                    # fakes over real services: in-memory model adapters/clients,
    └── conftest.py           # mocked ffmpeg/TTS/network; shared fakes + harness fixtures
```

Data layout (under `~/.local/share/entertainment-harness/` or `./data` for dev):

```
data/
├── harness.db                # SQLite: derived index, rebuildable from works/
├── voices/<name>.wav|.txt    # user voice library (reference samples + transcripts)
└── works/<series-slug>/      # one directory per manga, book, comic, or tiktok search
    ├── work.json             # id (ULID/source id), title, source, progress, rolling context
    ├── cover.*               # imported book/comic cover
    ├── tiktok/               # whole-work short-form video
    │   ├── out.mp4
    │   ├── script.json
    │   └── render_state.json
    └── chapters/ch-NNN/      # chapter number, e.g. ch-001, ch-001.5
        ├── chapter.json      # chapter_num, title, lang, pages, etc.
        ├── source/           # source pages / book text
        ├── translated/       # overlay-translated pages (eh recap --translated)
        ├── translation.json  # extracted bubbles + model attribution
        ├── recap.json        # chapter artifact (gist..full retelling) + model/detail/instruction attribution
        ├── narration.json    # legacy pre-merge retelling (no longer written; still indexed if present)
        ├── video-recap/      # recap-kind video artifacts (artifact detail < full)
        │   ├── out.mp4       # final video (+ out-<preset>.mp4 when compressed)
        │   ├── script.json
        │   ├── seg-*.wav
        │   └── render_state.json
        └── video-narration/  # narration-kind video artifacts (detail = full)
            └── ...           # one video per chapter: building one kind supersedes the other
```

Re-downloadable weights live in the cache dir (`$EH_CACHE_DIR`, else `$XDG_CACHE_HOME/entertainment-harness`, else `~/.cache/entertainment-harness`), not the data dir:

```
cache/
├── models/hf/<owner>/<repo>/*.gguf   # HF adapter downloads
├── models/sam_vit_b_01ec64.pth       # SAM checkpoint
├── tts/                              # Kokoro weights (kokoro-v1.0.onnx, voices-v1.0.bin)
└── colorize/                         # DDColor model
```

The filesystem is the source of truth: every artifact for a work lives under `data/works/<series-slug>/`. Work and chapter directories are human-readable (title slug, `ch-NNN`); the canonical ids stay inside `work.json`/`chapter.json`, and `library/works.py` resolves id → directory (with fallback to legacy id-named dirs), so all APIs and the remote store remain id-keyed. SQLite is a queryable cache; `eh index` rebuilds it by scanning `data/works/`. One-time migrations on startup move legacy data from `data/manga/`, `data/books/`, and `data/videos/` into `data/works/`, rename id-named dirs to slugs, and relocate model weights into the cache dir.

## Remote store (HF dataset repo or R2 bucket)

Rendered videos are the harness's heavy, expensive-to-regenerate artifacts — pages are re-creatable from the source, so the store holds videos only. `store/__init__.py` mirrors the local works layout under `works/<series-id>/chapters/<chapter-id>/video-recap|video-narration/` and `works/<series-id>/tiktok/` (final mp4s only, never intermediate clips). The backend is pluggable via `[store] provider`: `hf` (default; `store/hf.py` — private HF dataset repo from `[store] repo`, auth via `HF_TOKEN`) or `r2` (`store/r2.py` — S3-compatible Cloudflare R2 bucket from `[store.r2]`, credentials from config or `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` env vars). Once configured, the store is the default backing for the pipeline: `eh recap --video` pushes each chapter's rendered video after building it, and `eh play` pulls a missing video back before giving up. Backends only upload and download — nothing deletes remotely, so when a video build supersedes the chapter's other kind (one video per chapter, see "Video pipeline"), the old kind's mp4 may linger in the store. Models follow a similar lifecycle through their adapters: `ensure()` pulls on demand, `eh models rm <model> [--backend hf]` deletes the local copy.

## SQLite schema (derived index)

SQLite is a rebuildable index over `data/works/`. Run `eh index` to recreate it from the filesystem; the schema is:

- `series(id TEXT PK, title, alt_titles, source, source_id, status, added_at, kind)` — `kind`: `manga` (default) | `book` | `comic`; set by `eh import`
- `chapters(id TEXT PK, series_id FK, chapter_num, title, lang, pages, published_at, fetched_at)`
- `progress(series_id FK, last_read_chapter, updated_at)` — reading state
- `recaps(id INTEGER PK, chapter_id FK UNIQUE, summary TEXT, created_at, model, detail, instruction, standalone)` — the chapter's artifact at its detail grain (`gist`..`full`; `full` is the old narration); `model` stores the full tag incl. quant, so recap quality is attributable to the exact artifact; `instruction` records the steering direction the run was given (`''` = none); `standalone` marks artifacts generated without the rolling story-so-far (`--video` gap fill, below)
- `narrations(id INTEGER PK, chapter_id FK UNIQUE, text TEXT, created_at, model)` — legacy pre-merge retellings; no pipeline writes it anymore (narrations are `recaps` rows at detail `full`), but the fold-in migration and `eh index` still read rows/files that predate the merge
- `series_context(series_id PK FK, rolling_summary TEXT, through_chapter)` — cumulative "story so far"
- `videos(id INTEGER PK, series_id FK, from_chapter, to_chapter, path, duration_s, created_at, tts_engine, model, kind)` — rendered video record; `model` attributes the script/panel selection; `kind`: `recap` (default) | `narration`, following the artifact's detail grain (`full` → narration). One video per chapter: a new build supersedes the other kind
- `translations(chapter_id PK FK, pages, model, created_at)` — translated-pages record; `model` attributes the exact artifact that produced the overlay translations
- `online_summaries(series_id PK FK, provider, query, summary, sources_json, steering_prompt, created_at)` — synthesized web summaries from `eh online-summary`, mirrored into `series_context` for `eh tiktok`
- `characters(id INTEGER PK, series_id FK, name, aliases JSON, role, first_seen, last_seen, origin, edited, UNIQUE(series_id, name))` — the work's character registry (see "Recap flow"): canonical name + aliases + one-line role + the chapter span it was seen in, learned from judged chapter artifacts; `origin`: `observed` (extractor) | `user` (UI-saved); `edited=1` rows are extractor-locked

## Recap flow (chapter-by-chapter)

1. `eh recap <series>` picks the first chapter in `[library] langs` after `progress.last_read_chapter` that has no recap. `eh sync` stores chapters in those languages by default; use `--lang` to fetch a specific source language.
2. Download page images via MangaDex at-home server (cache under `works/<series-id>/chapters/<chapter-id>/source/`, skip if present).
3. Resolve the vision model through the registry: hardware probe → quant selection → user alert if quantized below Q8 → `ensure()` auto-pulls the selected artifact if missing.
4. Send pages in batches of N (N=4, tune by memory); each batch summarized, then combined into a chapter recap.
5. Prepend `series_context.rolling_summary` so the model has story context; prompt asks to continue the recap.
6. Store chapter recap in `recaps`, update `series_context` (rolling summary re-compressed by the text-role model from previous context + new chapter recap), advance `progress`.

**Detail grains** (`--detail gist|brief|standard|detailed|full`, or `[pipeline] detail`; default `standard`): one unified pipeline produces the chapter's artifact at the requested grain — from a one-liner gist up to `full`, the old narration (a complete in-order retelling). The grain selects the prompt pair (batch/combine) and is recorded on the artifact (`recaps.detail`, `recap.json`). Pending selection treats a chapter as done only when its artifact sits at or above the requested grain — chapters with a lower-grain artifact are upgrade candidates (ungated on `last_read`), so `eh recap --detail full` re-does already-recapped chapters without touching context ownership (see "Narration pipeline"). Books use the same grains with chapter-text prompts (`BOOK_RULES`) instead of page batches.

**Gap fill** (`--video` only): a chapter at or before `last_read` with no artifact and no usable video is a hole the frontier gate can never close. So `recap`/`narrate --video` adds these chapters to the pending set (they sort before the frontier chapters, so fold order is preserved). Chapters with a usable video row are skipped by this bucket (nothing is missing), and wiped rows count as missing, so a post-`wipe` re-run rebuilds them. Without `--video`, selection is unchanged.

**Standalone artifacts**: any chapter the run is about to generate an artifact for that sits at or before `last_read` with no artifact yet is generated **standalone** — the story-so-far block is withheld from every prompt (a rolling summary folded beyond the chapter would leak future events into the artifact), nothing is folded, `progress` is untouched, and the artifact is recorded with `recaps.standalone = 1` (`recap.json`, `eh show` header) so its provenance is visible. This applies to `--video` gap fill, to explicit selections (`--chapter`, `--chapters`), and to `--all`; it never applies to frontier chapters, which fold normally. The `on_recap` callback still fires, so a `--video` run builds the chapter's video right after — the hole gets a video without corrupting the context tape.

**Steering instructions** (`--instruction/-i "..."`, or `[pipeline] instructions`; flag wins over config): a free-form reader direction for the run ("skip the cold-open recap pages", "write in Gen Z slang"). A value starting with `@` reads the direction from the named file instead (curl-style; relative paths resolve from the cwd; an empty file means no direction) — the resolved contents, never the `@path`, are what prompts and attribution see. A non-empty instruction appends a `USER DIRECTION` block to the batch and combine prompts of every grain (book prompts included) — content it says to skip stays out of the artifact; a voice it asks for applies; it never overrides the base rules. The recap/narration judges get the same block with a mandated-omission rule: omissions and style choices the direction mandates are NOT issues, while every other criterion (faithfulness, the evidence-quoting rules) applies unchanged where the direction is silent. The instruction is recorded on the artifact (`recaps.instruction`, `recap.json`) and shown in the `eh show` header; it plays no part in pending selection, so changing it does not re-run completed chapters on its own (use `--all`/`--chapter` to redo). Empty instruction (the default) leaves every prompt byte-identical to before.

**Character registry** (`pipelines/characters.py`, the character-bible initiative; opt-out `[pipeline] characters = false`): the strict recap rules forbid any name not printed on the current pages (an anti-hallucination guard), and the ~300-word story-so-far compresses names out — so without help the model can never call a recurring character by name and narrations degrade into pronouns and epithets. The registry closes that gap: after each chapter's artifact is stored, the text-role model lists the chapter's cast as JSON (`[{name, aliases, role}]`), a dedicated judge verifies the list against the artifact at the same faithfulness bar (a failed or unparseable update keeps the previous registry — the never-destroy-accumulated-state guard), and a deterministic merge folds it into the per-work `characters` table — canonical names never flip-flop (a differing incoming name becomes an alias), aliases union, the latest non-empty role wins, user-edited rows stay locked. Later runs load the registry as a capped `CAST` block appended to the rules slot of every batch/combine prompt (book prompts and the video script prompt included) carrying its own scoped amendment: those names MAY be used for those characters only, overriding the printed-names rule for exactly that set. The artifact judges see the same list as pre-approved, so using a registry name is never an outside-knowledge failure; the translation prompt gets it as canonical-spelling guidance instead (see "Translation pipeline"). An empty registry renders no block at all — prompts stay byte-identical to the no-registry case. `eh cast <series>` lists the registry, `eh cast --rebuild` re-folds it from the stored artifacts in chapter order (backfill — the work view's Rebuild button runs the same fold as a background job), and the work view's Characters section edits it (saved rows land `origin='user', edited=1`: the extractor never rewrites them — only their last-seen chapter still advances).

## Narration pipeline (chapter narrations)

Narrations are not a separate pipeline anymore: a narration is simply the chapter's artifact at **detail `full`** — the unified recap pipeline (see "Recap flow") run with the full-retelling prompt pair (~1000+ words, every beat in sequence), stored in `recaps` with `detail = 'full'`. The merge's consequences:

- The old `narrations` table and `narration.json` are legacy: nothing writes them, but pre-merge rows/files are still read by the fold-in migration, `eh index`, and video backfill. Chapter selection is the unified `pending_chapters` rule — a narrated chapter is a recaps row at `full`, so recaps and narrations no longer coexist per chapter; the artifact just upgrades grain.
- Context ownership is unchanged in spirit: only bucket-(a) chapters (no artifact at all) advance `progress` and fold the rolling context. Upgrading a chapter to `full` re-folds the context only when the chapter is not already covered by it, and never touches progress; forced re-runs (`--all`/`--chapter`) touch neither; `--video` gap-fill chapters (see "Recap flow") are generated standalone and touch neither either.
- Judging uses dedicated prompts (`judge_narration`) that add a completeness axis — a narration must not compress beats away — on top of the recap judge's faithfulness/continuity/cohesion/form checks; thinking levels behave identically (high is vision-verified against pages), and steering instructions apply with the mandated-omission rule.
- Narration-kind videos (`detail='full'` artifacts with `--video`) skip the script-generation model call: the retelling is already spoken-form, so it is split verbatim into TTS segments (blank-line paragraphs, short ones merged forward, over-long ones split on sentence boundaries). They render into the `video-narration/` workdir with `videos.kind = 'narration'` — see "Video pipeline" for the one-video-per-chapter rule.

## Judge (eval loop)

`pipelines/judge.py` evaluates the recap pipeline's two text outputs before they are stored — verification is easier than generation, so even the small text-role model is a serviceable judge (the ch 25.5 crossover incident showed the capability emerging unprompted). The judge role defaults to the text model; `[models.judge]` pins a different one.

- **Chapter recap** — judged against the batch summaries (faithfulness: every name/event must come from them), the story so far (continuity), cohesion (no repetition loops), and form (no refusals/meta-commentary). When the run carries a steering instruction, the artifact judges (recap and narration, text and vision variants) also see the `USER DIRECTION` block under the mandated-omission rule (see "Recap flow"); the context judge never does — it checks fact preservation against the artifact as produced, where a mandated omission is correct by definition. When the work's character registry is non-empty (see "Recap flow"), the artifact judges likewise see its names as **pre-approved** — using one for a character that appears is not the outside-knowledge failure the faithfulness criterion guards against; the context judge again sees nothing (its inputs already carry the names).
- **Cast update** — the per-chapter character-registry extraction is judged against the chapter's stored artifact (`judge_cast`): every listed character must actually appear in it, aliases and roles must be supported by it, no outside knowledge — so registry entries are always traceable to a judged chapter. Persistent failure or garbage output keeps the previous registry.
- **Context update** — judged on preservation of the previous story-so-far, incorporation of the new chapter (with an explicit pass case for unrelated bonus/crossover chapters: keep the previous context unchanged), and no meta-commentary.

A failing verdict regenerates with the issues fed back into the prompt, bounded per thinking level (`ATTEMPTS`: medium = 3, high = 5). Final policy: a recap that still fails keeps its **first** attempt, stored with a warning — retries can over-comply with a mistaken critique (feedback poisoning: seen live, the judge false-rejected a good recap and the retry degenerated into an empty "no story content" output the confused judge then passed), so the unpoisoned first version is the safest fallback; a context update that still fails is **discarded in favor of the previous context** (never destroy accumulated state), with the regex `_suspicious_context` backstop on top. To make false rejections rarer, both judge prompts must quote the exact offending phrase for every issue they report — no quote, no issue. A judge that returns unparseable output counts as a pass — a broken judge must not stall the pipeline.

**Thinking levels** (`eh recap --thinking low|medium|high`, or `[pipeline] thinking` in config; default medium): low skips judging entirely; medium is the text judge described above; **high is vision-verified** for recaps — the recap is judged by the vision model against up to 12 evenly spaced chapter *pages* (`sample_pages`), catching hallucinations a text judge can't see. Context updates and page translations stay text-judged at every level (for translations, attaching the page image made the vision judge degenerate — it failed good translations with bare-quote or punctuation-pedantry issues, observed live with qwen3-vl-32b); high only widens their attempt budget to 5. Books have no pages, so high falls back to the text judge with the 5-attempt budget. High is deliberately the slow, expensive tier.

At medium the judge only sees text: it verifies internal consistency and provenance, not fidelity to the page images. At high it re-reads the pages (a sample of them) — the spot-check upgrade, realized.

## Translation pipeline (overlay scanlation)

The translation stage of `eh recap --translated` translates chapter pages into `[library] langs[0]` (default `en`) — the non-English sources' whole point. Chapters already in any preferred language are skipped. Three stages per page:

1. **Extract** (`pipelines/translate.py`) — the vision-role model reads one page and returns JSON: per bubble `{box, original}` with normalized (0–1) coordinates (pixel-coordinate responses are normalized as a fallback — models flip formats unpredictably). Prompt enforces one entry per bubble, no SFX (including untranslated kana SFX left in the art)/credits, and the strict no-outside-knowledge rules.
2. **Translate** — the `[models.translation]` role (default: the text-role model) converts the extracted originals into the target language, and is told to never leave a line empty or untranslated (leaked SFX become short English sound effects). This splits the language work from the OCR/grounding work so a strong text model can be used for translation without replacing the vision model. When the work's character registry is non-empty (see "Recap flow"), the prompt also carries it as canonical-spelling guidance — a recurring character romanizes the same way every chapter, under any of its registered aliases.
3. **Snap + render** — VLM grounding is right-neighborhood but loose, so each box is snapped with OpenCV to the dominant bright connected component inside it (dark caption boxes keep the VLM box and render inverted: black box, white text). Pillow draws a padded rounded rect with word-wrapped, auto-shrinking text.

Each page's result then goes through the same judge loop as recaps (`judge_translation`, text-only): the judge checks language (translations actually in the target language), faithfulness to the printed original (meaning only — not punctuation nuance), form (no refusals/meta-commentary inside a translation), and scope (credits/watermarks rejected; leaked SFX must be translated) — with the evidence-quoting rule hardened to full sentences naming the entry (bare quotes are not valid issues), plus the usual feedback retry and keep-first-on-persistent-failure policy. `--thinking low` disables it.

Output: `works/<series-id>/chapters/<chapter-id>/translated/page-NNN.png` + `translation.json` (raw bubbles + model attribution), recorded in the `translations` table. Chapters already in the target language are skipped; pages cache individually, `--force` redoes them.

`eh recap --video --translated` (or `[video] translated = true`) builds videos from translated pages: page picking and Ken Burns read the English overlays. Chapter resolution switches from "`lang = [library] lang`" to "has a `translations` row", since the source chapter is by definition in another language. `render_state.json` tracks `{colorize, translated, mode, detail, panel_first}` so toggling any of them re-renders instead of serving stale video. The video build refuses with guidance when the chapter was never translated.

Rejected alternative: full re-lettering (manga-image-translator: detect → OCR → translate → **inpaint** → typeset). Better typography in principle, but it pulls in PyTorch plus detection/OCR/inpainting model families — this project deliberately stays torch-free (Kokoro and DDColor both run via ONNX). Overlay boxes are uglier on text-over-art but need zero new runtime stack.

## Video pipeline (chapter-recap videos)

The video stage (`eh recap --video`) turns the chapter's artifact into a narrated video. **One video per chapter**: the video kind follows the artifact's detail grain — `full` (the old narration) renders in the `video-narration/` workdir with `videos.kind = 'narration'`, every lower grain in `video-recap/` with kind `recap` — except that panel-first (the default stage 1, below) always forces the narration kind — and a successful build supersedes the other kind (its workdir is pruned, its `videos` row deleted). Four staged, individually cached steps — each skips when its output artifact exists, so re-runs and partial regeneration are cheap:

1. **Script** (`video/script.py`) — with panel-first off (`--no-panel-first` / `[video] panel_first = false`; NOT the default — see the panel-first bullet), the text-role model turns the chapter artifact into a narration script: 25–40 short spoken beats (one sentence each, typically 12–22 words ≈ 4–8 s), each tagged with the chapter moment it covers; a deterministic splitter re-chunks any over-long segment at sentence boundaries. A non-empty character registry (see "Recap flow") appends a CAST block to the prompt, so beats may name characters the artifact leaves unnamed. Saved as `script.json`. A detail-`full` artifact is already spoken-form, so this stage skips the model call and splits the retelling verbatim into segments (blank-line paragraphs, short ones merged forward, over-long ones split on sentence boundaries).
   - **Panel-first** (`video/panelfirst.py`; ON BY DEFAULT — `--no-panel-first` or `[video] panel_first = false` opts out into the artifact script + assign + grounding path; anchored-scroll Phase 4): the default stage 1. The chapter's cached panel beats (`panels.json`; missing pages are extracted first — a page whose panel parse keeps failing gets one plain-text describe-the-page call that salvages a single whole-page panel, so story content is never silently dropped) are flattened into one ordered beat stream, and the text-role model groups ADJACENT panels into narration beats while writing one spoken line per group — every segment is born with its panel span (`pages` + padded `hold` regions in reading order), so page assignment and the grounding judge are skipped entirely and no artifact is required (the recaps row only supplies the steering instruction). Grouping validates to a strict partition of the beat stream (every panel in exactly one group, in order; overlaps clamp, gaps close by extending the previous group, trailing beats extend the last; unusable responses retry like `generate_script`). The grouped retelling goes through the narration judge (completeness axis) exactly like a full-detail artifact — per-page panel descriptions are the evidence batches, thinking `high` verifies against the page images, persistent failure keeps the first attempt — and `panelfirst.json` caches the grouping keyed by a hash of the beat sequence plus the instruction (panel boxes deliberately excluded: groups are `{"from","to"}` indices, so re-extracted boxes apply without regrouping). Panel-first forces `kind='narration'` (mutually exclusive with `source=`); because panel-first shares the `video-narration/` workdir with the artifact-split path, a cached `script.json` whose model tag doesn't match the requested path is regenerated, and `render_state.json` carries a `panel_first` key (older states default false). Videos run longer than recap-grain ones (reading pace); that's the point.
2. **Narration** (`video/tts.py`) — the TTS engine renders each segment to WAV. **Audio timing is authoritative**: segment durations come from the rendered audio, and visuals conform to them — this avoids sync drift. Engine mini-registry mirroring the model-adapter pattern: **Kokoro-82M** (default — via `kokoro-onnx`, since the `kokoro` package's spaCy dependency chain does not build on Python 3.13; model files auto-download on first use), macOS `say` (zero-dependency fallback), and **qwen3** voice cloning (below).
   - **Voice cloning** (`video/tts_qwen3.py` + `video/voices.py`, optional `qwen3` extra): Alibaba's Qwen3-TTS-12Hz **Base** models (Apache 2.0) clone any voice from ~3 s of clean reference speech via [mlx-audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon — no torch. `eh voices add NAME --audio sample.wav --text "..."` stores the sample + transcript in the voice library (`data/voices/`); with a transcript the engine clones in ICL mode (best quality), without one (`--x-vector`) it uses the speaker embedding only. Voices are named, so the `TTSEngine` protocol is unchanged: `--tts-engine qwen3 --voice NAME` (also `voice = "NAME"` in `[video]`). Per-engine settings (`model`, `language`, ...) live in `[tts.qwen3]` and are passed as constructor overrides (default `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`; the 1.7B variant is a config swap).
3. **Visuals** (`video/visuals.py`) — v1 uses **full pages** with Ken Burns motion (the vision model picks which pages match each segment; no cropping). Pages are shown as labeled contact sheets of **6 pages at a time**, and the prompt demands the EXACT narrated moment — same characters, same action, same place; thematic neighbors, same-character-different-moment pages, and flashbacks/forwards must be omitted (most segments match zero pages in a sheet; segments no sheet claims inherit a neighbor's page). Small sheets plus exactness keep assignments local — a 12-page-sheet run let one exposition page soak up 26 fight-arc beats (anchored-scroll Phase-3 gate). Per-panel crops are a gated upgrade: the vision model returns crop boxes only once grounding proves reliable on manga layouts (see "Feasibility & risks"). Pillow cuts any crops; each segment gets a motion spec (start/end rect) scaled to its audio duration.
   - **Optional colorization** (`video/colorize.py`, the `colorize` plugin category): `[video] colorize = true` or `eh recap --video --colorize` runs the `[colorize] provider` (default DDColor-tiny, ONNX, ~130 MB, auto-downloaded from HF Hub to the cache dir) over the selected pages before assembly. DDColor predicts chroma at 512²; it's recombined with the original-resolution luminance so line art stays crisp. Palettes are plausible, not canon — opt-in, never silent. `render_state.json` records the treatment so toggling it re-renders instead of serving a stale video.
4. **Assembly** (`video/assemble.py`) — ffmpeg (subprocess): `zoompan` renders each crop into a clip exactly matching its segment's audio length, clips are concatenated, narration is muxed, and the script is embedded as a `mov_text` soft-subtitle track (from SRT; the Homebrew ffmpeg build has no libass for burning in). Between beats the mux inserts a short silence (`pause_after_s`: 150 ms, or 300 ms where the moment label changes) — the last page holds through the pause while the subtitle cue ends at speech end — and the narration is loudness-normalized with `loudnorm` (EBU R128, −16 LUFS / −1.5 dBTP). A `pacing` key in `render_state.json` versions this finishing; a mismatch re-renders the cached video (before TTS, so pruned WAVs regenerate). `pacing_stats()` logs beats/words/cuts/avg seconds per visual at assembly time. With `[video] credits` on (the default), the muxer appends a static 4 s credits end card (`video/credits.py`) — work title, author when known (EPUB `dc:creator`), source, chapter, and "Made with Entertainment Harness — terragohan.com" — plus matching silence, so shared videos carry the attribution SHARING.md requires; the card PNG and its clip are cached by content hash in `clips/`, the `credits` key in `render_state.json` re-renders on toggle (pre-credits renders re-mux once to gain the card), and the returned duration includes the card. Output: one mp4 per chapter (or per `--range`), recorded in `videos`.
   - **Presentation mode** (`[video] mode` or `--video-mode`; default `kenburns`): six styles, all keyed into `render_state.json` so switching re-renders. Motion is restamped from the mode at render time rather than read from `script.json`, so cached scripts render identically either way.
     - `kenburns`: per-page pan/zoom clips (the v1 behavior).
     - `scroll`: a multi-page segment's assigned pages are stacked vertically in page order (centered/padded to the widest page, 24 px neutral-gray gutters) into a strip image cached at `clips/strip-NN.png`, and the whole segment renders as one clip — the same animated-crop pan descending the strip top-to-bottom over the segment's slot. One-page segments degrade to the usual `pan_down`.
     - `slideshow`: still center-crops with no drift; the pages of one segment dissolve into each other (`xfade`, 0.5 s) inside a single clip whose per-page holds are padded so the chain ends exactly on the segment slot — segment boundaries stay hard cuts, keeping narration and subtitles aligned. One-page segments render a plain static clip.
     - `cards`: no pages at all — stage 3 renders one caption card per segment (`video/cards.py`, the book-short renderer: moment label + spoken text over a blurred cover or dark gradient, fonts scaling with card width), the cards become the assembly page list, and colorize/translated/page-fetch are skipped. This is the only mode that works for **books** (no pages to pick): a book with no explicit mode choice auto-defaults to cards; an explicitly chosen non-cards mode on a book is an error. Cards are cleared whenever the script regenerates (baked text goes stale with it).
     - `panels`: a guided per-beat view — the grounding stage runs as for scroll (panel-first segments are born with spans), then each anchor renders as its own clip: hold anchors Ken-Burns within their cropped panel box (`clips/panel-NN-K.png`; degenerate boxes fall back to the full page), pan anchors Ken-Burns the full page, equal slot shares with hard cuts between panels. Ungrounded segments take the per-page path.
     - `animate`: `panels` with living panels — same anchors and crops, but each anchor's image is animated by a **frame animator** (`video/frames.py`, the `frames` plugin category; `[frames] provider`): the `runway` animator generates 2–`max_frames` frames per beat (spacing `seconds_per_frame`) via gen4_image text-to-image with the crop as a tagged `@panel` reference and a motion-phase prompt, keeping the chapter's own composition and art while the motion advances; the `openrouter` animator runs the same loop over the OpenAI-compatible endpoint (`[models.openai_compat]`, any image-generation model — `[frames] model`, e.g. `gpt-6-luna`; empty = provider default) via `video/gen/openrouter.py`, normalizing each frame to the target ratio so the chain stays ffmpeg-uniform. The frames stitch through the slideshow xfade chain into the slot-exact clip. Frames are content-addressed (`frames/f-<sha256>-<i>.png`) so re-renders and interrupted runs never regenerate them. A non-animated provider (`local`, the default) renders the panels-mode Ken Burns crop instead — `animate` degrades to `panels` per anchor, and a provider failure on one anchor (a Runway 400, a network error — all surfaced as `VideoError`) logs and degrades that anchor the same way rather than killing the render (the fallback clip stays out of the canonical clip cache, so the next run retries the generator instead of reusing Ken Burns); a configuration error (`VideoConfigError`, e.g. a model id that only outputs text) instead aborts the render immediately, since every anchor would fail identically. Cost note: a grounded chapter is typically 100+ gen4_image calls. `render_state.json` records the frames provider and the resolved image model (changing either re-renders); `videos` metadata attributes the provider when frames were generated.
     - `sequence`: frame-by-frame panel animation — same grounded anchors, but each anchor's panel crop is first **outpainted to the video frame** (image-model expansion, the missing border imagined from the panel), then the animator generates `round(share × fps)` **temporally chained** frames (each one generated from the previous frame plus the beat text, `[sequence] fps`/`max_frames` in `video/sequence.py`, the animator's registry entry declaring `frame-sequence` + `image-gen`). The built-in animators are `sequence` (Runway gen4_image) and `openrouter-sequence` (the OpenAI-compatible endpoint via `video/gen/openrouter.py` — `[sequence] model`, e.g. `gpt-6-luna`; empty = provider default); both share the expansion/chain loop, so the drift critic and caching behave identically. The chain concatenates at slot/N per frame and ffmpeg `minterpolate` motion-compensates up to `[sequence] interp_fps` (30) for smooth playback, staying slot-exact so narration and subtitles stay aligned. An optional **drift critic** (`[sequence] critic`, the vision-role model) compares each chained frame against the previous frame and original panel: feedback → one regeneration with the feedback appended; persistent drift truncates the chain at the last good frame (the truncated chain still fills the slot; cached frames are re-critiqued on re-runs). Expansion and frames are content-addressed under `sequence/` so re-renders never regenerate them. A non-animated provider degrades to the panels-mode Ken Burns crop, a provider failure on one anchor logs and degrades just that anchor the same way (the run continues, and the fallback clip stays out of the canonical clip cache so the next run retries the generator; a `VideoConfigError` — broken model/provider config — aborts instead, since every anchor would fail identically), and a vision-model failure drops the critic with a log line rather than killing the render. Cost note: ≈ `count` image calls per anchor (×2 worst case with the critic on) — a grounded chapter is typically 400–1000 image calls at fps 1.5. `render_state.json` records provider + resolved image model + fps (changing any of them re-renders); `videos` metadata attributes the provider when frames were generated.
     - `motion`: clips are generated per segment by the `[video_gen] provider` (`local` or `runway`) from the segment's first assigned page, following the short pipeline's seam — generated clips re-stamp `duration_s` from their real length, so subtitle cues follow the video. A non-animated provider (`local`) logs and falls back to the stills assembly. `render_state.json` also records the provider, so switching it re-renders; `videos` metadata records which provider produced the video.
   - **Grounded hold-and-glide** (scroll, panels, animate, and sequence modes, on the opt-in artifact script path — panel-first segments are born with their spans and skip this stage; anchored-scroll initiative): between page picking and assembly, a grounding stage (`video/grounding.py`) anchors every segment to panel regions. It ensures Phase 2's `panels.json` (extracting only assigned pages, cached per page), numbers each page's regions on an overlay, and asks the vision model which region(s) illustrate each beat — batched per page, matching by story content since the narration is a retelling. A grounding judge then verifies each choice on the highlighted overlay ("is this region the right illustration for this story beat", never "is the sentence visible"); rejections re-ground with feedback (bounded at 3 proposals per segment-page, the recap judge-loop idiom), and persistent failure, no assignment, or an unparseable judge verdict (strict — the inversion of the text judge's pass-on-garbage policy, so unverifiable choices never ship) **prunes the page from the segment** — rejected art never renders for that beat (safe over smooth, scoped to the page level). Only a segment whose every assigned page was pruned degrades to a whole-page pan over its first original page (a "segment fallback"). Two metrics come out of the stage: assignment quality (`ok / (ok + pruned)` over judged segment-pages — how many of page-picking's picks proved illustratable) and the rendered-slot hit rate (`verified hold anchors / (verified holds + segment-fallback pans)` — what fraction of the visual slots that ship show judge-verified art; vacuous pages, where only a whole-page region exists, are neither judged nor counted in either). The resolved anchors (`[{"page", "kind": "hold", "box"} | {"page", "kind": "pan"}]`, region boxes stamped so assembly never re-reads regions) live on the segments in `script.json` alongside the surviving (post-prune) pages, and are cached per chapter in `grounding.json` (versioned, keyed by a content hash of segment text + ORIGINAL pre-prune pages so cache hits re-apply the pruned pages — a regenerated script recomputes only changed segments). Pages no segment survives on simply aren't rendered. At assembly, a grounded segment renders one strip clip whose viewport y follows piecewise-linear keyframes: hold each anchored region (vertically centered) for an equal share of the slot, glide to the next in a 0.7 s slice; a pan anchor descends its whole page during its share, and an ungrounded segment takes today's linear descent, so fallbacks compose per page and per segment. Positions clamp to the strip and never scroll back up; anchors are capped to the 2.5 s/page readability pace. In panels, animate, and sequence modes the same anchors feed the per-panel crop clips instead of the strip. `render_state.json` carries a `grounded` key for scroll, panels, animate, and sequence renders (pre-grounding scroll videos re-render once; kenburns never carries it and is untouched).
5. **Optional compression** (`video/compress.py`) — quality-based CRF presets: `hd` (1080p h264 CRF 22, near-master), `balanced` (720p h264 CRF 25), `small` (720p HEVC CRF 28, `hvc1`-tagged for Apple players). Measured on a real 31-min narration master (1.15 G, 1080p30): 1080p CRF-only re-encodes save only ~25% — the wins come from resolution and codec: balanced lands at ~2.2× smaller, small at ~3.7× (≈310 M), at ~3 and ~6 minutes encode on Apple Silicon respectively. The preset comes from `--compress` or the `[video] compress=` config default (auto-compression on every render). With `[video] keep_master = false` the master is pruned once a compressed copy exists and the compressed copy becomes the deliverable (`eh play`, store); switching presets then re-renders the master first, so re-compression never degrades a compressed copy. A compressed copy that matches `render_state.json` short-circuits the whole pipeline (it *is* the cache). After a successful render or compress, concat inputs (`clips/`, `seg-*.wav`) are pruned automatically — both regenerate on demand (clips from the cached page/motion picks, WAVs from the TTS engine), and the TTS stage skips re-synthesis when audio was pruned but the video is cached (script durations are authoritative). `eh compress [series] [--chapter N] [--preset P] [--prune]` backfills existing libraries.

Notes:

- **Rights**: recap videos embed scanned manga pages — fine for personal viewing; publishing them has copyright implications the user owns.
- No background music in v1 (a `[video] bgm` slot is reserved in config).

## CLI commands (Typer)

- `eh search "<title>" [--source S]` — search a source (mangadex default; weebcentral for English scanlations), show results.
- `eh chapters <manga-id> [--lang L] [--source S]` — list hosted chapters for a series on a source.
- `eh add <id-or-search-term> [--source S]` — add series to library.
- `eh sync [series] [--lang LANG]` — fetch chapter list from the source into DB. By default only chapters in `[library] langs` are kept; use `--lang` to fetch a specific source language (e.g. `pt-br`) when a series is not available in your preferred languages.
- `eh list` — show library with read/artifact status incl. the detail grain (rich table).
- `eh remove <series> [--keep-files] [-y]` — remove a series (e.g. a duplicate) with its chapters, recaps, narrations, translations, videos, and local files (confirmed interactively; store backups are left untouched).
- `eh recap <series> [--all] [--chapter N] [--chapters SPEC] [--max-chapters N] [--detail D] [--instruction "…"] [--translated] [--video] [--thinking L] [--voice V] [--tts-engine E] [--colorize] [--compress P] [--panel-first|--no-panel-first] [--vision-model M] [--text-model M] [--backend B] [--quant Q]` — the single pipeline command. Scope: **no args recaps every unfinished chapter** (all unread after last-read, in order) plus lower-grain upgrade candidates; `--all` re-recaps **every synced chapter**, overwriting existing recaps; `--chapter N` re-runs a single chapter; `--chapters "1-3,4,6-10"` re-runs exactly the synced chapters inside the spec (side chapters like 9.5 match by containment; same chapter-spec syntax as `eh wipe`). Overwriting scopes (`--all`/`--chapter`/`--chapters`) leave rolling context and progress untouched — an out-of-order rewrite would corrupt them — and chapters at or before last-read with no artifact yet are generated standalone (story-so-far withheld, `recaps.standalone = 1`; see "Recap flow"). `--detail gist|brief|standard|detailed|full` sets the artifact grain (default `[pipeline] detail` or `standard`; `full` is the old narration). `--instruction "…"` steers the run with a reader direction (`@file` reads it from a file; default `[pipeline] instructions`; see "Recap flow"). `--max-chapters` (default 500) caps how many chapters one run processes. `--translated` runs the translation stage first for chapters not already in `[library] langs` (see "Translation pipeline"); recaps and rolling context land in the same tables, so plain and translated recaps form one story-so-far chain. `--video` renders each chapter's video right after its artifact (via the `on_recap` callback), pushes it to the configured store, then backfills videos for chapters that lack one — and gap-fills chapters behind the read frontier that have no artifact and no video with standalone artifacts (see "Recap flow"), so their videos can be built too; `--voice/--tts-engine/--colorize/--compress/--panel-first/--no-panel-first` are its video options (panel-first scripting — the video script comes from cached panel beats, segments born with their panel spans, no page assignment or grounding judge, narration kind forced — is ON BY DEFAULT; `--no-panel-first` scripts from the artifact and runs the grounded assign+grounding path instead; default `[video] panel_first=`). `--thinking low|medium|high` sets critiquing depth (see "Judge"). Runs display a live progress UI (`ui.py`): overall chapter bar + current-chapter stage (translate/recap/video, cache hits, output paths); non-terminal output falls back to plain log lines.
- `eh show <series> [--chapter N] [--narration]` — print each chapter's artifact at its current detail grain; the header carries the grain, model, any steering instruction, and a standalone marker when the artifact was generated without story-so-far context. `--narration` is deprecated (detail `full` is the narration).
- `eh cast <series> [--rebuild] [--thinking L]` — list the work's character registry (canonical name, aliases, role, chapter span, origin; see "Recap flow"). `--rebuild` clears the registry and re-folds it from the stored chapter artifacts in chapter order — the backfill for works processed before the registry existed.
- `eh play <series> [--chapter N] [--tiktok] [--narration]` — open the chapter's rendered video in the system player (one video per chapter; kind follows the artifact's detail). `--tiktok`: the whole-work short-form video; `--narration` is deprecated (restricts the lookup to narration-kind videos). If the file was cleaned locally, it is pulled back from the configured store first.
- `eh compress [series] [--preset P] [--prune]` — backfill compression for rendered chapter videos that lack an `out-<preset>.mp4` (default: `[video] compress=` or `balanced`). Encodes from the master when present; `--prune` also deletes clips/segment WAVs and, per `keep_master`, masters. Prints per-chapter before/after sizes.
- `eh wipe <series> --chapters "1-3,4,6-10" [--kind narration|recap|all]` — delete a chapter video's files and mark it wiped (`videos.wiped_at` + `video.json`), for reclaiming space after watching. Chapter numbers are floats, so ranges match side chapters (9.5) by containment. Wiped chapters keep their script/render caches — re-rendering reuses them and clears the mark (`_record_video` resets `wiped_at`). Wiping a chapter that has no artifact on file destroys the only remaining form of its story: it cannot be re-rendered from cache (a later `recap --video` can only mint a fresh standalone artifact), so each such chapter gets an explicit warning. `eh play` reports a wiped chapter distinctly; `eh concat` skips wiped chapters. Chapters whose files were deleted by hand are marked wiped on the next `eh wipe` run instead of being silently skipped.
- `eh concat <series> [--kind narration|recap] [--preset P] [--out PATH]` — build one long video from a series' chapter videos in chapter order (`works/<slug>/<kind>-full.mp4` by default). Picks `out-<preset>.mp4` when given, else the chapter's recorded `compress` preset, else the master `out.mp4`. When all parts share one codec/resolution/fps the concat is a lossless stream copy via the ffmpeg concat demuxer; mixed formats are re-encoded at 720p h264 CRF 25. Each chapter video carries its own credits end card (assembly stage above), so concatenated exports show the card between chapters and at the very end — the attribution travels with the export.
- `eh import <path-or-url> [--title T] [--kind book|comic]` — import a book (EPUB/TXT/MD) or comic (CBZ/CBR/image folder) from a local file or direct-file URL.
- `eh tiktok <title> [--voice V] [--tts-engine E] [--video-gen local|runway] [--steering-prompt P]` — render a whole-work vertical short-form (1080x1920, ~60–90 s) summary video from a title. Searches the web for summaries, synthesizes a rolling summary, and builds the video with generated caption cards. No library series is required.
- `eh anime-scene <series> --chapter N --pages "8-10" --instruction "..." [--stop-after plan|keyframes] [--regenerate-shot N]` — adapt one contiguous manga scene into one ≤30 s silent 16:9 anime scene (see "Anime scene (guided manga→anime slice)"). Staged and cached under `works/<series>/chapters/<ch>/anime-scene/`; `--stop-after plan` writes a human-editable `scene.json` before any paid Runway call.
- `eh online-summary <series> [--provider P] [--steering-prompt P] [--text-model M] [--backend B]` — fetch online summaries (blogs, Reddit, YouTube transcripts) for a series and store a synthesized summary (`online_summaries`) for `eh tiktok`.
- `eh quantize <repo> --quant Q [--yes]` — user-confirmed local quantization of a HF GGUF repo's FP16/BF16 source via llama.cpp (last resort; refuses when the quant is already published).
- `eh models` — hardware profile + configured models with quant, size, and fit verdict per adapter. `eh models rm <model> [--backend B]` deletes the local copy.
- `eh voices add|list|remove|preview` — manage the voice library for `--tts-engine qwen3`: `add NAME --audio sample.wav --text "..."` registers a reference sample (3+ s, one speaker; `--x-vector` skips the transcript at lower cloning quality), stored under `data/voices/`. Without `--audio`, `add` runs interactive onboarding instead: it hands you a line to read aloud (your `--text` if given, else a default line — the transcript must match your words for ICL mode), records up to 15 s from the microphone via macOS `afrecord`, and offers playback + re-record until you're happy. Onboarding then synthesizes 3 sample clips with the new voice (varied narration/dialogue/numbers; `--no-preview` to skip, `--play` to audition immediately, `eh voices preview NAME [--play]` to regenerate), stored under `data/voices/previews/` and deleted with the voice.
- `eh plugins` — list registered plugins per category (model backends, sources, TTS engines, video generators, frame animators, search providers) with declared capabilities and which is active per config; sources show enabled/disabled per `[sources].enabled`. `--check` runs the conformance kit (`testing.py`) on every plugin; `eh plugin check pkg.mod:Class --category <cat>` checks one class.
- `eh sources` — list registered content sources with origin (built-in vs entry point) and enabled/disabled state. `eh sources enable|disable NAME` updates `[sources].enabled` in config.toml via `save_config` (built-ins only can be disabled; the list is kept built-ins-first in registry order, extras after); `eh sources reset` restores the defaults (`["mangadex", "weebcentral"]`). A disabled built-in makes every source lookup (`eh search/add/sync --source`, pipeline page fetches) fail with a re-enable hint.
- `eh serve [--port N] [--host H]` — run the local HTTP API for the desktop UI (see "Local server"). Default: `127.0.0.1`, port 0 (ephemeral). The first stdout line is machine-readable — `listening <port>` — for the parent process that spawned it; all other logging goes to stderr.

## Local server (`eh serve`)

`eh serve` runs a FastAPI app (`server/`) under uvicorn on localhost — the backend half of the desktop UI (see `docs/initiatives/desktop-ui/`). It is purely additive: every endpoint reads and writes through the same seams as the CLI (`db.connect()` + `library/works.py` for library data and video paths, `config.load_config`/`save_config` for config.toml, the sources registry for source state), so the library, config file, and CLI behave identically whether or not the server is running. CORS is permissive by design (Electrobun webviews have opaque origins; the server only ever binds localhost). All endpoints live under `/api`:

- `GET /api/health` — `{"ok": true}`; the desktop parent polls this after spawn.
- `GET /api/library` — every work with its chapters: per-chapter artifact detail, `has_recap`, `has_video` (a non-wiped `videos` row), video kind/duration, and the chapter's stream URL when playable, plus the work's `auto` flag (the per-work auto-process toggle, false when never enabled). Read-only; sourced from the same tables as `eh list`/`eh show`. A chapter can hold several `videos` rows (one per kind is legal), so the query picks a single representative row — usable beats wiped, then newest — and never emits a chapter twice.
- `GET /api/videos/{work}/{chapter}/stream` — stream the chapter's mp4 with HTTP Range support (206 partial content, `Content-Range`/`Accept-Ranges` — WebKit `<video>` requires this for seeking; handled by Starlette's `FileResponse`). The path is resolved exactly like `eh play` (works layout, compressed copy preferred when recorded), with ids always looked up in the DB — never raw path joins — so traversal attempts 404. Only locally-present files are served: a wiped or store-only video 404s (pull it with `eh play` or re-render); the server does not fetch from the remote store. When a chapter has several `videos` rows, the usable one wins (wiped rows can't shadow a playable video with a 404).
- `POST /api/works/{work}/export` (202) — start a background **video export**: assemble the work's playable chapter videos into one mp4 under `~/Downloads/EntertainmentHarness` (`EH_EXPORT_DIR` overrides the directory; tests/smoke use it) and report completion by polling. Resolution mirrors the stream endpoint per chapter (usable video row wins, compressed deliverable preferred, missing files skipped); one part is copied as-is, several are concatenated by `video.assemble.concat_mp4s` — lossless stream copy when every input shares codec/resolution/fps, balanced 720p re-encode for mixed formats (the same helper `eh video concat` uses). One active export per work: a concurrent POST gets a 409; unknown work 404. The job ends `done` (`dest`, `total` assembled, `skipped` chapters without a usable local video) or `error` (e.g. nothing playable).
- `GET /api/exports` — export jobs, newest first (`{id, work, title, status, dest, total, skipped, error, created_at}`). In-memory like runs: gone when the backend restarts.
- `POST /api/runs` — start a background recap run: the same flow as `eh recap` (select chapters → pre-flight guard → `recap_series` with the per-chapter video callback → video backfill) in a worker thread, minus the TTY bits (no Rich console, no `--dry-run`; a steering `instruction` is used literally, the `@file` form is not resolved). `video: true` also turns on the `--video` gap-fill (chapters behind the read frontier with no artifact and no usable video join the run as standalone gap-fills), exactly as the CLI flag does. Body: `work` (id/slug) plus the headless recap options — `all_chapters`, `chapter`, `chapters` (a `--chapters`-style spec, e.g. `"1-3,4"` — exactly the synced chapters inside it, in order), `max_chapters`, `skip_done` (drop chapters that already have a recap at the requested grain — and a non-wiped video, when `video` is on — from a forced scope, so a range run only processes what is actually missing; a chapter kept only for its missing video is **not re-recapped** — the video callback fires directly and the loop moves on, so a failed video never costs a re-narration), `video`, `translated`, `thinking`, `detail`, `instruction`, `voice`, `tts_engine`, `colorize`, `compress`, `video_mode`, `panel_first`, `skip_preflight`. One active run per work: a concurrent POST for the same work gets a 409; unknown work 404; invalid option values 422 (a malformed `chapters` spec is rejected up front rather than mid-run). Returns the run (`{id, work, title, status, ...}`). The auto-process supervisor starts its runs through the same `RunManager` (see "Desktop app"); the endpoint itself is unchanged by it.
- `GET /api/runs`, `GET /api/runs/{id}` — run status (`running | done | error | cancelled`, with `error` message). Each run also carries `current`: a cursor `{event, chapter, stage, detail}` mirroring the latest non-log event (`run-start`/`chapter-start`/`stage`/`video-ready`/`chapter-done`), so polling clients can render "running — ch 12: narration" without streaming SSE; it is `null` in terminal states.
- `GET /api/runs/{id}/events` — SSE stream (`text/event-stream`). Buffered events are replayed first, so a late subscriber sees the whole run from the beginning; the stream ends after the terminal event. Progress reporting goes through `ServerProgress` (`server/progress.py`), a drop-in for the duck-typed `ui.py` interface (`start`/`chapter_start`/`stage`/`log`/`chapter_done`) that turns each call into a JSON-able event dict — the pipelines needed no changes. Each frame is `event: <type>` + `data: <json>`; the JSON carries `event`, `run_id`, `work`, `ts`, plus type-specific fields:

  ```
  event: stage
  data: {"event": "stage", "run_id": "9f3c…", "work": "01J…", "ts": "2026-09-20T10:15:30+00:00", "chapter": 3.0, "stage": "video", "detail": "narration"}
  ```

A run worker holds its `db.connect()` across the whole run, and Python's sqlite3 keeps an implicit write transaction open across the long LLM/TTS calls — so while a run is active the database is write-locked for minutes at a time. `db.connect()` must therefore stay read-only in the common case: the narrations fold-in migration only writes when a read-only pre-check finds un-folded rows, and defers silently (retrying on a later open) when the database is locked, rather than failing every endpoint with `database is locked`.

  Event types in run order: `run-start` (`chapters`), then per chapter `chapter-start` (`chapter`), `stage` (`chapter`, `stage`, `detail`), `log` (`chapter`, `message`), `video-ready` when the chapter's mp4 lands (`chapter`, `kind`, `duration_s`, `stream` — the UI can start playback immediately), `chapter-done`; terminal `run-done` (`chapters`), `run-cancelled` (`chapters` — how many finished before the stop; see below), or `run-error` (`message`).
- `POST /api/runs/{id}/stop` — cooperatively stop a running run. Stopping is flag-based, not a thread kill: `Run.request_stop()` sets a `threading.Event` that `ServerProgress` checks at its structural callbacks (`start`/`chapter_start`/`stage`/`chapter_done`), raising `RunCancelled` to unwind the pipeline; `log` never raises (pipelines call it from error paths). The in-flight model/ffmpeg call finishes, then the run aborts at the next chapter or stage boundary, emitting `run-cancelled` and landing in status `cancelled` (completed chapters stay recapped — only the remaining ones are skipped). Idempotent and asynchronous: the endpoint returns while the run is still unwinding (status may still be `running`), and stopping an already-finished run is a no-op returning the run. 404 for an unknown id.
- `PUT /api/works/{work}/auto` — the per-work auto-process toggle. Body: `{enabled, detail, instruction, skip_preflight, video_mode?, chapters?, all_chapters?}` — the scope is optional (neither set = the default pending selection). `detail` is validated against the recap grains and `video_mode` against the presentation styles (kenburns/scroll/slideshow/cards/panels/motion/animate/sequence); a malformed `chapters` spec (or `chapters` combined with `all_chapters`) is rejected, unknown work 404 (all errors 422/404 as appropriate). Enabling persists the toggle and its modifiers and, when no run is active for the work and it has unfinished chapters, starts one immediately; disabling persists and cooperatively stops any active run for the work. Returns `{work, enabled, run}` with `run` the started run's dict or null. State lives in `data_dir()/auto.json`, owned by the supervisor (`server/auto.py`), which also reconciles on a daemon timer: enabled works with no active run get a new run while unfinished work remains, and a work whose latest supervisor run errored is automatically disabled so a broken config can't loop. The toggle never auto-resumes after a restart: state loaded from disk at startup is **disarmed** (an in-memory `armed` flag, never persisted), so the library payload's `auto` flag reads false and no supervised run starts until the user flips the toggle on again — settings (scope, detail, instruction) are kept. This stops a crash loop (e.g. a TTS stack that `exit(1)`s the process) from re-arming itself on every launch. "Unfinished" mirrors a run with the stored options exactly: scope-less, the gap-aware pending set (`pending_chapters` at the configured detail/langs, `fill_gaps=True` — behind-frontier chapters with no artifact and no usable video included) plus recapped chapters lacking a playable video (`_chapters_missing_video`, the `--video` backfill set); scoped (`chapters`/`all_chapters`), any chapter inside the scope that `chapter_needs_work` keeps, since scoped runs are `skip_done` — only the unfinished chapters inside the scope are processed.
- `GET /api/config` — the effective config as JSON (`dataclasses.asdict(load_config())` minus `raw`, the parsed-toml mirror kept for plugin merges — sending it would duplicate the payload and invite edits to a bogus `[raw]` table).
- `GET /api/works/{work}/characters`, `PUT /api/works/{work}/characters` — the work's character registry (see "Recap flow"). GET lists it most-recently-seen-first (`name`, `aliases`, `role`, `first_seen`/`last_seen`, `origin`, `edited`); unknown work 404. PUT replaces the whole list from the work view's Characters section — body `{characters: [{name, aliases, role}]}` (blank aliases/roles are stripped, empty names 422) — and every saved row lands `origin='user', edited=1`, so the extractor never rewrites user-touched entries (only their last-seen chapter advances on later sightings).
- `POST /api/works/{work}/characters/rebuild` (202) — re-extract the registry from the work's stored chapter artifacts in the background: the work view's Rebuild button, running the same fold as `eh cast --rebuild` (`characters.rebuild_cast`, shared with the CLI) at the configured `[pipeline] thinking` level. Replaces the whole registry, manual edits included. A work with no recaps 422s before any model is touched; one active build per work (concurrent POST 409); unknown work 404. Poll `GET /api/cast-builds` — jobs (`{id, work, title, status, count, error, log, created_at}`, newest first, with an 8-line log tail for progress display) are in-memory like exports: a backend restart mid-build leaves the registry at the last committed chapter fold.
- `PUT /api/config` — body `{section: {key: value}}`, applied via `save_config` (validated against the `Config` dataclasses, comment-preserving, atomic); invalid updates 422 with the `ConfigWriteError` message and the file is untouched. Returns the new effective config.
- `GET /api/sources` — registered sources with `builtin`/`enabled` state (same data as `eh sources`).
- `PUT /api/sources` — body `{"enabled": [...]}` → `save_config` on `[sources].enabled` (unknown names 422; ordering normalized built-ins-first like the CLI). `POST /api/sources/reset` restores the defaults.

## Desktop app (Electrobun)

The desktop UI lives in `ui/` — a self-contained Electrobun project (Bun main process in TypeScript + native WebKit webview; own `package.json`, repo root stays Python-only). The app is a pure client of `eh serve`: the main process spawns the backend, parses the `listening <port>` line, polls `/api/health`, then hands the port to the webview, which talks HTTP directly to `127.0.0.1`. The backend child is killed on quit. Views: library (works → chapters with playable/in-progress state), work (run controls + live SSE progress: overall bar, current stage, scrolling log; chapters become playable as `video-ready` events land), player (`<video>` on the stream endpoint with autoadvance), settings (config.toml editor via `PUT /api/config` + sources toggles with restore-defaults). The settings editor is organized into **task-oriented groups** rather than raw config sections: an **API keys** panel (the four secrets — Runway, OpenAI-compatible, R2 access/secret — as write-only password inputs, never prefilled, blank = keep current), Library & recaps, Models (quant policy, the four model roles as subsections, endpoints), Video (style/tts/voice selects plus frames/sequence/colorize/anime/video_gen subsections), **Translation (scanlation)** — the active job is a select over the defined `[jobs.scanlation.<name>]` jobs and each job renders its extract/translate/render/judge stages as cards, with `models.lfm_model` co-located and the job→stage→LFM relationship explained in place — Storage, and Advanced. Nested config tables render as nested subsections (typed inputs per leaf — raw JSON text is only a last resort, e.g. a non-empty stage `extra`); enum-like fields are selects; anything the layout doesn't place (future config keys) still renders generically under "Other settings". Saving sends only changed fields as nested `{section: …}` updates; because `save_config` replaces free-form dict tables wholesale (`jobs.scanlation`, `tts`, `roles` — unlike dataclass tables, which merge key-by-key), the editor re-seeds those subtrees from the loaded config before applying edits so sibling entries survive.

- **Auto-process toggle**: the work view's run panel is built around a per-work **"Process unfinished chapters" toggle switch** — there is no Generate button; the toggle *is* the run control. Next to it sit the modifiers every auto run uses: a **detail-level select** (gist/brief/standard/detailed/full, defaulting to the config's `[pipeline] detail`) and a **video-style select** (kenburns/scroll/slideshow/cards/panels/motion/animate/sequence, defaulting to `[video] mode` — for book works, to `cards` when the config is at its kenburns default, mirroring the backend's book auto-default), an **optional chapter scope** — from/to inputs plus an **"all" checkbox** (both empty = the default pending selection from the read mark onward, gaps behind it filled in; a range — one box alone = that single chapter — or "all" limits processing to those chapters, still only the unfinished ones; persisted per work in webview localStorage as `eh.scope.<workId>`), a steering-instruction textarea (persisted per work in webview localStorage as `eh.instruction.<workId>`; pre-filled with a built-in default steering direction — names over pronouns, no chapter openers/closers, smooth transitions, vivid onomatopoeia — that the user can edit or clear per work, same semantics as `eh recap --instruction`: used literally, `@file` not resolved), and a skip-preflight checkbox. Flipping it on sends `PUT /api/works/{work}/auto {enabled: true, detail, video_mode, instruction, skip_preflight, chapters?, all_chapters?}`; the backend stores the toggle **server-side** (in `data_dir()/auto.json`, via the supervisor — the settings survive window closes *and* backend restarts, unlike the pre-Phase-11 localStorage toggle) and starts a run immediately when the work has unfinished chapters and none is active; the UI attaches to that run's SSE stream when one starts. Flipping it off sends `{enabled: false}`; the backend cooperatively stops any active run and the existing `run-cancelled` handling reports "Stopped — …". **After a backend restart nothing auto-resumes**: loaded state starts disarmed, the toggle reads off until the user flips it on again (scope/detail/instruction are kept), so a crash can never re-arm processing by itself. **"Unfinished"** means: no recap at the selected grain, no playable video, or a gap behind the read frontier with neither artifact nor video (scope-less), or any of that inside the chosen scope (scoped — the supervisor's `has_unfinished` mirrors a run with the stored options, so no no-op runs ever start, and emptied-out chapters behind the read mark still get filled in). While the toggle is on but nothing is running (everything done), the panel shows "Auto-processing on — nothing to do right now."
- **Characters section**: the work view carries a collapsible **Characters** panel between the header and the run panel — the work's character registry (see "Recap flow"), with the entry count in the summary and one editable row per character (name, comma-separated aliases, role) plus add/remove-row buttons. A single Save replaces the whole list via `PUT /api/works/{work}/characters` and re-renders from the response; saved rows are extractor-locked server-side (`origin='user', edited=1`), so one correction fixes every future chapter. A two-step **Rebuild** button (first click arms, second confirms — the fold replaces the whole registry, manual edits included) starts a background cast build via `POST /api/works/{work}/characters/rebuild` and polls `/api/cast-builds`, tailing the job's log in the status line and re-rendering the rows when it lands.
- **Auto supervisor** (`server/auto.py`): owns the persisted toggle state (`set()` is the only mutation entry; atomic temp+replace writes; a missing/corrupt `auto.json` means empty state) and reconciles on a daemon `threading.Timer` loop (30 s, started lazily on the first enable, never blocks requests): an enabled **and armed** work with no active run gets a new run while unfinished work remains. **Arming** happens only via an explicit `set(enabled=True)` — i.e. the user flipping the toggle; `load()` starts every entry disarmed, so a backend restart never auto-resumes processing. Error handling: when a supervisor-started run reaches terminal `error`, the toggle is disabled for that work (persisted) so a broken config can't loop forever — the loop checks `runs.list()` on every tick, plus a one-shot reconcile ~2 s after each supervisor-started run so errors turn the toggle off promptly. Supervisor runs are `RunOptions(work, video=True, detail/instruction/skip_preflight/video_mode=<stored>)` — scope-less by default (the gap-aware pending selection, `fill_gaps=True`, matching `eh recap --video`), or scoped with the stored `chapters`/`all_chapters` plus `skip_done=True` so only unfinished chapters in scope are processed; `POST /api/runs` itself is unchanged.
- **Video export**: the work view's run panel has an **"Export videos" button** — it assembles the work's playable chapter videos into one mp4 in `~/Downloads/EntertainmentHarness` (single chapter → copy; several → lossless concat, or a balanced re-encode for mixed formats) and the user is **notified when the export lands**. The webview asks the **main process** over RPC (`exportWorkVideos`) rather than POSTing itself: the main process owns the job — it starts the backend export, polls `GET /api/exports` to completion (concat can take minutes), fires a **native notification** (`Utils.showNotification`, which works even when the window is closed), and pushes the terminal state to the view (`exportDone`) for an in-app banner. The view also polls while it is open (button disables while an export for the work runs; re-entering the view resumes tracking).
- **Background runs**: while the toggle is on, the webview reports background mode over RPC (`backgroundModeChanged`) and closing the window keeps the app (backend + processing) alive instead of quitting; the main process re-checks `/api/runs` at close time, so a stale "on" never traps the app once processing has stopped. **Off** means closing the window quits as usual (and any active processing is stopped by the toggle itself). Electrobun 1.x cannot intercept a window close (the native side destroys the window, then emits a non-cancellable `close` event), so "close hides" is implemented via `runtime.exitOnLastWindowClosed: false` plus the per-close decision above. Reopening recreates the window (a destroyed Electrobun window cannot be re-shown) via the Dock icon (`reopen` event) or the "Show Entertainment Harness" app-menu item; the webview re-reads the toggle from the library payload, so it re-syncs after restarts. Cmd-Q / menu Quit always quits fully (no guardrails). Caveats: runs live in the backend process, so a full quit kills them; if the window is closed while backgrounded and processing finishes unseen, the app stays alive until an explicit quit (the next close after everything stops quits normally).
- **Run awareness without SSE**: the nav and the library cards poll `GET /api/runs` (shared ~3 s poller, only while the backend is ready) and render active runs from the `current` cursor — the nav shows "● 1 run — ch 12: narration" (click → the work view, ✕ → stop that run) and library cards get a "running — ch N: stage" line. The work view itself uses SSE for live progress; stopping its run is the auto toggle (off) rather than a per-run button, and on `run-cancelled` it reports "Stopped — N chapter(s) finished before stopping."
- **Dev mode**: `cd ui && EH_DATA_DIR=<scratch> bun run dev` spawns `uv run eh serve` from the repo root. `bun run smoke` runs a headless smoke test (linkedom DOM shim + EventSource client over a real spawned backend); `bun run typecheck` type-checks.
- **Builds**: dev-only — `bun run build` (or `bun run dev`) produces `build/dev-macos-arm64/`, loading app code from flat files, and the app always spawns `uv run eh serve` from the repo checkout (`backendCommand()` in `ui/src/bun/backend.ts`; the child's PATH gets `/opt/homebrew/bin:/usr/local/bin` prepended since Finder launches have no Homebrew, and generation shells out to ffmpeg/ffprobe). There is no packaged/frozen-backend distribution: the PyInstaller spec, stable channel, and tag-triggered DMG release workflow were removed (git history has them). Self-built apps are ad-hoc signed (first launch needs right-click → Open). The espeak-ng dylib stores its data path in a ~160-byte fixed buffer, so `video/tts.py` pins `ESPEAK_DATA_PATH`/`EspeakConfig` to a short tempdir symlink when the real path is too long (deep checkouts would otherwise make espeak `exit(1)`).
- Only run one app instance per data dir: concurrent `eh serve` processes against the same `harness.db` hit `database is locked` (see the run-worker note under "Local server").

## Import (books & comics)

`eh import <path-or-url> [--title T] [--kind book|comic]` brings local works into the library — the sources are user-provided files or direct-file URLs, never site scraping. `library/importer.py` detects the kind from the extension (`.epub/.txt/.md` → book; `.cbz/.cbr`/directory → comic) and inserts a `series` row with `source="import"` and `kind` set.

- **Books**: EPUBs are parsed with the stdlib (`zipfile` + `xml.etree` spine order + `html.parser` tag stripping — no new dependency); TXT/MD split on chapter headings, falling back to ~4 000-word chunks. Parts land in `works/<series-id>/chapters/<chapter-id>/source/ch-NNN.txt` (plus the EPUB cover as `cover.*` in the work root), one `chapters` row per part. Tiny spine items (title pages) merge into neighbors.
- **Comics**: CBZ/folders extract into the standard `works/<series-id>/chapters/<chapter-id>/source/` page layout (one file = one chapter), so recap/visuals paths work untouched. CBR needs a system `unar`/`unrar`.
- **URLs**: direct file links only (httpx download, 500 MB cap); HTML pages are refused with guidance.
- Book recaps skip the vision model entirely: the text-role model summarizes each part in ~3 000-word chunks and combines them; the rolling `series_context` works identically. Source clients are constructed lazily so `source="import"` never hits the client factory.

## Short-form (TikTok) videos

`eh tiktok <title>` renders one whole-work vertical video (1080x1920, ~60–90 s) in `video/short.py` from a web search, no library series required. It reuses the chapter pipeline's stages with three differences:

1. **Script**: `SHORT_SCRIPT_PROMPT` — 8–12 segments, 150–230 words, segment 1 is a hook; input is a synthesized online summary (`online_summaries` table, mirrored into `series_context`).
2. **Visuals**: generated caption cards (`video/cards.py`, Pillow — gradient backdrop, moment label + segment text; no vision model) because no chapter pages are available. Cut pacing is faster (`min_page_seconds` 1.5 vs 2.5).
3. **Assembly**: same ffmpeg path at 1080x1920; `videos` row with `from_chapter = to_chapter = NULL` marks whole-work videos (`eh play --tiktok`).

## Anime scene (guided manga→anime slice)

`eh anime-scene` (`anime/` package, initiative: `docs/initiatives/manga-to-anime-scene/`) adapts one manually selected, contiguous page range into one ≤30 s silent anime scene. Four cached stages, filesystem is the source of truth (no DB rows):

1. **plan** (`anime/planner.py`) — the configured vision-role model reads ALL selected pages plus the user's free-form instruction and returns one validated `SceneSpec`: 3–5 shots, each exactly 5 s, manga event order, page-bounds checked. Written to `scene.json`, which is deliberately human-editable — rerun resumes from the edited file (it is never silently regenerated; delete it to re-plan). A `Shot` carries `source_page`, optional normalized `source_crop` `[left, top, right, bottom]` (cropped with Pillow), `shot_type`, `action`, `composition`, `camera`, `continuity`, and separate `keyframe_prompt`/`animation_prompt`. Shots are NOT recap `Segment`s — no narration, no audio timing.
2. **keyframes** — Runway `gen4_image` (`POST /v1/text_to_image`) renders one 1280x720 frame per shot from tagged `referenceImages`: `@Source` (the manga page/crop), plus `@Anchor` (shot-00 keyframe) and `@Previous` (prior keyframe) for shot 1+ — the first generated frame anchors character/style identity for the scene.
3. **shots** — Runway image-to-video (`[anime] video_model`, default `gen4.5`): the keyframe is the first frame, 1280:720, exactly 5 s, prompt describes motion only.
4. **assembly** — ffmpeg normalizes each clip (1280x720 scale+pad, 30 fps, yuv420p, no audio) and hard-cuts them in storyboard order via the concat demuxer; `ffprobe` verifies ≤30 s. The recap narration muxer is not involved.

Caching: sha256 content fingerprints in `state.json` per stage (plan: page contents + instruction + planner model + prompt version; keyframe: shot fields + source/ref contents + image model; clip: keyframe content + animation prompt + video model + duration + ratio; final: ordered clip contents). Editing one shot in `scene.json` regenerates only that shot and whatever references its keyframe; `--regenerate-shot N` forces one shot's keyframe+clip. The Runway client (`video/gen/runway.py`) was modernized to `https://api.dev.runwayml.com` / `X-Runway-Version: 2024-11-06` with shared submit/poll/download/data-URI helpers; reference assets are padded (white bars, never cropped) into gen4_image's accepted 0.5–2.0 width/height window so tall panels don't 400, and transport failures surface as `VideoError` (the seam contract) so assembly's per-anchor degradation catches them; the legacy TikTok path (gen3a_turbo, 768:1280) is unchanged.

## Feasibility & risks

Verdict: doable — all components exist today; the work is integration, not research. Honest risk ranking:

**Low risk (proven, pure engineering):** MangaDex API fetching, SQLite state, Kokoro TTS, ffmpeg Ken Burns assembly, hardware fit (24 GB easily runs an 8B VLM at Q4–Q8 plus a small text model).

**Medium risk (quality, not feasibility):**

- **Recap quality on manga pages** — the load-bearing assumption. VLMs read manga imperfectly: irregular panel/bubble reading order, text-light action scenes, occasional hallucinated plot details (cf. the MangaLMM/MangaVQA research, which exists because Qwen2.5-VL-7B was only middling at manga). Target: "good enough for personal catch-up," not publishable. Prompt iteration matters more than model choice; the adapter layer exists partly so swapping models is cheap when a better one appears.
- **Panel-crop grounding for videos** — Qwen3-VL is trained for grounded box output, but reliability on irregular manga layouts is unproven. Mitigation: v1 videos use **full pages** with Ken Burns motion; per-panel crops are an upgrade, gated on the grounding proving reliable.

**Validation spike (before building the pipeline):** pull `qwen3-vl:8b-instruct`, feed it a few real manga pages, judge the recap quality by hand. ~1 hour, de-risks the entire project. If it disappoints, try `openbmb/minicpm-v4.5` and `qwen3-vl:8b-thinking` before writing pipeline code.

## Design decisions / trade-offs

- **Local inference by default, no fine-tuning**: recaps are a prompting problem; current local models (Qwen3-VL, MiniCPM-V 4.5) are strong enough at manga reading that training a model isn't warranted. Remote inference (hosted APIs, rented vLLM pods) is opt-in via the `openai_compat` backend for users who want 32B–72B-class quality.
- **Quantization as a first-class concept**: on 24 GB, quant choice is the difference between "8B at full quality" and "12B comfortably, 32B borderline". The harness detects hardware, selects the highest-quality quant that fits, and **alerts rather than silently degrades** — quantized artifacts can measurably hurt OCR of small manga text, and the user deserves to know which artifact produced each recap.
- **Qwen3-VL as the default vision model**: newest architecture with dynamic high-resolution input (tall manga pages stay intact) and best-in-class Japanese/CJK OCR — the two things manga recap needs most.
- **Adapter + registry over hard-coded Ollama**: model runtimes change fast; the adapter protocol isolates that churn. `ensure()`/`list_available()` make each adapter own its registry interaction (Ollama library, HF Hub), including per-quant size reporting.
- **Adapter + registry over pluggy**: one small shared `PluginRegistry` (plugins.py) instead of a plugin framework; entry-point discovery is adopted via stdlib `importlib.metadata` for third-party plugins.
- **Ollama as the default adapter**: already installed (v0.32.15) with models pulled; the `openai_compat` and `huggingface` adapters keep hosted APIs, llama.cpp/LM Studio/vLLM servers, and HF GGUFs usable without new code paths.
- **Separate vision/text roles**: page reading needs vision; rolling-summary compression and scriptwriting don't — using a small text model for them saves time and RAM.
- **Vision model over OCR+text-LLM**: manga OCR (speech-bubble extraction) is fragile; modern VLMs read pages directly. Cost: slower, more RAM — acceptable locally on Apple Silicon.
- **Video visuals: real pages + Ken Burns over AI-generated imagery**: authentic and cheap; local generative video can't keep characters consistent scene-to-scene; also the most defensible framing for personal use since no new derivative art is fabricated.
- **Audio-first timing**: TTS segment durations are authoritative and visuals conform to them — avoids narration/sync drift, the most common failure mode of auto-generated recap videos.
- **ffmpeg via subprocess over moviepy**: full filter control (`zoompan`, xfade, subtitles) with no heavy abstraction dependency.
- **Staged, cached video pipeline**: script → audio → crops → assembly artifacts all persist, so regenerating a video after tweaking a script costs seconds, not a full re-run.
- **MangaDex**: official public REST API, no auth needed for reading; chapter images via documented at-home endpoint.
- **Weeb Central as the scanlation source**: no official API, but its server-rendered HTMX endpoints (`/search/simple`, `/series/<id>/full-chapter-list`, `/chapters/<id>/images`) are stable and scrapeable with plain httpx + regex — covers English scanlations MangaDex doesn't host (e.g. Kenja no Mago). English-only by nature, so `Chapter.lang` is always "en". Fragility is contained in one module; regexes are the revisit point if the site restructures.
- **Source protocol** (`sources/`): keeps the door open for future aggregators without v1 over-engineering — one concrete source, one small interface.

## Future (out of scope for v1, noted for design continuity)

- Third-party plugin packages (entry-point discovery exists; no external plugins yet), `mlx` adapter, KV-cache-aware fit estimation, more sources (manga and other media), volume/arc rollups, video extras (background music, arc compilations, generated-art transitions), native experience app, scheduled auto-sync + auto-recap + auto-video.

## References

- [Best Local Vision Models 2026: LLaVA, Qwen3-VL & Ollama](https://www.promptquorum.com/power-local-llm/local-vision-models-llava-ollama-2026)
- [How to Run Qwen3-VL Locally with Ollama](https://apidog.com/blog/how-to-run-qwen-3-vl-locally-with-ollama/)
- [MiniCPM-V 4.5 on Ollama](https://ollama.com/openbmb/minicpm-v4.5)
- [Kokoro-82M TTS](https://huggingface.co/hexgrad/Kokoro-82M)
