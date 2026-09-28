"""Paths and config.toml loading.

Data dir defaults to ~/.local/share/entertainment-harness and can be
overridden with the EH_DATA_DIR environment variable (dev/tests).
Re-downloadable weights (models, TTS files) live in the cache dir, which
defaults to the XDG cache dir (~/.cache/entertainment-harness) and can be
overridden with EH_CACHE_DIR. If config.toml is absent, defaults are used
silently.

load_config() parses; save_config() writes (tomlkit, preserving hand-edited
comments/formatting, validated against the dataclasses, atomic replace).
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import tomlkit

DATA_DIR_ENV = "EH_DATA_DIR"
CACHE_DIR_ENV = "EH_CACHE_DIR"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "entertainment-harness"
CONFIG_FILENAME = "config.toml"

# Built-in content sources enabled by default (see SourcesConfig).
DEFAULT_ENABLED_SOURCES = ["mangadex", "weebcentral"]


def data_dir() -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_DATA_DIR


def cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "entertainment-harness"


@dataclass
class ModelRoleConfig:
    backend: str = "ollama"
    model: str = ""
    quant: str | None = None  # optional pin; None = auto-select


@dataclass
class StageConfig:
    """One stage of a scanlation job (extract, translate, render, judge)."""

    backend: str = ""
    model: str = ""
    # Any backend-specific keys that don't fit the generic shape (e.g.
    # include_text for the LFM backend) are preserved here.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanlationJobConfig:
    """A complete scanlation pipeline configured as named stages."""

    extract: StageConfig = field(default_factory=StageConfig)
    translate: StageConfig = field(default_factory=StageConfig)
    # When omitted, the judge stage defaults to the translate stage.
    judge: StageConfig | None = None
    render: StageConfig = field(default_factory=StageConfig)

    @property
    def judge_stage(self) -> StageConfig:
        return self.judge if self.judge is not None else self.translate


@dataclass
class ScanlationConfig:
    """Active scanlation job selection."""

    job: str = ""


@dataclass
class JobsConfig:
    """Named job pipelines. Currently only scanlation is supported."""

    scanlation: dict[str, ScanlationJobConfig] = field(default_factory=dict)


@dataclass
class OpenAICompatConfig:
    """Remote OpenAI-compatible endpoint (OpenRouter default, or a vLLM pod)."""
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""  # falls back to OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"


@dataclass
class ModelsConfig:
    quant_policy: str = "prefer-quality"  # or "prefer-speed"
    vision: ModelRoleConfig = field(
        default_factory=lambda: ModelRoleConfig(model="qwen3-vl:8b-instruct")
    )
    text: ModelRoleConfig = field(default_factory=lambda: ModelRoleConfig(model="qwen3:4b"))
    # Translation text model. Empty model = fall back to the text role.
    translation: ModelRoleConfig = field(default_factory=ModelRoleConfig)
    judge: ModelRoleConfig = field(default_factory=ModelRoleConfig)  # "" = use text role
    # Path to a SAM checkpoint. Empty string disables SAM segmentation.
    # Use "auto" to download the default ViT-B checkpoint on first use.
    sam_checkpoint: str = ""
    # LiquidAI LFM2.5-VL model ID for bubble localization/OCR.
    # Empty string disables LFM and uses the vision role model instead.
    lfm_model: str = ""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai_compat: OpenAICompatConfig = field(default_factory=OpenAICompatConfig)


@dataclass
class HardwareConfig:
    budget_gb: float | None = None  # optional override; default = auto-detected


@dataclass
class PreflightConfig:
    enabled: bool = True
    min_free_disk_gb: float | None = None  # overrides auto-derived disk need
    min_free_memory_gb: float | None = None  # overrides auto-derived memory need
    skip_gpu_check: bool = False


@dataclass
class VideoConfig:
    tts_engine: str = "kokoro"  # "kokoro", "say", or "qwen3" (voice cloning)
    voice: str = "af_heart"  # engine-specific voice id
    resolution: str = "1920x1080"
    # Presentation style: "kenburns" (per-page pan/zoom), "scroll" (viewport
    # descends a per-segment page strip), "slideshow" (still pages dissolving
    # within a segment), "cards" (caption cards; the only mode for books),
    # "panels" (per-beat panel crops), "motion" (generated clips via the
    # [video_gen] provider), "animate" (AI-generated frames of each beat's
    # panel, via the [frames] provider). --video-mode overrides.
    mode: str = "kenburns"
    colorize: bool = False  # opt-in DDColor page colorization for videos
    translated: bool = False  # opt-in: use translated pages (eh translate) in videos
    steering_prompt: str = ""  # custom direction for tiktok scripts
    # Default compression preset applied after every video render ("" = off;
    # --compress on the command line overrides). See video/compress.py.
    compress: str = ""
    # Keep the full-quality master out.mp4 when a compressed copy exists.
    # False makes the compressed copy the deliverable (eh play, store).
    keep_master: bool = True
    # Append the credits end card (work, author when known, source, chapter,
    # and the tool) to every rendered video. Sharing generated content
    # requires that attribution (see SHARING.md); turn off only for private
    # viewing. render_state.json records it, so toggling re-renders.
    credits: bool = True
    # Anchored-scroll Phase 4: build the video script from the chapter's
    # cached panel beats (segments born with their panel spans; no recap
    # artifact, page assignment, or grounding judge). This is the default
    # video script path; panel_first = false here (or --no-panel-first per
    # run) opts out into the grounded assign+grounding path. Forces the
    # narration kind.
    panel_first: bool = True


@dataclass
class SearchConfig:
    provider: str = "duckduckgo"  # internal registry key
    max_sources: int = 6


@dataclass
class RunwayConfig:
    api_key: str = ""  # falls back to RUNWAY_API_KEY env var
    model: str = "gen3a_turbo"


@dataclass
class VideoGenConfig:
    provider: str = "local"  # or "runway"
    runway: RunwayConfig = field(default_factory=RunwayConfig)


@dataclass
class FramesConfig:
    # Frame animation for [video] mode = "animate": an image generator
    # produces per-beat animation frames from the grounded panel crop.
    provider: str = "local"  # or "runway" / "openrouter"
    model: str = ""  # empty = the provider's default image model
    seconds_per_frame: float = 2.0  # generated-frame spacing at assembly
    max_frames: int = 6  # cost cap per beat


@dataclass
class ColorizeConfig:
    # Page colorization for [video] colorize = true, via the colorize plugin
    # registry (video/colorize.py).
    provider: str = "ddcolor"


@dataclass
class SequenceConfig:
    # Frame-by-frame panel animation for [video] mode = "sequence": the
    # panel is outpainted to video size, then frames chain off each other.
    provider: str = "sequence"  # frames-registry animator name
    model: str = ""  # empty = the provider's default image model
    fps: float = 1.5  # generated frames per second of slot share
    max_frames: int = 12  # cost cap per anchor
    interp_fps: int = 30  # minterpolate playback fps at assembly (= pipeline FPS)
    critic: bool = True  # vision-role drift critic per chained frame


@dataclass
class AnimeConfig:
    # Models for the guided manga→anime scene pipeline (eh anime-scene).
    provider: str = "runway"  # video_gen plugin with image-gen + image-to-video
    keyframe_model: str = "gen4_image"  # Runway text_to_image
    video_model: str = "gen4.5"  # Runway image_to_video (alt: seedance2_5)


@dataclass
class R2Config:
    """Cloudflare R2 (S3-compatible) video store."""
    bucket: str = ""
    account_id: str = ""  # endpoint = https://<account_id>.r2.cloudflarestorage.com
    access_key_id: str = ""  # falls back to R2_ACCESS_KEY_ID
    secret_access_key: str = ""  # falls back to R2_SECRET_ACCESS_KEY


@dataclass
class StoreConfig:
    provider: str = "hf"  # or "r2"
    repo: str | None = None  # HF dataset repo id ("owner/name") for page backup
    r2: R2Config = field(default_factory=R2Config)


@dataclass
class LibraryConfig:
    # Languages the user is happy to read / recap in. The first language is
    # the default translation target for `eh recap --translated`.
    langs: list[str] = field(default_factory=lambda: ["en"])


@dataclass
class PipelineConfig:
    thinking: str = "medium"  # low | medium | high — critiquing depth
    # gist | brief | standard | detailed | full — artifact grain for
    # 'eh recap' (full = the old narration); --detail wins over this default.
    detail: str = "standard"
    # steering direction injected into every grain's prompts ("" = none);
    # --instruction wins over this default. Recorded on artifacts for
    # attribution; changing it does not invalidate existing artifacts.
    instructions: str = ""
    # character bible: after each chapter artifact, extract/merge the cast
    # registry and inject it into later chapters' prompts (see
    # pipelines/characters.py). Off = no extra model calls.
    characters: bool = True


@dataclass
class SourcesConfig:
    # Built-in content sources that are enabled. Only built-in names absent
    # from this list are blocked; third-party (entry-point) sources are
    # always usable. Managed via `eh sources enable|disable|reset`.
    enabled: list[str] = field(default_factory=lambda: list(DEFAULT_ENABLED_SOURCES))


@dataclass
class PluginsConfig:
    # strict = true turns capability-fit warnings (a role bound to a model
    # that doesn't declare the role's required capabilities) into errors.
    strict: bool = False


@dataclass
class Config:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    scanlation: ScanlationConfig = field(default_factory=ScanlationConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    # Per-TTS-engine settings from [tts.<name>] tables, keyed by engine name
    # (e.g. config.tts["qwen3"]["model"]). Passed as constructor overrides.
    tts: dict[str, dict] = field(default_factory=dict)
    search: SearchConfig = field(default_factory=SearchConfig)
    video_gen: VideoGenConfig = field(default_factory=VideoGenConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    colorize: ColorizeConfig = field(default_factory=ColorizeConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    anime: AnimeConfig = field(default_factory=AnimeConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    # Role bindings from [roles.<name>] tables, keyed by role name. These win
    # over the legacy [models.vision|text|translation|judge] aliases; see
    # models/roles.py for resolution.
    roles: dict[str, ModelRoleConfig] = field(default_factory=dict)
    # The parsed config.toml, kept so PluginRegistry.create() can merge
    # per-plugin [<section>.<plugin>] tables into constructors generically —
    # plugins with settings need no parser changes here. Empty for Config()
    # built directly (tests), which disables the merge.
    raw: dict = field(default_factory=dict)


def _role_config(data: dict, default_model: str, default_backend: str = "ollama") -> ModelRoleConfig:
    return ModelRoleConfig(
        backend=data.get("backend", default_backend),
        model=data.get("model", default_model),
        quant=data.get("quant"),
    )


def _stage_config(data: dict) -> StageConfig:
    return StageConfig(
        backend=str(data.get("backend", "")),
        model=str(data.get("model", "")),
        extra={k: v for k, v in data.items() if k not in ("backend", "model")},
    )


def _scanlation_job_config(data: dict) -> ScanlationJobConfig:
    return ScanlationJobConfig(
        extract=_stage_config(data.get("extract", {})),
        translate=_stage_config(data.get("translate", {})),
        judge=_stage_config(data.get("judge", {})) if "judge" in data else None,
        render=_stage_config(data.get("render", {})),
    )


def load_config(path: Path | None = None) -> Config:
    config_path = path if path is not None else data_dir() / CONFIG_FILENAME
    if not config_path.exists():
        return Config()

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)
    return parse_config(raw)


def parse_config(raw: dict) -> Config:
    """Build a Config from an already-parsed TOML dict (missing keys = defaults)."""
    config = Config()
    config.raw = raw

    hardware = raw.get("hardware", {})
    config.hardware.budget_gb = hardware.get("budget_gb")

    models = raw.get("models", {})
    config.models.quant_policy = models.get("quant_policy", "prefer-quality")
    config.models.vision = _role_config(models.get("vision", {}), "qwen3-vl:8b-instruct")
    config.models.text = _role_config(models.get("text", {}), "qwen3:4b")
    config.models.translation = _role_config(models.get("translation", {}), "")
    config.models.judge = _role_config(models.get("judge", {}), "")
    config.models.sam_checkpoint = str(models.get("sam_checkpoint", ""))
    config.models.lfm_model = str(models.get("lfm_model", ""))
    config.models.ollama.base_url = str(
        models.get("ollama", {}).get("base_url", "http://localhost:11434")
    )
    openai_compat = models.get("openai_compat", {})
    config.models.openai_compat.base_url = str(
        openai_compat.get("base_url", "https://openrouter.ai/api/v1")
    )
    config.models.openai_compat.api_key = str(openai_compat.get("api_key", ""))

    roles = raw.get("roles", {})
    # An unset backend stays "" so role_config() can apply the RoleSpec's
    # default_backend (models/roles.py); legacy [models.*] keeps "ollama".
    config.roles = {
        str(name): _role_config(table, "", default_backend="")
        for name, table in roles.items() if isinstance(table, dict)
    }

    plugins = raw.get("plugins", {})
    config.plugins.strict = bool(plugins.get("strict", False))

    scanlation = raw.get("scanlation", {})
    config.scanlation.job = str(scanlation.get("job", ""))

    jobs = raw.get("jobs", {})
    scanlation_jobs = jobs.get("scanlation", {})
    if isinstance(scanlation_jobs, dict):
        default_name = scanlation_jobs.get("default")
        if isinstance(default_name, str):
            config.scanlation.job = config.scanlation.job or default_name
        for name, job_data in scanlation_jobs.items():
            if name == "default" or not isinstance(job_data, dict):
                continue
            config.jobs.scanlation[name] = _scanlation_job_config(job_data)

    video = raw.get("video", {})
    config.video.tts_engine = video.get("tts_engine", "kokoro")
    config.video.voice = video.get("voice", "af_heart")
    config.video.resolution = video.get("resolution", "1920x1080")
    config.video.mode = str(video.get("mode", "kenburns"))
    config.video.colorize = bool(video.get("colorize", False))
    config.video.translated = bool(video.get("translated", False))
    config.video.steering_prompt = str(video.get("steering_prompt", ""))
    config.video.compress = str(video.get("compress", ""))
    config.video.keep_master = bool(video.get("keep_master", True))
    config.video.credits = bool(video.get("credits", True))
    config.video.panel_first = bool(video.get("panel_first", True))

    tts = raw.get("tts", {})
    config.tts = {
        str(name): dict(table) for name, table in tts.items() if isinstance(table, dict)
    }

    search = raw.get("search", {})
    config.search.provider = search.get("provider", "duckduckgo")
    config.search.max_sources = int(search.get("max_sources", 6))

    video_gen = raw.get("video_gen", {})
    config.video_gen.provider = video_gen.get("provider", "local")
    runway = video_gen.get("runway", {})
    config.video_gen.runway.api_key = str(runway.get("api_key", ""))
    config.video_gen.runway.model = str(runway.get("model", "gen3a_turbo"))

    anime = raw.get("anime", {})
    config.anime.provider = str(anime.get("provider", "runway"))
    config.anime.keyframe_model = str(anime.get("keyframe_model", "gen4_image"))
    config.anime.video_model = str(anime.get("video_model", "gen4.5"))

    frames = raw.get("frames", {})
    config.frames.provider = frames.get("provider", "local")
    config.frames.model = str(frames.get("model", ""))
    config.frames.seconds_per_frame = float(frames.get("seconds_per_frame", 2.0))
    config.frames.max_frames = int(frames.get("max_frames", 6))

    colorize = raw.get("colorize", {})
    config.colorize.provider = str(colorize.get("provider", "ddcolor"))

    sequence = raw.get("sequence", {})
    config.sequence.provider = str(sequence.get("provider", "sequence"))
    config.sequence.model = str(sequence.get("model", ""))
    config.sequence.fps = float(sequence.get("fps", 1.5))
    config.sequence.max_frames = int(sequence.get("max_frames", 12))
    config.sequence.interp_fps = int(sequence.get("interp_fps", 30))
    config.sequence.critic = bool(sequence.get("critic", True))

    store = raw.get("store", {})
    config.store.provider = store.get("provider", "hf")
    config.store.repo = store.get("repo")
    r2 = store.get("r2", {})
    config.store.r2.bucket = str(r2.get("bucket", ""))
    config.store.r2.account_id = str(r2.get("account_id", ""))
    config.store.r2.access_key_id = str(r2.get("access_key_id", ""))
    config.store.r2.secret_access_key = str(r2.get("secret_access_key", ""))

    library = raw.get("library", {})
    if "langs" in library:
        config.library.langs = [str(v) for v in library["langs"] if v]
    elif "lang" in library:
        # Backwards compatibility: single language string becomes a one-item list.
        config.library.langs = [str(library["lang"])]
    else:
        config.library.langs = ["en"]
    config.pipeline.thinking = raw.get("pipeline", {}).get("thinking", "medium")
    config.pipeline.detail = str(raw.get("pipeline", {}).get("detail", "standard"))
    config.pipeline.instructions = str(
        raw.get("pipeline", {}).get("instructions", "")
    )
    config.pipeline.characters = bool(
        raw.get("pipeline", {}).get("characters", True)
    )

    preflight = raw.get("preflight", {})
    config.preflight.enabled = bool(preflight.get("enabled", True))
    config.preflight.min_free_disk_gb = preflight.get("min_free_disk_gb")
    config.preflight.min_free_memory_gb = preflight.get("min_free_memory_gb")
    config.preflight.skip_gpu_check = bool(preflight.get("skip_gpu_check", False))

    sources = raw.get("sources", {})
    if "enabled" in sources:
        config.sources.enabled = [str(v) for v in sources["enabled"] if v]
    return config


class ConfigWriteError(ValueError):
    """Invalid config.toml update (unknown section/key or wrong type).

    Raised before the file is touched; the existing config is never corrupted.
    """


def _validate_value(hint: Any, value: Any, path: str) -> None:
    """Check a single update value against a dataclass field's type hint."""
    if hint is Any:
        return
    origin = get_origin(hint)
    if origin in (Union, UnionType):
        args = get_args(hint)
        if value is None and type(None) in args:
            return
        for arg in args:
            if arg is type(None):
                continue
            try:
                _validate_value(arg, value, path)
                return
            except ConfigWriteError:
                continue
        raise ConfigWriteError(
            f"{path}: value {value!r} matches none of the allowed types"
        )
    if origin is list:
        if not isinstance(value, list):
            raise ConfigWriteError(
                f"{path} must be a list, got {type(value).__name__}"
            )
        args = get_args(hint)
        item_hint = args[0] if args else Any
        for item in value:
            _validate_value(item_hint, item, path)
        return
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigWriteError(
                f"{path} must be a table, got {type(value).__name__}"
            )
        return  # free-form table contents (e.g. [tts.<engine>], [jobs.scanlation])
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigWriteError(
                f"{path} must be a bool, got {type(value).__name__}"
            )
    elif hint is str:
        if not isinstance(value, str):
            raise ConfigWriteError(
                f"{path} must be a string, got {type(value).__name__}"
            )
    elif hint is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigWriteError(
                f"{path} must be an int, got {type(value).__name__}"
            )
    elif hint is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigWriteError(
                f"{path} must be a number, got {type(value).__name__}"
            )
    elif isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigWriteError(
                f"{path} must be a table, got {type(value).__name__}"
            )
        _validate_fields(hint, value, path)


def _validate_fields(cls: type, updates: dict, prefix: str) -> None:
    hints = get_type_hints(cls)
    for key, value in updates.items():
        if key not in hints:
            known = ", ".join(sorted(hints))
            raise ConfigWriteError(
                f"unknown config key '{prefix}.{key}' (known: {known})"
            )
        _validate_value(hints[key], value, f"{prefix}.{key}")


def _validate_updates(updates: dict[str, Any]) -> None:
    hints = get_type_hints(Config)
    for section, sub in updates.items():
        if section not in hints:
            known = ", ".join(sorted(hints))
            raise ConfigWriteError(
                f"unknown config section {section!r} (known: {known})"
            )
        if not isinstance(sub, dict):
            raise ConfigWriteError(
                f"update for [{section}] must be a table of key/value pairs"
            )
        _validate_value(hints[section], sub, section)


def _apply_table(table: dict, updates: dict, cls: type) -> None:
    """Merge updates into a tomlkit table; nested dataclass fields recurse so
    untouched keys in the same sub-table (and their comments) survive."""
    hints = get_type_hints(cls)
    for key, value in updates.items():
        hint = hints.get(key)
        if isinstance(value, dict) and isinstance(hint, type) and is_dataclass(hint):
            sub = table.get(key)
            if not isinstance(sub, dict):
                sub = tomlkit.table()
                table[key] = sub
            _apply_table(sub, value, hint)
        else:
            table[key] = value


def save_config(config_path: Path, updates: dict[str, Any]) -> None:
    """Apply nested updates ({section: {key: value}}) to config.toml.

    Comments and formatting of the hand-edited file are preserved (tomlkit).
    Updates are validated against the Config dataclasses and the result is
    round-tripped through the real parsing code (parse_config) BEFORE
    anything is written — invalid updates raise ConfigWriteError with the
    file untouched. The write itself is atomic (temp file + os.replace).
    """
    _validate_updates(updates)

    config_path = Path(config_path)
    if config_path.exists():
        doc = tomlkit.parse(config_path.read_text())
    else:
        doc = tomlkit.document()
    _apply_table(doc, updates, Config)

    rendered = tomlkit.dumps(doc)
    # Validation must not drift from loading: parse what we would write
    # through the same code load_config uses.
    parse_config(tomllib.loads(rendered))

    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=config_path.parent,
        prefix=config_path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(rendered)
        os.replace(tmp_path, config_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
