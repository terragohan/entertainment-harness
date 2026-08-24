"""Registry: config + HardwareProfile -> adapter instance + quant selection.

Adding a new backend is a one-line register() call plus its adapter module;
third-party backends can register via the entertainment_harness.model_backends
entry-point group (see plugins.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from entertainment_harness.config import Config, ModelRoleConfig
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import (
    ModelAdapter,
    ModelInfo,
    ModelNotFoundError,
    ModelTooLargeError,
)
from entertainment_harness.models.roles import ROLES, role_config
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginError, PluginRegistry

# Backend name (from config) -> adapter class. New adapters register here;
# dotted strings keep the registry import cheap (no backend imports until used).
REGISTRY = PluginRegistry("model_backend", ENTRY_POINT_GROUPS["model_backend"])
REGISTRY.register("ollama", "entertainment_harness.models.ollama:OllamaAdapter")
REGISTRY.register("huggingface", "entertainment_harness.models.huggingface:HFAdapter")
REGISTRY.register(
    "openai_compat", "entertainment_harness.models.openai_compat:OpenAICompatAdapter"
)
# Optional dataset extra; scanlation stages use backend = "lfm" (an alias for
# this registered backend, resolved like any other plugin).
REGISTRY.register("lfm", "entertainment_harness.models.lfm:LFMAdapter")

# Quality-first preference; reversed when quant_policy = "prefer-speed".
# F16/BF16 sources are deliberately absent: they rank last (they are
# quantization sources, not serving targets).
QUANT_PREFERENCE = ["Q8_0", "Q6_K", "Q4_K_M", "Q4_0"]

FULL_PRECISION = ("F16", "BF16", "F32")

TIGHT_HEADROOM_BYTES = int(2e9)  # <2 GB headroom => "tight"

# Smallest realistic quant is ~0.6 bytes per parameter (docs/design.md model
# table, Q4 column). Used as a floor estimate when no real size is known.
BYTES_PER_PARAM = 0.6e9

# Ollama tag convention: "name:8b", "name:1.5b", "name:8b-instruct".
_TAG_PARAMS_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)[bB]")


@dataclass
class Selection:
    adapter: ModelAdapter
    info: ModelInfo
    pinned: bool = False
    warning: str | None = None


def get_adapter(backend: str, config: Config | None = None) -> ModelAdapter:
    """Instantiate a backend adapter; unknown names raise PluginError."""
    return REGISTRY.create(backend, config)


def fit_verdict(size_bytes: int | None, budget_bytes: int) -> str:
    if size_bytes is None:
        return "unknown"
    if size_bytes > budget_bytes:
        return "too large"
    if budget_bytes - size_bytes < TIGHT_HEADROOM_BYTES:
        return "tight"
    return "fits"


def _base_name(tag: str) -> str:
    return tag.split(":", 1)[0]


def estimate_size_bytes(info: ModelInfo) -> int | None:
    """Known on-disk size, else a floor estimate from the parameter count.

    Params come from registry metadata when available, else from the tag
    itself ("qwen3-vl:32b" -> 32B) — Ollama 404s /api/show for non-local
    models, so the tag is the last-resort signal that keeps us from pulling
    a 20 GB artifact that cannot fit.
    """
    if info.size_bytes is not None:
        return info.size_bytes
    params = info.params
    if params is None and ":" in info.name:
        match = _TAG_PARAMS_RE.match(info.name.split(":", 1)[1])
        if match:
            params = float(match.group(1))
    return int(params * BYTES_PER_PARAM) if params else None


def _preference_order(policy: str) -> list[str]:
    if policy == "prefer-speed":
        return list(reversed(QUANT_PREFERENCE))
    return list(QUANT_PREFERENCE)


def select_quant(
    candidates: list[ModelInfo],
    budget_bytes: int,
    policy: str = "prefer-quality",
    alternatives: list[ModelInfo] | None = None,
) -> tuple[ModelInfo, str | None]:
    """Pick the highest-preference candidate that fits the budget.

    Returns (chosen, warning). Never silently downgrades: a below-Q8 pick or
    a tight fit comes back with a warning string. Raises ModelTooLargeError
    (listing what would fit) when nothing fits.
    """
    if not candidates:
        raise ModelTooLargeError("unknown", alternatives or [])

    order = _preference_order(policy)

    def rank(info: ModelInfo) -> int:
        quant = (info.quant or "").upper()
        return order.index(quant) if quant in order else len(order)

    fitting = [
        c
        for c in sorted(candidates, key=rank)
        if c.size_bytes is None or c.size_bytes <= budget_bytes
    ]
    if not fitting:
        model = candidates[0].name
        raise ModelTooLargeError(model, alternatives or [])

    chosen = fitting[0]
    warning: str | None = None
    quant = (chosen.quant or "").upper()
    if chosen.size_bytes is not None:
        headroom = budget_bytes - chosen.size_bytes
        if quant and quant != "Q8_0" and quant not in FULL_PRECISION:
            warning = (
                f"{chosen.name} resolved to {chosen.quant} "
                f"({chosen.size_gb:.1f} GB). Quantized below 8-bit — OCR "
                "accuracy on small text may degrade."
            )
            if headroom < TIGHT_HEADROOM_BYTES:
                warning += " Fit is tight on this machine (<2 GB headroom)."
        elif headroom < TIGHT_HEADROOM_BYTES:
            warning = (
                f"{chosen.name} ({chosen.size_gb:.1f} GB) fits, but with "
                "<2 GB headroom."
            )
    return chosen, warning


def resolve(
    role: ModelRoleConfig,
    profile: HardwareProfile,
    config: Config,
    policy: str = "prefer-quality",
) -> Selection:
    adapter = get_adapter(role.backend, config)

    if getattr(adapter, "remote", False):
        # Remote backends (openai_compat) have no local footprint — the
        # quant/memory-budget machinery must not refuse a 72B hosted model
        # over local RAM. role.quant is ignored for remote backends.
        return Selection(adapter=adapter, info=adapter.supports(role.model))

    if role.quant:
        # Explicit pin: skip auto-selection entirely.
        tag = (
            role.model
            if role.quant.lower() in role.model.lower()
            else f"{role.model}-{role.quant.lower()}"
        )
        try:
            info = adapter.supports(tag)
        except ModelNotFoundError:
            try:
                info = adapter.supports(role.model)
            except ModelNotFoundError:
                # Unknown to the registry until pulled (some Ollama versions
                # 404 /api/show for non-local models); ensure() will pull it.
                info = ModelInfo(name=tag, backend=role.backend)
        warnings = []
        if role.quant.upper() != "Q8_0":
            warnings.append(
                f"Pinned quant {role.quant} of {info.name} is below 8-bit — "
                "OCR accuracy on small text may degrade."
            )
        size = estimate_size_bytes(info)
        if size is not None and size > profile.budget_bytes:
            warnings.append(
                f"Pinned {info.name} (~{size / 1e9:.1f} GB) exceeds the memory"
                f" budget ({profile.budget_gb:.1f} GB)."
            )
        return Selection(
            adapter=adapter, info=info, pinned=True,
            warning=" ".join(warnings) or None,
        )

    available = adapter.list_available()
    if ":" in role.model:
        # An explicit tag requests that artifact; a different tag of the same
        # base name is a different model size, not a quant of it — never
        # silently substitute it.
        candidates = [i for i in available if i.name == role.model]
    else:
        candidates = [i for i in available if _base_name(i.name) == _base_name(role.model)]
    if not candidates:
        remote = adapter.list_remote(role.model)
        if remote is not None:
            # Registries with per-quant remote sizes (HF Hub) let the quant
            # policy run before anything is downloaded.
            if not remote:
                raise ModelNotFoundError(
                    f"{role.model!r} publishes no GGUF artifacts"
                )
            fitting_alternatives = [
                i for i in available
                if estimate_size_bytes(i) is not None
                and estimate_size_bytes(i) <= profile.budget_bytes
            ]
            chosen, warning = select_quant(
                remote, profile.budget_bytes, policy, fitting_alternatives
            )
            return Selection(adapter=adapter, info=chosen, warning=warning)
        # Not installed locally; supports() resolves registry metadata and
        # ensure() will pull it at generation time. Some Ollama versions 404
        # /api/show for non-local models — then fall back to a tag-based size
        # estimate so a 32B-class request is refused instead of pulled.
        try:
            info = adapter.supports(role.model)
        except ModelNotFoundError:
            info = ModelInfo(name=role.model, backend=role.backend)
        size = estimate_size_bytes(info)
        if size is not None and size > profile.budget_bytes:
            fitting = [
                i
                for i in available
                if estimate_size_bytes(i) is not None
                and estimate_size_bytes(i) <= profile.budget_bytes
            ]
            raise ModelTooLargeError(role.model, fitting)
        return Selection(adapter=adapter, info=info)

    fitting_alternatives = [
        i
        for i in available
        if _base_name(i.name) != _base_name(role.model)
        and i.size_bytes is not None
        and i.size_bytes <= profile.budget_bytes
    ]
    chosen, warning = select_quant(
        candidates, profile.budget_bytes, policy, fitting_alternatives
    )
    return Selection(adapter=adapter, info=chosen, warning=warning)


def resolve_role(name: str, config: Config, profile: HardwareProfile) -> Selection:
    """Resolve the model bound to a role (bindings: models/roles.py).

    The pick is validated against the role's declared capabilities: a known
    mismatch warns by default and raises PluginError under
    [plugins] strict = true. Models whose capabilities are unknown (empty
    frozenset — adapters that don't report them) pass without a warning.
    """
    role = role_config(name, config)
    selection = resolve(role, profile, config, config.models.quant_policy)
    spec = ROLES.get(name)
    missing = (
        spec.requires - selection.info.capabilities
        if spec is not None and selection.info.capabilities
        else frozenset()
    )
    if missing:
        message = (
            f"role {name!r} requires capabilities {sorted(missing)} but "
            f"{selection.info.name} declares {sorted(selection.info.capabilities)}"
        )
        if config.plugins.strict:
            raise PluginError(message)
        selection.warning = (
            f"{selection.warning} {message}" if selection.warning else message
        )
    return selection


def get_vision_model(config: Config, profile: HardwareProfile) -> Selection:
    return resolve_role("vision", config, profile)


def get_text_model(config: Config, profile: HardwareProfile) -> Selection:
    return resolve_role("text", config, profile)


def get_judge_model(config: Config, profile: HardwareProfile) -> Selection:
    """The judge role defaults to the text-role model when unconfigured."""
    return resolve_role("judge", config, profile)


def get_translation_model(config: Config, profile: HardwareProfile) -> Selection:
    """The translation role defaults to the text-role model when unconfigured."""
    return resolve_role("translation", config, profile)
