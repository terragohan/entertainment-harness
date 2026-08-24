"""Model roles as data.

Every place the harness needs a model binds to a named role instead of a
hardcoded model. A RoleSpec declares what the role requires (capabilities
the bound model must have), its default model, and the fallback role used
when it is left unconfigured; ROLES is the registry third-party plugins can
extend with register_role().

Config binds roles with [roles.<name>] tables. The legacy
[models.vision|text|translation|judge] tables are aliases for the built-in
roles — identical effective config; [roles.<name>] wins when both are set.
"""

from __future__ import annotations

from dataclasses import dataclass

from entertainment_harness.config import Config, ModelRoleConfig


@dataclass(frozen=True)
class RoleSpec:
    name: str
    requires: frozenset[str] = frozenset()  # capabilities the bound model must have
    default_model: str = ""
    default_backend: str = "ollama"
    fallback: str = ""  # role whose binding is used when this one is unconfigured


ROLES: dict[str, RoleSpec] = {}


def register_role(spec: RoleSpec) -> None:
    """Add or replace a role; entry-point plugins call this at import time."""
    ROLES[spec.name] = spec


register_role(
    RoleSpec(
        "vision",
        requires=frozenset({"vision"}),
        default_model="qwen3-vl:8b-instruct",
    )
)
register_role(RoleSpec("text", default_model="qwen3:4b"))
register_role(RoleSpec("translation", fallback="text"))
register_role(RoleSpec("judge", fallback="text"))

# Built-in roles whose [models.<name>] table is a legacy alias.
LEGACY_ALIASES = ("vision", "text", "translation", "judge")


def role_config(name: str, config: Config) -> ModelRoleConfig:
    """The effective binding for a role.

    Precedence: [roles.<name>] > the legacy [models.<name>] alias > RoleSpec
    defaults. An empty model follows the spec's fallback role, then the
    spec's default model. Unknown roles resolve to whatever was configured
    (empty binding = the resolve() default path).
    """
    spec = ROLES.get(name)
    default_backend = spec.default_backend if spec is not None else "ollama"
    if name in config.roles:
        role = config.roles[name]
    elif name in LEGACY_ALIASES:
        role = getattr(config.models, name)
    else:
        role = ModelRoleConfig()
    if role.model:
        if not role.backend:
            role = ModelRoleConfig(
                backend=default_backend, model=role.model, quant=role.quant
            )
        return role
    if spec is not None and spec.fallback:
        return role_config(spec.fallback, config)
    if spec is not None and spec.default_model:
        return ModelRoleConfig(
            backend=role.backend or default_backend,
            model=spec.default_model,
            quant=role.quant,
        )
    if not role.backend:
        role = ModelRoleConfig(
            backend=default_backend, model=role.model, quant=role.quant
        )
    return role
