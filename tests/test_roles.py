"""Roles-as-data tests: bindings, aliases, fallbacks, capability validation."""

from __future__ import annotations

import pytest

from entertainment_harness.config import Config, parse_config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import REGISTRY, resolve_role
from entertainment_harness.models.roles import (
    ROLES,
    RoleSpec,
    register_role,
    role_config,
)
from entertainment_harness.plugins import PluginError

GB = 10**9


class _FakeAdapter:
    """In-memory backend whose ModelInfo capabilities tests can set."""

    name = "fake-roles"
    remote = False
    capabilities = frozenset()
    caps: frozenset[str] = frozenset()  # reported on every ModelInfo

    def __init__(self, config: Config | None = None) -> None:
        pass

    def supports(self, model: str) -> ModelInfo:
        return ModelInfo(
            model, self.name, quant="Q8_0", size_bytes=GB, capabilities=self.caps
        )

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def remove(self, model: str) -> None:
        pass

    def list_available(self) -> list[ModelInfo]:
        return []

    def list_remote(self, model: str) -> list[ModelInfo] | None:
        return None

    def generate(self, model, prompt, images=None) -> str:
        return ""


REGISTRY.register("fake-roles", _FakeAdapter)


@pytest.fixture(autouse=True)
def _clean_role_state():
    saved = dict(ROLES)
    _FakeAdapter.caps = frozenset()
    yield
    ROLES.clear()
    ROLES.update(saved)
    _FakeAdapter.caps = frozenset()


def _profile() -> HardwareProfile:
    return HardwareProfile(
        chip="test", total_ram_bytes=64 * GB, budget_bytes=32 * GB, gpu_backend="cpu"
    )


# --- bindings, aliases, fallbacks ----------------------------------------------


def test_builtin_defaults_match_legacy_models_section():
    config = Config()
    assert role_config("vision", config).model == "qwen3-vl:8b-instruct"
    assert role_config("text", config).model == "qwen3:4b"
    # Unconfigured roles fall back to the text role, exactly as before.
    assert role_config("translation", config).model == "qwen3:4b"
    assert role_config("judge", config).model == "qwen3:4b"


def test_legacy_models_section_alias_still_binds():
    config = parse_config(
        {"models": {"text": {"backend": "fake-roles", "model": "legacy-text"}}}
    )
    assert role_config("text", config).model == "legacy-text"
    # The judge fallback follows the legacy text alias too.
    assert role_config("judge", config).model == "legacy-text"


def test_roles_table_wins_over_legacy_alias():
    config = parse_config({
        "models": {"vision": {"model": "legacy-vision"}},
        "roles": {"vision": {"backend": "fake-roles", "model": "new-vision"}},
    })
    assert role_config("vision", config).model == "new-vision"


def test_roles_table_binds_custom_role():
    register_role(RoleSpec("picker", default_backend="fake-roles"))
    config = parse_config({"roles": {"picker": {"model": "picker-model"}}})
    assert role_config("picker", config).model == "picker-model"


def test_custom_role_default_model_used_when_unbound():
    register_role(
        RoleSpec("picker", default_model="picker-default", default_backend="fake-roles")
    )
    assert role_config("picker", Config()).model == "picker-default"


# --- resolution + capability validation ------------------------------------------


def test_resolve_role_uses_roles_table_binding():
    config = parse_config({"roles": {"text": {"backend": "fake-roles", "model": "m1"}}})
    selection = resolve_role("text", config, _profile())
    assert selection.info.name == "m1"
    assert selection.warning is None


def test_capability_mismatch_warns_when_capabilities_known():
    register_role(
        RoleSpec("vision2", requires=frozenset({"vision"}), default_backend="fake-roles")
    )
    _FakeAdapter.caps = frozenset({"chat"})  # known, and missing "vision"
    config = parse_config({"roles": {"vision2": {"model": "m1"}}})
    selection = resolve_role("vision2", config, _profile())
    assert "vision2" in selection.warning
    assert "vision" in selection.warning


def test_unknown_capabilities_pass_without_warning():
    register_role(
        RoleSpec("vision2", requires=frozenset({"vision"}), default_backend="fake-roles")
    )
    _FakeAdapter.caps = frozenset()  # adapter does not report capabilities
    config = parse_config({"roles": {"vision2": {"model": "m1"}}})
    selection = resolve_role("vision2", config, _profile())
    assert selection.warning is None


def test_strict_mode_turns_mismatch_into_error():
    register_role(
        RoleSpec("vision2", requires=frozenset({"vision"}), default_backend="fake-roles")
    )
    _FakeAdapter.caps = frozenset({"chat"})
    config = parse_config({
        "roles": {"vision2": {"model": "m1"}},
        "plugins": {"strict": True},
    })
    assert config.plugins.strict is True
    with pytest.raises(PluginError, match="vision2"):
        resolve_role("vision2", config, _profile())


def test_builtin_vision_role_requires_vision_capability():
    assert ROLES["vision"].requires == frozenset({"vision"})
    assert ROLES["translation"].fallback == "text"
    assert ROLES["judge"].fallback == "text"
