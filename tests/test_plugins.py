"""PluginRegistry core tests + per-seam factory conformance."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest

from entertainment_harness.config import Config
from entertainment_harness.plugins import PluginError, PluginRegistry

UUID = "01234567-89ab-cdef-0123-456789abcdef"


class _Dummy:
    def __init__(self, config: Config | None = None, **overrides) -> None:
        self.config = config
        self.overrides = overrides


# --- PluginRegistry core -----------------------------------------------------


def test_register_create_names_roundtrip():
    reg = PluginRegistry("dummy", "entertainment_harness.test_dummy")
    reg.register("d", _Dummy)
    assert reg.names() == ["d"]
    plugin = reg.create("d", Config(), extra=1)
    assert isinstance(plugin, _Dummy)
    assert isinstance(plugin.config, Config)
    assert plugin.overrides == {"extra": 1}


def test_unknown_name_raises_plugin_error_listing_available():
    reg = PluginRegistry("dummy", "entertainment_harness.test_dummy")
    reg.register("d", _Dummy)
    with pytest.raises(PluginError, match=r"Unknown dummy plugin 'x'; available: d"):
        reg.create("x", Config())
    # PluginError subclasses ValueError so existing catches keep working.
    with pytest.raises(ValueError):
        reg.create("x", Config())


def test_lazy_dotted_registration_defers_import(monkeypatch):
    imported: list[str] = []
    real_import = importlib.import_module

    def tracking_import(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", tracking_import)
    reg = PluginRegistry("dummy", "entertainment_harness.test_lazy")
    reg.register("say", "entertainment_harness.video.tts:SayEngine")
    assert reg.names() == ["say"]
    assert "entertainment_harness.video.tts" not in imported
    assert reg.create("say", Config()).name == "say"
    assert "entertainment_harness.video.tts" in imported


# --- entry-point discovery ----------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, name: str, target) -> None:
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def test_entry_points_discovered_and_broken_skipped(monkeypatch):
    eps = [
        _FakeEntryPoint("external", _Dummy),
        _FakeEntryPoint("broken", ImportError("boom")),
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: eps if group == "entertainment_harness.test_ep" else [],
    )
    reg = PluginRegistry("dummy", "entertainment_harness.test_ep")
    reg.register("builtin", _Dummy)
    assert reg.names() == ["builtin", "external"]  # broken EP skipped
    assert reg.is_entry_point("external")
    assert not reg.is_entry_point("builtin")
    assert isinstance(reg.create("external", Config()), _Dummy)


# --- automatic per-plugin config (Phase 1) --------------------------------------


class _NamedParams:
    def __init__(
        self, config: Config | None = None, base_url: str = "x", timeout: int = 5
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout


class _ConfigOnly:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config


def _raw_config(table: dict) -> Config:
    config = Config()
    config.raw = {"video_gen": {"acme": table}}
    return config


def test_create_merges_plugin_config_section():
    reg = PluginRegistry("video_gen", "entertainment_harness.test_dummy")
    reg.register("acme", _NamedParams)
    plugin = reg.create("acme", _raw_config({"base_url": "http://h", "timeout": 9}))
    assert plugin.base_url == "http://h"
    assert plugin.timeout == 9


def test_create_explicit_overrides_beat_config_section():
    reg = PluginRegistry("video_gen", "entertainment_harness.test_dummy")
    reg.register("acme", _NamedParams)
    plugin = reg.create("acme", _raw_config({"base_url": "http://h"}), base_url="http://w")
    assert plugin.base_url == "http://w"


def test_create_drops_unknown_keys_for_named_params():
    reg = PluginRegistry("video_gen", "entertainment_harness.test_dummy")
    reg.register("acme", _NamedParams)
    plugin = reg.create("acme", _raw_config({"timeout": 9, "bogus": 1}))
    assert plugin.timeout == 9
    assert not hasattr(plugin, "bogus")


def test_create_var_kwargs_gets_all_keys():
    reg = PluginRegistry("video_gen", "entertainment_harness.test_dummy")
    reg.register("acme", _Dummy)
    plugin = reg.create("acme", _raw_config({"anything": 1, "goes": 2}))
    assert plugin.overrides == {"anything": 1, "goes": 2}


def test_create_config_only_constructor_gets_no_kwargs():
    reg = PluginRegistry("video_gen", "entertainment_harness.test_dummy")
    reg.register("acme", _ConfigOnly)
    plugin = reg.create("acme", _raw_config({"bogus": 1}))  # no TypeError
    assert isinstance(plugin.config, Config)


def test_create_no_raw_config_leaves_defaults():
    reg = PluginRegistry("video_gen", "entertainment_harness.test_dummy")
    reg.register("acme", _NamedParams)
    plugin = reg.create("acme", Config())
    assert plugin.base_url == "x"
    assert plugin.timeout == 5


def test_parse_config_keeps_raw_for_plugin_merge():
    from entertainment_harness.config import parse_config

    config = parse_config({"models": {"acme": {"base_url": "http://h"}}})
    reg = PluginRegistry("model_backend", "entertainment_harness.test_dummy")
    reg.register("acme", _NamedParams)
    plugin = reg.create("acme", config)
    assert plugin.base_url == "http://h"


# --- capability queries (Phase 2) ---------------------------------------------


class _Capable:
    capabilities = frozenset({"image-gen"})

    def __init__(self, config: Config | None = None) -> None:
        pass


def test_names_with_filters_by_declared_capability():
    reg = PluginRegistry("frames", "entertainment_harness.test_dummy")
    reg.register("gen", _Capable)
    reg.register("plain", _Dummy)  # no capabilities attribute
    assert reg.names_with("image-gen") == ["gen"]
    assert reg.names_with("teleport") == []


def test_builtin_capability_declarations():
    from entertainment_harness.video.frames import (
        LocalFrameAnimator,
        RunwayFrameAnimator,
    )
    from entertainment_harness.video.gen.local import LocalVideoGenProvider
    from entertainment_harness.video.gen.runway import RunwayProvider
    from entertainment_harness.video.tts import KokoroEngine, SayEngine
    from entertainment_harness.video.tts_qwen3 import Qwen3TTSCloneEngine

    assert SayEngine.capabilities == frozenset({"tts"})
    assert KokoroEngine.capabilities == frozenset({"tts"})
    assert Qwen3TTSCloneEngine.capabilities == frozenset({"tts", "voice-clone"})
    assert LocalVideoGenProvider.capabilities == frozenset({"stills"})
    assert RunwayProvider.capabilities == frozenset({"image-to-video", "image-gen"})
    assert RunwayFrameAnimator.capabilities == frozenset({"image-gen"})
    assert getattr(LocalFrameAnimator, "capabilities", frozenset()) == frozenset()


def test_tts_registry_names_with():
    from entertainment_harness.video.tts import REGISTRY

    assert REGISTRY.names_with("voice-clone") == ["qwen3"]
    assert REGISTRY.names_with("tts") == ["kokoro", "qwen3", "say"]


def test_model_info_capabilities_default_empty():
    from entertainment_harness.models.base import ModelInfo

    assert ModelInfo(name="m:1b", backend="ollama").capabilities == frozenset()


# --- per-seam factory conformance ---------------------------------------------


def test_model_backend_factory():
    from entertainment_harness.models.registry import get_adapter

    adapter = get_adapter("ollama", Config())
    assert adapter.name == "ollama"
    assert adapter.remote is False
    with pytest.raises(PluginError):
        get_adapter("nope", Config())


def test_source_factory():
    from entertainment_harness.sources import get_client, looks_like_id
    from entertainment_harness.sources.mangadex import MangaDexClient

    assert isinstance(get_client("mangadex", Config()), MangaDexClient)
    assert looks_like_id("mangadex", UUID)
    assert not looks_like_id("mangadex", "Kenja no Mago")
    with pytest.raises(PluginError):
        get_client("nope")
    with pytest.raises(PluginError):
        looks_like_id("nope", "x")


def test_tts_factory():
    from entertainment_harness.video.tts import get_engine

    assert get_engine("say", Config()).name == "say"
    with pytest.raises(PluginError):
        get_engine("nope")


def test_video_gen_factory():
    from entertainment_harness.video.gen import get_provider
    from entertainment_harness.video.gen.local import LocalVideoGenProvider

    provider = get_provider("local", Config())
    assert isinstance(provider, LocalVideoGenProvider)
    assert provider.animated is False
    with pytest.raises(PluginError):
        get_provider("nope", Config())


def test_search_factory():
    from entertainment_harness.search import get_provider

    assert get_provider("duckduckgo", Config()).name == "duckduckgo"
    with pytest.raises(PluginError):
        get_provider("nope")
