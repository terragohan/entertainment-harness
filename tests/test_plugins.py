"""PluginRegistry core tests + per-seam factory conformance."""

from __future__ import annotations

import importlib
import importlib.metadata
import json as _json

import httpx
import pytest
import respx

from entertainment_harness.config import Config
from entertainment_harness.plugins import PluginError, PluginRegistry
from entertainment_harness.video.script import VideoError
from entertainment_harness.video.tts import FishEngine

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
    from entertainment_harness.video.tts import FishEngine, KokoroEngine, SayEngine
    from entertainment_harness.video.tts_qwen3 import Qwen3TTSCloneEngine

    assert SayEngine.capabilities == frozenset({"tts"})
    assert KokoroEngine.capabilities == frozenset({"tts"})
    assert FishEngine.capabilities == frozenset({"tts"})
    assert Qwen3TTSCloneEngine.capabilities == frozenset({"tts", "voice-clone"})
    assert LocalVideoGenProvider.capabilities == frozenset({"stills"})
    assert RunwayProvider.capabilities == frozenset({"image-to-video", "image-gen"})
    assert RunwayFrameAnimator.capabilities == frozenset({"image-gen"})
    assert getattr(LocalFrameAnimator, "capabilities", frozenset()) == frozenset()


def test_tts_registry_names_with():
    from entertainment_harness.video.tts import REGISTRY

    assert REGISTRY.names_with("voice-clone") == ["qwen3"]
    assert REGISTRY.names_with("tts") == ["fish", "kokoro", "qwen3", "say"]


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


# --- Fish engine (hosted API) ----------------------------------------------------


FISH_API = "https://api.fish.audio/v1/tts"


def _wav_bytes() -> bytes:
    """A minimal valid PCM WAV (so wav_duration can read it back)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00\x00" * 4410)  # 0.1 s
    return buf.getvalue()


@respx.mock
def test_fish_synthesize_posts_wav_request(tmp_path):
    route = respx.post(FISH_API).mock(
        return_value=httpx.Response(200, content=_wav_bytes())
    )
    engine = FishEngine(api_key="tok", model="s2.1-pro-free")
    dest = tmp_path / "seg-00.wav"
    engine.synthesize("Hello there.", "voice-ref-1", dest)

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer tok"
    assert request.headers["model"] == "s2.1-pro-free"
    body = _json.loads(request.content)
    assert body["text"] == "Hello there."
    assert body["format"] == "wav"
    assert body["reference_id"] == "voice-ref-1"
    assert dest.read_bytes() == _wav_bytes()
    from entertainment_harness.video.tts import wav_duration

    assert 0.09 < wav_duration(dest) < 0.11


@respx.mock
def test_fish_synthesize_omits_reference_id_for_default_voice(tmp_path):
    route = respx.post(FISH_API).mock(
        return_value=httpx.Response(200, content=_wav_bytes())
    )
    FishEngine(api_key="tok").synthesize("Hi.", "", tmp_path / "s.wav")
    body = _json.loads(route.calls[0].request.content)
    assert "reference_id" not in body


@respx.mock
def test_fish_synthesize_api_error_mentions_detail(tmp_path):
    respx.post(FISH_API).mock(return_value=httpx.Response(400, json={
        "status": 400, "message": "text is too long", "reason": "invalid",
    }))
    engine = FishEngine(api_key="tok")
    with pytest.raises(VideoError, match="text is too long"):
        engine.synthesize("x" * 10_000, "", tmp_path / "s.wav")


@respx.mock
def test_fish_synthesize_401_hints_at_key(tmp_path):
    respx.post(FISH_API).mock(
        return_value=httpx.Response(401, json={"message": "unauthorized"})
    )
    engine = FishEngine(api_key="bad")
    with pytest.raises(VideoError, match="check the API key"):
        engine.synthesize("Hi.", "", tmp_path / "s.wav")


def test_fish_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    engine = FishEngine()
    with pytest.raises(VideoError, match="FISH_AUDIO_API_KEY"):
        engine.synthesize("Hi.", "", tmp_path / "s.wav")


def test_fish_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "env-tok")
    assert FishEngine()._api_key == "env-tok"
    assert FishEngine(api_key="explicit")._api_key == "explicit"


def test_fish_settings_from_tts_table():
    """[tts.fish] keys flow through get_engine as constructor overrides."""
    from entertainment_harness.video.tts import get_engine

    config = Config()
    config.tts["fish"] = {"model": "s2.1-pro-free", "api_key": "k"}
    engine = get_engine("fish", config)
    assert engine.name == "fish"
    assert engine._model == "s2.1-pro-free"
    assert engine._api_key == "k"


def test_fish_drama_model_default_and_guidance():
    """The default fish model is drama-3, whose delivery guidance steers
    the script prompts; non-drama models keep the plain wording."""
    from entertainment_harness.video.script import (
        SCRIPT_GUIDANCE_DRAMA,
        SCRIPT_GUIDANCE_PLAIN,
    )

    engine = FishEngine(api_key="k")
    assert engine._model == "drama-3"
    assert engine.script_guidance == SCRIPT_GUIDANCE_DRAMA
    s21 = FishEngine(api_key="k", model="s2.1-pro")
    assert s21.script_guidance == SCRIPT_GUIDANCE_PLAIN
    assert s21.default_voice == ""


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
