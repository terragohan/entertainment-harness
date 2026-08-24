"""OpenAICompatAdapter contract tests (respx) + config/registry integration."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from entertainment_harness.models.base import ModelError
from entertainment_harness.models.openai_compat import OpenAICompatAdapter

BASE = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen2.5-vl-72b-instruct"


def _adapter() -> OpenAICompatAdapter:
    return OpenAICompatAdapter(base_url=BASE, api_key="test-key")


# --- generate() ------------------------------------------------------------


@respx.mock
def test_generate_text_only():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(
            200, json={"choices": [{"message": {"content": "translated"}}]}
        )
    )
    assert _adapter().generate(MODEL, "translate this") == "translated"
    payload = route.calls.last.request
    assert payload.headers["Authorization"] == "Bearer test-key"
    import json

    body = json.loads(payload.content)
    assert body["model"] == MODEL
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "translate this"}]}
    ]


@respx.mock
def test_generate_with_images_sends_base64_data_urls(tmp_path):
    from PIL import Image

    page = tmp_path / "page.png"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(page, format="PNG")
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    assert _adapter().generate(MODEL, "ocr", images=[page]) == "ok"
    import base64
    import json

    body = json.loads(route.calls.last.request.content)
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "ocr"}
    assert parts[1]["type"] == "image_url"
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:4] == b"\x89PNG"


@respx.mock
def test_generate_uses_actual_image_mime_not_extension(tmp_path):
    from PIL import Image

    # Save a JPEG but give it a .png filename — the adapter must send image/jpeg.
    page = tmp_path / "page.png"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(page, format="JPEG")

    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    _adapter().generate(MODEL, "ocr", images=[page])
    import json

    body = json.loads(route.calls.last.request.content)
    parts = body["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@respx.mock
def test_generate_http_error_propagates(monkeypatch):
    monkeypatch.setattr(
        "entertainment_harness.models.openai_compat.time.sleep", lambda s: None
    )
    respx.post(f"{BASE}/chat/completions").mock(return_value=Response(500))
    with pytest.raises(ModelError, match="request failed: 500"):
        _adapter().generate(MODEL, "boom")


# --- generate() retries + defensive parsing -----------------------------------


def _no_sleep(monkeypatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "entertainment_harness.models.openai_compat.time.sleep", sleeps.append
    )
    return sleeps


@respx.mock
def test_generate_500_then_200_succeeds_on_retry(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            Response(500, text="gateway boom"),
            Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    assert _adapter().generate(MODEL, "hi") == "ok"
    assert route.call_count == 2
    assert sleeps == [2.0]  # first backoff step


@respx.mock
def test_generate_persistent_500_raises_after_all_attempts(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(500, text="gateway boom")
    )
    with pytest.raises(ModelError, match="request failed: 500"):
        _adapter().generate(MODEL, "boom")
    assert route.call_count == 4  # 1 initial + 3 retries
    assert sleeps == [2.0, 8.0, 30.0]  # exponential backoff, capped


@respx.mock
def test_generate_400_fails_fast_without_retry(monkeypatch):
    _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(400, text='{"error": "bad request"}')
    )
    with pytest.raises(ModelError, match="request failed: 400"):
        _adapter().generate(MODEL, "boom")
    assert route.call_count == 1  # permanent 4xx: no retry


@respx.mock
def test_generate_429_retried_honoring_retry_after(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            Response(429, headers={"Retry-After": "0.01"}),
            Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    assert _adapter().generate(MODEL, "hi") == "ok"
    assert route.call_count == 2
    assert sleeps == [0.01]


@respx.mock
def test_generate_network_timeout_retried(monkeypatch):
    import httpx

    sleeps = _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.TimeoutException("timed out"),
            Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    assert _adapter().generate(MODEL, "hi") == "ok"
    assert route.call_count == 2
    assert sleeps == [2.0]


@respx.mock
def test_generate_persistent_timeout_raises_model_error(monkeypatch):
    import httpx

    _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(ModelError, match="request failed"):
        _adapter().generate(MODEL, "boom")
    assert route.call_count == 4


@respx.mock
def test_generate_malformed_200_retried_then_model_error(monkeypatch):
    """HTTP 200 with an error body (gateway hiccup) retries, then surfaces a
    ModelError carrying the model name and a body excerpt."""
    _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, json={"error": {"message": "upstream blew up"}})
    )
    with pytest.raises(ModelError, match="malformed response") as excinfo:
        _adapter().generate(MODEL, "boom")
    assert route.call_count == 4
    assert MODEL in str(excinfo.value)
    assert "upstream blew up" in str(excinfo.value)  # body excerpt


@respx.mock
def test_generate_200_without_choices_raises_model_error_not_keyerror(
    monkeypatch,
):
    """Regression: the original crash — 200 with no "choices" key raised a raw
    KeyError that killed a long recap run; it must be a ModelError."""
    _no_sleep(monkeypatch)
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, json={"id": "chatcmpl-x", "usage": {}})
    )
    with pytest.raises(ModelError):
        _adapter().generate(MODEL, "boom")


@respx.mock
def test_generate_empty_choices_raises_model_error(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, json={"choices": []})
    )
    with pytest.raises(ModelError, match="malformed response"):
        _adapter().generate(MODEL, "boom")


@respx.mock
def test_generate_missing_content_raises_model_error(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, json={"choices": [{"message": {}}]})
    )
    with pytest.raises(ModelError, match="malformed response"):
        _adapter().generate(MODEL, "boom")


@respx.mock
def test_generate_non_json_200_raises_model_error(monkeypatch):
    _no_sleep(monkeypatch)
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=Response(200, text="<html>bad gateway</html>")
    )
    with pytest.raises(ModelError, match="unreadable response"):
        _adapter().generate(MODEL, "boom")


@respx.mock
def test_generate_malformed_200_recovers_on_retry(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            Response(200, json={"error": "no choices here"}),
            Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    assert _adapter().generate(MODEL, "hi") == "ok"
    assert route.call_count == 2
    assert sleeps == [2.0]


# --- ensure() / remove() / list_available() ---------------------------------


@respx.mock
def test_ensure_is_a_health_check():
    route = respx.get(f"{BASE}/models").mock(
        return_value=Response(200, json={"data": []})
    )
    _adapter().ensure(MODEL)
    assert route.called


@respx.mock
def test_ensure_raises_model_error_when_unreachable():
    import httpx

    respx.get(f"{BASE}/models").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(ModelError, match="not reachable"):
        _adapter().ensure(MODEL)


def test_remove_rejected_for_remote_backend():
    with pytest.raises(ModelError, match="remote backend"):
        _adapter().remove(MODEL)


@respx.mock
def test_list_available_maps_model_ids():
    respx.get(f"{BASE}/models").mock(
        return_value=Response(
            200,
            json={"data": [{"id": MODEL}, {"id": "qwen/qwen3-32b"}]},
        )
    )
    available = _adapter().list_available()
    assert [i.name for i in available] == [MODEL, "qwen/qwen3-32b"]
    assert all(i.backend == "openai_compat" for i in available)
    assert available[0].params == 72.0  # parsed from the id, display-only


@respx.mock
def test_list_available_malformed_body_raises_model_error():
    respx.get(f"{BASE}/models").mock(
        return_value=Response(200, json={"data": [{"no_id": 1}]})
    )
    with pytest.raises(ModelError, match="malformed response"):
        _adapter().list_available()


@respx.mock
def test_list_available_non_json_raises_model_error():
    respx.get(f"{BASE}/models").mock(
        return_value=Response(200, text="<html>bad gateway</html>")
    )
    with pytest.raises(ModelError, match="unreadable response"):
        _adapter().list_available()


@respx.mock
def test_list_available_http_error_raises_model_error():
    respx.get(f"{BASE}/models").mock(return_value=Response(503))
    with pytest.raises(ModelError, match="model listing failed"):
        _adapter().list_available()


# --- constructor credentials -------------------------------------------------


def test_missing_api_key_raises_with_guidance(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ModelError, match="OPENROUTER_API_KEY"):
        OpenAICompatAdapter(base_url=BASE, api_key="")


def test_local_endpoint_needs_no_key():
    adapter = OpenAICompatAdapter(base_url="http://localhost:8000/v1", api_key="")
    assert adapter._client.headers["Authorization"] == "Bearer no-key"


def test_env_fallbacks(monkeypatch):
    from entertainment_harness import config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: config_mod.Config())
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-openrouter")
    adapter = OpenAICompatAdapter()
    assert adapter._client.headers["Authorization"] == "Bearer from-openrouter"

    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "from-generic")
    adapter = OpenAICompatAdapter()
    assert adapter._client.headers["Authorization"] == "Bearer from-generic"


# --- config parsing ----------------------------------------------------------


def test_load_config_parses_openai_compat_section(tmp_path):
    from entertainment_harness.config import load_config

    path = tmp_path / "config.toml"
    path.write_text(
        "[models.openai_compat]\n"
        'base_url = "https://deepinfra.com/v1/openai"\n'
        'api_key = "cfg-key"\n'
    )
    cfg = load_config(path)
    assert cfg.models.openai_compat.base_url == "https://deepinfra.com/v1/openai"
    assert cfg.models.openai_compat.api_key == "cfg-key"


def test_load_config_openai_compat_defaults(tmp_path):
    from entertainment_harness.config import load_config

    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.models.openai_compat.base_url == "https://openrouter.ai/api/v1"
    assert cfg.models.openai_compat.api_key == ""


# --- registry integration ----------------------------------------------------


def test_resolve_remote_backend_skips_budget(monkeypatch):
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.hardware import HardwareProfile
    from entertainment_harness.models.registry import resolve

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    profile = HardwareProfile("Apple M4", 16 * 10**9, 16 * 10**9, "metal")
    # A 72B model would be refused by the local-budget path; remote must pass.
    role = ModelRoleConfig(backend="openai_compat", model=MODEL)
    selection = resolve(role, profile, Config())
    assert selection.info.name == MODEL
    assert selection.info.backend == "openai_compat"
    assert selection.warning is None
    assert selection.pinned is False


def test_unknown_backend_error_lists_openai_compat():
    from entertainment_harness.config import Config
    from entertainment_harness.models.registry import get_adapter
    from entertainment_harness.plugins import PluginError

    with pytest.raises(PluginError, match="openai_compat"):
        get_adapter("nope", Config())
