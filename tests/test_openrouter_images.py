"""OpenRouter image-client tests: credential resolution, the chat-completions
request shape, response parsing (data URL and http), output normalization to
the requested ratio, and error paths. No live API."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from entertainment_harness.config import Config
from entertainment_harness.video.gen.openrouter import (
    DEFAULT_IMAGE_MODEL,
    OpenRouterImageProvider,
    _ratio_pixels,
)
from entertainment_harness.video.script import VideoConfigError, VideoError


def _png_bytes(size: tuple[int, int] = (100, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, format="PNG")
    return buf.getvalue()


def _data_url(size: tuple[int, int] = (100, 50)) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(size)).decode()


class _Resp:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = content
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeHttpxClient:
    """httpx.Client stand-in: records POSTs, serves canned responses."""

    posts: list[dict] = []
    gets: list[str] = []
    post_response: _Resp = _Resp(payload={})
    post_responses: list = []  # when non-empty, pop(0) per POST
    get_response: _Resp = _Resp(content=b"")

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, headers=None, json=None, timeout=None):
        type(self).posts.append({"url": url, "headers": headers, "json": json})
        if type(self).post_responses:
            return type(self).post_responses.pop(0)
        return type(self).post_response

    def get(self, url, timeout=None):
        type(self).gets.append(url)
        return type(self).get_response


@pytest.fixture
def http(monkeypatch):
    FakeHttpxClient.posts = []
    FakeHttpxClient.gets = []
    FakeHttpxClient.post_responses = []
    monkeypatch.setattr(
        "entertainment_harness.video.gen.openrouter.httpx.Client",
        FakeHttpxClient,
    )
    return FakeHttpxClient


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "entertainment_harness.video.gen.openrouter.time.sleep",
        lambda s: None,  # no real backoff in tests
    )
    config = Config()
    config.models.openai_compat.api_key = "test-or-key"
    return OpenRouterImageProvider(config)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(VideoError, match="api_key"):
        OpenRouterImageProvider(Config())


def test_env_key_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    provider = OpenRouterImageProvider(Config())
    assert provider.api_key == "env-key"
    assert provider.base_url == "https://openrouter.ai/api/v1"


def test_ratio_pixels_parses_and_falls_back():
    assert _ratio_pixels("1920:1080") == (1920, 1080)
    assert _ratio_pixels("garbage") == (1920, 1080)


def test_text_to_image_request_shape_and_normalization(http, provider, tmp_path):
    panel = tmp_path / "panel.png"
    Image.new("RGB", (80, 60), "blue").save(panel)
    http.post_response = _Resp(payload={
        "choices": [{"message": {"images": [
            {"type": "image_url", "image_url": {"url": _data_url((100, 50))}}
        ]}}],
    })
    dest = tmp_path / "out" / "frame.png"
    provider.text_to_image(
        "Redraw the manga panel @panel as one frame", [("panel", panel)],
        model="gpt-6-luna", ratio="1920:1080", ref_ratio=(80, 60), dest=dest,
    )
    (call,) = http.posts
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-or-key"
    payload = call["json"]
    assert payload["model"] == "gpt-6-luna"
    assert payload["modalities"] == ["image", "text"]
    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "text", "text": "Redraw the manga panel @panel as one frame",
    }
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert http.gets == []  # data URL: no download round-trip
    with Image.open(dest) as img:
        assert img.size == (1920, 1080)  # normalized for ffmpeg


def test_text_to_image_downloads_http_results(http, provider, tmp_path):
    http.post_response = _Resp(payload={
        "choices": [{"message": {"images": [
            {"image_url": {"url": "https://cdn.example/frame.png"}}
        ]}}],
    })
    http.get_response = _Resp(content=_png_bytes((1080, 1920)))
    dest = tmp_path / "frame.png"
    provider.text_to_image(
        "p", [], model="m", ratio="1920:1080", ref_ratio=(0, 0), dest=dest
    )
    assert http.gets == ["https://cdn.example/frame.png"]
    with Image.open(dest) as img:
        assert img.size == (1920, 1080)  # tall result center-cropped to 16:9


def test_no_image_in_response_is_videoerror(http, provider, tmp_path):
    http.post_response = _Resp(payload={
        "choices": [{"message": {"content": "I cannot draw that."}}],
    })
    with pytest.raises(VideoError, match="no image"):
        provider.text_to_image(
            "p", [], model="gpt-4o", ratio="1920:1080",
            ref_ratio=(0, 0), dest=tmp_path / "f.png",
        )


def test_no_image_retries_then_succeeds(http, provider, tmp_path):
    """The endpoint intermittently answers 200 with no image payload; the
    client retries before failing the anchor."""
    http.post_responses = [
        _Resp(payload={"choices": [{"message": {"content": "busy"}}]}),
        _Resp(payload={"choices": [{"message": {"images": [
            {"type": "image_url", "image_url": {"url": _data_url((100, 50))}}
        ]}}]}),
    ]
    dest = tmp_path / "f.png"
    provider.text_to_image(
        "p", [], model="m", ratio="1920:1080", ref_ratio=(0, 0), dest=dest
    )
    assert len(http.posts) == 2  # one empty answer, then the image
    with Image.open(dest) as img:
        assert img.size == (1920, 1080)


def test_no_image_persistent_failure_raises_after_attempts(
    http, provider, tmp_path
):
    http.post_response = _Resp(
        payload={"choices": [{"message": {"content": "no"}}]}
    )
    with pytest.raises(VideoError, match="no image after 3 attempts"):
        provider.text_to_image(
            "p", [], model="m", ratio="1920:1080",
            ref_ratio=(0, 0), dest=tmp_path / "f.png",
        )
    assert len(http.posts) == 3


def test_http_error_is_videoerror(http, provider, tmp_path):
    http.post_response = _Resp(status_code=401, payload={"error": "bad key"})
    with pytest.raises(VideoError, match="401"):
        provider.text_to_image(
            "p", [], model="m", ratio="1920:1080",
            ref_ratio=(0, 0), dest=tmp_path / "f.png",
        )


def test_default_model_constant_is_an_openrouter_id():
    assert "/" in DEFAULT_IMAGE_MODEL


def test_text_only_model_404_is_config_error(http, provider, tmp_path):
    # OpenRouter routing failure: the model exists but no endpoint serves
    # image output for it (e.g. gpt-6-luna is text+image->text). A config
    # error, not a per-panel failure — it must abort, not degrade.
    http.post_response = _Resp(status_code=404, payload={
        "error": {"message": "No endpoints found that support the requested"
                             " output modalities: image, text", "code": 404},
    })
    with pytest.raises(VideoConfigError, match="cannot generate images"):
        provider.text_to_image(
            "p", [], model="gpt-6-luna", ratio="1920:1080",
            ref_ratio=(0, 0), dest=tmp_path / "f.png",
        )


def test_plain_404_stays_a_plain_videoerror(http, provider, tmp_path):
    http.post_response = _Resp(
        status_code=404, payload={"error": {"message": "model not found"}}
    )
    with pytest.raises(VideoError) as exc:
        provider.text_to_image(
            "p", [], model="m", ratio="1920:1080",
            ref_ratio=(0, 0), dest=tmp_path / "f.png",
        )
    assert not isinstance(exc.value, VideoConfigError)
