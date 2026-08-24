"""OpenAICompatAdapter — any OpenAI-compatible chat-completions endpoint.

Covers hosted per-token APIs (OpenRouter default, DeepInfra, Together,
Fireworks) and self-hosted vLLM pods (RunPod, Vast.ai) — the wire protocol
is the same; only base_url/api_key differ.

generate() posts {base_url}/chat/completions with content parts
[{"type": "text", ...}, {"type": "image_url", "image_url": {"url":
"data:<mime>;base64,..."}}] — the same shape HFAdapter uses against
llama-server (huggingface.py).

Failures surface as ModelError, never raw httpx/KeyError/JSONDecodeError.
generate() retries transient faults with exponential backoff (2s → 8s →
30s, 3 retries = 4 attempts; Retry-After honored): network/timeout errors,
5xx, 408/409/425/429, and malformed 200 bodies (a gateway hiccup can return
HTTP 200 with no "choices"). Other 4xx are permanent config/auth errors and
fail fast without retrying.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from entertainment_harness.config import Config
from PIL import Image

from entertainment_harness.models.base import ModelError, ModelInfo

# Map Pillow format names to IANA MIME types.
_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}


def _mime_type(image: Path) -> str:
    """Return the MIME type for an image path based on its actual contents."""
    try:
        with Image.open(image) as im:
            return _MIME_BY_FORMAT.get(im.format, "image/jpeg")
    except Exception:
        return "image/jpeg"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

MAX_RETRIES = 3  # retries after the first attempt (4 attempts total)
_INITIAL_DELAY = 2.0
_MAX_DELAY = 30.0
# Transient HTTP statuses worth retrying (5xx is always retried); every
# other 4xx is a permanent config/auth error and fails fast.
_RETRYABLE_STATUS = {408, 409, 425, 429}

_PARAMS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)b(?:-|$|[^a-z])", re.IGNORECASE)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _is_local(base_url: str) -> bool:
    return urlparse(base_url).hostname in _LOCAL_HOSTS


def _excerpt(resp: httpx.Response, limit: int = 200) -> str:
    return resp.text[:limit].strip()


def _parse_completion(resp: httpx.Response, model: str) -> str:
    """Extract the assistant text from a chat-completions body.

    Any deviation from the expected shape — undecodable JSON, missing/empty
    "choices", missing "message"/"content" — raises ModelError carrying the
    model name and a body excerpt; no raw KeyError/IndexError/JSONDecodeError
    escapes the adapter.
    """
    try:
        data = resp.json()
    except ValueError as exc:
        raise ModelError(
            f"{model} returned an unreadable response:"
            f" {_excerpt(resp)}"
        ) from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(
            f"{model} returned a malformed response:"
            f" {_excerpt(resp)}"
        ) from exc
    if not isinstance(content, str) or not content:
        raise ModelError(
            f"{model} returned an empty response:"
            f" {_excerpt(resp)}"
        )
    return content


class OpenAICompatAdapter:
    name = "openai_compat"
    remote = True  # registry.resolve() skips local quant/budget selection

    def __init__(
        self,
        config: Config | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        if base_url is None or api_key is None:
            # Explicit kwargs win; the rest comes from config/env.
            from entertainment_harness.config import load_config

            remote = (config or load_config()).models.openai_compat
            if base_url is None:
                base_url = remote.base_url
            if api_key is None:
                api_key = (
                    remote.api_key
                    or os.environ.get("OPENAI_COMPAT_API_KEY", "")
                    or os.environ.get("OPENROUTER_API_KEY", "")
                )
        if not api_key:
            if _is_local(base_url):
                api_key = "no-key"  # local vLLM ignores the bearer token
            else:
                raise ModelError(
                    "openai_compat backend needs an API key: set api_key under"
                    " [models.openai_compat] in config.toml, or export"
                    " OPENAI_COMPAT_API_KEY / OPENROUTER_API_KEY."
                )
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def supports(self, model: str) -> ModelInfo:
        # Remote models carry no local footprint; params are parsed from the
        # id ("qwen/qwen2.5-vl-72b-instruct" -> 72) for display only.
        match = _PARAMS_RE.search(model)
        return ModelInfo(
            name=model,
            backend=self.name,
            params=float(match.group(1)) if match else None,
        )

    def ensure(self, model: str, quant: str | None = None) -> None:
        # Nothing to pull remotely — but fail fast on a dead/misconfigured
        # endpoint at the same point pipelines call ensure() for local ones.
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(
                f"openai_compat endpoint {self._base_url} is not reachable: {exc}"
            ) from exc

    def remove(self, model: str) -> None:
        raise ModelError("Nothing to remove — openai_compat is a remote backend.")

    def list_available(self) -> list[ModelInfo]:
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise ModelError(
                f"openai_compat model listing failed: {exc}"
            ) from exc
        except ValueError as exc:
            raise ModelError(
                "openai_compat /models returned an unreadable response:"
                f" {_excerpt(resp)}"
            ) from exc
        entries = data.get("data", [])
        if not isinstance(entries, list):
            raise ModelError(
                "openai_compat /models returned a malformed response:"
                f" {_excerpt(resp)}"
            )
        try:
            return [self.supports(m["id"]) for m in entries]
        except (KeyError, TypeError) as exc:
            raise ModelError(
                "openai_compat /models returned a malformed response:"
                f" {_excerpt(resp)}"
            ) from exc

    def list_remote(self, model: str) -> list[ModelInfo] | None:
        # Hosted endpoint: no local artifacts, so no remote per-quant view
        # (remote = True short-circuits the quant machinery before this).
        return None

    def generate(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        num_ctx: int = 16384,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images or []:
            b64 = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:{_mime_type(image)};base64,{b64}"}}
            )
        payload = {"model": model,
                   "messages": [{"role": "user", "content": content}]}
        delay = _INITIAL_DELAY
        last_error: ModelError | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.post(
                    "/chat/completions", json=payload, timeout=None,
                )
            except httpx.HTTPError as exc:
                # Network/timeout failure: transient by definition.
                last_error = ModelError(
                    f"{model} request failed: {exc}"
                )
                wait = delay
            else:
                if resp.status_code >= 500 or resp.status_code in _RETRYABLE_STATUS:
                    last_error = ModelError(
                        f"{model} request failed: {resp.status_code}"
                        f" from {self._base_url}: {_excerpt(resp, 500)}"
                    )
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else delay
                    except ValueError:
                        wait = delay
                elif resp.status_code >= 400:
                    # 400/401/403/404 etc: permanent config/auth error — fail fast.
                    raise ModelError(
                        f"{model} request failed: {resp.status_code}"
                        f" from {self._base_url}: {_excerpt(resp, 500)}"
                    )
                else:
                    try:
                        return _parse_completion(resp, model)
                    except ModelError as exc:
                        # HTTP 200 without a usable completion — a transient
                        # provider/gateway hiccup; worth retrying like a 5xx.
                        last_error = exc
                        wait = delay
            if attempt == MAX_RETRIES:
                break
            time.sleep(wait)
            delay = min(delay * 4, _MAX_DELAY)
        raise last_error
