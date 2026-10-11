"""Eval-scoped hosted image-to-image clients: fal.ai and Replicate.

These providers host the instruction-editing and open-weights image models
the OpenRouter catalog lacks (FLUX Kontext, Qwen-Image-Edit, Seedream 4,
OmniGen 2, ...). An arm declares the endpoint/model id directly:
`model = "fal-ai/flux-pro/kontext"` or `model = "black-forest-labs/flux-kontext-dev"`.
Replicate community models need a version: `"lucataco/omnigen2:<version>"`.

Both clients duck-type the product's `text_to_image` interface; the FIRST
reference image is the edit source (the @panel crop, or the chained @prev
frame — the loops always lead with the image being animated). Results are
normalized with the product's own helpers so frames stay byte-comparable
with the OpenRouter arms. Auth is env-only (evals, no config.toml changes):
FAL_KEY (FAL_API_KEY also accepted) and REPLICATE_API_TOKEN
(REPLICATE_API_KEY also accepted).
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import httpx
from PIL import Image

from entertainment_harness.config import Config
from entertainment_harness.video.gen.openrouter import _data_uri, _ratio_pixels
from entertainment_harness.video.script import VideoError

from evals.imagegen.clients import ClientError

POLL_INTERVAL = 2.0
MAX_POLL_SECONDS = 600.0
#: Some fal endpoints (BFL's edit models observed 2026-10-11) run a flaky
#: content checker: the identical prompt bounces between pass and
#: content_policy_violation across requests. Retry only that verdict.
CONTENT_POLICY_ATTEMPTS = 5
CONTENT_POLICY_RETRY_DELAY = 5.0


def _write_normalized(raw: bytes, ratio: str, dest: Path) -> None:
    """Center-crop to the ratio's aspect and resize to its pixels, so every
    frame in a chain has identical dimensions for ffmpeg. Mirrors
    OpenRouterImageProvider._write_normalized (kept in lockstep on purpose:
    cross-provider comparability depends on identical normalization)."""
    target_w, target_h = _ratio_pixels(ratio)
    with Image.open(io.BytesIO(raw)) as img:
        src_w, src_h = img.size
        target = target_w / target_h
        src = src_w / src_h
        if src > target:  # too wide: crop the sides
            new_w = int(src_h * target)
            left = (src_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, src_h))
        elif src < target:  # too tall: crop top/bottom
            new_h = int(src_w / target)
            top = (src_h - new_h) // 2
            img = img.crop((0, top, src_w, top + new_h))
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="PNG")


def _download(url: str, what: str) -> bytes:
    try:
        with httpx.Client() as client:
            resp = client.get(url, timeout=120.0)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise VideoError(f"{what} download failed: {exc}") from exc
    return resp.content


class FalImageClient:
    """fal.ai queue API: POST https://queue.fal.run/<endpoint-id>, poll the
    status URL, fetch the result URL. Most edit endpoints take `image_urls`
    (a list, data URIs accepted); the two known singular exceptions are
    listed below."""

    name = "fal"
    uses_references = True
    QUEUE_BASE = "https://queue.fal.run"

    #: Endpoints whose input schema takes ONE image URL string; every other
    #: endpoint takes the `image_urls` list (fal's dominant edit shape —
    #: verified against the live input schemas, 2026-10-10).
    _SINGULAR_IMAGE_URL = frozenset({
        "fal-ai/qwen-image-edit",
        "fal-ai/flux-pro/kontext",
        "ideogram/v4.5/edit",
    })

    def __init__(self, config: Config | None = None) -> None:
        self.api_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if not self.api_key:
            raise ClientError(
                "the fal provider needs an API key: set the FAL_KEY"
                " environment variable (https://fal.ai/dashboard/keys)"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.api_key}"}

    def text_to_image(
        self,
        prompt: str,
        references: list[tuple[str, Path]],
        *,
        model: str,
        ratio: str,
        ref_ratio: tuple[int, int] = (0, 0),  # parity with the product clients
        dest: Path,
    ) -> Path:
        if not model:
            raise VideoError(
                "fal arms need a model = the fal endpoint id, e.g."
                ' "fal-ai/flux-pro/kontext"'
            )
        if not references:
            raise VideoError("fal image-to-image needs a reference image")
        image_url = _data_uri(Path(references[0][1]))
        payload: dict = {"prompt": prompt}
        if model in self._SINGULAR_IMAGE_URL:
            payload["image_url"] = image_url
        else:
            payload["image_urls"] = [image_url]
        try:
            with httpx.Client() as client:
                for attempt in range(1, CONTENT_POLICY_ATTEMPTS + 1):
                    try:
                        return self._submit_and_wait(
                            client, model, payload, ratio, dest
                        )
                    except _ContentPolicyRejected:
                        if attempt == CONTENT_POLICY_ATTEMPTS:
                            raise VideoError(
                                "fal content checker rejected the prompt"
                                f" {CONTENT_POLICY_ATTEMPTS}x (nondeterministic"
                                " — same prompt passes on retry); giving up"
                            ) from None
                        time.sleep(CONTENT_POLICY_RETRY_DELAY)
        except httpx.HTTPError as exc:
            raise VideoError(f"fal request failed: {exc}") from exc
        raise VideoError(f"fal generation timed out after {MAX_POLL_SECONDS:g}s")

    def _submit_and_wait(self, client, model, payload, ratio, dest):
        resp = client.post(
            f"{self.QUEUE_BASE}/{model}",
            headers=self._headers(),
            json=payload,
            timeout=60.0,
        )
        if resp.status_code != 200:
            raise VideoError(
                f"fal generation failed ({resp.status_code}): {resp.text}"
            )
        queued = resp.json()
        status_url = queued.get("status_url")
        response_url = queued.get("response_url")
        if not status_url or not response_url:
            raise VideoError(f"fal response missing queue URLs: {queued}")
        start = time.time()
        while time.time() - start < MAX_POLL_SECONDS:
            poll = client.get(status_url, headers=self._headers(),
                              timeout=30.0)
            poll.raise_for_status()
            status = str(poll.json().get("status", "")).upper()
            if status == "COMPLETED":
                result = client.get(response_url,
                                    headers=self._headers(), timeout=30.0)
                if result.status_code == 422 and (
                    "content_policy_violation" in result.text
                ):
                    raise _ContentPolicyRejected()
                result.raise_for_status()
                url = _first_image_url(result.json())
                if not url:
                    raise VideoError(
                        "fal result carried no image URL:"
                        f" {result.text[:300]}"
                    )
                _write_normalized(_download(url, "fal output"), ratio, dest)
                return Path(dest)
            if status in {"FAILED", "CANCELED", "CANCELLED"}:
                raise VideoError(f"fal queue ended with status {status}")
            time.sleep(POLL_INTERVAL)
        raise VideoError(f"fal generation timed out after {MAX_POLL_SECONDS:g}s")


class _ContentPolicyRejected(Exception):
    """The fal worker's content checker rejected the prompt (422
    content_policy_violation). Observed to be nondeterministic — the same
    prompt passes on another request — so the caller retries."""


class ReplicateImageClient:
    """Replicate predictions API. Official models POST to
    /v1/models/<owner>/<name>/predictions; community models need a version
    hash — declare those as "<owner>/<name>:<version>". Input is prompt +
    image (data URIs accepted); poll the prediction's GET URL until it
    succeeds."""

    name = "replicate"
    uses_references = True
    API_BASE = "https://api.replicate.com/v1"

    def __init__(self, config: Config | None = None) -> None:
        self.token = os.getenv("REPLICATE_API_TOKEN") or os.getenv(
            "REPLICATE_API_KEY"
        )
        if not self.token:
            raise ClientError(
                "the replicate provider needs an API key: set the"
                " REPLICATE_API_TOKEN environment variable"
                " (https://replicate.com/account/api-tokens)"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def text_to_image(
        self,
        prompt: str,
        references: list[tuple[str, Path]],
        *,
        model: str,
        ratio: str,
        ref_ratio: tuple[int, int] = (0, 0),  # parity with the product clients
        dest: Path,
    ) -> Path:
        if not model:
            raise VideoError(
                "replicate arms need a model = the model id, e.g."
                ' "black-forest-labs/flux-kontext-dev"'
            )
        if not references:
            raise VideoError("replicate image-to-image needs a reference image")
        input_payload = {"prompt": prompt, "image": _data_uri(Path(references[0][1]))}
        name, _, version = model.partition(":")
        if version:
            url = f"{self.API_BASE}/predictions"
            payload = {"version": version, "input": input_payload}
        else:
            url = f"{self.API_BASE}/models/{name}/predictions"
            payload = {"input": input_payload}
        try:
            with httpx.Client() as client:
                resp = client.post(
                    url, headers=self._headers(), json=payload, timeout=60.0
                )
                if resp.status_code != 201:
                    raise VideoError(
                        f"replicate prediction failed ({resp.status_code}):"
                        f" {resp.text}"
                    )
                prediction = resp.json()
                get_url = (prediction.get("urls") or {}).get("get")
                if not get_url:
                    raise VideoError(
                        f"replicate response missing poll URL: {prediction}"
                    )
                start = time.time()
                while time.time() - start < MAX_POLL_SECONDS:
                    poll = client.get(get_url, headers=self._headers(),
                                      timeout=30.0)
                    poll.raise_for_status()
                    data = poll.json()
                    status = str(data.get("status", "")).lower()
                    if status == "succeeded":
                        url = _first_image_url(data.get("output"))
                        if not url:
                            raise VideoError(
                                f"replicate output carried no image URL: {data}"
                            )
                        _write_normalized(
                            _download(url, "replicate output"), ratio, dest
                        )
                        return Path(dest)
                    if status in {"failed", "canceled", "cancelled"}:
                        error = (data.get("error") or {}).get("message", status)
                        raise VideoError(f"replicate prediction {status}: {error}")
                    time.sleep(POLL_INTERVAL)
        except httpx.HTTPError as exc:
            raise VideoError(f"replicate request failed: {exc}") from exc
        raise VideoError(f"replicate prediction timed out after {MAX_POLL_SECONDS:g}s")


def _first_image_url(output) -> str | None:
    """fal results nest under image/images; replicate predictions return the
    output directly (a URL string, a list of URLs, or list of file dicts)."""
    if isinstance(output, str) and output:
        return output
    if isinstance(output, dict):
        image = output.get("image")
        if isinstance(image, dict) and image.get("url"):
            return str(image["url"])
        for key in ("images", "output"):
            url = _first_image_url(output.get(key))
            if url:
                return url
        if output.get("url"):
            return str(output["url"])
        if output.get("file"):
            return str(output["file"])
    if isinstance(output, list):
        for item in output:
            url = _first_image_url(item)
            if url:
                return url
    return None
