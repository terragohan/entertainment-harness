"""Provider -> image client construction for the eval harness.

The hosted providers reuse the product clients (duck-typed
`text_to_image`); "diffusers" is eval-scoped: local open-weights models
through the same interface, lazily imported so the product never depends
on diffusers/torch. Local text-to-image models cannot consume the
@panel/@prev reference images (no server-side tagging, no IP-Adapter in
scope): the client accepts-and-ignores references and exposes
`uses_references = False`, which the runner records in the manifest.
Their faithfulness scores therefore measure blind generation — the gap
reference conditioning closes — and the gallery labels those arms.
"""

from __future__ import annotations

from pathlib import Path

from entertainment_harness.config import Config

PROVIDERS = ("openrouter", "runway", "diffusers", "fal", "replicate")

#: Default local model when a diffusers arm leaves `model` empty.
DEFAULT_DIFFUSERS_MODEL = "stabilityai/sdxl-turbo"

#: Cap for local generation: the longest side in pixels (aspect preserved,
#: both dims a multiple of 8 for the VAE).
MAX_LOCAL_SIDE = 1024


class ClientError(RuntimeError):
    """An eval client could not be constructed (missing extras, bad provider)."""


def build_client(provider: str, config: Config):
    """Construct the image client for a provider; constructor failures
    (missing API key, missing diffusers, OOM loading weights later)
    propagate so the runner can record them per arm."""
    if provider == "openrouter":
        from entertainment_harness.video.gen.openrouter import (
            OpenRouterImageProvider,
        )

        return OpenRouterImageProvider(config)
    if provider == "runway":
        from entertainment_harness.video.gen.runway import RunwayProvider

        return RunwayProvider(config)
    if provider == "diffusers":
        return DiffusersImageClient(config)
    if provider in ("fal", "replicate"):
        from evals.imagegen.hosted import FalImageClient, ReplicateImageClient

        return (FalImageClient if provider == "fal" else ReplicateImageClient)(
            config
        )
    raise ClientError(
        f"unknown provider {provider!r} — expected one of"
        f" {', '.join(PROVIDERS)}"
    )


class DiffusersImageClient:
    """Local text-to-image via Hugging Face diffusers (eval-scoped).

    `model` is a HF repo id (e.g. "stabilityai/sdxl-turbo"). The pipeline
    loads once per client, keyed by model id; MPS is preferred over CUDA
    over CPU, float16 on accelerators. Reference images are accepted and
    ignored — open T2I models cannot condition on them here.
    """

    name = "diffusers"
    uses_references = False

    def __init__(self, config: Config | None = None) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ClientError(
                "the diffusers provider needs the diffusers and torch"
                " packages, which are not product dependencies — run with:"
                " uv run --with diffusers --with torch python -m evals.imagegen ..."
            ) from exc
        self._torch = torch
        self._pipeline_cls = AutoPipelineForText2Image
        self._pipelines: dict[str, object] = {}

    def _device_and_dtype(self):
        torch = self._torch
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32

    def _pipeline(self, model: str):
        pipe = self._pipelines.get(model)
        if pipe is None:
            device, dtype = self._device_and_dtype()
            if dtype is self._torch.float16:
                # fp16 weight files are half the download (~7 GB vs ~14 GB
                # for SDXL-Turbo) — ask for them. Not every repo ships an
                # fp16 variant, so any variant-load failure retries plain.
                try:
                    pipe = self._pipeline_cls.from_pretrained(
                        model, torch_dtype=dtype, variant="fp16"
                    )
                except Exception:
                    pipe = None
            if pipe is None:
                pipe = self._pipeline_cls.from_pretrained(
                    model, torch_dtype=dtype
                )
            pipe.to(device)
            self._pipelines[model] = pipe
        return pipe

    @staticmethod
    def _pixels(ratio: str) -> tuple[int, int]:
        """"W:H" -> pixel dims: longest side capped at MAX_LOCAL_SIDE,
        aspect preserved, both dims a multiple of 8."""
        w, _, h = ratio.partition(":")
        try:
            width, height = int(w), int(h)
        except ValueError:
            width, height = MAX_LOCAL_SIDE, MAX_LOCAL_SIDE
        scale = min(1.0, MAX_LOCAL_SIDE / max(width, height, 1))
        width = max(8, round(width * scale / 8) * 8)
        height = max(8, round(height * scale / 8) * 8)
        return width, height

    @staticmethod
    def _settings(model: str) -> dict:
        """Per-model-id heuristics: distilled turbo/schnell models want a
        few steps and no CFG; everything else gets the common default."""
        mid = model.lower()
        if "turbo" in mid or "schnell" in mid:
            return {"num_inference_steps": 4, "guidance_scale": 0.0}
        return {"num_inference_steps": 20, "guidance_scale": 7.5}

    def text_to_image(
        self,
        prompt: str,
        references: list[tuple[str, Path]],
        *,
        model: str,
        ratio: str,
        ref_ratio: tuple[int, int] = (0, 0),  # parity with the hosted clients
        dest: Path,
    ) -> Path:
        model = model or DEFAULT_DIFFUSERS_MODEL
        width, height = self._pixels(ratio)
        pipe = self._pipeline(model)
        result = pipe(
            prompt=prompt, width=width, height=height, **self._settings(model)
        )
        image = result.images[0]
        if image.mode != "RGB":
            image = image.convert("RGB")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="PNG")
        return dest
