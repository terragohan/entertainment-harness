"""HFAdapter — Hugging Face Hub GGUF repos, served by llama.cpp.

Model ids are HF repo ids ("owner/name-GGUF"); a specific artifact is
"repo:QUANT" (e.g. "Qwen/Qwen3-0.6B-GGUF:Q4_K_M"). Quant sizes come from the
Hub API (?blobs=true), so the registry can pick a quant before anything is
downloaded; ensure() then fetches just that file. Inference runs through a
managed llama-server subprocess speaking the OpenAI chat-completions API.
Vision models need the repo's mmproj file, downloaded alongside the weights.
"""

from __future__ import annotations

import atexit
import base64
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx

from entertainment_harness.config import Config, cache_dir
from entertainment_harness.models.base import ModelInfo, ModelNotFoundError

API_URL = "https://huggingface.co"
SERVER_STARTUP_TIMEOUT = 180.0

# Quant tag = filename stem after the last "-", e.g. "Qwen3-0.6B-Q4_K_M.gguf".
_QUANT_TAG_RE = re.compile(r"^I?Q[0-9]|^Q[0-9]|^BF16$|^F16$|^F32$", re.IGNORECASE)
_PARAMS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([BM])", re.IGNORECASE)


def parse_model_ref(model: str) -> tuple[str, str | None]:
    """Split "repo[:QUANT]" (also tolerating ollama-style "repo-quant")."""
    if ":" in model:
        repo, quant = model.rsplit(":", 1)
        return repo, quant.upper()
    stem = model.rsplit("-", 1)
    if len(stem) == 2 and _QUANT_TAG_RE.match(stem[1]):
        return stem[0], stem[1].upper()
    return model, None


def quant_from_filename(filename: str) -> str | None:
    if not filename.endswith(".gguf"):
        return None
    tag = filename[: -len(".gguf")].rsplit("-", 1)[-1]
    return tag.upper() if _QUANT_TAG_RE.match(tag) else None


def _parse_params(text: str) -> float | None:
    match = _PARAMS_RE.search(text)
    if not match:
        return None
    number = float(match.group(1))
    return number / 1000 if match.group(2).upper() == "M" else number


class _LlamaServer:
    """A managed llama-server process for one model at a time."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self.model_key: str | None = None
        self.base_url: str = ""
        atexit.register(self.stop)

    def _free_port(self) -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def start(self, key: str, model_path: Path, mmproj_path: Path | None,
              num_ctx: int) -> None:
        if self._proc is not None and self.model_key == key:
            return
        self.stop()
        port = self._free_port()
        cmd = [
            "llama-server", "--model", str(model_path),
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(num_ctx),
        ]
        if mmproj_path is not None:
            cmd += ["--mmproj", str(mmproj_path)]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup ({cmd})")
            try:
                if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                    self.model_key = key
                    self.base_url = url
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"llama-server did not become healthy ({cmd})")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            self.model_key = None


class HFAdapter:
    name = "huggingface"
    remote = False

    def __init__(
        self,
        config: Config | None = None,
        models_dir: Path | None = None,
        timeout: float = 60.0,
    ) -> None:
        # config is accepted for the shared plugin constructor contract; there
        # is no [models.huggingface] section yet.
        self._client = httpx.Client(base_url=API_URL, timeout=timeout)
        self._models_dir = models_dir or cache_dir() / "models" / "hf"
        self._server = _LlamaServer()

    # --- Hub API -------------------------------------------------------------

    def _list_files(self, repo: str) -> list[dict]:
        resp = self._client.get(f"/api/models/{repo}", params={"blobs": "true"})
        if resp.status_code == 404:
            raise ModelNotFoundError(f"Hugging Face has no repo {repo!r}")
        resp.raise_for_status()
        return resp.json().get("siblings", [])

    def _gguf_files(self, repo: str) -> tuple[list[dict], list[str]]:
        """(weight files, mmproj filenames) of a repo."""
        weights, mmprojs = [], []
        for s in self._list_files(repo):
            fname = s.get("rfilename", "")
            if not fname.endswith(".gguf"):
                continue
            if fname.lower().startswith("mmproj"):
                mmprojs.append(fname)
            else:
                weights.append(s)
        return weights, mmprojs

    def _file_info(self, repo: str, entry: dict, vision: bool = False) -> ModelInfo:
        fname = entry["rfilename"]
        return ModelInfo(
            name=f"{repo}:{quant_from_filename(fname) or fname}",
            backend=self.name,
            params=_parse_params(fname),
            quant=quant_from_filename(fname),
            size_bytes=entry.get("size"),
            capabilities=frozenset({"vision"}) if vision else frozenset(),
        )

    def list_remote(self, model: str) -> list[ModelInfo]:
        """All weight quants a repo publishes, with sizes (registry policy
        input — nothing is downloaded)."""
        repo, quant = parse_model_ref(model)
        files, mmprojs = self._gguf_files(repo)
        infos = [self._file_info(repo, e, vision=bool(mmprojs)) for e in files]
        if quant:
            infos = [i for i in infos if i.quant == quant]
        return infos

    # --- adapter protocol ------------------------------------------------------

    def supports(self, model: str) -> ModelInfo:
        # Local download is authoritative; else resolve against the Hub.
        for info in self.list_available():
            if info.name == model:
                return info
        infos = self.list_remote(model)
        if not infos:
            raise ModelNotFoundError(f"No GGUF found for {model!r}")
        _, quant = parse_model_ref(model)
        if quant and len(infos) != 1:
            raise ModelNotFoundError(f"No {quant} quant in {model!r}")
        info = infos[0]
        if quant is None and len(infos) > 1:
            # Bare repo with several quants: no single artifact to point at.
            quants = ", ".join(sorted(i.quant or "?" for i in infos))
            raise ModelNotFoundError(
                f"{model!r} publishes several quants ({quants}); pick one"
                f" with {model}:<QUANT>"
            )
        return info

    def list_available(self) -> list[ModelInfo]:
        infos = []
        for path in sorted(self._models_dir.glob("*/*/*.gguf")):
            if path.name.lower().startswith("mmproj"):
                continue
            repo = f"{path.parent.parent.name}/{path.parent.name}"
            # A sibling mmproj file means the download can read images.
            vision = any(path.parent.glob("mmproj*.gguf"))
            infos.append(
                ModelInfo(
                    name=f"{repo}:{quant_from_filename(path.name) or path.name}",
                    backend=self.name,
                    params=_parse_params(path.name),
                    quant=quant_from_filename(path.name),
                    size_bytes=path.stat().st_size,
                    capabilities=frozenset({"vision"}) if vision else frozenset(),
                )
            )
        return infos

    def ensure(self, model: str, quant: str | None = None) -> None:
        repo, tag = parse_model_ref(model)
        tag = tag or (quant.upper() if quant else None)
        files, mmprojs = self._gguf_files(repo)
        matches = [f for f in files if tag is None
                   or quant_from_filename(f["rfilename"]) == tag]
        if not matches:
            raise ModelNotFoundError(
                f"No {tag or 'GGUF'} artifact found in repo {repo!r}"
            )
        entry = matches[0]
        self._download(repo, entry["rfilename"], entry.get("size"))
        # Vision repos ship a separate projector; grab the closest quant.
        if mmprojs:
            chosen_mm = next(
                (m for m in mmprojs if quant_from_filename(m) == tag),
                mmprojs[0],
            )
            self._download(repo, chosen_mm, None)

    def _download(self, repo: str, filename: str, expected_size: int | None) -> Path:
        dest = self._models_dir / repo / filename
        if dest.exists() and (expected_size is None
                              or dest.stat().st_size == expected_size):
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream(
            "GET", f"/{repo}/resolve/main/{filename}",
            follow_redirects=True, timeout=None,
        ) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        return dest

    def _paths_for(self, model: str) -> tuple[Path, Path | None]:
        repo, quant = parse_model_ref(model)
        folder = self._models_dir / repo
        weights = [p for p in folder.glob("*.gguf")
                   if not p.name.lower().startswith("mmproj")]
        if quant:
            weights = [p for p in weights
                       if quant_from_filename(p.name) == quant]
        if not weights:
            raise ModelNotFoundError(f"{model!r} is not downloaded; ensure() first")
        mmprojs = sorted(folder.glob("mmproj*.gguf"))
        return weights[0], (mmprojs[0] if mmprojs else None)

    def remove(self, model: str) -> None:
        """Delete local files for "repo" (whole repo) or "repo:QUANT"."""
        self._server.stop()  # never delete weights out from under a server
        repo, quant = parse_model_ref(model)
        folder = self._models_dir / repo
        if not folder.is_dir():
            raise ModelNotFoundError(f"{model!r} is not downloaded")
        if quant is None:
            shutil.rmtree(folder)
            return
        matches = [p for p in folder.glob("*.gguf")
                   if quant_from_filename(p.name) == quant
                   and not p.name.lower().startswith("mmproj")]
        if not matches:
            raise ModelNotFoundError(f"{model!r} is not downloaded")
        for path in matches:
            path.unlink()
        weights = [p for p in folder.glob("*.gguf")
                   if not p.name.lower().startswith("mmproj")]
        if not weights:
            shutil.rmtree(folder)  # nothing left but mmproj files

    def generate(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        num_ctx: int = 16384,
    ) -> str:
        model_path, mmproj_path = self._paths_for(model)
        if images and mmproj_path is None:
            raise ModelNotFoundError(
                f"Repo for {model!r} has no mmproj file — cannot read images"
            )
        self._server.start(model, model_path, mmproj_path, num_ctx)

        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images or []:
            mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        resp = httpx.post(
            f"{self._server.base_url}/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": content}]},
            timeout=None,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
