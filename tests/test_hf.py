"""HFAdapter + quantize tests (respx-mocked Hub API, mocked llama-server /
llama-quantize subprocesses). No network, no llama.cpp needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import respx
from httpx import Response

from entertainment_harness.models.base import ModelNotFoundError, ModelTooLargeError
from entertainment_harness.models.huggingface import (
    HFAdapter,
    parse_model_ref,
    quant_from_filename,
)
from entertainment_harness.models.quantize import QuantizeError, quantize

GB = 10**9
API = "https://huggingface.co"
REPO = "Qwen/Qwen3-0.6B-GGUF"

SIBLINGS = {
    "id": REPO,
    "siblings": [
        {"rfilename": "README.md", "size": 100},
        {"rfilename": "Qwen3-0.6B-BF16.gguf", "size": int(1.2 * GB)},
        {"rfilename": "Qwen3-0.6B-Q8_0.gguf", "size": int(0.64 * GB)},
        {"rfilename": "Qwen3-0.6B-Q4_K_M.gguf", "size": int(0.4 * GB)},
        {"rfilename": "mmproj-Qwen3-0.6B-F16.gguf", "size": int(0.1 * GB)},
    ],
}


def _mock_repo(payload=SIBLINGS):
    respx.get(f"{API}/api/models/{REPO}").mock(
        return_value=Response(200, json=payload)
    )


# --- name parsing ------------------------------------------------------------


def test_parse_model_ref():
    assert parse_model_ref("a/b-GGUF") == ("a/b-GGUF", None)
    assert parse_model_ref("a/b-GGUF:Q4_K_M") == ("a/b-GGUF", "Q4_K_M")
    assert parse_model_ref("a/b-GGUF:q4_k_m") == ("a/b-GGUF", "Q4_K_M")
    assert parse_model_ref("a/b-GGUF-q4_k_m") == ("a/b-GGUF", "Q4_K_M")


def test_quant_from_filename():
    assert quant_from_filename("Qwen3-0.6B-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_from_filename("Qwen3-0.6B-UD-Q4_K_XL.gguf") == "Q4_K_XL"
    assert quant_from_filename("model-BF16.gguf") == "BF16"
    assert quant_from_filename("README.md") is None
    assert quant_from_filename("model-weird.gguf") is None


# --- adapter ------------------------------------------------------------------


@respx.mock
def test_list_remote_returns_quants_with_sizes(tmp_path):
    _mock_repo()
    adapter = HFAdapter(models_dir=tmp_path)
    infos = adapter.list_remote(REPO)
    assert {i.quant for i in infos} == {"BF16", "Q8_0", "Q4_K_M"}  # no mmproj
    q8 = next(i for i in infos if i.quant == "Q8_0")
    assert q8.name == f"{REPO}:Q8_0"
    assert q8.size_bytes == int(0.64 * GB)
    assert q8.params == pytest.approx(0.6)


@respx.mock
def test_supports_specific_quant(tmp_path):
    _mock_repo()
    adapter = HFAdapter(models_dir=tmp_path)
    info = adapter.supports(f"{REPO}:Q4_K_M")
    assert info.quant == "Q4_K_M"
    assert info.size_bytes == int(0.4 * GB)


@respx.mock
def test_supports_bare_repo_with_multiple_quants_is_ambiguous(tmp_path):
    _mock_repo()
    adapter = HFAdapter(models_dir=tmp_path)
    with pytest.raises(ModelNotFoundError, match="several quants"):
        adapter.supports(REPO)


@respx.mock
def test_unknown_repo_raises(tmp_path):
    respx.get(f"{API}/api/models/nope/nothing").mock(return_value=Response(404))
    adapter = HFAdapter(models_dir=tmp_path)
    with pytest.raises(ModelNotFoundError):
        adapter.list_remote("nope/nothing")


@respx.mock
def test_ensure_downloads_weights_and_mmproj(tmp_path):
    # small declared sizes so the mock content length matches: the second
    # ensure() must then skip the download (size check)
    _mock_repo({
        "id": REPO,
        "siblings": [
            {"rfilename": "Qwen3-0.6B-Q4_K_M.gguf", "size": len(b"weights")},
            {"rfilename": "mmproj-Qwen3-0.6B-F16.gguf", "size": len(b"mmproj")},
        ],
    })
    weights = respx.get(f"{API}/{REPO}/resolve/main/Qwen3-0.6B-Q4_K_M.gguf").mock(
        return_value=Response(200, content=b"weights")
    )
    mmproj = respx.get(
        f"{API}/{REPO}/resolve/main/mmproj-Qwen3-0.6B-F16.gguf"
    ).mock(return_value=Response(200, content=b"mmproj"))
    adapter = HFAdapter(models_dir=tmp_path)
    adapter.ensure(f"{REPO}:Q4_K_M")
    assert (tmp_path / REPO / "Qwen3-0.6B-Q4_K_M.gguf").read_bytes() == b"weights"
    assert (tmp_path / REPO / "mmproj-Qwen3-0.6B-F16.gguf").read_bytes() == b"mmproj"
    assert weights.called and mmproj.called

    # second ensure: size matches -> no re-download
    adapter.ensure(f"{REPO}:Q4_K_M")
    assert weights.call_count == 1


def test_list_available_scans_models_dir(tmp_path):
    folder = tmp_path / REPO
    folder.mkdir(parents=True)
    (folder / "Qwen3-0.6B-Q4_K_M.gguf").write_bytes(b"x" * 100)
    (folder / "mmproj-Qwen3-0.6B-F16.gguf").write_bytes(b"proj")
    adapter = HFAdapter(models_dir=tmp_path)
    infos = adapter.list_available()
    assert len(infos) == 1
    assert infos[0].name == f"{REPO}:Q4_K_M"
    assert infos[0].size_bytes == 100


# --- capabilities ---------------------------------------------------------------


@respx.mock
def test_list_remote_marks_vision_when_repo_ships_mmproj(tmp_path):
    _mock_repo()
    adapter = HFAdapter(models_dir=tmp_path)
    infos = adapter.list_remote(REPO)
    assert all(i.capabilities == frozenset({"vision"}) for i in infos)


@respx.mock
def test_list_remote_no_mmproj_means_no_capabilities(tmp_path):
    _mock_repo({
        "id": REPO,
        "siblings": [{"rfilename": "Qwen3-0.6B-Q4_K_M.gguf", "size": int(0.4 * GB)}],
    })
    adapter = HFAdapter(models_dir=tmp_path)
    (info,) = adapter.list_remote(REPO)
    assert info.capabilities == frozenset()


def test_list_available_marks_vision_from_sibling_mmproj(tmp_path):
    folder = tmp_path / REPO
    folder.mkdir(parents=True)
    (folder / "Qwen3-0.6B-Q4_K_M.gguf").write_bytes(b"x")
    (folder / "mmproj-Qwen3-0.6B-F16.gguf").write_bytes(b"proj")
    adapter = HFAdapter(models_dir=tmp_path)
    (info,) = adapter.list_available()
    assert info.capabilities == frozenset({"vision"})


def test_list_available_without_mmproj_has_no_capabilities(tmp_path):
    folder = tmp_path / REPO
    folder.mkdir(parents=True)
    (folder / "Qwen3-0.6B-Q4_K_M.gguf").write_bytes(b"x")
    adapter = HFAdapter(models_dir=tmp_path)
    (info,) = adapter.list_available()
    assert info.capabilities == frozenset()


@respx.mock
def test_generate_via_llama_server(tmp_path, monkeypatch):
    _mock_repo()
    respx.get(f"{API}/{REPO}/resolve/main/Qwen3-0.6B-Q4_K_M.gguf").mock(
        return_value=Response(200, content=b"weights")
    )
    respx.get(
        f"{API}/{REPO}/resolve/main/mmproj-Qwen3-0.6B-F16.gguf"
    ).mock(return_value=Response(200, content=b"mmproj"))
    adapter = HFAdapter(models_dir=tmp_path)
    adapter.ensure(f"{REPO}:Q4_K_M")

    started: list[dict] = []

    class FakeServer:
        base_url = "http://fake-llama"

        def start(self, key, model_path, mmproj_path, num_ctx):
            started.append({"key": key, "mmproj": mmproj_path, "ctx": num_ctx})

        def stop(self):
            pass

    monkeypatch.setattr(adapter, "_server", FakeServer())
    chat = respx.post("http://fake-llama/v1/chat/completions").mock(
        return_value=Response(
            200, json={"choices": [{"message": {"content": "Hi there."}}]}
        )
    )
    image = tmp_path / "page.jpg"
    image.write_bytes(b"\xff\xd8fake")
    out = adapter.generate(f"{REPO}:Q4_K_M", "Summarize.", images=[image])
    assert out == "Hi there."
    assert started[0]["key"] == f"{REPO}:Q4_K_M"
    assert started[0]["mmproj"] is not None  # mmproj was downloaded by ensure
    import base64, json

    body = json.loads(chat.calls.last.request.content)
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Summarize."}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8fake"


# --- registry integration ------------------------------------------------------


@respx.mock
def test_registry_selects_remote_quant_before_download(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))  # isolate from real data dir
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.hardware import HardwareProfile
    from entertainment_harness.models.registry import resolve

    _mock_repo()
    profile = HardwareProfile("Apple M4 Pro", 24 * GB, int(0.5 * GB), "metal")
    role = ModelRoleConfig(backend="huggingface", model=REPO)
    selection = resolve(role, profile, Config())
    # budget 0.5 GB: Q8_0 (0.64) too large -> Q4_K_M, with below-Q8 warning
    assert selection.info.name == f"{REPO}:Q4_K_M"
    assert "below 8-bit" in selection.warning


@respx.mock
def test_registry_refuses_when_no_remote_quant_fits(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))  # isolate from real data dir
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.hardware import HardwareProfile
    from entertainment_harness.models.registry import resolve

    _mock_repo()
    profile = HardwareProfile("Apple M4 Pro", 24 * GB, int(0.1 * GB), "metal")
    role = ModelRoleConfig(backend="huggingface", model=REPO)
    with pytest.raises(ModelTooLargeError):
        resolve(role, profile, Config())


# --- quantize ------------------------------------------------------------------


@respx.mock
def test_quantize_refuses_published_quant(tmp_path):
    _mock_repo()
    with pytest.raises(QuantizeError, match="already publishes"):
        quantize(REPO, "Q4_K_M", adapter=HFAdapter(models_dir=tmp_path),
                 confirm=lambda msg: True, log=lambda m: None)


@respx.mock
def test_quantize_requires_source_and_confirmation(tmp_path):
    _mock_repo()
    adapter = HFAdapter(models_dir=tmp_path)
    # user declines
    with pytest.raises(QuantizeError, match="Aborted"):
        quantize(REPO, "Q6_K", adapter=adapter,
                 confirm=lambda msg: False, log=lambda m: None)
    # repo without FP16/BF16 source
    respx.get(f"{API}/api/models/x/quants-only").mock(
        return_value=Response(200, json={
            "id": "x/quants-only",
            "siblings": [{"rfilename": "m-Q4_K_M.gguf", "size": 100}],
        })
    )
    with pytest.raises(QuantizeError, match="no FP16/BF16 source"):
        quantize("x/quants-only", "Q6_K", adapter=adapter,
                 confirm=lambda msg: True, log=lambda m: None)


@respx.mock
def test_quantize_runs_llama_quantize(tmp_path, monkeypatch):
    _mock_repo()
    respx.get(f"{API}/{REPO}/resolve/main/Qwen3-0.6B-BF16.gguf").mock(
        return_value=Response(200, content=b"bf16-weights")
    )
    adapter = HFAdapter(models_dir=tmp_path)

    ran: list[list[str]] = []

    class FakeResult:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd, **kwargs):
        ran.append(cmd)
        Path(cmd[2]).write_bytes(b"q6k-output")
        return FakeResult()

    monkeypatch.setattr("entertainment_harness.models.quantize.subprocess.run", fake_run)
    monkeypatch.setattr(
        "entertainment_harness.models.quantize.shutil.which", lambda name: "/usr/bin/llama-quantize"
    )
    out = quantize(REPO, "Q6_K", adapter=adapter,
                   confirm=lambda msg: True, log=lambda m: None)
    assert out.name == "Qwen3-0.6B-Q6_K.gguf"
    assert ran[0][0] == "llama-quantize"
    assert ran[0][3] == "Q6_K"
    # result is discoverable by the adapter
    assert f"{REPO}:Q6_K" in [i.name for i in adapter.list_available()]


# --- remove -------------------------------------------------------------------


def _seed_repo(tmp_path: Path) -> Path:
    folder = tmp_path / REPO
    folder.mkdir(parents=True)
    (folder / "Qwen3-0.6B-Q4_K_M.gguf").write_bytes(b"q4")
    (folder / "Qwen3-0.6B-Q8_0.gguf").write_bytes(b"q8")
    (folder / "mmproj-Qwen3-0.6B-F16.gguf").write_bytes(b"proj")
    return folder


def test_remove_quant_deletes_only_that_file(tmp_path):
    folder = _seed_repo(tmp_path)
    HFAdapter(models_dir=tmp_path).remove(f"{REPO}:Q4_K_M")
    assert not (folder / "Qwen3-0.6B-Q4_K_M.gguf").exists()
    assert (folder / "Qwen3-0.6B-Q8_0.gguf").exists()
    assert (folder / "mmproj-Qwen3-0.6B-F16.gguf").exists()


def test_remove_last_quant_removes_repo_folder(tmp_path):
    folder = _seed_repo(tmp_path)
    adapter = HFAdapter(models_dir=tmp_path)
    adapter.remove(f"{REPO}:Q4_K_M")
    adapter.remove(f"{REPO}:Q8_0")
    assert not folder.exists()  # mmproj goes with it


def test_remove_bare_repo_deletes_everything(tmp_path):
    folder = _seed_repo(tmp_path)
    HFAdapter(models_dir=tmp_path).remove(REPO)
    assert not folder.exists()


def test_remove_missing_raises(tmp_path):
    adapter = HFAdapter(models_dir=tmp_path)
    with pytest.raises(ModelNotFoundError):
        adapter.remove(f"{REPO}:Q4_K_M")
    _seed_repo(tmp_path)
    with pytest.raises(ModelNotFoundError, match="not downloaded"):
        adapter.remove(f"{REPO}:Q6_K")
