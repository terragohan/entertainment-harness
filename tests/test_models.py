"""Registry quant-selection tests + OllamaAdapter contract tests (respx)."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from entertainment_harness.models.base import ModelInfo, ModelTooLargeError
from entertainment_harness.models.ollama import OllamaAdapter
from entertainment_harness.models.registry import fit_verdict, select_quant

GB = 10**9
BUDGET = 16 * GB

TAGS_RESPONSE = {
    "models": [
        {
            "name": "qwen3-vl:8b-instruct",
            "size": int(6.1 * GB),
            "details": {
                "parameter_size": "8.8B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "bakllava:latest",
            "size": int(4.7 * GB),
            "details": {"parameter_size": "7B", "quantization_level": "Q4_0"},
        },
    ]
}


def _candidate(quant: str, size_gb: float, name: str = "m:8b") -> ModelInfo:
    return ModelInfo(
        name=name, backend="ollama", params=8.0, quant=quant, size_bytes=int(size_gb * GB)
    )


# --- quant selection -------------------------------------------------------


def test_selects_q8_when_it_fits():
    candidates = [_candidate("Q4_K_M", 5), _candidate("Q8_0", 8.5)]
    chosen, warning = select_quant(candidates, BUDGET)
    assert chosen.quant == "Q8_0"
    assert warning is None


def test_falls_back_to_q4_with_warning():
    candidates = [_candidate("Q8_0", 20), _candidate("Q4_K_M", 12)]
    chosen, warning = select_quant(candidates, BUDGET)
    assert chosen.quant == "Q4_K_M"
    assert warning is not None
    assert "below 8-bit" in warning


def test_prefer_speed_picks_smallest_first():
    candidates = [_candidate("Q8_0", 8.5), _candidate("Q4_K_M", 5)]
    chosen, _ = select_quant(candidates, BUDGET, policy="prefer-speed")
    assert chosen.quant == "Q4_K_M"


def test_tight_fit_warns():
    candidates = [_candidate("Q8_0", 15)]  # <2 GB headroom on 16 GB budget
    chosen, warning = select_quant(candidates, BUDGET)
    assert chosen.quant == "Q8_0"
    assert warning is not None
    assert "headroom" in warning


def test_nothing_fits_raises_with_alternatives():
    alternatives = [_candidate("Q4_K_M", 5, name="qwen3:4b")]
    candidates = [_candidate("Q4_K_M", 19, name="qwen3-vl:32b")]
    with pytest.raises(ModelTooLargeError) as exc_info:
        select_quant(candidates, BUDGET, alternatives=alternatives)
    message = str(exc_info.value)
    assert "qwen3-vl:32b" in message
    assert "qwen3:4b" in message  # lists what would fit


@respx.mock
def test_explicit_pin_skips_selection():
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.hardware import HardwareProfile
    from entertainment_harness.models.registry import resolve

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    profile = HardwareProfile(
        chip="Apple M4 Pro",
        total_ram_bytes=24 * GB,
        budget_bytes=BUDGET,
        gpu_backend="metal",
    )
    # Local install is Q4_K_M; pinning q8_0 must not silently "upgrade" to a
    # different local artifact — the pinned tag is what gets resolved.
    role = ModelRoleConfig(backend="ollama", model="qwen3-vl:8b-instruct", quant="q8_0")
    selection = resolve(role, profile, Config())
    assert selection.pinned is True
    # pinned tag not local -> falls back to the base model's local info
    assert selection.info.name == "qwen3-vl:8b-instruct"


def test_fit_verdict():
    assert fit_verdict(5 * GB, BUDGET) == "fits"
    assert fit_verdict(15 * GB, BUDGET) == "tight"
    assert fit_verdict(20 * GB, BUDGET) == "too large"
    assert fit_verdict(None, BUDGET) == "unknown"


# --- resolve(): pinned quant + oversized refusal ---------------------------


def _profile() -> "HardwareProfile":
    from entertainment_harness.hardware import HardwareProfile

    return HardwareProfile("Apple M4 Pro", 24 * GB, BUDGET, "metal")


@respx.mock
def test_pinned_below_q8_warns():
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.models.registry import resolve

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    role = ModelRoleConfig(
        backend="ollama", model="qwen3-vl:8b-instruct", quant="q4_k_m"
    )
    selection = resolve(role, _profile(), Config())
    assert selection.pinned is True
    assert selection.warning is not None
    assert "below 8-bit" in selection.warning


@respx.mock
def test_pinned_q8_has_no_quality_warning():
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.models.registry import resolve

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    role = ModelRoleConfig(
        backend="ollama", model="qwen3-vl:8b-instruct", quant="q8_0"
    )
    selection = resolve(role, _profile(), Config())
    assert selection.warning is None


@respx.mock
def test_oversized_tag_refused_with_alternatives():
    """qwen3-vl:32b is not local and /api/show 404s; the tag's 32B params put
    even the smallest realistic quant over budget -> refuse, list what fits.
    Crucially it must NOT silently select the installed qwen3-vl:8b-instruct."""
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.models.registry import resolve

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    role = ModelRoleConfig(backend="ollama", model="qwen3-vl:32b")
    with pytest.raises(ModelTooLargeError) as exc_info:
        resolve(role, _profile(), Config())
    message = str(exc_info.value)
    assert "qwen3-vl:32b" in message
    assert "qwen3-vl:8b-instruct" in message  # listed as what would fit
    assert "bakllava:latest" in message


@respx.mock
def test_unknown_size_tag_proceeds_to_pull():
    """No size, no params in the tag ('latest') -> cannot estimate; let
    ensure() pull rather than refuse blindly."""
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.models.registry import resolve

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json={"models": []})
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    role = ModelRoleConfig(backend="ollama", model="bakllava:latest")
    selection = resolve(role, _profile(), Config())
    assert selection.info.name == "bakllava:latest"


# --- OllamaAdapter contract (mocked HTTP) ----------------------------------


@respx.mock
def test_list_available_parses_quant_and_size():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    adapter = OllamaAdapter()
    infos = adapter.list_available()
    assert len(infos) == 2
    first = infos[0]
    assert first.name == "qwen3-vl:8b-instruct"
    assert first.backend == "ollama"
    assert first.params == pytest.approx(8.8)
    assert first.quant == "Q4_K_M"
    assert first.size_bytes == int(6.1 * GB)


@respx.mock
def test_supports_uses_local_data():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    adapter = OllamaAdapter()
    info = adapter.supports("qwen3-vl:8b-instruct")
    assert info.quant == "Q4_K_M"
    assert info.size_bytes == int(6.1 * GB)


@respx.mock
def test_supports_falls_back_to_show_for_remote_model():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(
        return_value=Response(
            200,
            json={
                "details": {
                    "parameter_size": "4.0B",
                    "quantization_level": "Q4_K_M",
                }
            },
        )
    )
    adapter = OllamaAdapter()
    info = adapter.supports("qwen3:4b")
    assert info.params == pytest.approx(4.0)
    assert info.size_bytes is None  # not local; size unknown until pulled


@respx.mock
def test_ensure_pulls_missing_model_with_quant_tag():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    pull = respx.post("http://localhost:11434/api/pull").mock(
        return_value=Response(200, json={"status": "success"})
    )
    adapter = OllamaAdapter()
    adapter.ensure("qwen3-vl:8b-instruct", quant="q8_0")
    assert pull.called
    import json

    body = json.loads(pull.calls.last.request.content)
    assert body == {"model": "qwen3-vl:8b-instruct-q8_0", "stream": False}


@respx.mock
def test_ensure_skips_pull_when_present():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    pull = respx.post("http://localhost:11434/api/pull").mock(
        return_value=Response(200, json={"status": "success"})
    )
    adapter = OllamaAdapter()
    adapter.ensure("bakllava:latest")
    assert not pull.called


@respx.mock
def test_generate_sends_base64_images(tmp_path):
    image = tmp_path / "page-001.jpg"
    image.write_bytes(b"\xff\xd8fake-jpeg")
    chat = respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"role": "assistant", "content": "A recap."}}
        )
    )
    adapter = OllamaAdapter()
    out = adapter.generate("qwen3-vl:8b-instruct", "Summarize.", images=[image])
    assert out == "A recap."
    import base64, json

    body = json.loads(chat.calls.last.request.content)
    assert body["model"] == "qwen3-vl:8b-instruct"
    assert body["stream"] is False
    assert body["options"]["num_ctx"] >= 8192  # pages exceed the 4096 default
    images = body["messages"][0]["images"]
    assert base64.b64decode(images[0]) == b"\xff\xd8fake-jpeg"


@respx.mock
def test_resolve_tolerates_unknown_remote_model():
    """Ollama 404s /api/show for non-local models; resolve() must not crash —
    ensure() pulls the model at generation time."""
    from entertainment_harness.config import Config, ModelRoleConfig
    from entertainment_harness.hardware import HardwareProfile
    from entertainment_harness.models.registry import resolve

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json=TAGS_RESPONSE)
    )
    respx.post("http://localhost:11434/api/show").mock(return_value=Response(404))
    profile = HardwareProfile("Apple M4 Pro", 24 * GB, BUDGET, "metal")
    role = ModelRoleConfig(backend="ollama", model="qwen3:4b")
    selection = resolve(role, profile, Config())
    assert selection.info.name == "qwen3:4b"
    assert selection.info.size_bytes is None


@respx.mock
def test_remove_deletes_local_model():
    delete = respx.delete("http://localhost:11434/api/delete").mock(
        return_value=Response(200, json={})
    )
    OllamaAdapter().remove("bakllava:latest")
    import json

    assert json.loads(delete.calls.last.request.content) == {"model": "bakllava:latest"}


@respx.mock
def test_remove_unknown_model_raises():
    from entertainment_harness.models.base import ModelNotFoundError

    respx.delete("http://localhost:11434/api/delete").mock(
        return_value=Response(404, json={"error": "model not found"})
    )
    with pytest.raises(ModelNotFoundError):
        OllamaAdapter().remove("nope:latest")
