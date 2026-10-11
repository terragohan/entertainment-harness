"""Eval-harness tests (frame-gen-evals): experiment TOML validation, the
runner over both generation loops with a fake image client, judge scoring
with a scripted vision adapter, the gallery, and the CLI. Offline only —
no network, no DB, no diffusers/torch, no ollama."""

from __future__ import annotations

import hashlib
import json
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.imagegen import clients as clients_mod  # noqa: E402
from evals.imagegen import gallery as gallery_mod  # noqa: E402
from evals.imagegen import judge as judge_mod  # noqa: E402
from evals.imagegen import runner as runner_mod  # noqa: E402
from evals.imagegen.__main__ import app  # noqa: E402
from evals.imagegen.experiment import (  # noqa: E402
    ExperimentError,
    estimated_generations,
    load_experiment,
)

from entertainment_harness.config import Config  # noqa: E402
from entertainment_harness.video.gen.openrouter import DEFAULT_IMAGE_MODEL  # noqa: E402

cli = CliRunner()


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """load_config() must never see the developer's real config.toml."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path / "eh-data"))


def _write_panel(path: Path, size: tuple[int, int] = (800, 600),
                 color: tuple[int, int, int] = (40, 60, 90)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


class FakeImageClient:
    """Duck-typed text_to_image client: deterministic Pillow PNGs, call
    recording, failure on demand — always (`fail`) or per source panel
    (`fail_panels`, matched on the panel-tagged reference's filename)."""

    def __init__(self, *, uses_references: bool = True, fail: bool = False,
                 fail_panels=()):
        self.uses_references = uses_references
        self.fail = fail
        self.fail_panels = frozenset(fail_panels)
        self.calls: list[dict] = []

    def text_to_image(self, prompt, references, *, model, ratio,
                      ref_ratio=(0, 0), dest):
        self.calls.append({
            "prompt": prompt, "references": list(references),
            "model": model, "ratio": ratio, "dest": Path(dest),
        })
        panel = next((p for tag, p in references if tag == "panel"), None)
        if self.fail or (panel is not None and panel.name in self.fail_panels):
            raise RuntimeError("fake provider boom")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        seed = int(hashlib.sha256((prompt + model).encode()).hexdigest()[:6], 16)
        Image.new("RGB", (96, 54),
                  (seed & 255, (seed >> 8) & 255, (seed >> 16) & 255)).save(dest)
        return dest


class ScriptedVision:
    """Scripted judge adapter: per-dimension reply queues dispatched on the
    prompt; when a queue runs out, its last reply repeats."""

    def __init__(self, faith=("OK",), drift=("OK",), motion=("ADVANCING",)):
        self._queues = {
            "faith": list(faith), "drift": list(drift), "motion": list(motion),
        }
        self.calls: list[dict] = []

    def generate(self, model, prompt, images=None):
        self.calls.append({"model": model, "prompt": prompt, "images": images or []})
        dim = ("motion" if "ADVANCING" in prompt
               else "faith" if "original manga panel" in prompt
               else "drift")
        queue = self._queues[dim]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def ensure(self, model, quant=None):
        pass


def _client_factory(made: list, **by_provider):
    """A client_factory recording construction; per-provider fakes via
    by_provider (default: a plain FakeImageClient)."""

    def factory(provider, config):
        client = by_provider.get(provider) or FakeImageClient()
        made.append({"provider": provider, "config": config, "client": client})
        return client

    return factory


def _write_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "exp.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _make_experiment(tmp_path: Path, *, mode="video-animation", arms, duration=4.0,
                     judge=False, panel_count=1, name="exp",
                     context="test scene"):
    panels = [
        # distinct bytes per panel: identical images share cache keys
        _write_panel(tmp_path / f"panel-{i}.jpg",
                     color=(40 + 50 * i, 60, 90))
        for i in range(panel_count)
    ]
    lines = [
        f'name = "{name}"',
        f'mode = "{mode}"',
        f"duration = {duration}",
        "panels = [" + ", ".join(f'"{p.name}"' for p in panels) + "]",
        f'context = "{context}"',
        f"judge = {'true' if judge else 'false'}",
    ]
    for arm in arms:
        lines.append("\n[[arms]]")
        lines.extend(arm)
    return load_experiment(_write_toml(tmp_path, "\n".join(lines) + "\n"))


# --- experiment TOML validation -------------------------------------------------


def test_valid_experiment_loads_with_defaults(tmp_path):
    panel = _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\n'
        '[[arms]]\nname = "a1"\n'
    ))
    exp = load_experiment(path)
    assert exp.name == "exp" and exp.mode == "video-animation"
    assert exp.duration == 4.0 and exp.judge is True  # defaults
    assert exp.context == ""
    assert exp.panels == [panel]  # resolved relative to the toml's dir
    (arm,) = exp.arms
    assert arm.provider == "openrouter" and arm.model == "" and arm.knobs == {}


def test_unknown_top_level_key_rejected(tmp_path):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\nbogus = 1\n'
        '[[arms]]\nname = "a1"\n'
    ))
    with pytest.raises(ExperimentError, match="unknown top-level"):
        load_experiment(path)


def test_unknown_arm_key_rejected(tmp_path):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\n'
        '[[arms]]\nname = "a1"\nbogus = 1\n'
    ))
    with pytest.raises(ExperimentError, match="unknown arm key"):
        load_experiment(path)


def test_bad_mode_rejected(tmp_path):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "spin"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\n[[arms]]\nname = "a1"\n'
    ))
    with pytest.raises(ExperimentError, match="mode must be one of"):
        load_experiment(path)


def test_duplicate_arm_names_rejected(tmp_path):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\n'
        '[[arms]]\nname = "a1"\n[[arms]]\nname = "a1"\n'
    ))
    with pytest.raises(ExperimentError, match="duplicate arm name"):
        load_experiment(path)


@pytest.mark.parametrize("bad", ["Bad Name", "a/b", "UPPER", "..", ""])
def test_path_unsafe_names_rejected(tmp_path, bad):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        f'panels = ["panel.jpg"]\n[[arms]]\nname = "{bad}"\n'
    ))
    with pytest.raises(ExperimentError, match="slug"):
        load_experiment(path)


@pytest.mark.parametrize("duration", ["0", "-1.5", "true", '"4"'])
def test_non_positive_or_typed_duration_rejected(tmp_path, duration):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\n'
        f"duration = {duration}\n"
        'panels = ["panel.jpg"]\n[[arms]]\nname = "a1"\n'
    ))
    with pytest.raises(ExperimentError, match="duration"):
        load_experiment(path)


@pytest.mark.parametrize("knob", ["fps = 1.5", "interp_fps = 30",
                                  "critic = true", 'resolution = "1280x720"'])
def test_sequence_knobs_rejected_on_animate_arms(tmp_path, knob):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        f'panels = ["panel.jpg"]\n[[arms]]\nname = "a1"\n{knob}\n'
    ))
    with pytest.raises(ExperimentError, match="frame-animation-mode knob"):
        load_experiment(path)


def test_animate_knob_rejected_on_sequence_arms(tmp_path):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "frame-animation"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\n[[arms]]\nname = "a1"\n'
        "seconds_per_frame = 2.0\n"
    ))
    with pytest.raises(ExperimentError, match="video-animation-mode knob"):
        load_experiment(path)


def test_missing_panel_rejected(tmp_path):
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "video-animation"\nduration = 4.0\n'
        'panels = ["nope.jpg"]\n[[arms]]\nname = "a1"\n'
    ))
    with pytest.raises(ExperimentError, match="not found"):
        load_experiment(path)


def test_bad_knob_values_rejected(tmp_path):
    _write_panel(tmp_path / "panel.jpg")
    path = _write_toml(tmp_path, (
        'name = "exp"\nmode = "frame-animation"\nduration = 4.0\n'
        'panels = ["panel.jpg"]\n[[arms]]\nname = "a1"\nfps = -1.0\n'
    ))
    with pytest.raises(ExperimentError, match="fps"):
        load_experiment(path)


# --- clients --------------------------------------------------------------------


def test_build_client_rejects_unknown_provider():
    with pytest.raises(clients_mod.ClientError, match="unknown provider"):
        clients_mod.build_client("bogus", Config())


def test_diffusers_client_errors_clearly_without_the_extra():
    # The test venv has no diffusers/torch (they are not product deps).
    with pytest.raises(clients_mod.ClientError, match="--with diffusers"):
        clients_mod.build_client("diffusers", Config())


def test_diffusers_pixel_math_and_settings():
    client = clients_mod.DiffusersImageClient
    assert client._pixels("1920:1080") == (1024, 576)  # capped, aspect kept
    w, h = client._pixels("1080:1920")
    assert (w, h) == (576, 1024)
    assert w % 8 == 0 and h % 8 == 0
    assert client._pixels("garbage") == (1024, 1024)
    assert client._settings("stabilityai/sdxl-turbo") == {
        "num_inference_steps": 4, "guidance_scale": 0.0,
    }
    assert client._settings("black-forest-labs/FLUX.1-schnell")[
        "num_inference_steps"] == 4
    assert client._settings("some/plain-model") == {
        "num_inference_steps": 20, "guidance_scale": 7.5,
    }
    assert client.uses_references is False


class _FakeTorch:
    """Minimal torch stand-in for DiffusersImageClient's device/dtype logic."""

    float16 = "float16"
    float32 = "float32"

    def __init__(self, *, mps: bool = False, cuda: bool = False):
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps)
        )
        self.cuda = SimpleNamespace(is_available=lambda: cuda)


class _FakePipe:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device


def _fake_pipeline_cls(*, fail_on_variant: bool = False):
    calls = []

    class FakePipelineCls:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append({"model": model, **kwargs})
            if fail_on_variant and kwargs.get("variant"):
                raise OSError("repo has no fp16 variant")
            return _FakePipe()

    return FakePipelineCls, calls


def _diffusers_client(torch_fake, pipeline_cls):
    """A DiffusersImageClient with fakes injected, bypassing __init__
    (which imports torch/diffusers — absent in the test env)."""
    client = object.__new__(clients_mod.DiffusersImageClient)
    client._torch = torch_fake
    client._pipeline_cls = pipeline_cls
    client._pipelines = {}
    return client


def test_diffusers_pipeline_prefers_fp16_variant_on_accelerators():
    cls, calls = _fake_pipeline_cls()
    client = _diffusers_client(_FakeTorch(mps=True), cls)
    pipe = client._pipeline("stabilityai/sdxl-turbo")
    assert calls == [
        {"model": "stabilityai/sdxl-turbo",
         "torch_dtype": "float16", "variant": "fp16"},
    ]
    assert pipe.device == "mps"
    assert client._pipeline("stabilityai/sdxl-turbo") is pipe  # cached
    assert len(calls) == 1


def test_diffusers_pipeline_falls_back_when_variant_load_fails():
    cls, calls = _fake_pipeline_cls(fail_on_variant=True)
    client = _diffusers_client(_FakeTorch(cuda=True), cls)
    pipe = client._pipeline("some/model")
    assert calls == [
        {"model": "some/model", "torch_dtype": "float16", "variant": "fp16"},
        {"model": "some/model", "torch_dtype": "float16"},  # retried plain
    ]
    assert pipe.device == "cuda"


def test_diffusers_pipeline_cpu_loads_without_variant():
    cls, calls = _fake_pipeline_cls()
    client = _diffusers_client(_FakeTorch(), cls)
    pipe = client._pipeline("some/model")
    assert calls == [{"model": "some/model", "torch_dtype": "float32"}]
    assert pipe.device == "cpu"


# --- runner: video-animation ------------------------------------------------------------


def test_runner_animate_counts_and_arm_isolation(tmp_path):
    exp = _make_experiment(tmp_path, arms=[
        ['name = "aaa"', "seconds_per_frame = 1.0", "max_frames = 4"],
        ['name = "bbb"', 'model = "x/y"'],
    ], panel_count=2)
    made = []
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory(made),
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["experiment"]["name"] == "exp"
    assert manifest["experiment"]["mode"] == "video-animation"
    assert manifest["judge"] == {"enabled": False}
    assert manifest["base_config"]["frames"]["provider"] == "local"
    assert manifest["base_config"]["video"]["resolution"] == "1920x1080"

    aaa, bbb = manifest["arms"]
    assert aaa["name"] == "aaa" and aaa["model"] == DEFAULT_IMAGE_MODEL
    assert bbb["name"] == "bbb" and bbb["model"] == "x/y"
    assert aaa["references_used"] is True
    assert aaa["overrides"]["seconds_per_frame"] == 1.0

    # frame counts follow frame_count math; arms land in separate dirs
    assert len(aaa["panels"]) == 2 and len(bbb["panels"]) == 2
    for panel in aaa["panels"]:
        assert panel["metrics"]["requested"] == 4  # frame_count(4.0, 1.0, 4)
        assert panel["metrics"]["produced"] == 4
        assert panel["metrics"]["generated"] == 4  # fresh workdir
        assert panel["metrics"]["cached"] == 0
        assert panel["metrics"]["ratio"] == "1440:1080"  # 800x600 panel
        assert all(f.startswith("arms/aaa/frames/") for f in panel["frames"])
        assert panel["image"].startswith("panels/")
    for panel in bbb["panels"]:
        assert panel["metrics"]["requested"] == 2  # frame_count(4.0, 2.0, 6)
        assert panel["metrics"]["produced"] == 2
        assert all(f.startswith("arms/bbb/frames/") for f in panel["frames"])
    assert (run_dir / "arms/aaa/frames").is_dir()
    assert (run_dir / "arms/bbb/frames").is_dir()
    assert (run_dir / "index.html").is_file()
    # the fake client saw the resolved models
    aaa_client = next(m["client"] for m in made if m["provider"] == "openrouter")
    assert {c["model"] for c in aaa_client.calls} == {DEFAULT_IMAGE_MODEL}


def test_runner_caching_counted_on_repeat_run(tmp_path):
    exp = _make_experiment(tmp_path, arms=[['name = "solo"']], duration=4.0)
    made = []
    run_dir = tmp_path / "run"
    runner_mod.run_experiment(
        exp, run_dir=run_dir, client_factory=_client_factory(made),
        log=lambda m: None,
    )
    calls_after_first = sum(len(m["client"].calls) for m in made)
    assert calls_after_first == 2

    runner_mod.run_experiment(
        exp, run_dir=run_dir, client_factory=_client_factory(made),
        log=lambda m: None,
    )
    calls_after_second = sum(len(m["client"].calls) for m in made)
    assert calls_after_second == calls_after_first  # all cached
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (panel,) = manifest["arms"][0]["panels"]
    assert panel["metrics"]["generated"] == 0
    assert panel["metrics"]["cached"] == panel["metrics"]["produced"] == 2


# --- runner: frame-animation -----------------------------------------------------------


def test_runner_sequence_counts_and_resolution(tmp_path):
    exp = _make_experiment(tmp_path, mode="frame-animation", duration=1.0, arms=[
        ['name = "seq"', "fps = 3.0", "max_frames = 3", "critic = false",
         'resolution = "1440x1080"'],
    ])
    made = []
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory(made),
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (panel,) = manifest["arms"][0]["panels"]
    assert panel["metrics"]["requested"] == 3  # sequence frame_count(1.0, 3, 3)
    assert panel["metrics"]["produced"] == 3
    assert "truncated" not in panel["metrics"]
    assert panel["metrics"]["ratio"] == "1440:1080"  # resolution override
    assert all(f.startswith("arms/seq/sequence/") for f in panel["frames"])
    client = made[0]["client"]
    assert [c["ratio"] for c in client.calls] == ["1440:1080"] * 3
    # chained frames reference the previous frame plus the original panel
    assert [t for t, _ in client.calls[1]["references"]] == ["prev", "panel"]


def test_runner_sequence_truncation_with_always_drift_critic(tmp_path):
    exp = _make_experiment(tmp_path, mode="frame-animation", duration=4.0,
                           judge=False, arms=[
        ['name = "crit"', "fps = 1.5", "max_frames = 6", "critic = true"],
    ])
    vision = ScriptedVision(drift=("DRIFT: the face melted",))
    made = []
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory(made),
        judge_resolver=lambda c, p: (vision, "scripted:1b"), log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # the critic used the adapter even though scoring is off
    assert manifest["judge"]["model"] == "scripted:1b"
    assert "warning" not in manifest["judge"]
    (panel,) = manifest["arms"][0]["panels"]
    assert panel["metrics"]["requested"] == 6  # round(4.0 * 1.5)
    assert panel["metrics"]["produced"] == 1   # truncated at the expansion
    assert panel["metrics"]["truncated"] is True
    assert "judge" not in panel  # scoring disabled
    # expand + first chain + its rejected retry = 3 generations
    assert len(made[0]["client"].calls) == 3


# --- runner: failure isolation ----------------------------------------------------


def test_runner_failing_arm_recorded_and_run_continues(tmp_path):
    """Client construction (or config/model resolution) fails: the arm-level
    backstop records one arm error, panels never start."""
    exp = _make_experiment(tmp_path, arms=[
        ['name = "bad"', 'provider = "runway"'],
        ['name = "good"', 'provider = "openrouter"'],
    ])

    def factory(provider, config):
        if provider == "runway":
            raise RuntimeError("no runway key")
        return FakeImageClient()

    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=factory,
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    bad, good = manifest["arms"]
    assert "no runway key" in bad["error"]
    assert bad["panels"] == []
    assert "error" not in good
    assert good["panels"][0]["metrics"]["produced"] == 2
    html = (run_dir / "index.html").read_text()
    assert "arm failed" in html and "no runway key" in html


def test_runner_panel_failure_recorded_and_arm_continues(tmp_path):
    """A transient failure on one panel (timeout, dropped connection) gets
    its own panel entry and the arm runs its remaining panels."""
    exp = _make_experiment(tmp_path, judge=True, arms=[['name = "flaky"']],
                           panel_count=2)
    vision = ScriptedVision()
    made = []
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run",
        client_factory=_client_factory(
            made, openrouter=FakeImageClient(fail_panels={"panel-0.jpg"})
        ),
        judge_resolver=lambda c, p: (vision, "scripted:1b"),
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (arm,) = manifest["arms"]
    assert "error" not in arm  # no arm-level error
    failed, ok = arm["panels"]
    assert "fake provider boom" in failed["error"]
    assert failed["frames"] == []
    assert failed["source"].endswith("panel-0.jpg")
    assert failed["image"].startswith("panels/")
    assert "metrics" not in failed and "judge" not in failed  # never scored
    assert ok["metrics"]["produced"] == 2 and ok["frames"]
    assert "judge" in ok  # the surviving panel still got scored
    # the arm summary aggregates only the scored panel
    assert arm["judge_summary"]["panels"] == 1
    html = (run_dir / "index.html").read_text()
    assert "panel failed" in html and "fake provider boom" in html
    assert "arms/flaky/frames/" in html  # the successful panel's images


def test_runner_limit_counts_arm_panel_pairs(tmp_path):
    exp = _make_experiment(tmp_path, arms=[
        ['name = "one"'], ['name = "two"'],
    ], panel_count=2)
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", limit=1,
        client_factory=_client_factory([]), log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["limit"] == 1
    assert [a["name"] for a in manifest["arms"]] == ["one"]
    assert len(manifest["arms"][0]["panels"]) == 1


def test_runner_limit_counts_failed_panels(tmp_path):
    """--limit counts attempted pairs: a failed panel still consumes one."""
    exp = _make_experiment(tmp_path, arms=[
        ['name = "flaky"'], ['name = "next"'],
    ], panel_count=1)
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", limit=1,
        client_factory=_client_factory(
            [], openrouter=FakeImageClient(fail=True)
        ),
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [a["name"] for a in manifest["arms"]] == ["flaky"]
    (panel,) = manifest["arms"][0]["panels"]
    assert "fake provider boom" in panel["error"]


# --- runner: generation timeout ---------------------------------------------------


class _SlowFirstPanelClient(FakeImageClient):
    """Sleeps on the first panel; SIGALRM interrupts the sleep, so the
    timeout test itself stays fast."""

    def text_to_image(self, prompt, references, *, model, ratio,
                      ref_ratio=(0, 0), dest):
        panel = next((p for tag, p in references if tag == "panel"), None)
        if panel is not None and panel.name == "panel-0.jpg":
            time.sleep(5)
        return super().text_to_image(
            prompt, references, model=model, ratio=ratio,
            ref_ratio=ref_ratio, dest=dest,
        )


def test_runner_gen_timeout_records_timeout_and_continues(tmp_path):
    if not hasattr(signal, "setitimer"):
        pytest.skip("signal.setitimer unavailable on this platform")
    exp = _make_experiment(tmp_path, arms=[['name = "slow"']], panel_count=2)
    old_handler = signal.getsignal(signal.SIGALRM)
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run",
        client_factory=_client_factory([], openrouter=_SlowFirstPanelClient()),
        gen_timeout=0.2,
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (arm,) = manifest["arms"]
    assert "error" not in arm
    timed_out, ok = arm["panels"]
    assert "TimeoutError" in timed_out["error"]
    assert "0.2s deadline" in timed_out["error"]
    assert timed_out["frames"] == []
    assert ok["metrics"]["produced"] == 2  # the arm's other panel ran
    # the pre-existing alarm state is fully restored
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) is old_handler


@pytest.mark.parametrize("gen_timeout", [None, 0])
def test_runner_gen_timeout_disabled_runs_untimed(tmp_path, gen_timeout):
    exp = _make_experiment(tmp_path, arms=[['name = "solo"']])
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run",
        client_factory=_client_factory([]), gen_timeout=gen_timeout,
        log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["arms"][0]["panels"][0]["metrics"]["produced"] == 2
    if hasattr(signal, "setitimer"):
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


# --- judge ------------------------------------------------------------------------


def test_score_frames_verdicts_and_aggregates(tmp_path):
    panel = _write_panel(tmp_path / "p.jpg")
    frames = [_write_panel(tmp_path / f"f{i}.png", (96, 54)) for i in range(3)]
    vision = ScriptedVision(
        faith=("OK", "DRIFT: the face changed", "total garbage"),
        drift=("OK", "DRIFT: background moved"),
        motion=("ADVANCING", "STAGNANT: nothing moves"),
    )
    scored = judge_mod.score_frames(
        vision, "m", panel, frames, rel=lambda p: p.name
    )
    assert [v["verdict"] for v in scored["frames"]] == [
        "ok", "drift", "inconclusive",
    ]
    assert scored["frames"][1]["note"] == "the face changed"
    assert scored["frames"][2]["note"] == "total garbage"  # garbage tolerated
    assert [(p["drift"], p["motion"]) for p in scored["pairs"]] == [
        ("ok", "advancing"), ("drift", "stagnant"),
    ]
    agg = scored["aggregates"]
    assert agg["frames"] == 3 and agg["faithfulness_ok"] == 1
    assert agg["faithfulness_pass_rate"] == round(1 / 3, 3)
    assert agg["drift_incidents"] == 1
    assert agg["stagnation_rate"] == 0.5
    # prompts: faithfulness compares frame vs panel; drift is verbatim reuse
    faith_call = vision.calls[0]
    assert [p.name for p in faith_call["images"]] == ["p.jpg", "f0.png"]
    from entertainment_harness.video.sequence import DRIFT_PROMPT
    assert vision.calls[3]["prompt"] == DRIFT_PROMPT


def test_runner_judge_scores_land_in_manifest_and_rescore_cached(tmp_path):
    exp = _make_experiment(tmp_path, judge=True, arms=[['name = "solo"']])
    vision = ScriptedVision()
    made = []
    run_dir = tmp_path / "run"
    resolver = lambda c, p: (vision, "scripted:1b")
    runner_mod.run_experiment(
        exp, run_dir=run_dir, client_factory=_client_factory(made),
        judge_resolver=resolver, log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["judge"] == {"enabled": True, "model": "scripted:1b"}
    (panel,) = manifest["arms"][0]["panels"]
    assert [v["verdict"] for v in panel["judge"]["frames"]] == ["ok", "ok"]
    assert panel["judge"]["aggregates"]["faithfulness_pass_rate"] == 1.0
    assert panel["judge"]["aggregates"]["stagnation_rate"] == 0.0
    assert manifest["arms"][0]["judge_summary"]["drift_incidents"] == 0
    calls_after_first = len(vision.calls)

    # a re-run of the fully cached arm still gets scored
    runner_mod.run_experiment(
        exp, run_dir=run_dir, client_factory=_client_factory(made),
        judge_resolver=resolver, log=lambda m: None,
    )
    assert len(vision.calls) > calls_after_first
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (panel,) = manifest["arms"][0]["panels"]
    assert panel["metrics"]["generated"] == 0
    assert panel["judge"]["aggregates"]["faithfulness_pass_rate"] == 1.0


def test_runner_judge_unavailable_records_warning(tmp_path):
    exp = _make_experiment(tmp_path, judge=True, arms=[['name = "solo"']])

    def boom(config, profile):
        raise RuntimeError("ollama is down")

    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory([]),
        judge_resolver=boom, log=lambda m: None,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "ollama is down" in manifest["judge"]["warning"]
    (panel,) = manifest["arms"][0]["panels"]
    assert "judge" not in panel
    assert panel["metrics"]["produced"] == 2  # the run itself completed
    assert (run_dir / "index.html").is_file()


# --- gallery ------------------------------------------------------------------------


def _gallery_run(tmp_path):
    """One real run with a good arm, a references-less arm, and an arm that
    fails at client construction — judged with drift + stagnation so
    badges appear."""
    exp = _make_experiment(tmp_path, judge=True, arms=[
        ['name = "good"', 'provider = "openrouter"'],
        ['name = "local"', 'provider = "diffusers"',
         'model = "stabilityai/sdxl-turbo"'],
        ['name = "bad"', 'provider = "runway"'],
    ])
    vision = ScriptedVision(
        faith=("OK", "DRIFT: the face changed"),
        motion=("STAGNANT: nothing moves",),
    )

    def factory(provider, config):
        if provider == "runway":
            raise RuntimeError("no runway key")
        if provider == "diffusers":
            return FakeImageClient(uses_references=False)
        return FakeImageClient()

    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run",
        client_factory=factory,
        judge_resolver=lambda c, p: (vision, "scripted:1b"),
        log=lambda m: None,
    )
    return run_dir


def test_gallery_from_real_manifest(tmp_path):
    run_dir = _gallery_run(tmp_path)
    out = gallery_mod.render_gallery(run_dir, log=lambda m: None)
    assert out == run_dir / "index.html"
    html = out.read_text()
    assert "id='arm-good'" in html and "id='arm-local'" in html
    assert "id='arm-bad'" in html
    assert "arms/good/frames/" in html  # relative img refs
    assert "panels/00-" in html  # source panel thumb
    assert ">drift<" in html and ">stagnant<" in html and ">ok<" in html
    assert "references_used: false" in html  # blind-generation label
    assert "arm failed" in html and "no runway key" in html
    assert "class='diff'" in html  # provider override differs from base
    assert "scripted:1b" in html  # judge model in the header
    assert "<script" not in html  # no JS


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_gallery_animate_clip(tmp_path):
    exp = _make_experiment(tmp_path, arms=[['name = "solo"']])
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory([]),
        log=lambda m: None,
    )
    gallery_mod.render_gallery(run_dir, clips=True, log=lambda m: None)
    clip = run_dir / "arms/solo/clip-00.mp4"
    assert clip.is_file() and clip.stat().st_size > 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["arms"][0]["panels"][0]["clip"] == "arms/solo/clip-00.mp4"
    assert "<video" in (run_dir / "index.html").read_text()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_gallery_sequence_clip_uses_build_sequence_clip(tmp_path):
    exp = _make_experiment(tmp_path, mode="frame-animation", duration=1.0, arms=[
        ['name = "seq"', "fps = 1.5", "max_frames = 2", "critic = false",
         'resolution = "320x180"'],
    ])
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory([]),
        log=lambda m: None,
    )
    gallery_mod.render_gallery(run_dir, clips=True, log=lambda m: None)
    clip = run_dir / "arms/seq/clip-00.mp4"
    assert clip.is_file() and clip.stat().st_size > 0
    assert "<video" in (run_dir / "index.html").read_text()


# --- CLI ------------------------------------------------------------------------


def test_cli_list_shows_seed_experiments_with_estimates(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo_root)  # panel paths are repo-root relative
    from evals.imagegen.experiment import load_experiments

    expected = {
        exp.name: estimated_generations(exp)
        for exp in load_experiments(repo_root / "evals/experiments")
    }
    result = cli.invoke(app, ["list"])
    assert result.exit_code == 0, result.output
    for name, estimate in expected.items():
        assert name in result.output
        assert str(estimate) in result.output


def test_seed_experiments_load_and_estimate(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo_root)
    from evals.imagegen.experiment import load_experiments

    experiments = {
        exp.name: exp
        for exp in load_experiments(repo_root / "evals/experiments")
    }
    assert set(experiments) == {
        "models-frames", "cadence-video-animation", "cadence-frame-animation",
        "models-i2i",
    }
    assert experiments["models-frames"].mode == "video-animation"
    assert experiments["cadence-video-animation"].mode == "video-animation"
    assert experiments["cadence-frame-animation"].mode == "frame-animation"
    assert experiments["models-i2i"].mode == "video-animation"
    assert {a.provider for a in experiments["models-i2i"].arms} == {"fal"}
    for exp in experiments.values():
        assert exp.arms and exp.panels
        # frame_count clamps to >= 2 per arm x panel in both modes
        assert estimated_generations(exp) >= 2 * len(exp.arms) * len(exp.panels)


def test_cli_run_end_to_end_with_injected_client(tmp_path, monkeypatch):
    exp = _make_experiment(tmp_path, judge=True, arms=[
        ['name = "solo"', 'model = "fake/model"'],
    ])
    made = []
    monkeypatch.setattr(
        clients_mod, "build_client", _client_factory(made)
    )
    result = cli.invoke(app, [
        "run", str(exp.path), "--out", str(tmp_path / "runs"), "--no-judge",
        "--gen-timeout", "0",  # 0 disables the per-panel deadline
    ])
    assert result.exit_code == 0, result.output
    assert len(made) == 1 and made[0]["provider"] == "openrouter"
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "index.html").is_file()
    assert str(run_dir) in result.output and "index.html" in result.output
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (arm,) = manifest["arms"]
    assert arm["name"] == "solo" and arm["model"] == "fake/model"
    assert manifest["judge"] == {"enabled": False}  # --no-judge won over TOML


def test_cli_render_renders_from_manifest(tmp_path):
    exp = _make_experiment(tmp_path, arms=[['name = "solo"']])
    run_dir = runner_mod.run_experiment(
        exp, run_dir=tmp_path / "run", client_factory=_client_factory([]),
        log=lambda m: None,
    )
    (run_dir / "index.html").write_text("stale")
    result = cli.invoke(app, ["render", str(run_dir)])
    assert result.exit_code == 0, result.output
    html = (run_dir / "index.html").read_text()
    assert "stale" not in html and "id='arm-solo'" in html


def test_cli_run_rejects_invalid_experiment(tmp_path):
    path = _write_toml(tmp_path, 'name = "exp"\nmode = "nope"\n')
    result = cli.invoke(app, ["run", str(path), "--out", str(tmp_path / "r")])
    assert result.exit_code == 1
    assert "mode" in result.output


# --- hosted i2i clients (fal / replicate) ------------------------------------

from evals.imagegen import hosted as hosted_mod  # noqa: E402
from entertainment_harness.video.script import VideoError  # noqa: E402


def _png_bytes(size=(1200, 900), color=(90, 120, 60)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, status_code, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPError(f"HTTP {self.status_code}")


class _FakeHttp:
    """Scripted httpx.Client stand-in: routes match (method, URL substring)
    and are consumed in order; every request is recorded for assertions."""

    def __init__(self, routes):
        self.routes = list(routes)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, **kw):
        return self._next("POST", url, kw)

    def get(self, url, **kw):
        return self._next("GET", url, kw)

    def _next(self, method, url, kw):
        self.requests.append((method, url, kw))
        for i, (m, sub, response) in enumerate(self.routes):
            if m == method and sub in url:
                self.routes.pop(i)
                return response() if callable(response) else response
        raise AssertionError(f"unexpected request: {method} {url}")


def _patch_httpx(monkeypatch, fake):
    monkeypatch.setattr(hosted_mod.httpx, "Client", lambda *a, **k: fake)


class TestFalClient:
    def test_roundtrip_normalizes_and_uses_first_reference(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.setenv("FAL_KEY", "test-fal-key")
        panel = _write_panel(tmp_path / "panel.png")
        fake = _FakeHttp([
            ("POST", "queue.fal.run/fal-ai/flux-pro/kontext",
             _FakeResponse(200, {
                 "status_url": "https://queue.fal.run/fal-ai/flux-pro/kontext/requests/r1/status",
                 "response_url": "https://queue.fal.run/fal-ai/flux-pro/kontext/requests/r1",
             })),
            ("GET", "/status", _FakeResponse(200, {"status": "IN_QUEUE"})),
            ("GET", "/status", _FakeResponse(200, {"status": "COMPLETED"})),
            ("GET", "requests/r1",
             _FakeResponse(200, {"image": {"url": "https://cdn.fal.ai/out.png"}})),
            ("GET", "cdn.fal.ai", _FakeResponse(200, content=_png_bytes())),
        ])
        _patch_httpx(monkeypatch, fake)

        dest = tmp_path / "frames" / "f.png"
        out = hosted_mod.FalImageClient().text_to_image(
            "a motion phase", [("panel", panel)],
            model="fal-ai/flux-pro/kontext", ratio="1280:720", dest=dest,
        )
        assert out == dest and dest.is_file()
        with Image.open(dest) as img:
            assert img.size == (1280, 720)  # normalized to the ratio
        method, url, kw = fake.requests[0]
        assert kw["headers"]["Authorization"] == "Key test-fal-key"
        assert kw["json"]["prompt"] == "a motion phase"
        assert kw["json"]["image_url"].startswith("data:image/jpeg;base64,")

    def test_seedream_takes_image_urls_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "k")
        panel = _write_panel(tmp_path / "panel.png")
        fake = _FakeHttp([
            ("POST", "seedream", _FakeResponse(200, {
                "status_url": "https://queue.fal.run/x/status",
                "response_url": "https://queue.fal.run/x",
            })),
            ("GET", "/status", _FakeResponse(200, {"status": "COMPLETED"})),
            ("GET", "queue.fal.run/x",
             _FakeResponse(200, {"images": [{"url": "https://cdn.fal.ai/o.png"}]})),
            ("GET", "cdn.fal.ai", _FakeResponse(200, content=_png_bytes())),
        ])
        _patch_httpx(monkeypatch, fake)
        hosted_mod.FalImageClient().text_to_image(
            "p", [("panel", panel)],
            model="fal-ai/bytedance/seedream/v4/edit", ratio="1024:1024",
            dest=tmp_path / "f.png",
        )
        assert "image_urls" in fake.requests[0][2]["json"]
        assert "image_url" not in fake.requests[0][2]["json"]

    def test_missing_key_is_a_client_error(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.delenv("FAL_API_KEY", raising=False)
        with pytest.raises(clients_mod.ClientError, match="FAL_KEY"):
            hosted_mod.FalImageClient()

    def test_empty_model_raises_video_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "k")
        with pytest.raises(VideoError, match="endpoint id"):
            hosted_mod.FalImageClient().text_to_image(
                "p", [("panel", _write_panel(tmp_path / "p.png"))],
                model="", ratio="1024:1024", dest=tmp_path / "f.png",
            )


class TestReplicateClient:
    def test_official_model_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
        panel = _write_panel(tmp_path / "panel.png")
        fake = _FakeHttp([
            ("POST", "/v1/models/black-forest-labs/flux-kontext-dev/predictions",
             _FakeResponse(201, {
                 "status": "starting",
                 "urls": {"get": "https://api.replicate.com/v1/predictions/p1"},
             })),
            ("GET", "predictions/p1", _FakeResponse(200, {"status": "processing"})),
            ("GET", "predictions/p1",
             _FakeResponse(200, {
                 "status": "succeeded",
                 "output": "https://replicate.delivery/x.png",
             })),
            ("GET", "replicate.delivery", _FakeResponse(200, content=_png_bytes())),
        ])
        _patch_httpx(monkeypatch, fake)

        dest = tmp_path / "f.png"
        hosted_mod.ReplicateImageClient().text_to_image(
            "edit it", [("prev", panel)],
            model="black-forest-labs/flux-kontext-dev", ratio="1024:1024",
            dest=dest,
        )
        assert dest.is_file()
        method, url, kw = fake.requests[0]
        assert kw["headers"]["Authorization"] == "Bearer test-token"
        assert "version" not in kw["json"]
        assert kw["json"]["input"]["image"].startswith("data:image/jpeg;base64,")

    def test_community_model_uses_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "t")
        panel = _write_panel(tmp_path / "panel.png")
        fake = _FakeHttp([
            ("POST", "/v1/predictions",
             _FakeResponse(201, {
                 "status": "starting",
                 "urls": {"get": "https://api.replicate.com/v1/predictions/p2"},
             })),
            ("GET", "predictions/p2", _FakeResponse(200, {"status": "succeeded",
                                                          "output": ["https://r.d/x.png"]})),
            ("GET", "r.d", _FakeResponse(200, content=_png_bytes())),
        ])
        _patch_httpx(monkeypatch, fake)
        hosted_mod.ReplicateImageClient().text_to_image(
            "p", [("panel", panel)], model="lucataco/omnigen2:abc123",
            ratio="1024:1024", dest=tmp_path / "f.png",
        )
        _, _, kw = fake.requests[0]
        assert kw["json"]["version"] == "abc123"

    def test_failed_prediction_surfaces_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "t")
        panel = _write_panel(tmp_path / "panel.png")
        fake = _FakeHttp([
            ("POST", "/v1/predictions",
             _FakeResponse(201, {
                 "status": "starting",
                 "urls": {"get": "https://api.replicate.com/v1/predictions/p3"},
             })),
            ("GET", "predictions/p3", _FakeResponse(200, {
                "status": "failed", "error": {"message": "out of memory"},
            })),
        ])
        _patch_httpx(monkeypatch, fake)
        with pytest.raises(VideoError, match="out of memory"):
            hosted_mod.ReplicateImageClient().text_to_image(
                "p", [("panel", panel)], model="lucataco/omnigen2:abc123",
                ratio="1024:1024", dest=tmp_path / "f.png",
            )

    def test_missing_token_is_a_client_error(self, monkeypatch):
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
        monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
        with pytest.raises(clients_mod.ClientError, match="REPLICATE_API_TOKEN"):
            hosted_mod.ReplicateImageClient()


def test_build_client_dispatches_new_providers(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    monkeypatch.setenv("REPLICATE_API_TOKEN", "t")
    assert clients_mod.build_client("fal", Config()).name == "fal"
    assert clients_mod.build_client("replicate", Config()).name == "replicate"
