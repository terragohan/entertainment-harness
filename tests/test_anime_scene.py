"""Tests for the manga→anime-scene vertical slice (eh anime-scene).

Covers the SceneSpec domain (parse/validate/crop), the planner contract,
Runway request shapes (fully mocked — no paid calls), and the staged
pipeline's caching/invalidation behavior. Real ffmpeg is used only for the
assembly test, guarded by shutil.which like test_video_gen.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import respx
from PIL import Image

from entertainment_harness.anime import pipeline as anime_pipeline
from entertainment_harness.anime.pipeline import (
    build_scene,
    parse_page_spec,
)
from entertainment_harness.anime.planner import plan_scene
from entertainment_harness.anime.scene import (
    SceneError,
    SceneSpec,
    Shot,
    crop_page,
    load_scene,
    parse_scene,
    save_scene,
    validate_scene,
)
from entertainment_harness.config import Config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection
from entertainment_harness.video.gen.runway import RunwayProvider

GB = 10**9


def _shot(index: int, page: int = 8, **overrides) -> Shot:
    fields = dict(
        index=index,
        source_page=page,
        action=f"action {index}",
        keyframe_prompt=f"keyframe {index}",
        animation_prompt=f"motion {index}",
    )
    fields.update(overrides)
    return Shot(**fields)


def _spec(shots: list[Shot] | None = None, pages=(8, 9, 10)) -> SceneSpec:
    return SceneSpec(
        title="Kenja no Mago",
        chapter=0,
        source_pages=list(pages),
        instruction="stay faithful",
        scene_summary="summary",
        continuity="continuity",
        shots=shots if shots is not None else [_shot(i) for i in range(3)],
    )


def _scene_json(shots: list[Shot] | None = None, **overrides) -> str:
    payload = _spec(shots).to_dict()
    payload.update(overrides)
    return json.dumps(payload)


# --- SceneSpec parsing / validation ----------------------------------------


def test_parse_scene_tolerates_fences_and_preamble():
    raw = "Here is the storyboard:\n```json\n" + _scene_json() + "\n```\n"
    spec = parse_scene(raw)
    assert spec.scene_summary == "summary"
    assert len(spec.shots) == 3
    assert spec.shots[0].duration_s == 5


def test_parse_scene_rejects_non_json():
    with pytest.raises(SceneError, match="JSON object"):
        parse_scene("no json here")


def test_validate_rejects_too_few_shots():
    with pytest.raises(SceneError, match="3-5 shots"):
        validate_scene(_spec(shots=[_shot(i) for i in range(2)]))


def test_validate_rejects_too_many_shots():
    with pytest.raises(SceneError, match="3-5 shots"):
        validate_scene(_spec(shots=[_shot(i) for i in range(6)]))


def test_validate_rejects_non_sequential_indices():
    shots = [_shot(0), _shot(2), _shot(1)]
    with pytest.raises(SceneError, match="sequential"):
        validate_scene(_spec(shots=shots))


def test_validate_rejects_non_five_second_shots():
    shots = [_shot(0), _shot(1, duration_s=4.0), _shot(2)]
    with pytest.raises(SceneError, match="exactly 5s"):
        validate_scene(_spec(shots=shots))


def test_validate_enforces_max_total_duration(monkeypatch):
    # 5x5s can never exceed 30s, so shrink the cap to exercise the check.
    monkeypatch.setattr(
        "entertainment_harness.anime.scene.MAX_SCENE_SECONDS", 10
    )
    with pytest.raises(SceneError, match="max is 10s"):
        validate_scene(_spec())


def test_validate_rejects_page_outside_selection():
    shots = [_shot(0, page=8), _shot(1, page=12), _shot(2, page=9)]
    with pytest.raises(SceneError, match="outside the selected pages"):
        validate_scene(_spec(shots=shots))


def test_validate_rejects_empty_action_or_prompts():
    for field_name in ("action", "keyframe_prompt", "animation_prompt"):
        data = _shot(1).to_dict()
        data[field_name] = "  "  # from_dict strips to empty
        shots = [_shot(0), Shot.from_dict(data), _shot(2)]
        with pytest.raises(SceneError, match=f"empty {field_name}"):
            validate_scene(_spec(shots=shots))


def test_crop_validation():
    bad_crops = [
        [0.1, 0.1, 0.9],            # wrong length
        [-0.1, 0.0, 0.5, 0.5],      # out of range
        [0.5, 0.0, 0.5, 0.9],       # zero width
        [0.1, 0.8, 0.9, 0.2],       # bottom < top
    ]
    for crop in bad_crops:
        shots = [_shot(0), _shot(1, source_crop=crop), _shot(2)]
        with pytest.raises(SceneError, match="crop"):
            validate_scene(_spec(shots=shots))


def test_crop_page_produces_normalized_box(tmp_path):
    page = tmp_path / "page.png"
    Image.new("RGB", (100, 200), "white").save(page)
    dest = crop_page(page, [0.1, 0.2, 0.6, 0.8], tmp_path / "crop.png")
    with Image.open(dest) as img:
        assert img.size == (50, 120)


def test_crop_page_full_page_when_no_crop(tmp_path):
    page = tmp_path / "page.png"
    Image.new("RGB", (100, 200), "white").save(page)
    dest = crop_page(page, None, tmp_path / "full.png")
    with Image.open(dest) as img:
        assert img.size == (100, 200)


def test_save_load_roundtrip_and_edit_revalidation(tmp_path):
    path = tmp_path / "scene.json"
    save_scene(_spec(), path)
    spec = load_scene(path)
    assert spec.total_seconds == 15
    # A bad human edit is caught on load, before any paid generation.
    payload = json.loads(path.read_text())
    payload["shots"][1]["source_page"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(SceneError, match="outside the selected pages"):
        load_scene(path)


def test_parse_page_spec_ranges_and_bounds():
    assert parse_page_spec("8-10", 11) == [8, 9, 10]
    assert parse_page_spec("8,10", 11) == [8, 10]
    with pytest.raises(SceneError, match="out of range"):
        parse_page_spec("8-12", 11)
    with pytest.raises(SceneError, match="Bad page"):
        parse_page_spec("x", 11)


# --- Stage 1: planner contract ----------------------------------------------

class PlannerAdapter:
    """Records generate() calls; returns canned scene JSON."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[dict] = []

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def generate(self, model, prompt, images=None):
        self.calls.append({"model": model, "prompt": prompt, "images": images or []})
        return self.output


def _pages(directory: Path, numbers=(8, 9, 10)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in numbers:
        p = directory / f"page-{n:02d}.png"
        Image.new("RGB", (32, 32), "white").save(p)
        paths.append(p)
    return paths


def test_planner_sends_all_selected_pages_as_images(tmp_path):
    pages = _pages(tmp_path)
    adapter = PlannerAdapter(_scene_json())
    spec = plan_scene(adapter, "fake-vl", pages, [8, 9, 10], "Kenja no Mago", 0, "serious tone")
    assert adapter.calls[0]["images"] == pages
    assert [s.source_page for s in spec.shots] == [8, 8, 8]


def test_planner_includes_user_instruction_and_identity(tmp_path):
    pages = _pages(tmp_path)
    adapter = PlannerAdapter(_scene_json())
    spec = plan_scene(adapter, "fake-vl", pages, [8, 9, 10], "Kenja no Mago", 0, "Emphasize the attack.")
    prompt = adapter.calls[0]["prompt"]
    assert "Emphasize the attack." in prompt
    assert "Kenja no Mago" in prompt
    assert spec.instruction == "Emphasize the attack."
    assert spec.title == "Kenja no Mago"


def test_planner_retries_once_on_invalid_json(tmp_path):
    pages = _pages(tmp_path)
    adapter = PlannerAdapter("not json")
    with pytest.raises(SceneError, match="after 2 attempts"):
        plan_scene(adapter, "fake-vl", pages, [8, 9, 10], "t", 0, "i")
    assert len(adapter.calls) == 2


def test_planner_stamps_caller_source_pages(tmp_path):
    pages = _pages(tmp_path)
    # Model claims pages it wasn't given; the caller's selection wins.
    adapter = PlannerAdapter(_scene_json(source_pages=[1, 2, 3]))
    spec = plan_scene(adapter, "fake-vl", pages, [8, 9, 10], "t", 0, "i")
    assert spec.source_pages == [8, 9, 10]


def test_planner_single_page_coerces_misnumbered_shots(tmp_path):
    pages = _pages(tmp_path, numbers=(9,))
    payload = json.loads(_scene_json(source_pages=[1, 1, 1]))
    adapter = PlannerAdapter(json.dumps(payload))
    spec = plan_scene(adapter, "fake-vl", pages, [9], "t", 0, "i")
    assert [s.source_page for s in spec.shots] == [9, 9, 9]


# --- Runway request shapes (fully mocked; no paid calls) ---------------------

def _runway() -> RunwayProvider:
    config = Config()
    config.video_gen.runway.api_key = "rk_test"
    return RunwayProvider(config)


def _mock_task(respx_mock, endpoint: str, output_url: str, captured: dict):
    def capture(request):
        captured["payload"] = json.loads(request.content.decode())
        return respx.MockResponse(200, json={"id": "task-x"})

    respx_mock.post(f"https://api.dev.runwayml.com/v1/{endpoint}").mock(
        side_effect=capture
    )
    respx_mock.get("https://api.dev.runwayml.com/v1/tasks/task-x").mock(
        return_value=respx.MockResponse(
            200, json={"id": "task-x", "status": "SUCCEEDED", "output": [output_url]}
        )
    )
    respx_mock.get(output_url).mock(
        return_value=respx.MockResponse(200, content=b"generated-bytes")
    )


def test_runway_keyframe_request_shape(tmp_path, respx_mock):
    provider = _runway()
    src = tmp_path / "src.png"
    Image.new("RGB", (800, 1200), "white").save(src)
    captured: dict = {}
    _mock_task(respx_mock, "text_to_image", "https://cdn.runwayml.com/k.png", captured)

    dest = provider.text_to_image(
        "frame for @Source", [("Source", src)],
        model="gen4_image", ratio="1280:720", ref_ratio=(1280, 720),
        dest=tmp_path / "out.png",
    )

    assert dest.read_bytes() == b"generated-bytes"
    payload = captured["payload"]
    assert payload["model"] == "gen4_image"
    assert payload["ratio"] == "1280:720"
    refs = payload["referenceImages"]
    assert [r["tag"] for r in refs] == ["Source"]
    assert refs[0]["uri"].startswith("data:image/")
    assert payload["promptText"] == "frame for @Source"


def test_runway_keyframe_prompt_capped_at_api_limit(tmp_path, respx_mock):
    provider = _runway()
    src = tmp_path / "src.png"
    Image.new("RGB", (64, 64), "white").save(src)
    captured: dict = {}
    _mock_task(respx_mock, "text_to_image", "https://cdn.runwayml.com/k.png", captured)

    provider.text_to_image(
        "x" * 1500, [("Source", src)],
        model="gen4_image", ratio="1280:720", ref_ratio=(1280, 720),
        dest=tmp_path / "out.png",
    )

    assert len(captured["payload"]["promptText"]) == 1000


def test_runway_keyframe_later_shots_carry_continuity_refs(tmp_path, respx_mock):
    provider = _runway()
    imgs = []
    for name in ("src", "anchor", "prev"):
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (64, 64), "white").save(p)
        imgs.append(p)
    captured: dict = {}
    _mock_task(respx_mock, "text_to_image", "https://cdn.runwayml.com/k.png", captured)

    provider.text_to_image(
        "frame for @Source guided by @Anchor and @Previous",
        [("Source", imgs[0]), ("Anchor", imgs[1]), ("Previous", imgs[2])],
        model="gen4_image", ratio="1280:720", ref_ratio=(1280, 720),
        dest=tmp_path / "out.png",
    )

    refs = captured["payload"]["referenceImages"]
    assert [r["tag"] for r in refs] == ["Source", "Anchor", "Previous"]


def test_runway_video_request_shape(tmp_path, respx_mock):
    provider = _runway()
    keyframe = tmp_path / "keyframe.png"
    Image.new("RGB", (1280, 720), "white").save(keyframe)
    captured: dict = {}
    _mock_task(respx_mock, "image_to_video", "https://cdn.runwayml.com/v.mp4", captured)

    provider.image_to_video(
        keyframe, "animate", duration=5, ratio="1280:720",
        model="gen4.5", ref_ratio=(1280, 720), dest=tmp_path / "out.mp4",
    )

    payload = captured["payload"]
    assert payload["model"] == "gen4.5"
    assert payload["ratio"] == "1280:720"
    assert payload["duration"] == 5
    assert payload["promptImage"].startswith("data:image/")
    assert payload["promptText"] == "animate"


def test_runway_video_configurable_alternate_model(tmp_path, respx_mock):
    provider = _runway()
    keyframe = tmp_path / "keyframe.png"
    Image.new("RGB", (1280, 720), "white").save(keyframe)
    captured: dict = {}
    _mock_task(respx_mock, "image_to_video", "https://cdn.runwayml.com/v.mp4", captured)

    provider.image_to_video(
        keyframe, "animate", duration=5, ratio="1280:720",
        model="seedance2_5", ref_ratio=(1280, 720), dest=tmp_path / "out.mp4",
    )
    assert captured["payload"]["model"] == "seedance2_5"


# --- Pipeline: staged caching, invalidation, assembly ------------------------

class FakeProvider:
    """Deterministic stand-in for RunwayProvider: writes content derived
    from its inputs so fingerprints change exactly when inputs change."""

    def __init__(self) -> None:
        self.keyframe_calls: list[dict] = []
        self.video_calls: list[dict] = []

    def text_to_image(self, prompt, references, *, model, ratio, ref_ratio, dest):
        self.keyframe_calls.append(
            {"prompt": prompt, "tags": [t for t, _ in references],
             "model": model, "ratio": ratio}
        )
        blob = prompt + "|" + "|".join(
            t + Path(p).read_bytes().hex() for t, p in references
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(hashlib.sha256(blob.encode()).digest())
        return dest

    def image_to_video(self, image, prompt, *, duration, ratio, model=None,
                       ref_ratio=None, dest):
        self.video_calls.append(
            {"image": Path(image).name, "prompt": prompt,
             "model": model, "ratio": ratio, "duration": duration}
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(
            hashlib.sha256(Path(image).read_bytes() + prompt.encode()).digest()
        )
        return dest


@pytest.fixture
def scene_env(tmp_path, monkeypatch):
    """Isolated works dir, fake planner, fake provider, no real ffmpeg."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    pages = {n: p for n, p in zip((8, 9, 10), _pages(tmp_path / "src", (8, 9, 10)))}
    planner = PlannerAdapter(_scene_json())
    info = ModelInfo("fake-vl", "fake", 8.0, "Q8_0", 5 * GB)
    monkeypatch.setattr(
        anime_pipeline, "get_vision_model",
        lambda config, profile: Selection(adapter=planner, info=info),
    )
    provider = FakeProvider()

    def fake_assemble(clips, workdir):
        fake_assemble.orders.append([c.name for c in clips])
        out = workdir / "out.mp4"
        out.write_bytes(b"".join(c.read_bytes() for c in clips))
        return out

    fake_assemble.orders: list[list[str]] = []
    monkeypatch.setattr(anime_pipeline, "assemble_scene", fake_assemble)
    monkeypatch.setattr(anime_pipeline, "video_duration", lambda p: 15.0)
    config = Config()
    profile = HardwareProfile("fake", 24 * GB, 16 * GB, "cpu")
    return pages, planner, provider, config, profile


def _build(env, **kwargs):
    pages, _, provider, config, profile = env
    return build_scene(
        "s1", "ch-0", 0, "Kenja no Mago", pages, config, profile,
        "stay faithful", provider=provider, **kwargs,
    )


def test_stop_after_plan_writes_editable_scene_and_spends_nothing(scene_env):
    pages, planner, provider, _, _ = scene_env
    out = _build(scene_env, stop_after="plan")
    assert out.name == "scene.json"
    spec = load_scene(out)
    assert len(spec.shots) == 3
    assert len(planner.calls) == 1
    assert planner.calls[0]["images"] == [pages[8], pages[9], pages[10]]
    assert provider.keyframe_calls == [] and provider.video_calls == []


def test_full_build_generates_keyframes_clips_and_out(scene_env):
    _, _, provider, _, _ = scene_env
    out = _build(scene_env)
    assert out.name == "out.mp4"
    workdir = out.parent
    assert len(list((workdir / "keyframes").glob("shot-*.png"))) == 3
    assert len(list((workdir / "clips").glob("shot-*.mp4"))) == 3
    # shot 0 references only @Source; later shots carry continuity refs
    # (shot 1's previous IS the anchor, so @Previous appears from shot 2 on)
    assert provider.keyframe_calls[0]["tags"] == ["Source"]
    assert provider.keyframe_calls[1]["tags"] == ["Source", "Anchor"]
    assert provider.keyframe_calls[2]["tags"] == ["Source", "Anchor", "Previous"]
    # every video call: configured model, landscape ratio, exactly 5s
    for call in provider.video_calls:
        assert call["model"] == "gen4.5"
        assert call["ratio"] == "1280:720"
        assert call["duration"] == 5
    # assembly receives clips in storyboard order
    assert anime_pipeline.assemble_scene.orders == [  # type: ignore[attr-defined]
        ["shot-00.mp4", "shot-01.mp4", "shot-02.mp4"]
    ]


def test_rerun_is_fully_cached(scene_env):
    _build(scene_env)
    _, planner, provider, _, _ = scene_env
    planner.calls.clear()
    provider.keyframe_calls.clear()
    provider.video_calls.clear()
    _build(scene_env)
    assert planner.calls == []  # scene.json loaded, never regenerated
    assert provider.keyframe_calls == []
    assert provider.video_calls == []


def test_missing_clip_regenerates_only_that_clip(scene_env):
    workdir = _build(scene_env).parent
    (workdir / "clips" / "shot-01.mp4").unlink()
    _, _, provider, _, _ = scene_env
    provider.keyframe_calls.clear()
    provider.video_calls.clear()
    _build(scene_env)
    assert provider.keyframe_calls == []
    assert [c["image"] for c in provider.video_calls] == ["shot-01.png"]


def test_editing_last_shot_keeps_earlier_shots_cached(scene_env):
    workdir = _build(scene_env).parent
    scene_path = workdir / "scene.json"
    payload = json.loads(scene_path.read_text())
    payload["shots"][2]["keyframe_prompt"] = "edited keyframe"
    scene_path.write_text(json.dumps(payload))
    _, _, provider, _, _ = scene_env
    provider.keyframe_calls.clear()
    provider.video_calls.clear()
    _build(scene_env)
    # only shot 2's inputs changed: shots 0-1 untouched
    assert len(provider.keyframe_calls) == 1
    assert "edited keyframe" in provider.keyframe_calls[0]["prompt"]
    assert [c["image"] for c in provider.video_calls] == ["shot-02.png"]


def test_regenerate_shot_forces_regeneration(scene_env):
    _build(scene_env)
    _, _, provider, _, _ = scene_env
    provider.keyframe_calls.clear()
    provider.video_calls.clear()
    _build(scene_env, regenerate_shots={2})
    assert len(provider.keyframe_calls) == 1
    assert "keyframe 2" in provider.keyframe_calls[0]["prompt"]
    assert [c["image"] for c in provider.video_calls] == ["shot-02.png"]


def test_prompt_templates():
    spec = _spec()
    shot0, shot1 = spec.shots[0], spec.shots[1]
    p0 = anime_pipeline._keyframe_prompt(shot0, "serious tone")
    p1 = anime_pipeline._keyframe_prompt(shot1, "serious tone")
    assert "@Source" in p0 and "@Anchor" not in p0
    assert "@Anchor" in p1 and "@Previous" in p1
    assert "serious tone" in p0
    anim = anime_pipeline._animation_prompt(shot1)
    assert "5 seconds" in anim
    assert "motion 1" in anim


def test_keyframe_refs_chain(tmp_path):
    spec = _spec()
    assert anime_pipeline._keyframe_refs(spec.shots[0], tmp_path) == []
    refs1 = anime_pipeline._keyframe_refs(spec.shots[1], tmp_path)
    assert [t for t, _ in refs1] == ["Anchor"]
    refs2 = anime_pipeline._keyframe_refs(spec.shots[2], tmp_path)
    assert [t for t, _ in refs2] == ["Anchor", "Previous"]


# --- Assembly with real ffmpeg (order + duration) ----------------------------

def _color_clip(dest: Path, color: str, seconds: float = 1.0):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c={color}:s=640x360:d={seconds}:r=30",
         "-pix_fmt", "yuv420p", str(dest)],
        capture_output=True, check=True,
    )


def test_assemble_scene_orders_and_caps_duration(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not installed")
    clips = []
    for i, color in enumerate(("red", "green", "blue")):
        clip = tmp_path / f"shot-{i:02d}.mp4"
        _color_clip(clip, color)
        clips.append(clip)
    out = anime_pipeline.assemble_scene(clips, tmp_path)
    assert (tmp_path / "clips.txt").read_text().splitlines() == [
        f"file '{(tmp_path / 'clips' / f'norm-shot-{i:02d}.mp4').as_posix()}'"
        for i in range(3)
    ]
    duration = anime_pipeline.video_duration(out)
    assert 2.5 < duration <= 30

    # order check: first frame is red, last is blue
    def frame_at(t: float) -> tuple[int, int, int]:
        raw = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(out),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        ).stdout
        w, h = 1280, 720
        mid = (h // 2) * w * 3 + (w // 2) * 3
        return raw[mid], raw[mid + 1], raw[mid + 2]

    r, g, b = frame_at(0.2)
    assert r > 150 and g < 100 and b < 100
    r, g, b = frame_at(duration - 0.2)
    assert b > 150 and r < 100 and g < 100


def test_assemble_scene_rejects_overlong_output(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    clip = tmp_path / "shot-00.mp4"
    _color_clip(clip, "red", seconds=1.0)
    monkeypatch.setattr(anime_pipeline, "video_duration", lambda p: 31.0)
    with pytest.raises(SceneError, match="max is 30s"):
        anime_pipeline.assemble_scene([clip, clip, clip], tmp_path)


# --- scene provider resolution (Phase 5) ----------------------------------------


def test_get_scene_provider_resolves_runway():
    from entertainment_harness.video.gen.runway import RunwayProvider

    config = Config()
    config.video_gen.runway.api_key = "k"
    provider = anime_pipeline.get_scene_provider(config)
    assert isinstance(provider, RunwayProvider)


def test_get_scene_provider_rejects_incapable_provider():
    config = Config()
    config.anime.provider = "local"  # stills only — no image-gen/image-to-video
    with pytest.raises(SceneError, match="image-gen"):
        anime_pipeline.get_scene_provider(config)


def test_anime_provider_config_parsed():
    from entertainment_harness.config import parse_config

    assert parse_config({}).anime.provider == "runway"
    assert parse_config({"anime": {"provider": "acme"}}).anime.provider == "acme"
