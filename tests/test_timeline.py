"""Tests for the sparse timeline domain model and the deterministic
sequencer (animation-conductor initiative, Phase 1).

Pure domain tests: no images, no ffmpeg, no models. The renderer/ffmpeg
integration lives in tests/test_anime_render.py.
"""

from __future__ import annotations

import json

import pytest

from entertainment_harness.anime.sequencer import camera_at, state_at
from entertainment_harness.anime.timeline import (
    SCHEMA_VERSION,
    AssetEvent,
    AudioClip,
    CameraKey,
    Layer,
    MouthEvent,
    Timeline,
    TimelineError,
    load_timeline,
    save_timeline,
    validate_timeline,
)

ASSETS = {
    "base": "assets/base.png",
    "blink": "assets/blink.png",
    "small": "assets/mouth-small.png",
    "open": "assets/mouth-open.png",
    "pose": "assets/pose.png",
}


def _timeline(**overrides) -> Timeline:
    fields = dict(
        fps=24,
        width=160,
        height=90,
        duration_frames=120,
        assets=dict(ASSETS),
        layers=[Layer(id="char", asset="base")],
        camera=[],
        events=[],
        audio=[],
    )
    fields.update(overrides)
    return Timeline(**fields)


def _blink_events() -> list:
    return [
        AssetEvent(frame=36, target="char", asset="blink"),
        AssetEvent(frame=39, target="char", asset="base"),
    ]


def _mouth_event(**overrides) -> MouthEvent:
    fields = dict(
        start_frame=60, end_frame=71, target="char",
        states=["base", "small", "open"], period_frames=2,
    )
    fields.update(overrides)
    return MouthEvent(**fields)


# --- Serialization ------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    timeline = _timeline(
        camera=[CameraKey(frame=0), CameraKey(frame=119, x=0.6, scale=2.0)],
        events=[*_blink_events(), _mouth_event()],
        audio=[AudioClip(start_frame=48, path="audio/tone.wav", volume=0.8)],
    )
    path = tmp_path / "timeline.json"
    save_timeline(timeline, path)
    loaded = load_timeline(path)
    assert loaded.to_dict() == timeline.to_dict()
    assert loaded.duration_seconds == 5.0
    assert loaded.frame_seconds(48) == 2.0


def test_human_edited_file_revalidated_on_load(tmp_path):
    path = tmp_path / "timeline.json"
    save_timeline(_timeline(), path)
    payload = json.loads(path.read_text())
    payload["events"] = [
        {"frame": 200, "type": "asset", "target": "char", "asset": "blink"}
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(TimelineError, match="outside the valid range"):
        load_timeline(path)


def test_malformed_files_fail_before_rendering(tmp_path):
    path = tmp_path / "timeline.json"
    path.write_text("this is not json")
    with pytest.raises(TimelineError, match="Cannot read"):
        load_timeline(path)
    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(TimelineError, match="not a JSON object"):
        load_timeline(path)
    path.write_text(json.dumps({
        "fps": 24, "width": 160, "height": 90, "duration_frames": 120,
        "assets": ["not", "a", "map"],
    }))
    with pytest.raises(TimelineError, match="Malformed timeline"):
        load_timeline(path)
    with pytest.raises(TimelineError, match="Cannot read"):
        load_timeline(tmp_path / "missing.json")


# --- Validation ---------------------------------------------------------------


def test_validate_rejects_unsupported_version():
    with pytest.raises(TimelineError, match="Unsupported timeline version 2"):
        validate_timeline(_timeline(version=2))
    validate_timeline(_timeline(version=SCHEMA_VERSION))


def test_validate_rejects_bad_fps():
    with pytest.raises(TimelineError, match="fps must be > 0"):
        validate_timeline(_timeline(fps=0))
    with pytest.raises(TimelineError, match="fps must be > 0"):
        validate_timeline(_timeline(fps=-24))
    with pytest.raises(TimelineError, match="fps must be an integer"):
        validate_timeline(_timeline(fps=23.976))


def test_validate_rejects_bad_dimensions():
    with pytest.raises(TimelineError, match="width/height must be > 0"):
        validate_timeline(_timeline(width=0))
    with pytest.raises(TimelineError, match="width/height must be > 0"):
        validate_timeline(_timeline(height=-90))


def test_validate_rejects_bad_duration():
    with pytest.raises(TimelineError, match="duration_frames must be > 0"):
        validate_timeline(_timeline(duration_frames=0))


def test_validate_rejects_out_of_range_event_frames():
    for frame in (-1, 120):
        events = [AssetEvent(frame=frame, target="char", asset="blink")]
        with pytest.raises(TimelineError, match="outside the valid range 0..119"):
            validate_timeline(_timeline(events=events))


def test_validate_rejects_invalid_mouth_ranges():
    with pytest.raises(TimelineError, match="before it starts"):
        validate_timeline(
            _timeline(events=[_mouth_event(start_frame=70, end_frame=60)])
        )
    with pytest.raises(TimelineError, match="outside the valid range"):
        validate_timeline(_timeline(events=[_mouth_event(end_frame=120)]))
    with pytest.raises(TimelineError, match="period_frames must be"):
        validate_timeline(_timeline(events=[_mouth_event(period_frames=0)]))
    with pytest.raises(TimelineError, match="no states"):
        validate_timeline(_timeline(events=[_mouth_event(states=[])]))


def test_validate_rejects_unknown_asset_references():
    with pytest.raises(TimelineError, match="unknown asset 'nope'"):
        validate_timeline(_timeline(layers=[Layer(id="char", asset="nope")]))
    events = [AssetEvent(frame=10, target="char", asset="nope")]
    with pytest.raises(TimelineError, match="unknown asset 'nope'"):
        validate_timeline(_timeline(events=events))
    with pytest.raises(TimelineError, match="unknown asset 'nope'"):
        validate_timeline(_timeline(events=[_mouth_event(states=["base", "nope"])]))


def test_validate_rejects_unknown_targets():
    events = [AssetEvent(frame=10, target="ghost", asset="blink")]
    with pytest.raises(TimelineError, match="unknown layer 'ghost'"):
        validate_timeline(_timeline(events=events))
    with pytest.raises(TimelineError, match="unknown layer 'ghost'"):
        validate_timeline(_timeline(events=[_mouth_event(target="ghost")]))


def test_validate_rejects_duplicate_and_overlapping_events():
    dupes = [
        AssetEvent(frame=36, target="char", asset="blink"),
        AssetEvent(frame=36, target="char", asset="pose"),
    ]
    with pytest.raises(TimelineError, match="ambiguous"):
        validate_timeline(_timeline(events=dupes))
    overlap = [
        _mouth_event(start_frame=60, end_frame=80),
        _mouth_event(start_frame=80, end_frame=100),
    ]
    with pytest.raises(TimelineError, match="Overlapping mouth events"):
        validate_timeline(_timeline(events=overlap))


def test_validate_rejects_unordered_or_duplicate_camera_keyframes():
    unordered = [CameraKey(frame=50), CameraKey(frame=10)]
    with pytest.raises(TimelineError, match="ordered by frame"):
        validate_timeline(_timeline(camera=unordered))
    dupes = [CameraKey(frame=10), CameraKey(frame=10, scale=2.0)]
    with pytest.raises(TimelineError, match="ordered by frame"):
        validate_timeline(_timeline(camera=dupes))
    with pytest.raises(TimelineError, match="outside the valid range"):
        validate_timeline(_timeline(camera=[CameraKey(frame=120)]))
    with pytest.raises(TimelineError, match="scale must be > 0"):
        validate_timeline(_timeline(camera=[CameraKey(frame=0, scale=0.0)]))


def test_validate_rejects_invalid_audio():
    with pytest.raises(TimelineError, match="outside the valid range"):
        validate_timeline(
            _timeline(audio=[AudioClip(start_frame=120, path="a.wav")])
        )
    with pytest.raises(TimelineError, match="non-empty path"):
        validate_timeline(_timeline(audio=[AudioClip(start_frame=0, path="")]))
    with pytest.raises(TimelineError, match="volume must be >= 0"):
        validate_timeline(
            _timeline(audio=[AudioClip(start_frame=0, path="a.wav", volume=-1)])
        )


def test_validate_rejects_duplicate_layers_and_bad_placement():
    layers = [Layer(id="char", asset="base"), Layer(id="char", asset="blink")]
    with pytest.raises(TimelineError, match="Duplicate layer id"):
        validate_timeline(_timeline(layers=layers))
    with pytest.raises(TimelineError, match="normalized"):
        validate_timeline(_timeline(layers=[Layer(id="char", asset="base", x=1.5)]))
    with pytest.raises(TimelineError, match="scale must be > 0"):
        validate_timeline(
            _timeline(layers=[Layer(id="char", asset="base", scale=0.0)])
        )


def test_validate_rejects_unknown_event_type():
    data = _timeline().to_dict()
    data["events"] = [{"type": "explode", "frame": 10, "target": "char"}]
    with pytest.raises(TimelineError, match="Unknown event type"):
        Timeline.from_dict(data)


def test_empty_timeline_is_valid():
    validate_timeline(_timeline())  # no layers/camera/events/audio at all


# --- Sequencer: persistent state and exact boundaries --------------------------


def test_asset_event_applies_at_its_exact_frame():
    timeline = _timeline(events=_blink_events())
    assert state_at(timeline, 35).layers[0].asset == "base"
    assert state_at(timeline, 36).layers[0].asset == "blink"


def test_asset_event_is_persistent_until_the_next_event():
    timeline = _timeline(events=[
        AssetEvent(frame=40, target="char", asset="pose"),
    ])
    assert state_at(timeline, 0).layers[0].asset == "base"
    assert state_at(timeline, 39).layers[0].asset == "base"
    assert state_at(timeline, 40).layers[0].asset == "pose"
    assert state_at(timeline, 119).layers[0].asset == "pose"  # holds to the end


def test_blink_applies_only_to_the_intended_frames():
    timeline = _timeline(events=_blink_events())
    for frame in (*range(0, 36), *range(39, 120)):
        assert state_at(timeline, frame).layers[0].asset == "base", frame
    for frame in (36, 37, 38):
        assert state_at(timeline, frame).layers[0].asset == "blink", frame


def test_state_at_rejects_frames_outside_the_timeline():
    timeline = _timeline()
    for frame in (-1, 120, 121):
        with pytest.raises(TimelineError, match="outside the timeline"):
            state_at(timeline, frame)


def test_layers_keep_their_placement_in_state():
    timeline = _timeline(
        layers=[Layer(id="bg", asset="base", x=0.2, y=0.3, scale=1.0),
                Layer(id="char", asset="blink", x=0.6, y=0.7, scale=0.5)],
    )
    state = state_at(timeline, 0)
    assert [layer.id for layer in state.layers] == ["bg", "char"]
    char = state.layers[1]
    assert (char.x, char.y, char.scale) == (0.6, 0.7, 0.5)


# --- Sequencer: camera interpolation -------------------------------------------


def test_camera_interpolates_linearly_between_keyframes():
    timeline = _timeline(camera=[
        CameraKey(frame=0, x=0.25, y=0.25, scale=1.0),
        CameraKey(frame=100, x=0.75, y=0.75, scale=3.0),
    ])
    start = camera_at(timeline, 0)
    assert (start.x, start.y, start.scale) == (0.25, 0.25, 1.0)
    mid = camera_at(timeline, 50)
    assert mid.x == pytest.approx(0.5)
    assert mid.y == pytest.approx(0.5)
    assert mid.scale == pytest.approx(2.0)
    end = camera_at(timeline, 100)
    assert (end.x, end.y, end.scale) == (0.75, 0.75, 3.0)


def test_camera_holds_first_and_last_keyframe_beyond_their_frames():
    timeline = _timeline(camera=[
        CameraKey(frame=10, x=0.2, scale=1.5),
        CameraKey(frame=20, x=0.8, scale=2.5),
    ])
    assert camera_at(timeline, 0).x == 0.2
    assert camera_at(timeline, 9).x == 0.2
    assert camera_at(timeline, 119).scale == 2.5


def test_camera_defaults_to_static_full_frame():
    camera = camera_at(_timeline(), 42)
    assert (camera.x, camera.y, camera.scale) == (0.5, 0.5, 1.0)


# --- Sequencer: mouth cycling ---------------------------------------------------


def test_mouth_cycle_is_deterministic_and_exact():
    timeline = _timeline(events=[_mouth_event()])
    # offsets 0..11, period 2, states [base, small, open]:
    expected = [
        "base", "base", "small", "small", "open", "open",
        "base", "base", "small", "small", "open", "open",
    ]
    for i, asset in enumerate(expected):
        frame = 60 + i
        assert state_at(timeline, frame).layers[0].asset == asset, frame
    # outside the interval the persistent asset applies
    assert state_at(timeline, 59).layers[0].asset == "base"
    assert state_at(timeline, 72).layers[0].asset == "base"
    # determinism: the same frame always resolves identically
    assert state_at(timeline, 64) == state_at(timeline, 64)


def test_mouth_interval_overrides_persistent_asset_but_not_forever():
    timeline = _timeline(events=[
        _mouth_event(start_frame=60, end_frame=71, states=["open"],
                     period_frames=1),
        AssetEvent(frame=65, target="char", asset="pose"),
    ])
    assert state_at(timeline, 59).layers[0].asset == "base"   # pre-interval hold
    assert state_at(timeline, 64).layers[0].asset == "open"   # mouth wins
    assert state_at(timeline, 66).layers[0].asset == "open"   # mouth wins over swap
    assert state_at(timeline, 72).layers[0].asset == "pose"   # swap survives


def test_pose_swap_at_exact_frame():
    timeline = _timeline(events=[
        _mouth_event(),
        AssetEvent(frame=100, target="char", asset="pose"),
    ])
    assert state_at(timeline, 99).layers[0].asset == "base"
    assert state_at(timeline, 100).layers[0].asset == "pose"
