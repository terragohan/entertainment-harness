"""Video pipeline tests: script parsing, TTS caching, page assignment,
SRT timing, clip filters, and staged-skip orchestration. No network, Ollama,
or ffmpeg (assembly is monkeypatched).
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from PIL import Image

from entertainment_harness import db
from entertainment_harness.library import works
from entertainment_harness.config import Config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection
from entertainment_harness.video import pipeline
from entertainment_harness.video.script import (
    BEAT_PAUSE_S,
    SCRIPT_PROMPT,
    TRANSITION_PAUSE_S,
    Segment,
    VideoConfigError,
    VideoError,
    apply_pauses,
    generate_script,
    load_script,
    parse_script,
    save_script,
    segments_from_narration,
    split_long_segments,
)
from entertainment_harness.video.tts import render_narration, wav_duration
from entertainment_harness.video.visuals import _fill_gaps, parse_assignments
from entertainment_harness.video import assemble as assemble_mod
from entertainment_harness.video.assemble import (
    STRIP_BACKGROUND,
    STRIP_GUTTER_PX,
    _clip_filter,
    _gap_wav,
    _segment_pages,
    build_strip,
    pacing_stats,
    write_srt,
)

GB = 10**9


def _artifact_config() -> Config:
    """Config for artifact-path video builds: panel-first (the default video
    script path) is off, so stage 1 scripts from the recap artifact."""
    config = Config()
    config.video.panel_first = False
    return config


def _write_wav(path: Path, seconds: float = 0.5, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))


def _write_page(path: Path, size: tuple[int, int] = (100, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


# --- script parsing ---------------------------------------------------------


def test_parse_script_clean_json():
    segments = parse_script('[{"text": "Shin hunts.", "moment": "The hunt"}]')
    assert segments == [Segment(index=0, text="Shin hunts.", moment="The hunt")]


def test_parse_script_tolerates_fences_and_preamble():
    raw = 'Here is the script:\n```json\n[{"text": "A"}, {"text": "B"}]\n```\nDone.'
    segments = parse_script(raw)
    assert [s.text for s in segments] == ["A", "B"]
    assert [s.index for s in segments] == [0, 1]


def test_parse_script_rejects_garbage():
    with pytest.raises(VideoError):
        parse_script("I cannot help with that.")
    with pytest.raises(VideoError):
        parse_script('[{"moment": "no text field"}]')
    with pytest.raises(VideoError):
        parse_script("[]")


def test_parse_script_salvages_truncated_array():
    raw = '[{"text": "First beat.", "moment": "A"}, {"text": "Second beat.", "mo'
    segments = parse_script(raw)
    assert [s.text for s in segments] == ["First beat."]  # complete objects kept
    with pytest.raises(VideoError):
        parse_script(raw, salvage=False)  # callers can opt for a retry instead


def test_generate_script_retries_flaky_endpoint():
    class FlakyAdapter:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.calls = 0

        def generate(self, model, prompt):
            self.calls += 1
            return self.outputs.pop(0)

    good = '[{"text": "A.", "moment": "x"}]'
    flaky = FlakyAdapter(['not json at all', good])
    segments = generate_script(flaky, "m", "recap", "T", 1.0)
    assert [s.text for s in segments] == ["A."]
    assert flaky.calls == 2

    dead = FlakyAdapter(['garbage'] * 3)
    with pytest.raises(VideoError, match="after 3 attempts"):
        generate_script(dead, "m", "recap", "T", 1.0)
    assert dead.calls == 3


def test_generate_script_settles_for_salvage_on_final_attempt():
    class TruncatingAdapter:
        calls = 0

        def generate(self, model, prompt):
            self.calls += 1
            return '[{"text": "Only beat."}, {"text": "Lost to trunc'

    adapter = TruncatingAdapter()
    segments = generate_script(adapter, "m", "recap", "T", 1.0, max_attempts=2)
    assert [s.text for s in segments] == ["Only beat."]
    assert adapter.calls == 2  # retried once for completeness, then salvaged


def test_script_roundtrip(tmp_path):
    segments = [Segment(index=0, text="A", moment="m", pages=[3], motion="pan_down",
                        duration_s=4.2)]
    path = tmp_path / "script.json"
    save_script(path, segments, "T", 1.0, "model-x (Q8_0)")
    loaded, model = load_script(path)
    assert loaded == segments
    assert model == "model-x (Q8_0)"


def test_script_roundtrip_old_format_without_pause(tmp_path):
    """Scripts saved before pause_after_s existed still load (default 0)."""
    path = tmp_path / "script.json"
    path.write_text(
        '{"chapter": 1, "title": "T", "model": "m",'
        ' "segments": [{"index": 0, "text": "A", "moment": "x",'
        ' "pages": [2], "motion": "zoom_in", "duration_s": 3.0}]}'
    )
    loaded, _ = load_script(path)
    assert loaded == [Segment(index=0, text="A", moment="x", pages=[2],
                              motion="zoom_in", duration_s=3.0,
                              pause_after_s=0.0)]


# --- script pacing ------------------------------------------------------------


def test_script_prompt_asks_for_short_natural_beats():
    assert "25 to 40" in SCRIPT_PROMPT
    assert "ONE spoken sentence" in SCRIPT_PROMPT
    assert "450-600 words" in SCRIPT_PROMPT
    # the old long-chunk guidance is gone
    assert "at least 25 words" not in SCRIPT_PROMPT
    assert "14 to 20" not in SCRIPT_PROMPT


def test_split_long_segments_splits_at_sentences():
    text = " ".join(["alpha"] * 20) + ". " + " ".join(["omega"] * 20) + "."
    segments = split_long_segments([Segment(index=0, text=text, moment="M")])
    assert [s.text for s in segments] == [
        " ".join(["alpha"] * 20) + ".",
        " ".join(["omega"] * 20) + ".",
    ]
    assert all(s.moment == "M" for s in segments)  # label copied to pieces
    assert [s.index for s in segments] == [0, 1]  # re-indexed


def test_split_long_segments_keeps_single_long_sentence():
    text = " ".join(["word"] * 40)  # no sentence boundary to split at
    segments = split_long_segments([Segment(index=0, text=text)])
    assert [s.text for s in segments] == [text]


def test_split_long_segments_leaves_short_segments_alone():
    segments = split_long_segments([Segment(index=0, text="Short one.")])
    assert segments == [Segment(index=0, text="Short one.")]


def test_apply_pauses_short_beat_longer_on_scene_change():
    segments = [
        Segment(index=0, text="a", moment="X"),
        Segment(index=1, text="b", moment="X"),
        Segment(index=2, text="c", moment="Y"),
    ]
    apply_pauses(segments)
    assert segments[0].pause_after_s == BEAT_PAUSE_S  # same scene continues
    assert segments[1].pause_after_s == TRANSITION_PAUSE_S  # moment changes
    assert segments[2].pause_after_s == 0.0  # nothing follows the last beat


def test_apply_pauses_unlabeled_segments_get_beat_pause():
    # narration-sourced segments carry no moment labels
    segments = [Segment(index=0, text="a"), Segment(index=1, text="b")]
    apply_pauses(segments)
    assert segments[0].pause_after_s == BEAT_PAUSE_S
    assert segments[1].pause_after_s == 0.0


# --- page assignment --------------------------------------------------------


def test_parse_assignments_filters_to_chunk_pages():
    raw = 'Sure! {"0": [3, 99], "2": [7], "x": [1], "4": "nope"}'
    result = parse_assignments(raw, valid_pages={1, 2, 3, 4, 5, 6, 7})
    assert result == {0: [3], 2: [7]}


def test_fill_gaps_inherits_neighbor_pages():
    segments = [
        Segment(index=0, text="a", pages=[5]),
        Segment(index=1, text="b"),
        Segment(index=2, text="c", pages=[9, 10]),
        Segment(index=3, text="d"),
    ]
    _fill_gaps(segments, page_count=12)
    assert segments[1].pages == [5]  # previous neighbor wins
    assert segments[3].pages == [9]


def test_assign_pages_chunks_by_six_and_demands_exact_moments(tmp_path):
    """The Phase-3 gate fix: small contact sheets (6 pages) and a prompt
    requiring the EXACT narrated moment — thematic matches must be omitted."""
    from entertainment_harness.video.visuals import CHUNK_SIZE, assign_pages

    assert CHUNK_SIZE == 6
    page_paths = [tmp_path / f"page-{i:03d}.jpg" for i in range(1, 14)]
    for p in page_paths:
        _write_page(p)
    prompts = []

    class Adapter:
        name = "fake"

        def generate(self, model, prompt, images=None):
            prompts.append(prompt)
            return "{}"

    segments = [Segment(index=0, text="Shin fights the boar.", moment="fight")]
    assign_pages(Adapter(), "m", segments, page_paths, tmp_path, "Kenja", 1.0)
    assert len(prompts) == 3  # 13 pages -> 6 + 6 + 1 sheets
    assert "pages 1-6" in prompts[0] and "pages 13-13" in prompts[2]
    assert "EXACT moment" in prompts[0]
    assert "thematically related" in prompts[0]
    assert "when in doubt, omit" in prompts[0]
    assert segments[0].pages == [1]  # no matches anywhere -> _fill_gaps


# --- tts --------------------------------------------------------------------


class FakeEngine:
    name = "fake"
    default_voice = "fake-voice"

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def synthesize(self, text: str, voice: str, dest: Path) -> None:
        self.synthesized.append(text)
        _write_wav(dest, seconds=0.5)


def test_render_narration_stamps_durations_and_caches(tmp_path):
    segments = [Segment(index=0, text="one"), Segment(index=1, text="two")]
    engine = FakeEngine()
    render_narration(segments, engine, "fake-voice", tmp_path, log=lambda m: None)
    assert engine.synthesized == ["one", "two"]
    assert all(s.duration_s == pytest.approx(0.5) for s in segments)
    assert wav_duration(tmp_path / "seg-00.wav") == pytest.approx(0.5)

    segments[0].duration_s = 0.0
    render_narration(segments, engine, "fake-voice", tmp_path, log=lambda m: None)
    assert engine.synthesized == ["one", "two"]  # no re-synthesis
    assert segments[0].duration_s == pytest.approx(0.5)  # re-stamped from file


# --- assembly helpers -------------------------------------------------------


def test_write_srt_cumulative_timing(tmp_path):
    segments = [
        Segment(index=0, text="First line.", duration_s=2.0),
        Segment(index=1, text="Second line.", duration_s=3.5),
    ]
    dest = tmp_path / "subs.srt"
    write_srt(segments, dest)
    content = dest.read_text()
    assert "00:00:00,000 --> 00:00:02,000\nFirst line." in content
    assert "00:00:02,000 --> 00:00:05,500\nSecond line." in content


def test_write_srt_pauses_become_gaps_between_cues(tmp_path):
    """The cue covers speech only; the pause is a gap, so subtitles don't
    linger over silence."""
    segments = [
        Segment(index=0, text="First line.", duration_s=2.0, pause_after_s=0.3),
        Segment(index=1, text="Second line.", duration_s=3.0),
    ]
    dest = tmp_path / "subs.srt"
    write_srt(segments, dest)
    content = dest.read_text()
    assert "00:00:00,000 --> 00:00:02,000\nFirst line." in content
    assert "00:00:02,300 --> 00:00:05,300\nSecond line." in content


def test_gap_wav_matches_segment_format(tmp_path):
    template = tmp_path / "seg-00.wav"
    _write_wav(template, seconds=1.0, rate=8000)
    gap = _gap_wav(tmp_path, 0.15, template)
    assert wav_duration(gap) == pytest.approx(0.15, abs=0.01)
    with wave.open(str(gap), "rb") as wav:
        assert wav.getframerate() == 8000
    assert _gap_wav(tmp_path, 0.15, template) == gap  # cached on second call



def test_clip_filter_pan_for_tall_page(tmp_path):
    page = tmp_path / "tall.jpg"
    _write_page(page, size=(100, 400))
    filtergraph, frames = _clip_filter(page, "pan_down", 4.0, 1920, 1080)
    assert "crop=1920:1080" in filtergraph
    assert frames == 120


def test_clip_filter_zoom_when_page_not_tall(tmp_path):
    page = tmp_path / "wide.jpg"
    _write_page(page, size=(400, 200))
    filtergraph, _ = _clip_filter(page, "pan_down", 4.0, 1920, 1080)
    assert "zoompan" in filtergraph  # falls back to zoom


def test_segment_pages_show_all_when_pace_allows():
    seg = Segment(index=0, text="x", pages=[5, 3, 3, 9], duration_s=30.0)
    assert _segment_pages(seg) == [3, 5, 9]  # dedup + sort


def test_segment_pages_cap_spans_the_range():
    # 10s -> at most 4 pages, evenly spaced across all 9 assignments
    # (not just the first 4, which would camp on the early chapter)
    seg = Segment(index=0, text="x", pages=[4, 13, 25, 26, 27, 28, 37, 38, 49],
                  duration_s=10.0)
    assert _segment_pages(seg) == [4, 26, 28, 49]


def test_segment_pages_single_page_when_short():
    seg = Segment(index=0, text="x", pages=[8, 20, 40], duration_s=2.0)
    assert _segment_pages(seg) == [8]
    seg2 = Segment(index=1, text="x", pages=[], duration_s=5.0)
    assert _segment_pages(seg2) == [1]  # fallback


def test_segment_pages_pause_extends_reading_time():
    # 4.9s of speech fits one page; with the pause the slot reaches 5.2s -> two
    seg = Segment(index=0, text="x", pages=[3, 5], duration_s=4.9,
                  pause_after_s=0.3)
    assert _segment_pages(seg) == [3, 5]
    same_without_pause = Segment(index=1, text="x", pages=[3, 5], duration_s=4.9)
    assert _segment_pages(same_without_pause) == [3]


def test_pacing_stats_counts_visual_changes():
    segments = [
        Segment(index=0, text="one two three", duration_s=5.0,
                pause_after_s=0.3, pages=[1, 2]),
        Segment(index=1, text="four five", duration_s=6.0, pages=[3]),
    ]
    stats = pacing_stats(segments)
    assert stats["segments"] == 2
    assert stats["words"] == 5
    assert stats["speech_s"] == 11.0
    assert stats["pauses_s"] == 0.3
    assert stats["total_s"] == 11.3
    assert stats["avg_seg_s"] == 5.5
    assert stats["median_seg_s"] == 5.5
    assert stats["max_seg_s"] == 6.0
    assert stats["visuals_shown"] == 3  # [1, 2] then [3]
    assert stats["visual_cuts"] == 2
    assert stats["avg_s_per_visual"] == pytest.approx(3.77, abs=0.01)


# --- scroll mode: strips + assembly -------------------------------------------


def _write_color_page(path: Path, size: tuple[int, int], color: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_build_strip_stacks_pages_in_order_with_gutter(tmp_path):
    p1 = tmp_path / "p1.png"
    p2 = tmp_path / "p2.png"
    _write_color_page(p1, (100, 200), (255, 0, 0))
    _write_color_page(p2, (60, 100), (0, 255, 0))
    dest = tmp_path / "strip.png"
    assert build_strip([p1, p2], dest) == dest
    with Image.open(dest) as strip:
        assert strip.width == 100  # the widest page wins
        assert strip.height == 200 + STRIP_GUTTER_PX + 100
        assert strip.getpixel((50, 100)) == (255, 0, 0)  # page 1 on top
        # neutral gutter between the pages
        assert (strip.getpixel((50, 200 + STRIP_GUTTER_PX // 2))
                == STRIP_BACKGROUND)
        # page 2 below the gutter, centered: (100 - 60) // 2 = 20 px inset
        assert (strip.getpixel((50, 200 + STRIP_GUTTER_PX + 50))
                == (0, 255, 0))
        # the narrower page is padded to strip width
        assert (strip.getpixel((5, 200 + STRIP_GUTTER_PX + 50))
                == STRIP_BACKGROUND)


def test_build_strip_reuses_cache(tmp_path):
    p1 = tmp_path / "p1.png"
    p2 = tmp_path / "p2.png"
    _write_color_page(p1, (100, 200), (255, 0, 0))
    _write_color_page(p2, (100, 100), (0, 255, 0))
    dest = tmp_path / "strip.png"
    build_strip([p1, p2], dest)
    before = dest.stat().st_mtime_ns
    p1.write_bytes(b"garbage")  # would explode if the pages were reopened
    assert build_strip([p1, p2], dest) == dest
    assert dest.stat().st_mtime_ns == before


def _capture_clips(monkeypatch):
    """Stub out ffmpeg: record build_clip calls, fake the mux."""
    calls: list[dict] = []

    def fake_build_clip(page, duration, motion, resolution, dest):
        calls.append({"page": page, "duration": duration,
                      "motion": motion, "dest": dest})
        dest.write_bytes(b"clip")

    def fake_mux(segments, clips, workdir, log=lambda m: None):
        out = workdir / "out.mp4"
        out.write_bytes(b"out")
        return out, sum(s.duration_s + s.pause_after_s for s in segments)

    monkeypatch.setattr(assemble_mod, "build_clip", fake_build_clip)
    monkeypatch.setattr(assemble_mod, "mux_clips", fake_mux)
    return calls


def _three_pages(tmp_path) -> list[Path]:
    pages_dir = tmp_path / "pages"
    paths = [pages_dir / f"page-{i:03d}.png" for i in range(1, 4)]
    for p in paths:
        _write_page(p, size=(120, 300))
    return paths


def test_assemble_scroll_multi_page_segment_renders_one_strip_clip(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[1, 2, 3], motion="scroll",
                  duration_s=9.0)
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert len(calls) == 1  # one clip for the whole segment
    call = calls[0]
    assert call["dest"].name == "clip-00-0.mp4"
    assert call["motion"] == "pan_down"  # existing filter reused over the strip
    assert call["duration"] == pytest.approx(9.0)  # the segment's full slot
    strip = call["page"]
    assert strip == workdir / "clips" / "strip-00.png"
    assert strip.exists()
    with Image.open(strip) as img:
        assert img.height == 300 * 3 + STRIP_GUTTER_PX * 2


def test_assemble_scroll_single_page_segment_uses_normal_path(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[2], motion="scroll", duration_s=2.0)
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert len(calls) == 1
    assert calls[0]["page"] == page_paths[1]  # the page itself, not a strip
    assert calls[0]["motion"] == "pan_down"  # degrades to today's pan behavior
    assert not (workdir / "clips" / "strip-00.png").exists()


def test_assemble_kenburns_renders_per_page_clips(tmp_path, monkeypatch):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[1, 2], motion="zoom_in",
                  duration_s=8.0)
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert [c["dest"].name for c in calls] == ["clip-00-0.mp4", "clip-00-1.mp4"]
    assert all(c["motion"] == "zoom_in" for c in calls)
    assert all(c["duration"] == pytest.approx(4.0) for c in calls)
    assert not list((workdir / "clips").glob("strip-*.png"))


def test_clip_filter_static_is_an_unmoving_cover_crop(tmp_path):
    page = tmp_path / "p.png"
    _write_page(page, size=(100, 200))
    graph, frames = _clip_filter(page, "static", 3.0, 360, 640)
    assert "zoompan" not in graph
    assert graph == (
        "fps=30,scale=360:640:force_original_aspect_ratio=increase,"
        "crop=360:640"
    )
    assert frames == 90


def test_build_xfade_clip_pads_holds_so_chain_ends_on_the_slot(
    tmp_path, monkeypatch
):
    cmds: list[list[str]] = []
    monkeypatch.setattr(assemble_mod, "_run", cmds.append)
    pages = _three_pages(tmp_path)
    dest = tmp_path / "out.mp4"
    assemble_mod.build_xfade_clip(pages, 9.0, (360, 640), dest)
    (cmd,) = cmds
    # 3 pages, 9s slot, 0.5s fades: per-page hold = (9 + 2*0.5)/3 = 3.333s;
    # xfade offsets at 2.833s and 5.667s -> total = 3*3.333 - 2*0.5 = 9.0s.
    assert cmd.count("-loop") == 3
    assert cmd[cmd.index("-t") + 1] == "3.333"
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("xfade=transition=fade:duration=0.500") == 2
    assert "offset=2.833" in graph and "offset=5.667" in graph
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    assert cmd[cmd.index("-frames:v") + 1] == "270"


def test_build_xfade_clip_caps_fade_so_it_never_swallows_a_page(
    tmp_path, monkeypatch
):
    cmds: list[list[str]] = []
    monkeypatch.setattr(assemble_mod, "_run", cmds.append)
    pages = _three_pages(tmp_path)
    assemble_mod.build_xfade_clip(pages, 2.0, (360, 640), tmp_path / "o.mp4")
    graph = cmds[0][cmds[0].index("-filter_complex") + 1]
    # slot/(2n) = 0.333s < the 0.5s default
    assert "duration=0.333" in graph


def test_build_xfade_clip_single_page_is_a_static_clip(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_build_clip(page, duration, motion, resolution, dest):
        calls.append({"page": page, "motion": motion, "duration": duration})

    monkeypatch.setattr(assemble_mod, "build_clip", fake_build_clip)
    (page,) = _three_pages(tmp_path)[:1]
    assemble_mod.build_xfade_clip([page], 4.0, (360, 640), tmp_path / "o.mp4")
    assert calls == [{"page": page, "motion": "static", "duration": 4.0}]


def test_assemble_slideshow_segment_renders_one_xfade_clip(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)
    xfades: list[dict] = []

    def fake_xfade(pages, slot, resolution, dest, fade_s=0.5):
        xfades.append({"pages": pages, "slot": slot, "dest": dest})
        dest.write_bytes(b"clip")

    monkeypatch.setattr(assemble_mod, "build_xfade_clip", fake_xfade)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[1, 2, 3], motion="slideshow",
                  duration_s=9.0)
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert calls == []  # no per-page Ken Burns clips
    assert len(xfades) == 1
    assert xfades[0]["pages"] == page_paths
    assert xfades[0]["slot"] == pytest.approx(9.0)
    assert xfades[0]["dest"].name == "clip-00-0.mp4"


def test_assemble_panels_hold_anchors_render_panel_crops(tmp_path, monkeypatch):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)  # 120x300 pages
    seg = Segment(
        index=0, text="x", pages=[1, 2], motion="panels", duration_s=8.0,
        regions=[
            {"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]},
            {"page": 2, "kind": "hold", "box": [0.25, 0.5, 1.0, 1.0]},
        ],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert [c["dest"].name for c in calls] == ["clip-00-0.mp4", "clip-00-1.mp4"]
    assert all(c["motion"] == "zoom_in" for c in calls)
    assert all(c["duration"] == pytest.approx(4.0) for c in calls)  # shares
    crop0, crop1 = (workdir / "clips" / f"panel-00-{k}.png" for k in (0, 1))
    assert [c["page"] for c in calls] == [crop0, crop1]
    with Image.open(crop0) as img:
        assert img.size == (60, 150)  # 0.5 x 0.5 of 120x300
    with Image.open(crop1) as img:
        assert img.size == (90, 150)  # 0.75 x 0.5 of 120x300


def test_assemble_panels_pan_anchor_uses_the_full_page(tmp_path, monkeypatch):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[2], motion="panels", duration_s=4.0,
        regions=[{"page": 2, "kind": "pan"}],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert len(calls) == 1
    assert calls[0]["page"] == page_paths[1]  # no crop built for a pan
    assert not list((workdir / "clips").glob("panel-*.png"))


def test_assemble_panels_ungrounded_segment_falls_back_to_pages(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[1, 2], motion="panels",
                  duration_s=8.0)
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert [c["page"] for c in calls] == page_paths[:2]
    # the fallback filter for "panels" is the plain zoom (no pan/zoompan
    # special case), same visual as kenburns' zoom_in
    assert all(c["motion"] == "panels" for c in calls)


def test_panel_crop_degenerate_box_returns_the_page(tmp_path):
    (page,) = _three_pages(tmp_path)[:1]
    dest = tmp_path / "crop.png"
    assert assemble_mod._panel_crop(page, [0.0, 0.0, 0.05, 0.5], dest) == page
    assert not dest.exists()  # no sliver crop was written



# --- pipeline orchestration --------------------------------------------------


class FakeAdapter:
    def __init__(self, output: str, model_name: str) -> None:
        self.output = output
        self.model_name = model_name
        self.calls = 0

    def supports(self, model: str) -> ModelInfo:
        return ModelInfo(model, "fake", 4.0, "Q8_0", 3 * GB)

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def list_available(self) -> list[ModelInfo]:
        return [self.supports(self.model_name)]

    def generate(self, model, prompt, images=None) -> str:
        self.calls += 1
        return self.output


class FakeClient:
    def download_pages(self, chapter_id: str, dest_dir: Path) -> list[Path]:
        for i in range(1, 4):
            _write_page(dest_dir / f"page-{i:03d}.jpg")
        return sorted(dest_dir.iterdir())


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'Ch 1', 'pt-br', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('ch-1', 'Shin does things.', 'now', 'qwen3-vl')"
    )
    conn.commit()

    text = FakeAdapter(
        '[{"text": "Segment one.", "moment": "A"},'
        ' {"text": "Segment two.", "moment": "B"}]',
        "fake-text:4b",
    )
    vision = FakeAdapter("{}", "fake-vision:8b")
    info = ModelInfo("fake:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        pipeline, "get_text_model", lambda c, p: Selection(adapter=text, info=info)
    )
    monkeypatch.setattr(
        pipeline, "get_vision_model", lambda c, p: Selection(adapter=vision, info=info)
    )
    engine = FakeEngine()
    monkeypatch.setattr(pipeline, "get_engine", lambda name, config=None: engine)

    assembled: list[int] = []

    def fake_assemble(segments, page_paths, workdir, resolution, log,
                      frame_animator=None, sequence_interp_fps=30,
                      sequence_critic=None):
        assembled.append(len(segments))
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", fake_assemble)

    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()
    chapter = conn.execute("SELECT * FROM chapters WHERE id = 'ch-1'").fetchone()
    profile = HardwareProfile("fake", 24 * GB, 16 * GB, "cpu")
    yield conn, series, chapter, profile, text, vision, engine, assembled
    conn.close()


def test_build_video_end_to_end_and_cached_second_run(harness):
    conn, series, chapter, profile, text, vision, engine, assembled = harness
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out.name == "out.mp4" and out.exists()
    assert text.calls == 1  # script generated
    assert vision.calls >= 1  # pages assigned
    assert engine.synthesized == ["Segment one.", "Segment two."]
    assert assembled == [2]
    row = conn.execute("SELECT * FROM videos WHERE series_id = 's1'").fetchone()
    assert row is not None
    assert row["from_chapter"] == 1.0 and row["to_chapter"] == 1.0
    assert row["tts_engine"] == "fake"
    assert row["duration_s"] == pytest.approx(1.0)

    # second run: every stage skips, nothing re-generated, no duplicate row
    out2 = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out2 == out
    assert text.calls == 1
    assert vision.calls == 1
    assert engine.synthesized == ["Segment one.", "Segment two."]
    assert assembled == [2]
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM videos WHERE series_id = 's1'"
    ).fetchone()["n"]
    assert count == 1


def test_build_video_requires_recap(harness):
    conn, series, chapter, profile, *_ = harness
    conn.execute("DELETE FROM recaps WHERE chapter_id = 'ch-1'")
    conn.commit()
    with pytest.raises(VideoError, match="no recap"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile,
            client=FakeClient(), log=lambda m: None,
        )


def test_build_video_slideshow_mode_stamps_motion(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    seen: dict = {}

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        seen["motions"] = [s.motion for s in segments]
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None, video_mode="slideshow",
    )
    assert seen["motions"] == ["slideshow", "slideshow"]
    import json
    state = json.loads(
        (works.video_dir_for_kind("s1", "ch-1", "recap") / "render_state.json")
        .read_text()
    )
    assert state["mode"] == "slideshow"


def _make_book(conn):
    conn.execute("UPDATE series SET kind = 'book' WHERE id = 's1'")
    conn.commit()
    return conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()


def _recording_assemble(monkeypatch, seen: dict):
    def recording(segments, page_paths, workdir, resolution, log):
        seen["paths"] = list(page_paths)
        seen["pages"] = [list(s.pages) for s in segments]
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording)


def test_build_video_book_defaults_to_cards(harness, monkeypatch):
    conn, series, chapter, profile, text, vision, engine, _ = harness
    series = _make_book(conn)
    seen: dict = {}
    _recording_assemble(monkeypatch, seen)
    logs: list[str] = []
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile, log=logs.append,
    )
    assert any("cards" in m for m in logs)  # the default is announced
    workdir = works.video_dir_for_kind("s1", "ch-1", "recap")
    cards = sorted((workdir / "cards").glob("page-*.png"))
    assert len(cards) == 2  # one per segment
    assert seen["paths"] == cards  # assembly renders the cards, not pages
    assert seen["pages"] == [[1], [2]]
    assert vision.calls == 0  # no page picking without pages


def test_build_video_manga_cards_mode_skips_page_picking(harness, monkeypatch):
    conn, series, chapter, profile, text, vision, engine, _ = harness
    seen: dict = {}
    _recording_assemble(monkeypatch, seen)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None, video_mode="cards",
    )
    workdir = works.video_dir_for_kind("s1", "ch-1", "recap")
    assert len(list((workdir / "cards").glob("page-*.png"))) == 2
    assert vision.calls == 0  # cards replace the vision page-assignment stage


def test_build_video_book_explicit_non_cards_mode_errors(harness):
    conn, series, chapter, profile, *_ = harness
    series = _make_book(conn)
    with pytest.raises(VideoError, match="video mode 'cards'"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile,
            log=lambda m: None, video_mode="scroll",
        )


def test_build_video_book_configured_non_cards_mode_errors(harness):
    conn, series, chapter, profile, *_ = harness
    series = _make_book(conn)
    config = _artifact_config()
    config.video.mode = "scroll"
    with pytest.raises(VideoError, match="video mode 'cards'"):
        pipeline.build_video(
            conn, series, chapter, config, profile, log=lambda m: None,
        )


def test_clear_downstream_narration_reset_clears_cards(tmp_path):
    workdir = tmp_path / "w"
    (workdir / "cards").mkdir(parents=True)
    (workdir / "cards" / "page-001.png").write_bytes(b"card")
    (workdir / "clips").mkdir()
    (workdir / "clips" / "clip-00-0.mp4").write_bytes(b"clip")
    (workdir / "clips" / "panel-00-0.png").write_bytes(b"crop")
    pipeline.clear_downstream(workdir)  # clip-level reset keeps cards
    assert (workdir / "cards" / "page-001.png").exists()
    assert not (workdir / "clips" / "clip-00-0.mp4").exists()
    assert not (workdir / "clips" / "panel-00-0.png").exists()  # stale crops go
    pipeline.clear_downstream(workdir, narration=True)  # script changed
    assert not (workdir / "cards").exists()


def test_build_video_panels_mode_grounds_and_stamps(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    grounded: list[int] = []

    def fake_ground(adapter, model, segments, page_paths, series_id,
                    chapter_id, title, chapter_num, log=lambda m: None):
        grounded.append(len(segments))
        for s in segments:
            s.regions = [{"page": 1, "kind": "hold", "box": [0, 0, 0.5, 0.5]}]

    monkeypatch.setattr(pipeline, "ground_segments", fake_ground)
    seen: dict = {}

    def capture(segments, page_paths, workdir, resolution, log):
        seen["motions"] = [s.motion for s in segments]
        seen["regions"] = [s.regions for s in segments]
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", capture)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None, video_mode="panels",
    )
    assert grounded == [2]  # grounding ran for both segments
    assert seen["motions"] == ["panels", "panels"]
    assert all(seen["regions"])  # anchors survived to assembly
    import json
    state = json.loads(
        (works.video_dir_for_kind("s1", "ch-1", "recap") / "render_state.json")
        .read_text()
    )
    assert state["mode"] == "panels" and state["grounded"] is True


class FakeAnimatedGen:
    name = "fake-gen"
    animated = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_segment(self, image, segment, duration, workdir):
        self.calls.append({"image": image, "index": segment.index,
                           "duration": duration})
        clip = workdir / f"gen-{segment.index:02d}.mp4"
        clip.write_bytes(b"clip")
        return clip


def test_build_video_motion_mode_uses_animated_provider(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    gen = FakeAnimatedGen()
    monkeypatch.setattr(pipeline, "get_video_gen_provider",
                        lambda name, config: gen)
    monkeypatch.setattr(pipeline, "video_duration", lambda path: 1.5)
    muxed: dict = {}

    def fake_mux(segments, clips, workdir, log=lambda m: None):
        muxed["durations"] = [s.duration_s for s in segments]
        muxed["clips"] = list(clips)
        out = workdir / "out.mp4"
        out.write_bytes(b"out")
        return out, sum(s.duration_s + s.pause_after_s for s in segments)

    monkeypatch.setattr(pipeline, "mux_clips", fake_mux)
    config = _artifact_config()
    config.video_gen.provider = "fake-gen"
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None, video_mode="motion",
    )
    assert [c["index"] for c in gen.calls] == [0, 1]
    # each segment is generated from its first assigned page
    manga_dir = works.source_dir("s1", "ch-1")
    assert all(c["image"].parent == manga_dir for c in gen.calls)
    assert muxed["durations"] == [1.5, 1.5]  # re-stamped from the real clips
    assert len(muxed["clips"]) == 2
    import json
    state = json.loads(
        (works.video_dir_for_kind("s1", "ch-1", "recap") / "render_state.json")
        .read_text()
    )
    assert state["mode"] == "motion" and state["video_gen"] == "fake-gen"
    meta = works.read_video_metadata("s1", "ch-1", "recap")
    assert meta.video_gen_provider == "fake-gen"


def test_build_video_motion_mode_stills_provider_falls_back(harness):
    conn, series, chapter, profile, text, vision, engine, assembled = harness
    logs: list[str] = []
    # [video_gen] provider defaults to "local" (stills, not animated)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=logs.append, video_mode="motion",
    )
    assert any("makes stills" in m for m in logs)
    assert assembled == [2]  # the normal assembly ran instead


class FakeFrameAnimator:
    name = "fake-frames"
    animated = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_frames(self, image, segment, duration, workdir,
                        log=lambda m: None):
        self.calls.append({"image": image, "index": segment.index,
                           "duration": duration})
        return [image, image, image]  # frame paths stand in for generations


def test_assemble_animate_stitches_generated_frames(tmp_path, monkeypatch):
    _capture_clips(monkeypatch)
    xfades: list[dict] = []

    def fake_xfade(pages, slot, resolution, dest, fade_s=0.5):
        xfades.append({"pages": pages, "slot": slot, "dest": dest})
        dest.write_bytes(b"clip")

    monkeypatch.setattr(assemble_mod, "build_xfade_clip", fake_xfade)
    animator = FakeFrameAnimator()
    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1, 2], motion="animate", duration_s=8.0,
        regions=[
            {"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]},
            {"page": 2, "kind": "pan"},
        ],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=animator)
    assert len(animator.calls) == 2
    # hold anchor animates its panel crop, pan anchor the full page
    assert animator.calls[0]["image"] == workdir / "clips" / "panel-00-0.png"
    assert animator.calls[1]["image"] == page_paths[1]
    assert all(c["duration"] == pytest.approx(4.0) for c in animator.calls)
    assert [x["dest"].name for x in xfades] == ["clip-00-0.mp4", "clip-00-1.mp4"]
    assert all(x["slot"] == pytest.approx(4.0) for x in xfades)
    assert all(len(x["pages"]) == 3 for x in xfades)  # the generated frames


def test_assemble_animate_stills_animator_falls_back_to_panels(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)

    class StillsAnimator(FakeFrameAnimator):
        animated = False

    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1], motion="animate", duration_s=4.0,
        regions=[{"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]}],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=StillsAnimator())
    assert len(calls) == 1  # the panels-mode Ken Burns crop clip
    assert calls[0]["motion"] == "zoom_in"
    assert calls[0]["page"] == workdir / "clips" / "panel-00-0.png"


def test_build_video_animate_mode_grounds_and_passes_animator(
    harness, monkeypatch
):
    conn, series, chapter, profile, *_ = harness

    def fake_ground(adapter, model, segments, page_paths, series_id,
                    chapter_id, title, chapter_num, log=lambda m: None):
        for s in segments:
            s.regions = [{"page": 1, "kind": "hold", "box": [0, 0, 0.5, 0.5]}]

    monkeypatch.setattr(pipeline, "ground_segments", fake_ground)
    animator = FakeFrameAnimator()
    monkeypatch.setattr(pipeline, "get_frame_animator",
                        lambda name, config: animator)
    seen: dict = {}

    def capture(segments, page_paths, workdir, resolution, log,
                frame_animator=None):
        seen["motions"] = [s.motion for s in segments]
        seen["animator"] = frame_animator
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", capture)
    config = _artifact_config()
    config.frames.provider = "fake-frames"
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None, video_mode="animate",
    )
    assert seen["motions"] == ["animate", "animate"]
    assert seen["animator"] is animator
    import json
    state = json.loads(
        (works.video_dir_for_kind("s1", "ch-1", "recap") / "render_state.json")
        .read_text()
    )
    assert state["mode"] == "animate"
    assert state["grounded"] is True
    assert state["frames"] == "fake-frames:"
    meta = works.read_video_metadata("s1", "ch-1", "recap")
    assert meta.video_gen_provider == "fake-frames"


def test_build_video_animate_local_provider_renders_as_panels(harness):
    conn, series, chapter, profile, *_ = harness
    logs: list[str] = []
    # [frames] provider defaults to "local" — no image generator configured
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=logs.append, video_mode="animate",
    )
    assert any("generates no frames" in m for m in logs)


class FakeSequenceAnimator:
    name = "fake-seq"
    animated = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_frames(self, image, segment, duration, workdir,
                        log=lambda m: None, critic=None):
        self.calls.append({"image": image, "index": segment.index,
                           "duration": duration, "critic": critic})
        return [image, image, image, image]  # expansion + 3 chained frames


def test_assemble_sequence_stitches_generated_frames(tmp_path, monkeypatch):
    _capture_clips(monkeypatch)
    seqs: list[dict] = []

    def fake_seq(frames, slot, resolution, dest, interp_fps=30):
        seqs.append({"frames": frames, "slot": slot, "dest": dest,
                     "interp_fps": interp_fps})
        dest.write_bytes(b"clip")

    monkeypatch.setattr(assemble_mod, "build_sequence_clip", fake_seq)
    animator = FakeSequenceAnimator()
    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1, 2], motion="sequence", duration_s=8.0,
        regions=[
            {"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]},
            {"page": 2, "kind": "pan"},
        ],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=animator, sequence_interp_fps=24,
                          sequence_critic="critic-fn")
    assert len(animator.calls) == 2
    # hold anchor sequences its panel crop, pan anchor the full page
    assert animator.calls[0]["image"] == workdir / "clips" / "panel-00-0.png"
    assert animator.calls[1]["image"] == page_paths[1]
    assert all(c["duration"] == pytest.approx(4.0) for c in animator.calls)
    assert all(c["critic"] == "critic-fn" for c in animator.calls)
    assert [s["dest"].name for s in seqs] == ["clip-00-0.mp4", "clip-00-1.mp4"]
    assert all(s["slot"] == pytest.approx(4.0) for s in seqs)
    assert all(s["interp_fps"] == 24 for s in seqs)
    assert all(len(s["frames"]) == 4 for s in seqs)  # the generated frames


def test_assemble_sequence_stills_animator_falls_back_to_panels(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)

    class StillsAnimator(FakeSequenceAnimator):
        animated = False

    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1], motion="sequence", duration_s=4.0,
        regions=[{"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]}],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=StillsAnimator())
    assert len(calls) == 1  # the panels-mode Ken Burns crop clip
    assert calls[0]["motion"] == "zoom_in"
    assert calls[0]["page"] == workdir / "clips" / "panel-00-0.png"


def test_assemble_sequence_generation_failure_falls_back_to_panels(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)

    class FailingAnimator(FakeSequenceAnimator):
        def generate_frames(self, image, segment, duration, workdir,
                            log=lambda m: None, critic=None):
            raise VideoError("Runway generation failed (400): bad aspect")

    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1], motion="sequence", duration_s=4.0,
        regions=[{"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]}],
    )
    workdir = tmp_path / "work"
    logs: list[str] = []
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=FailingAnimator(), log=logs.append)
    assert len(calls) == 1  # the panels-mode Ken Burns crop clip
    assert calls[0]["motion"] == "zoom_in"
    assert calls[0]["page"] == workdir / "clips" / "panel-00-0.png"
    assert any("panels fallback" in m for m in logs)


def test_assemble_animate_generation_failure_falls_back_to_panels(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)

    class FailingAnimator(FakeFrameAnimator):
        def generate_frames(self, image, segment, duration, workdir,
                            log=lambda m: None):
            raise VideoError("boom")

    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1], motion="animate", duration_s=4.0,
        regions=[{"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]}],
    )
    workdir = tmp_path / "work"
    logs: list[str] = []
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=FailingAnimator(), log=logs.append)
    assert len(calls) == 1  # the panels-mode Ken Burns crop clip
    assert calls[0]["motion"] == "zoom_in"
    assert calls[0]["page"] == workdir / "clips" / "panel-00-0.png"
    assert any("panels fallback" in m for m in logs)


def test_assemble_sequence_config_error_aborts_render(tmp_path, monkeypatch):
    """A VideoConfigError (a model that can't generate images) is not a
    per-panel failure: the render aborts instead of degrading every anchor."""
    calls = _capture_clips(monkeypatch)

    class MisconfiguredAnimator(FakeSequenceAnimator):
        def generate_frames(self, image, segment, duration, workdir,
                            log=lambda m: None, critic=None):
            raise VideoConfigError("'gpt-6-luna' cannot generate images")

    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1], motion="sequence", duration_s=4.0,
        regions=[{"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]}],
    )
    with pytest.raises(VideoConfigError, match="cannot generate images"):
        assemble_mod.assemble([seg], page_paths, tmp_path / "work",
                              resolution=(360, 640),
                              frame_animator=MisconfiguredAnimator(),
                              log=lambda m: None)
    assert calls == []  # no panels fallback, no clip built


def test_assemble_sequence_fallback_clip_is_not_cached(tmp_path, monkeypatch):
    """A degraded fallback clip is not the provider's product: it must not
    satisfy the canonical clip-*.mp4 cache, so the next run retries the
    generator for that anchor."""
    calls = _capture_clips(monkeypatch)
    seqs: list[dict] = []

    def fake_seq(frames, slot, resolution, dest, interp_fps=30):
        seqs.append({"dest": dest})
        dest.write_bytes(b"clip")

    monkeypatch.setattr(assemble_mod, "build_sequence_clip", fake_seq)

    class FailingAnimator(FakeSequenceAnimator):
        def generate_frames(self, image, segment, duration, workdir,
                            log=lambda m: None, critic=None):
            raise VideoError("provider down")

    page_paths = _three_pages(tmp_path)
    seg = Segment(
        index=0, text="x", pages=[1], motion="sequence", duration_s=4.0,
        regions=[{"page": 1, "kind": "hold", "box": [0.0, 0.0, 0.5, 0.5]}],
    )
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=FailingAnimator(), log=lambda m: None)
    assert len(calls) == 1
    assert calls[0]["dest"].name == "clip-00-0.fallback.mp4"
    assert not (workdir / "clips" / "clip-00-0.mp4").exists()

    # A later run with a healthy provider regenerates the anchor instead of
    # reusing the fallback clip.
    animator = FakeSequenceAnimator()
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640),
                          frame_animator=animator, log=lambda m: None)
    assert len(animator.calls) == 1  # retried, not served from the cache
    assert [s["dest"].name for s in seqs] == ["clip-00-0.mp4"]


def _fake_ground(adapter, model, segments, page_paths, series_id,
                 chapter_id, title, chapter_num, log=lambda m: None):
    for s in segments:
        s.regions = [{"page": 1, "kind": "hold", "box": [0, 0, 0.5, 0.5]}]


def _sequence_capture(seen):
    def capture(segments, page_paths, workdir, resolution, log,
                frame_animator=None, sequence_interp_fps=30,
                sequence_critic=None):
        seen["motions"] = [s.motion for s in segments]
        seen["animator"] = frame_animator
        seen["interp_fps"] = sequence_interp_fps
        seen["critic"] = sequence_critic
        seen["renders"] = seen.get("renders", 0) + 1
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    return capture


def _sequence_harness(harness, monkeypatch, critic=False):
    """Wire a fake sequence animator into the pipeline; returns (args, seen)."""
    conn, series, chapter, profile = harness[:4]
    monkeypatch.setattr(pipeline, "ground_segments", _fake_ground)
    animator = FakeSequenceAnimator()
    monkeypatch.setattr(pipeline, "get_frame_animator",
                        lambda name, config: animator)
    seen: dict = {}
    monkeypatch.setattr(pipeline, "assemble", _sequence_capture(seen))
    config = _artifact_config()
    config.sequence.provider = "fake-seq"
    config.sequence.critic = critic
    return (conn, series, chapter, config, profile), seen, animator


def test_build_video_sequence_mode_grounds_and_passes_animator(
    harness, monkeypatch
):
    args, seen, animator = _sequence_harness(harness, monkeypatch)
    conn, series, chapter, config, profile = args
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None, video_mode="sequence",
    )
    assert seen["motions"] == ["sequence", "sequence"]
    assert seen["animator"] is animator
    assert seen["interp_fps"] == config.sequence.interp_fps
    assert seen["critic"] is None  # critic off → nothing passed through
    import json
    state = json.loads(
        (works.video_dir_for_kind("s1", "ch-1", "recap") / "render_state.json")
        .read_text()
    )
    assert state["mode"] == "sequence"
    assert state["grounded"] is True
    assert state["sequence"] == "fake-seq::1.5"
    meta = works.read_video_metadata("s1", "ch-1", "recap")
    assert meta.video_gen_provider == "fake-seq"


def test_build_video_sequence_critic_built_when_enabled(harness, monkeypatch):
    args, seen, _ = _sequence_harness(harness, monkeypatch, critic=True)
    conn, series, chapter, config, profile = args
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None, video_mode="sequence",
    )
    assert callable(seen["critic"])


def test_build_video_sequence_fps_change_re_renders(harness, monkeypatch):
    args, seen, _ = _sequence_harness(harness, monkeypatch)
    conn, series, chapter, config, profile = args
    for _ in range(2):
        pipeline.build_video(
            conn, series, chapter, config, profile,
            client=FakeClient(), log=lambda m: None, video_mode="sequence",
        )
    assert seen["renders"] == 1  # second build served the cached render
    config.sequence.fps = 2.0
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None, video_mode="sequence",
    )
    assert seen["renders"] == 2  # fps is part of the render state


def test_build_video_sequence_model_change_re_renders(harness, monkeypatch):
    args, seen, _ = _sequence_harness(harness, monkeypatch)
    conn, series, chapter, config, profile = args
    for _ in range(2):
        pipeline.build_video(
            conn, series, chapter, config, profile,
            client=FakeClient(), log=lambda m: None, video_mode="sequence",
        )
    assert seen["renders"] == 1  # second build served the cached render
    config.sequence.model = "google/gemini-3-pro-image"
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None, video_mode="sequence",
    )
    assert seen["renders"] == 2  # the image model is part of the render state


def test_build_video_sequence_stills_provider_renders_as_panels(harness):
    conn, series, chapter, profile, *_ = harness
    logs: list[str] = []
    config = _artifact_config()
    config.sequence.provider = "local"  # no image generator behind it
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=logs.append, video_mode="sequence",
    )
    assert any("generates no frames" in m for m in logs)


def test_build_sequence_clip_is_slot_exact(tmp_path):
    import shutil

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not installed")
    frames = [tmp_path / f"frame-{i}.png" for i in range(3)]
    for p in frames:
        _write_page(p, size=(160, 90))
    dest = tmp_path / "seq.mp4"
    assemble_mod.build_sequence_clip(frames, 2.4, (320, 180), dest,
                                     interp_fps=10)
    assert assemble_mod.video_duration(dest) == pytest.approx(2.4, abs=0.15)


def test_build_sequence_clip_single_frame_is_static(tmp_path, monkeypatch):
    clips: list[dict] = []

    def fake_build_clip(page, duration, motion, resolution, dest):
        clips.append({"page": page, "duration": duration, "motion": motion})
        dest.write_bytes(b"clip")

    monkeypatch.setattr(assemble_mod, "build_clip", fake_build_clip)
    frame = tmp_path / "frame.png"
    _write_page(frame)
    assemble_mod.build_sequence_clip([frame], 3.0, (320, 180),
                                     tmp_path / "seq.mp4")
    assert clips == [{"page": frame, "duration": 3.0, "motion": "static"}]


class FakeColorizer:
    """Copies pages into the colorized dir instead of running DDColor."""

    def __init__(self, config=None, model_dir=None, **kw) -> None:
        pass

    def colorize_pages(self, pages, dest_dir, log=lambda m: None):
        mapping = {}
        for src in pages:
            dest = dest_dir / f"{src.stem}.png"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"colorized-" + src.read_bytes())
            mapping[src] = dest
        return mapping


def _fake_colorizer(monkeypatch):
    from entertainment_harness.video import colorize

    monkeypatch.setattr(colorize, "Colorizer", FakeColorizer)
    # The pipeline resolves the provider through the registry, whose lazy
    # load caches the real class — patch create() too, order-independently.
    monkeypatch.setattr(
        colorize.REGISTRY, "create", lambda name, config=None, **kw: FakeColorizer()
    )


def test_build_video_colorize_uses_colorized_pages(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    _fake_colorizer(monkeypatch)
    seen: list[list[Path]] = []

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        seen.append(list(page_paths))
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)
    workdir = works.video_recap_dir("s1", "ch-1")

    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), colorize=True, log=lambda m: None,
    )
    colorized = [p for p in seen[0] if "colorized" in p.parts]
    assert colorized  # used pages were swapped for colorized copies
    assert all(p.read_bytes().startswith(b"colorized-") for p in colorized)
    import json

    state = json.loads((workdir / "render_state.json").read_text())
    assert state == {"colorize": True, "translated": False, "mode": "kenburns",
                     "detail": "standard", "pacing": pipeline.PACING_VERSION,
                     "panel_first": False}


def test_colorize_toggle_triggers_rerender(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    _fake_colorizer(monkeypatch)
    renders: list[list[Path]] = []

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        renders.append(list(page_paths))
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)

    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), colorize=True, log=lambda m: None,
    )
    pipeline.build_video(  # cached: same treatment, no re-render
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), colorize=True, log=lambda m: None,
    )
    assert len(renders) == 1
    pipeline.build_video(  # treatment changed: re-render with raw pages
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), colorize=False, log=lambda m: None,
    )
    assert len(renders) == 2
    assert not any("colorized" in p.parts for p in renders[1])


def _make_translated_pages(harness):
    """Create the eh-translate output layout for the harness chapter."""
    translated = works.translated_dir("s1", "ch-1")
    translated.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        _write_page(translated / f"page-{i:03d}.png")
    return translated


def test_build_video_translated_requires_translation(harness):
    conn, series, chapter, profile, *_ = harness
    with pytest.raises(VideoError, match="eh recap --translated"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile,
            client=FakeClient(), translated=True, log=lambda m: None,
        )


def test_build_video_translated_uses_translated_pages(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    _make_translated_pages(harness)
    seen: list[list[Path]] = []

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        seen.append(list(page_paths))
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), translated=True, log=lambda m: None,
    )
    assert all("translated" in p.parts for p in seen[0])
    import json

    workdir = works.video_recap_dir("s1", "ch-1")
    state = json.loads((workdir / "render_state.json").read_text())
    assert state == {"colorize": False, "translated": True, "mode": "kenburns",
                     "detail": "standard", "pacing": pipeline.PACING_VERSION,
                     "panel_first": False}


def test_translated_toggle_triggers_rerender(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    renders = 0

    def counting_assemble(segments, page_paths, workdir, resolution, log):
        nonlocal renders
        renders += 1
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", counting_assemble)
    pipeline.build_video(  # normal build
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    pipeline.build_video(  # cached
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert renders == 1
    _make_translated_pages(harness)
    pipeline.build_video(  # treatment changed -> re-render
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), translated=True, log=lambda m: None,
    )
    assert renders == 2


# --- video mode (kenburns / scroll) -------------------------------------------


def test_load_config_parses_video_mode(tmp_path):
    from entertainment_harness.config import load_config

    path = tmp_path / "config.toml"
    path.write_text('[video]\nmode = "scroll"\n')
    assert load_config(path).video.mode == "scroll"
    assert load_config(tmp_path / "missing.toml").video.mode == "kenburns"


def test_load_config_parses_panel_first(tmp_path):
    from entertainment_harness.config import load_config

    path = tmp_path / "config.toml"
    # panel-first is the default video script path; config can only opt out.
    assert load_config(tmp_path / "missing.toml").video.panel_first is True
    path.write_text("[video]\npanel_first = false\n")
    assert load_config(path).video.panel_first is False
    path.write_text("[video]\npanel_first = true\n")
    assert load_config(path).video.panel_first is True


def test_clear_downstream_removes_strips(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip-00-0.mp4").write_bytes(b"x")
    (clips / "strip-00.png").write_bytes(b"x")
    (tmp_path / "out.mp4").write_bytes(b"x")
    pipeline.clear_downstream(tmp_path)
    assert not list(clips.iterdir())
    assert not (tmp_path / "out.mp4").exists()


def test_build_video_rejects_bad_mode(harness):
    conn, series, chapter, profile, *_ = harness
    with pytest.raises(VideoError, match="Bad video mode"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile,
            client=FakeClient(), video_mode="diagonal", log=lambda m: None,
        )


def test_build_video_scroll_mode_sets_motions_and_state(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    seen: list[list[str]] = []

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        seen.append([s.motion for s in segments])
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), video_mode="scroll", log=lambda m: None,
    )
    assert seen == [["scroll", "scroll"]]
    import json

    workdir = works.video_recap_dir("s1", "ch-1")
    state = json.loads((workdir / "render_state.json").read_text())
    assert state == {"colorize": False, "translated": False, "mode": "scroll",
                     "detail": "standard", "pacing": pipeline.PACING_VERSION,
                     "grounded": True, "panel_first": False}


def test_build_video_mode_config_default_applies(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    seen: list[list[str]] = []

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        seen.append([s.motion for s in segments])
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)
    config = Config()
    config.video.panel_first = False
    config.video.mode = "scroll"
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert seen == [["scroll", "scroll"]]


def test_mode_toggle_triggers_rerender_and_restamps_motions(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    renders: list[list[str]] = []

    def recording_assemble(segments, page_paths, workdir, resolution, log):
        renders.append([s.motion for s in segments])
        out = workdir / "out.mp4"
        out.write_bytes(b"fake-mp4")
        return out, sum(s.duration_s for s in segments)

    monkeypatch.setattr(pipeline, "assemble", recording_assemble)
    pipeline.build_video(  # kenburns: alternating motions by index
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    pipeline.build_video(  # cached: same mode, no re-render
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert renders == [["pan_down", "zoom_in"]]
    pipeline.build_video(  # mode changed -> re-render, every segment scrolls
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), video_mode="scroll", log=lambda m: None,
    )
    assert renders == [["pan_down", "zoom_in"], ["scroll", "scroll"]]
    pipeline.build_video(  # back to kenburns -> re-render, motions restamped
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert renders[-1] == ["pan_down", "zoom_in"]


# --- compression --------------------------------------------------------------


def _fake_compress(monkeypatch):
    calls: list[str] = []

    def fake_compress(src, preset, workdir):
        calls.append(preset)
        dest = workdir / f"out-{preset}.mp4"
        dest.write_bytes(b"compressed-" + preset.encode())
        return dest

    monkeypatch.setattr(pipeline, "compress_video", fake_compress)
    return calls


def test_build_video_compress_returns_compressed_copy(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    calls = _fake_compress(monkeypatch)
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), compress="balanced", log=lambda m: None,
    )
    assert out.name == "out-balanced.mp4" and out.exists()
    assert calls == ["balanced"]
    master = out.with_name("out.mp4")
    assert master.exists()  # master preserved for future re-compression
    row = conn.execute("SELECT * FROM videos WHERE series_id = 's1'").fetchone()
    assert row["path"] == str(out)


def test_build_video_compress_caches_and_reswitches(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    calls = _fake_compress(monkeypatch)
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), compress="balanced", log=lambda m: None,
    )
    out2 = pipeline.build_video(  # cached: no re-encode
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), compress="balanced", log=lambda m: None,
    )
    assert out2 == out
    assert calls == ["balanced"]

    out3 = pipeline.build_video(  # different preset: re-encode from master
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), compress="small", log=lambda m: None,
    )
    assert calls == ["balanced", "small"]
    assert out3.name == "out-small.mp4"
    row = conn.execute("SELECT * FROM videos WHERE series_id = 's1'").fetchone()
    assert row["path"] == str(out3)


# --- automated compression + pruning -----------------------------------------


def test_build_video_config_default_compress_applies(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    calls = _fake_compress(monkeypatch)
    config = Config()
    config.video.panel_first = False
    config.video.compress = "balanced"
    out = pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out.name == "out-balanced.mp4"
    assert calls == ["balanced"]


def test_build_video_prunes_intermediates_after_assembly(harness, monkeypatch):
    conn, series, chapter, profile, *_rest = harness
    engine = _rest[2]
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    from entertainment_harness.library import works

    workdir = works.video_recap_dir("s1", "ch-1")
    assert engine.synthesized  # TTS ran during assembly
    assert not list(workdir.glob("seg-*.wav"))  # segment WAVs pruned
    assert not (workdir / "clips").exists()
    assert (workdir / "out.mp4").exists()  # master kept by default


def test_build_video_keep_master_false_prunes_master(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    calls = _fake_compress(monkeypatch)
    config = Config()
    config.video.panel_first = False
    config.video.compress = "balanced"
    config.video.keep_master = False
    out = pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out.name == "out-balanced.mp4"
    from entertainment_harness.library import works

    workdir = works.video_recap_dir("s1", "ch-1")
    assert not (workdir / "out.mp4").exists()  # master pruned
    assert calls == ["balanced"]


def test_build_video_preset_switch_after_master_pruned(harness, monkeypatch):
    conn, series, chapter, profile, *_ = harness
    calls = _fake_compress(monkeypatch)
    config = Config()
    config.video.panel_first = False
    config.video.compress = "balanced"
    config.video.keep_master = False
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None,
    )
    out = pipeline.build_video(  # new preset, master pruned: re-render, then
        conn, series, chapter, config, profile,  # compress the fresh master
        client=FakeClient(), compress="small", log=lambda m: None,
    )
    assert out.name == "out-small.mp4"
    assert calls == ["balanced", "small"]


def test_build_video_cached_compressed_copy_short_circuits(harness, monkeypatch):
    conn, series, chapter, profile, *_rest = harness
    engine = _rest[2]
    calls = _fake_compress(monkeypatch)
    config = Config()
    config.video.panel_first = False
    config.video.compress = "balanced"
    config.video.keep_master = False
    pipeline.build_video(
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None,
    )
    synthesized = len(engine.synthesized)
    out = pipeline.build_video(  # same preset: compressed copy is the cache
        conn, series, chapter, config, profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out.name == "out-balanced.mp4"
    assert calls == ["balanced"]  # no re-encode
    assert len(engine.synthesized) == synthesized  # no re-TTS/re-assembly


def test_tts_skipped_when_audio_pruned_and_video_cached(harness, monkeypatch):
    conn, series, chapter, profile, *_rest = harness
    engine = _rest[2]
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    synthesized = len(engine.synthesized)
    from entertainment_harness.library import works

    workdir = works.video_recap_dir("s1", "ch-1")
    for wav in workdir.glob("seg-*.wav"):  # simulate post-assembly pruning
        wav.unlink()
    logs: list[str] = []
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=logs.append,
    )
    assert out.name == "out.mp4"  # cached video untouched
    assert len(engine.synthesized) == synthesized  # no re-TTS
    assert any("pruned (video cached)" in m for m in logs)


def test_pacing_version_bump_rerenders_and_regenerates_pruned_wav(
    harness, monkeypatch
):
    """A new PACING_VERSION invalidates the cached video even after concat
    inputs were pruned: the clear happens before TTS, so segment WAVs are
    re-synthesized instead of failing at mux time."""
    conn, series, chapter, profile, text, vision, engine, assembled = harness
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_recap_dir("s1", "ch-1")
    assert not list(workdir.glob("seg-*.wav"))  # pruned after assembly

    monkeypatch.setattr(pipeline, "PACING_VERSION", pipeline.PACING_VERSION + 1)
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out.exists()
    assert assembled == [2, 2]  # re-rendered
    assert len(engine.synthesized) == 4  # WAVs re-synthesized
    assert text.calls == 1 and vision.calls == 1  # script/pages stayed cached

    out2 = pipeline.build_video(  # new version is the new cache: skips again
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out2 == out
    assert assembled == [2, 2]
    assert len(engine.synthesized) == 4


def test_pauses_applied_from_cached_script(harness):
    """Pauses are deterministic from the script: a cached script.json gets
    them on load, and they persist back to disk."""
    conn, series, chapter, profile, *_ = harness
    import json

    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_recap_dir("s1", "ch-1")
    payload = json.loads((workdir / "script.json").read_text())
    pauses = [s["pause_after_s"] for s in payload["segments"]]
    # fixture script has moments A -> B: one transition pause, then none
    assert pauses == [TRANSITION_PAUSE_S, 0.0]



# --- segments_from_narration ---------------------------------------------------


def _words(word: str, n: int) -> str:
    return " ".join([word] * n)


def test_segments_from_narration_empty_raises():
    with pytest.raises(VideoError, match="Narration is empty"):
        segments_from_narration("")
    with pytest.raises(VideoError, match="Narration is empty"):
        segments_from_narration("  \n\n  ")


def test_segments_from_narration_merges_short_paragraphs():
    # three 3-word paragraphs merge forward into one 9-word segment
    segments = segments_from_narration("\n\n".join([_words("w", 3)] * 3))
    assert len(segments) == 1
    assert len(segments[0].text.split()) == 9

    # a paragraph at min_words starts a new segment instead of merging
    text = "\n\n".join([_words("a", 25), _words("b", 10), _words("c", 10)])
    segments = segments_from_narration(text, min_words=25)
    assert [len(s.text.split()) for s in segments] == [25, 20]


def test_segments_from_narration_splits_long_paragraph():
    # 150 words in one paragraph (> 2 * target_words) split at sentences
    paragraph = " ".join(_words("word", 15) + "." for _ in range(10))
    segments = segments_from_narration(paragraph, target_words=60)
    assert len(segments) == 3  # 60 + 60 + 30 words
    assert " ".join(s.text for s in segments).split() == paragraph.split()


def test_segments_from_narration_sequential_indices():
    text = "\n\n".join([_words("w", 30)] * 3)
    segments = segments_from_narration(text)
    assert [s.index for s in segments] == [0, 1, 2]


# --- detail-derived build_video kind ------------------------------------------

NARRATION_TEXT = _words("alpha", 40) + "\n\n" + _words("omega", 40)


def _seed_narration(conn, chapter_id="ch-1", text=NARRATION_TEXT):
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES (?, ?, 'now', 'fake-vision:8b (Q8_0)')",
        (chapter_id, text),
    )
    conn.commit()


def _seed_full_recap(conn, chapter_id="ch-1", text=NARRATION_TEXT):
    """Post-merge home of a full-detail artifact: recaps at detail='full'
    (the harness seeds a standard recaps row; upgrade it in place)."""
    conn.execute(
        "UPDATE recaps SET summary = ?, model = 'fake-vision:8b (Q8_0)',"
        " detail = 'full' WHERE chapter_id = ?",
        (text, chapter_id),
    )
    conn.commit()


def test_build_video_full_detail_derives_narration_kind(harness):
    """The video kind derives from the artifact's detail: a full-detail
    recaps row renders verbatim-split segments (no script model) in the
    narration workdir, recorded with kind='narration'."""
    import json

    conn, series, chapter, profile, text, vision, engine, assembled = harness
    _seed_full_recap(conn)
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_narration_dir("s1", "ch-1")
    assert out == workdir / "out.mp4" and out.exists()
    segments, script_model = load_script(workdir / "script.json")
    assert any("alpha" in s.text for s in segments)  # from the recaps row
    assert "fake-vision:8b" in script_model
    assert text.calls == 0  # the script model is never invoked
    row = conn.execute(
        "SELECT kind FROM videos WHERE series_id = 's1'"
    ).fetchone()
    assert row["kind"] == "narration"
    state = json.loads((workdir / "render_state.json").read_text())
    assert state["detail"] == "full"


def test_build_video_recaps_full_wins_over_legacy_narrations(harness):
    """With both a full-detail recaps row and a legacy narrations row, the
    recaps artifact is authoritative."""
    conn, series, chapter, profile, text, vision, engine, assembled = harness
    _seed_narration(conn, text=_words("legacy", 80))
    _seed_full_recap(conn)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_narration_dir("s1", "ch-1")
    segments, _ = load_script(workdir / "script.json")
    joined = " ".join(s.text for s in segments)
    assert "alpha" in joined
    assert "legacy" not in joined


def test_build_video_source_override_keeps_legacy_narration_path(harness):
    """source='narration' forces the narration kind for callers that pass it;
    with no full-detail recaps row the legacy narrations table is read."""
    conn, series, chapter, profile, text, vision, engine, assembled = harness
    _seed_narration(conn)
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile, source="narration",
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_narration_dir("s1", "ch-1")
    assert out == workdir / "out.mp4" and out.exists()
    assert (workdir / "script.json").exists()
    assert text.calls == 0  # the script model is never invoked
    assert engine.synthesized  # segments went through TTS
    row = conn.execute(
        "SELECT * FROM videos WHERE series_id = 's1' AND kind = 'narration'"
    ).fetchone()
    assert row is not None
    assert row["from_chapter"] == 1.0 and row["to_chapter"] == 1.0


def test_build_video_narration_requires_full_artifact(harness):
    conn, series, chapter, profile, *_ = harness
    with pytest.raises(VideoError, match="no full-detail artifact"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile, source="narration",
            client=FakeClient(), log=lambda m: None,
        )


def test_build_video_rejects_bad_source(harness):
    conn, series, chapter, profile, *_ = harness
    with pytest.raises(VideoError, match="Bad video source"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile, source="bogus",
            client=FakeClient(), log=lambda m: None,
        )


def test_build_video_supersedes_other_kind_on_detail_change(harness):
    """One video per chapter: re-rendering after a detail upgrade prunes the
    old kind's workdir and leaves a single videos row at the new kind."""
    conn, series, chapter, profile, *_ = harness
    out_recap = pipeline.build_video(  # standard artifact -> recap kind
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out_recap == works.video_recap_dir("s1", "ch-1") / "out.mp4"
    assert out_recap.exists()
    _seed_full_recap(conn)  # detail upgrade standard -> full
    out_narration = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out_narration == works.video_narration_dir("s1", "ch-1") / "out.mp4"
    assert out_narration.exists()
    assert not works.video_recap_dir("s1", "ch-1").exists()  # files pruned
    rows = conn.execute(
        "SELECT kind FROM videos WHERE series_id = 's1'"
    ).fetchall()
    assert [r["kind"] for r in rows] == ["narration"]  # one row, new kind


def test_build_video_render_state_detail_change_rerenders(harness):
    """render_state.json records the artifact's detail: a grain change within
    the same kind (standard -> detailed, both recap) re-renders instead of
    serving the stale video (same pattern as mode/colorize)."""
    import json

    conn, series, chapter, profile, text, vision, engine, assembled = harness
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_recap_dir("s1", "ch-1")
    state = json.loads((workdir / "render_state.json").read_text())
    assert state["detail"] == "standard"

    conn.execute(
        "UPDATE recaps SET detail = 'detailed' WHERE chapter_id = 'ch-1'"
    )
    conn.commit()
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert assembled == [2, 2]  # re-rendered
    assert text.calls == 1  # script.json cache still serves (as on any re-recap)
    state = json.loads((workdir / "render_state.json").read_text())
    assert state["detail"] == "detailed"


# --- panel-first narration (anchored scroll, Phase 4) ------------------------

PF_GROUPS = (
    '[{"from": 1, "to": 1, "moment": "wake", "text": "Shin wakes up."},'
    ' {"from": 2, "to": 3, "moment": "fight", "text": "They fight and win."}]'
)


def _seed_panels_for_s1() -> None:
    from entertainment_harness.video.panels import Panel, store_panels

    src = works.source_dir("s1", "ch-1")
    for i in range(1, 4):
        _write_page(src / f"page-{i:03d}.jpg")
    store_panels("s1", "ch-1", {
        f"page-{i:03d}.jpg": [Panel([0.0, 0.0, 1.0, 1.0], f"Panel {i}.")]
        for i in range(1, 4)
    })


def _patch_panel_first_models(monkeypatch, judge_output='{"pass": true}'):
    grouping = FakeAdapter(PF_GROUPS, "fake-text:4b")
    judge = FakeAdapter(judge_output, "fake-judge:4b")
    info = ModelInfo("fake:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        pipeline, "get_text_model",
        lambda c, p: Selection(adapter=grouping, info=info),
    )
    monkeypatch.setattr(
        pipeline, "get_judge_model",
        lambda c, p: Selection(adapter=judge, info=info),
    )
    return grouping, judge


def test_build_video_panel_first(harness, monkeypatch):
    """Panel-first (Phase 4): the script is grouped from cached panel beats —
    segments are born with pages+regions (stages 3/3.5 never run), the kind
    is forced to narration, and render_state records the flag. The [video]
    panel_first= config default applies when no flag is passed."""
    import json

    conn, series, chapter, profile, text, vision, engine, assembled = harness
    _seed_panels_for_s1()
    grouping, judge = _patch_panel_first_models(monkeypatch)
    config = Config()
    config.video.panel_first = True  # config default, no CLI flag
    out = pipeline.build_video(
        conn, series, chapter, config, profile, video_mode="scroll",
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_narration_dir("s1", "ch-1")  # kind forced
    assert out == workdir / "out.mp4" and out.exists()
    segments, script_model = load_script(workdir / "script.json")
    assert script_model.startswith("panel-first")
    assert [s.text for s in segments] == [
        "Shin wakes up.", "They fight and win."
    ]
    assert segments[0].pages == [1] and segments[1].pages == [2, 3]
    assert all(s.regions for s in segments)
    assert grouping.calls == 1
    assert judge.calls == 1   # the narration judge ran (thinking medium)
    assert vision.calls == 0  # no assign_pages, no grounding judge
    assert not (works.chapter_dir("s1", "ch-1") / "grounding.json").exists()
    assert (works.chapter_dir("s1", "ch-1") / "panelfirst.json").exists()
    row = conn.execute(
        "SELECT kind FROM videos WHERE series_id = 's1'"
    ).fetchone()
    assert row["kind"] == "narration"
    state = json.loads((workdir / "render_state.json").read_text())
    assert state["panel_first"] is True and state["mode"] == "scroll"


def test_build_video_panel_first_script_provenance_regenerates(
    harness, monkeypatch
):
    """Panel-first shares the narration workdir with the artifact-split path:
    a plain full-detail build after a panel-first render regenerates the
    script instead of reusing the panel-first one."""
    conn, series, chapter, profile, text, vision, engine, assembled = harness
    _seed_panels_for_s1()
    _patch_panel_first_models(monkeypatch)
    _seed_full_recap(conn)
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile, panel_first=True,
        client=FakeClient(), log=lambda m: None,
    )
    workdir = works.video_narration_dir("s1", "ch-1")
    _, script_model = load_script(workdir / "script.json")
    assert script_model.startswith("panel-first")

    pipeline.build_video(  # plain build: same workdir, other stage-1 path
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    segments, script_model = load_script(workdir / "script.json")
    assert script_model.startswith("narration")
    assert any("alpha" in s.text for s in segments)


def test_build_video_panel_first_rejects_source_override(harness):
    conn, series, chapter, profile, *_ = harness
    with pytest.raises(VideoError, match="mutually exclusive"):
        pipeline.build_video(
            conn, series, chapter, _artifact_config(), profile,
            panel_first=True, source="recap",
            client=FakeClient(), log=lambda m: None,
        )


def test_build_video_panel_first_supersedes_recap_video(harness, monkeypatch):
    """One video per chapter still holds: the panel-first (narration-kind)
    build prunes the recap kind's workdir and its videos row."""
    conn, series, chapter, profile, *_ = harness
    out_recap = pipeline.build_video(  # plain standard build -> recap kind
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    assert out_recap.exists()
    _seed_panels_for_s1()
    _patch_panel_first_models(monkeypatch)
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile, panel_first=True,
        client=FakeClient(), log=lambda m: None,
    )
    assert out == works.video_narration_dir("s1", "ch-1") / "out.mp4"
    assert not works.video_recap_dir("s1", "ch-1").exists()
    rows = conn.execute(
        "SELECT kind FROM videos WHERE series_id = 's1'"
    ).fetchall()
    assert [r["kind"] for r in rows] == ["narration"]


# --- hold-and-glide (anchored scroll, Phase 3) ------------------------------

from entertainment_harness.video.assemble import (
    GLIDE_S,
    _segment_anchors,
    _y_expression,
    hold_glide_keyframes,
    strip_layout,
)


def _layout(pages):
    return strip_layout(pages)


def test_segment_anchors_validation_and_reading_order():
    seg = Segment(index=0, text="x", pages=[1, 2], duration_s=30.0, regions=[
        {"page": 2, "kind": "hold", "box": [0, 0.2, 1, 0.5]},
        {"page": 1, "kind": "hold", "box": [0, 0.5, 1, 0.8]},
        {"page": 1, "kind": "pan"},
        {"page": 9, "kind": "pan"},                    # out of range
        {"page": 1, "kind": "hold"},                   # no box
        {"page": 1, "kind": "weird"},                  # unknown kind
        "junk",
    ])
    anchors = _segment_anchors(seg, 30.0, page_count=3)
    assert [a["kind"] for a in anchors] == ["hold", "pan", "hold"]
    assert [a["page"] for a in anchors] == [1, 1, 2]  # page-sorted


def test_segment_anchors_cap_picks_evenly_spaced():
    regions = [{"page": i + 1, "kind": "pan"} for i in range(5)]
    seg = Segment(index=0, text="x", duration_s=6.0, regions=regions)
    anchors = _segment_anchors(seg, 6.0, page_count=5)  # 6.0 // 2.5 -> 2
    assert [a["page"] for a in anchors] == [1, 5]
    seg.duration_s = 2.0
    assert len(_segment_anchors(seg, 2.0, page_count=5)) == 1


def test_hold_glide_keyframes_holds_with_glide_between(tmp_path):
    pages = _three_pages(tmp_path)
    layout = _layout(pages[:2])  # 120x300 pages: scale 3 at out_w 360
    anchors = [
        {"page": 1, "kind": "hold", "box": [0, 0.5, 1, 0.8]},
        {"page": 2, "kind": "hold", "box": [0, 0.2, 1, 0.5]},
    ]
    kf = hold_glide_keyframes(anchors, [1, 2], layout, 360, 640, 9.0)
    # hold share = (9 - 0.7) / 2 = 4.15; holds center the region in the
    # viewport: strip y 195 (page 1) -> 585 scaled -> 265; y 429 -> 967.
    assert kf[0] == (0.0, pytest.approx(265.0))
    assert kf[1] == (pytest.approx(4.15), kf[0][1])
    assert kf[2] == (pytest.approx(4.15 + GLIDE_S), pytest.approx(967.0))
    assert kf[3] == (pytest.approx(9.0), kf[2][1])


def test_hold_glide_keyframes_pan_anchor_descends_its_page(tmp_path):
    pages = _three_pages(tmp_path)
    layout = _layout(pages[1:2])  # the strip holds only the anchor's page
    anchors = [{"page": 2, "kind": "pan"}]
    kf = hold_glide_keyframes(anchors, [2], layout, 360, 640, 5.0)
    # single-page strip 300 px -> 900 scaled; viewport 640: pan 0 -> 260.
    assert kf == [(0.0, pytest.approx(0.0)), (pytest.approx(5.0), pytest.approx(260.0))]


def test_hold_glide_keyframes_never_scrolls_back_up(tmp_path):
    pages = _three_pages(tmp_path)
    layout = _layout(pages[:1])
    anchors = [
        {"page": 1, "kind": "hold", "box": [0, 0.6, 1, 0.9]},  # y = 355
        {"page": 1, "kind": "hold", "box": [0, 0.0, 1, 0.3]},  # raw y clamps to 0
    ]
    kf = hold_glide_keyframes(anchors, [1], layout, 360, 640, 9.0)
    ys = [y for _, y in kf]
    assert ys == sorted(ys)
    assert kf[2][1] == kf[1][1]  # second hold raised to the first's y


def test_hold_glide_keyframes_single_anchor_holds_whole_slot(tmp_path):
    pages = _three_pages(tmp_path)
    layout = _layout(pages[:1])
    anchors = [{"page": 1, "kind": "hold", "box": [0, 0.5, 1, 0.8]}]
    kf = hold_glide_keyframes(anchors, [1], layout, 360, 640, 7.5)
    assert kf[0] == (0.0, kf[0][1])
    assert kf[-1] == (pytest.approx(7.5), kf[0][1])
    assert all(y == kf[0][1] for _, y in kf)


def test_y_expression_piecewise_structure():
    kf = [(0.0, 265.0), (4.15, 265.0), (4.85, 967.0), (9.0, 967.0)]
    expr = _y_expression(kf)
    assert "if(lt(t,4.150),265.0," in expr          # first hold is constant
    assert "+1002.8571*(t-4.150)" in expr           # glide slope 702/0.7
    assert "if(lt(t,9.000),967.0,967.0)" in expr    # final hold to clip end
    assert _y_expression([(0.0, 42.0)]) == "42.0"


def test_assemble_scroll_grounded_segment_renders_hold_glide(tmp_path, monkeypatch):
    captured = {}

    def fake_hold_glide(strip, duration, keyframes, resolution, dest):
        captured.update(strip=strip, duration=duration, keyframes=keyframes)
        dest.write_bytes(b"clip")

    def fake_mux(segments, clips, workdir, log=lambda m: None):
        out = workdir / "out.mp4"
        out.write_bytes(b"out")
        return out, 0.0

    monkeypatch.setattr(assemble_mod, "build_hold_glide_clip", fake_hold_glide)
    monkeypatch.setattr(assemble_mod, "mux_clips", fake_mux)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[1, 2], motion="scroll",
                  duration_s=9.0, regions=[
                      {"page": 1, "kind": "hold", "box": [0, 0.5, 1, 0.8]},
                      {"page": 2, "kind": "pan"},
                  ])
    workdir = tmp_path / "work"
    assemble_mod.assemble([seg], page_paths, workdir, resolution=(360, 640))
    assert captured["strip"] == workdir / "clips" / "strip-00.png"
    assert captured["duration"] == pytest.approx(9.0)
    kf = captured["keyframes"]
    assert kf[0][1] == pytest.approx(265.0)   # hold y on page 1's region
    assert kf[-1] == (pytest.approx(9.0), pytest.approx(1872 - 640))  # pan end
    with Image.open(captured["strip"]) as img:
        assert img.height == 300 * 2 + STRIP_GUTTER_PX  # only anchor pages


def test_assemble_scroll_malformed_regions_fall_back_to_linear(
    tmp_path, monkeypatch
):
    calls = _capture_clips(monkeypatch)
    page_paths = _three_pages(tmp_path)
    seg = Segment(index=0, text="x", pages=[2], motion="scroll",
                  duration_s=2.0, regions=[{"page": 2, "kind": "hold"}])  # no box
    assemble_mod.assemble([seg], page_paths, tmp_path / "work",
                          resolution=(360, 640))
    assert len(calls) == 1
    assert calls[0]["page"] == page_paths[1]  # today's per-page path
    assert calls[0]["motion"] == "pan_down"


def test_script_roundtrip_preserves_regions(tmp_path):
    segs = [Segment(index=0, text="a", moment="m", pages=[1], motion="scroll",
                    duration_s=1.0, pause_after_s=0.15,
                    regions=[{"page": 1, "kind": "hold", "box": [0, 0, 1, 1]},
                             {"page": 2, "kind": "pan"}])]
    from entertainment_harness.video.script import save_script, load_script

    path = tmp_path / "script.json"
    save_script(path, segs, "T", 1.0, "m")
    loaded, _ = load_script(path)
    assert loaded[0].regions == segs[0].regions


def test_script_roundtrip_old_format_without_regions(tmp_path):
    import json

    path = tmp_path / "script.json"
    path.write_text(json.dumps({
        "chapter": 1.0, "title": "T", "model": "m",
        "segments": [{"index": 0, "text": "a", "moment": "", "pages": [],
                      "motion": "", "duration_s": 0.0, "pause_after_s": 0.0}],
    }))
    from entertainment_harness.video.script import load_script

    loaded, _ = load_script(path)
    assert loaded[0].regions == []


def test_build_video_scroll_mode_grounds_segments(harness, monkeypatch):
    """Scroll mode runs the grounding stage (panel extraction -> grounded
    anchors -> judge), stamps segments, and records 'grounded' in
    render_state.json; the second run serves everything from cache."""
    import json

    PANELS = json.dumps([
        {"box": [0.05, 0.04, 0.95, 0.30], "description": "Top panel."},
        {"box": [0.55, 0.35, 0.95, 0.60], "description": "Right panel."},
        {"box": [0.05, 0.35, 0.50, 0.60], "description": "Left panel."},
    ])

    class Router(FakeAdapter):
        def __init__(self):
            super().__init__("{}", "fake-vision:8b")
            self.calls = 0

        def generate(self, model, prompt, images=None):
            self.calls += 1
            if "contact sheet" in prompt:
                return "{}"  # no page matches -> _fill_gaps assigns page 1
            if "Divide the page into its panels" in prompt:
                return PANELS
            if "Verdict task" in prompt:
                return '{"pass": true}'
            return '{"0": [1], "1": [2]}'

    from entertainment_harness.video.panels import parse_panels
    from entertainment_harness.video.regions import regions_for_panels

    expected = regions_for_panels(parse_panels(PANELS))
    conn, series, chapter, profile, text, _vision, engine, assembled = harness
    router = Router()
    info = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    monkeypatch.setattr(
        pipeline, "get_vision_model", lambda c, p: Selection(adapter=router, info=info)
    )
    out = pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), video_mode="scroll", log=lambda m: None,
    )
    assert out.exists()
    workdir = works.video_recap_dir("s1", "ch-1")
    script_segs, _ = load_script(workdir / "script.json")
    assert script_segs[0].regions == [
        {"page": 1, "kind": "hold", "box": list(expected[0].box)}
    ]
    assert script_segs[1].regions == [
        {"page": 1, "kind": "hold", "box": list(expected[1].box)}
    ]
    state = json.loads((workdir / "render_state.json").read_text())
    assert state["grounded"] is True and state["mode"] == "scroll"
    assert (works.chapter_dir("s1", "ch-1") / "grounding.json").exists()
    assert (works.chapter_dir("s1", "ch-1") / "panels.json").exists()
    calls = router.calls

    # Second run: script carries regions -> grounding stage skips entirely.
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), video_mode="scroll", log=lambda m: None,
    )
    assert router.calls == calls


def test_build_video_kenburns_skips_grounding(harness):
    conn, series, chapter, profile, *_ = harness
    pipeline.build_video(
        conn, series, chapter, _artifact_config(), profile,
        client=FakeClient(), log=lambda m: None,
    )
    import json

    workdir = works.video_recap_dir("s1", "ch-1")
    state = json.loads((workdir / "render_state.json").read_text())
    assert "grounded" not in state  # kenburns assembly untouched
    assert not (works.chapter_dir("s1", "ch-1") / "grounding.json").exists()
    script_segs, _ = load_script(workdir / "script.json")
    assert all(s.regions == [] for s in script_segs)
