"""Video pipeline orchestration: script -> narration -> visuals -> assembly.

Five staged, individually cached steps under
works/<series-id>/chapters/<chapter-id>/video-recap|video-narration/
(script, TTS narration, page assignment, assembly, optional compression):
each stage skips when its output artifact exists, so re-runs and partial
regeneration are cheap. Mirrors recap.py: models are resolved via the
registry only. An optional colorization pass (DDColor, off by default) sits
between visuals and assembly; render_state.json records which treatment
produced out.mp4 so toggling it re-renders instead of serving stale video.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.config import Config
from entertainment_harness.db import utcnow
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.registry import (
    get_judge_model,
    get_text_model,
    get_vision_model,
)
from entertainment_harness.pipelines.judge import ATTEMPTS, MAX_ATTEMPTS
from entertainment_harness.pipelines.recap import (
    chapter_pages,
    model_tag,
    video_kind_for_detail,
)
from entertainment_harness.sources.mangadex import MangaDexClient
from entertainment_harness.video.assemble import (
    assemble,
    first_page_image,
    mux_clips,
    pacing_stats,
    video_duration,
)
from entertainment_harness.video.cards import render_cards
from entertainment_harness.video.compress import compress as compress_video
from entertainment_harness.video.credits import CREDITS_SECONDS, Credits
from entertainment_harness.video.gen import (
    get_provider as get_video_gen_provider,
)
from entertainment_harness.video.frames import (
    get_animator as get_frame_animator,
    resolve_image_model,
)
from entertainment_harness.video.grounding import ground_segments
from entertainment_harness.video.panelfirst import build_panel_first_script
from entertainment_harness.video.script import (
    VideoError,
    apply_pauses,
    generate_script,
    load_script,
    save_script,
    segments_from_narration,
)
from entertainment_harness.video.tts import get_engine, render_narration
from entertainment_harness.video.visuals import MOTIONS, assign_pages


def _parse_resolution(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    except ValueError:
        raise VideoError(f"Bad resolution {value!r}; expected e.g. 1920x1080")


def _card_paths(workdir: Path, count: int) -> list[Path]:
    """Caption-card images, numbered like pages so assembly can treat cards
    as the segment's page list (render_cards writes the same layout)."""
    return [workdir / "cards" / f"page-{i + 1:03d}.png" for i in range(count)]


# Assembly-level finishing cached out.mp4 files don't reflect. Bump when it
# changes (inter-beat pauses, loudness normalization); a render_state.json
# mismatch re-renders from the cached script/pages instead of serving stale.
PACING_VERSION = 1

VIDEO_MODES = (
    "kenburns", "scroll", "slideshow", "cards", "panels", "motion", "animate",
    "sequence",
)


def clear_downstream(workdir: Path, narration: bool = False) -> None:
    """Delete clips/out.mp4 (and optionally narration WAVs) after an upstream
    stage re-ran, so stale artifacts are never reused. A narration reset means
    the script changed, so caption cards (baked segment text) go too."""
    if narration:
        for wav in workdir.glob("seg-*.wav"):
            wav.unlink()
        for gap in workdir.glob("gap-*.wav"):
            gap.unlink()
        shutil.rmtree(workdir / "cards", ignore_errors=True)
    for clip in (workdir / "clips").glob("clip-*.mp4"):
        clip.unlink()
    for strip in (workdir / "clips").glob("strip-*.png"):
        strip.unlink()
    for panel in (workdir / "clips").glob("panel-*.png"):
        panel.unlink()
    for stale in (workdir / "out.mp4", workdir / "subs.srt"):
        if stale.exists():
            stale.unlink()


def log_word_count(segments, log) -> None:
    """Log the segment/word-count summary after a fresh script is written."""
    total_words = sum(len(s.text.split()) for s in segments)
    log(
        f"  {len(segments)} segments, ~{total_words} words "
        f"(~{total_words / 150:.1f} min narrated)"
    )


def tts_stage(
    segments, script_path: Path, title: str, chapter_num: float,
    script_model: str, engine, workdir: Path, voice: str | None, log,
) -> None:
    """Stage 2 (TTS narration), shared by build_video and build_short:
    render missing segment audio, clear downstream when new audio timing
    landed, and persist durations to the script."""
    chosen_voice = voice or engine.default_voice
    wavs = [workdir / f"seg-{s.index:02d}.wav" for s in segments]
    if all(w.exists() for w in wavs):
        log(f"Stage 2/4 narration ({engine.name}): cached")
    elif (workdir / "out.mp4").exists() and all(s.duration_s for s in segments):
        # Segment audio was pruned after a successful assembly. The script
        # carries the authoritative durations; re-synthesizing here would
        # invalidate the cached video for nothing.
        log(f"Stage 2/4 narration ({engine.name}): pruned (video cached)")
        return
    else:
        log(f"Stage 2/4 narration ({engine.name}): rendering, voice={chosen_voice}...")
    if render_narration(segments, engine, chosen_voice, workdir, log):
        clear_downstream(workdir)  # new audio timing invalidates clips/out
    save_script(script_path, segments, title, chapter_num, script_model)


def prune_intermediates(workdir: Path, log=lambda m: None) -> None:
    """Delete concat inputs (per-page clips, segment WAVs) once a final video
    exists. Both are regenerable — clips re-render from the cached page/motion
    picks, WAVs from the TTS engine — and their absence is what the existing
    re-render paths key on."""
    clips_dir = workdir / "clips"
    if clips_dir.is_dir():
        shutil.rmtree(clips_dir)
        log("  pruned clips/")
    wavs = list(workdir.glob("seg-*.wav")) + list(workdir.glob("gap-*.wav"))
    for wav in wavs:
        wav.unlink()
    if wavs:
        log(f"  pruned {len(wavs)} segment WAV(s)")


def _record_video(
    conn: sqlite3.Connection,
    series_id: str,
    chapter_id: str,
    chapter_num: float,
    kind: str,
    final_path: Path,
    duration: float,
    engine_name: str,
    script_model: str,
    rendered: bool,
    video_gen_provider: str | None = None,
    compress: str | None = None,
) -> None:
    """Upsert the videos row for one chapter video after a render/compress."""
    now = utcnow()
    existing = conn.execute(
        "SELECT id, path FROM videos WHERE series_id = ? AND from_chapter = ?"
        " AND to_chapter = ? AND kind = ?",
        (series_id, chapter_num, chapter_num, kind),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, tts_engine, model, kind)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (series_id, chapter_num, chapter_num, str(final_path), duration,
             now, engine_name, script_model, kind),
        )
    elif existing["path"] != str(final_path) or rendered:
        conn.execute(
            "UPDATE videos SET path = ?, duration_s = ?, created_at = ?,"
            " tts_engine = ?, model = ?, kind = ?, wiped_at = NULL WHERE id = ?",
            (str(final_path), duration, now, engine_name, script_model,
             kind, existing["id"]),
        )
    conn.commit()
    works.write_video_metadata(
        series_id,
        chapter_id,
        works.VideoMetadata(
            kind=kind,
            duration_s=duration,
            model=script_model,
            tts_engine=engine_name,
            created_at=now,
            video_gen_provider=video_gen_provider,
            compress=compress,
        ),
    )


def _supersede_other_kind(
    conn: sqlite3.Connection,
    series_id: str,
    chapter_id: str,
    chapter_num: float,
    kind: str,
    log,
) -> None:
    """Enforce one video per chapter after a build at `kind`: the other
    kind's workdir is pruned and its videos row deleted — a detail change
    supersedes the old grain's video."""
    other = "recap" if kind == "narration" else "narration"
    other_dir = works.video_dir_for_kind(series_id, chapter_id, other)
    had_dir = other_dir.is_dir()
    if had_dir:
        shutil.rmtree(other_dir)
    cur = conn.execute(
        "DELETE FROM videos WHERE series_id = ? AND from_chapter = ?"
        " AND to_chapter = ? AND kind = ?",
        (series_id, chapter_num, chapter_num, other),
    )
    if had_dir or cur.rowcount:
        conn.commit()
        log(f"  superseded the {other} video (one video per chapter)")


def build_video(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    chapter: sqlite3.Row,
    config: Config,
    profile: HardwareProfile,
    voice: str | None = None,
    engine_name: str | None = None,
    colorize: bool | None = None,
    translated: bool | None = None,
    compress: str | None = None,
    video_mode: str | None = None,
    source: str | None = None,
    panel_first: bool | None = None,
    client: MangaDexClient | None = None,
    progress=None,
    log=lambda m: None,
) -> Path:
    """Build (or reuse) the chapter's one video. Returns the final mp4.

    One video per chapter (unify-recap-narrate): the video kind follows the
    chapter artifact's detail (recaps table). detail='full' is the old
    narration behavior — the spoken-form retelling is split into segments
    verbatim, no script model — and renders in the video-narration workdir
    with videos.kind = 'narration'; every lower grain goes through the
    script model in video-recap. `source` ("recap"/"narration") overrides
    the derived kind for callers that need to force it. `panel_first`
    (anchored-scroll Phase 4; ON BY DEFAULT — [video] panel_first=false or
    --no-panel-first opts out) forces
    kind='narration' and replaces stage 1: the chapter's cached panel beats
    are grouped into narration segments born with their panel spans, so
    page assignment and region grounding are skipped; no artifact is
    required (the recap row only supplies the steering instruction).
    panel_first and source= are mutually exclusive. A successful build
    supersedes the other kind's video: its workdir is pruned and its videos
    row deleted, so one row per chapter remains.

    With translated=True, stages 3-4 read the chapter's translated pages
    (works/<slug>/chapters/ch-NNN/translated/, produced by the recap
    pipeline's translation step) instead of the originals; the chapter must
    be translated first. With a compression preset (see video/compress.py),
    the master out.mp4 is re-encoded to out-<preset>.mp4 and the compressed
    file is returned instead; the preset comes from --compress or the
    [video] compress= config default. The presentation mode (kenburns or
    scroll) comes from --video-mode or [video] mode=; render_state.json
    records it plus the artifact's detail, so toggling either re-renders.
    Concat inputs (clips/, seg-*.wav) are pruned once a final video exists;
    with keep_master = false the master is pruned too. progress, if given,
    receives stage() updates for the live UI (see ui.py).
    """
    if source not in (None, "recap", "narration"):
        raise VideoError(f"Bad video source {source!r}; expected recap or narration")
    panel_first_enabled = (
        config.video.panel_first if panel_first is None else panel_first
    )
    if panel_first_enabled and source is not None:
        raise VideoError("panel_first and source= are mutually exclusive")
    chapter_num = chapter["chapter_num"]
    title = series["title"]

    # The chapter's artifact (recaps row at its detail grain) drives the
    # video kind; the legacy narrations table only backs pre-merge rows the
    # fold-in migration hasn't touched. The artifact's steering instruction
    # feeds the panel-first grouping (Phase 4).
    artifact = conn.execute(
        "SELECT summary, detail, model, instruction FROM recaps"
        " WHERE chapter_id = ?",
        (chapter["id"],),
    ).fetchone()
    legacy = conn.execute(
        "SELECT text, model FROM narrations WHERE chapter_id = ?",
        (chapter["id"],),
    ).fetchone()
    detail = (
        artifact["detail"] if artifact is not None
        else "full" if legacy is not None
        else "standard"
    )
    instruction = (
        str(artifact["instruction"] or "") if artifact is not None else ""
    )
    kind = (
        "narration" if panel_first_enabled
        else source or video_kind_for_detail(detail)
    )
    workdir = works.video_dir_for_kind(series["id"], chapter["id"], kind)
    script_path = workdir / "script.json"
    out_path = workdir / "out.mp4"

    # CLI flag wins; otherwise the [video] compress= default applies.
    preset = compress if compress is not None else (config.video.compress or None)
    colorize_enabled = config.video.colorize if colorize is None else colorize
    translated_enabled = config.video.translated if translated is None else translated
    mode = video_mode or config.video.mode
    if mode not in VIDEO_MODES:
        raise VideoError(
            f"Bad video mode {mode!r}; expected one of: {', '.join(VIDEO_MODES)}"
        )
    is_book = series["kind"] == "book" if "kind" in series.keys() else False
    if is_book and mode != "cards":
        if video_mode is None and config.video.mode == "kenburns":
            # Books have no pages; caption cards are the one rendering that
            # works. Default to them unless the user chose a mode.
            mode = "cards"
            log("Book work: defaulting to video mode 'cards'"
                " (caption cards — books have no pages).")
        else:
            raise VideoError(
                "Books have no pages to render — use video mode 'cards'"
                f" (got {mode!r})."
            )
    credits_enabled = config.video.credits
    wanted = {
        "colorize": colorize_enabled,
        "translated": translated_enabled,
        "mode": mode,
        "detail": detail,
        "pacing": PACING_VERSION,
        # Render states from before the flag existed default to False via
        # state.get(k, False), so plain builds don't spuriously re-render.
        "panel_first": panel_first_enabled,
        # Old renders predate the end card: with credits on they mismatch
        # once and re-mux to gain it (clips/TTS are cached — assembly only).
        "credits": credits_enabled,
    }
    if mode in ("scroll", "panels", "animate", "sequence"):
        # Anchored renders (Phase 3 grounding): scroll renders from before
        # grounding existed lack the key and re-render once; kenburns
        # assembly is untouched, so it never carries the key. Panels and
        # animate renders are grounded by definition — no legacy ones.
        wanted["grounded"] = True
    if mode == "motion":
        # Generated clips depend on the provider too: switching providers
        # re-renders, like switching modes.
        wanted["video_gen"] = config.video_gen.provider
    if mode == "animate":
        # Same for the frame animator behind animate mode: the provider and
        # the image model both change the generated frames.
        wanted["frames"] = (
            f"{config.frames.provider}:"
            f"{resolve_image_model(config.frames.provider, config.frames.model)}"
        )
    if mode == "sequence":
        # Same for the sequence animator: provider, image model, and frame
        # rate all change the generated frames.
        wanted["sequence"] = (
            f"{config.sequence.provider}:"
            f"{resolve_image_model(config.sequence.provider, config.sequence.model)}"
            f":{config.sequence.fps}"
        )
    state_path = workdir / "render_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    if preset is not None and script_path.exists():
        # A matching compressed copy with an intact script is the deliverable;
        # without this early-out, a pruned master (keep_master = false) would
        # needlessly re-run TTS + assembly just to be compressed again.
        compressed_path = workdir / f"out-{preset}.mp4"
        if (
            compressed_path.exists()
            and state.get("compress") == preset
            and {k: state.get(k, False) for k in wanted} == wanted
        ):
            log(f"compress ({preset}): cached")
            _supersede_other_kind(
                conn, series["id"], chapter["id"], chapter_num, kind, log
            )
            return compressed_path

    manga_dir = works.source_dir(series["id"], chapter["id"])
    translated_dir = works.translated_dir(series["id"], chapter["id"])
    if translated_enabled and mode != "cards" and (
        not translated_dir.is_dir() or not any(translated_dir.glob("*.png"))
    ):
        raise VideoError(
            f"Chapter {chapter_num:g} has no translated pages —"
            " run 'eh recap --translated' first."
        )

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def _page_paths() -> list[Path]:
        if translated_enabled:
            return sorted(
                p for p in translated_dir.glob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        return sorted(
            p for p in manga_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    # --- stage 1: narration script -----------------------------------------
    if progress is not None:
        progress.stage("video", "script")
    cached = None
    if script_path.exists():
        loaded_segments, loaded_model = load_script(script_path)
        if loaded_model.startswith("panel-first") != panel_first_enabled:
            # Panel-first shares the video-narration workdir with the
            # artifact-split path; a script written by the other stage-1
            # path is not this run's script — regenerate it.
            log("Stage 1/4 script: cached script is from the other stage-1"
                " path, regenerating...")
            script_path.unlink()
        else:
            cached = (loaded_segments, loaded_model)
    if cached is not None:
        segments, script_model = cached
        log(f"Stage 1/4 script: cached ({len(segments)} segments)")
    elif panel_first_enabled:
        # Anchored-scroll Phase 4: the chapter's cached panel beats are
        # grouped into narration segments born with their panel spans — no
        # artifact, page assignment, or grounding judge required.
        clear_downstream(workdir, narration=True)
        vision = get_vision_model(config, profile)
        text = get_text_model(config, profile)
        for selection in (vision, text):
            if selection.warning:
                log(f"Warning: {selection.warning}")
        thinking = config.pipeline.thinking
        judge = None
        if thinking != "low":
            judge = get_judge_model(config, profile)
            if judge.warning:
                log(f"Warning: {judge.warning}")
        log(f"Stage 1/4 script: panel-first grouping with"
            f" {model_tag(text.info)}...")
        text.adapter.ensure(text.info.name)
        segments, status = build_panel_first_script(
            vision, text, judge, _page_paths(), series, chapter,
            instruction=instruction,
            max_attempts=ATTEMPTS.get(thinking, MAX_ATTEMPTS),
            vision_verify=thinking == "high",
            log=log,
        )
        script_model = f"panel-first ({model_tag(text.info)})"
        save_script(script_path, segments, title, chapter_num, script_model)
        log(f"  panel-first grouping: {status}")
        log_word_count(segments, log)
    elif kind == "narration":
        # detail='full': the artifact is already spoken-form — split it into
        # segments verbatim (no script model). The recaps row is
        # authoritative; the legacy narrations row is the pre-merge fallback.
        if artifact is not None and artifact["detail"] == "full":
            narration_text, narration_model = artifact["summary"], artifact["model"]
        elif legacy is not None:
            narration_text, narration_model = legacy["text"], legacy["model"]
        else:
            raise VideoError(
                f"Chapter {chapter_num:g} has no full-detail artifact"
                " — run 'eh recap --detail full' first."
            )
        # A new script invalidates everything downstream of it.
        clear_downstream(workdir, narration=True)
        log("Stage 1/4 script: splitting full-detail artifact into segments"
            " (no model)...")
        segments = segments_from_narration(narration_text)
        script_model = (
            f"narration ({narration_model})" if narration_model else "narration"
        )
        save_script(script_path, segments, title, chapter_num, script_model)
        log_word_count(segments, log)
    else:
        if artifact is None:
            raise VideoError(
                f"Chapter {chapter_num:g} has no recap — run 'eh recap' first."
            )
        # A new script invalidates everything downstream of it.
        clear_downstream(workdir, narration=True)
        text = get_text_model(config, profile)
        if text.warning:
            log(f"Warning: {text.warning}")
        log(f"Stage 1/4 script: writing with {model_tag(text.info)}...")
        text.adapter.ensure(text.info.name)
        cast = ""
        if config.pipeline.characters:
            # Registry names (character-bible): beats keep canonical names
            # even where the recap leaves a recurring character unnamed.
            from entertainment_harness.pipelines.characters import (
                cast_script_block,
                load_cast_block,
            )
            cast = load_cast_block(conn, series["id"], block=cast_script_block)
        segments = generate_script(
            text.adapter, text.info.name, artifact["summary"], title,
            chapter_num, cast=cast,
        )
        script_model = model_tag(text.info)
        save_script(script_path, segments, title, chapter_num, script_model)
        log_word_count(segments, log)

    apply_pauses(segments)  # breathing room between beats (deterministic)

    # Render-setting changes invalidate the finished video here — before TTS —
    # so segment WAVs pruned after a previous assembly regenerate instead of
    # failing the re-render at mux time.
    if out_path.exists() and {k: state.get(k, False) for k in wanted} != wanted:
        log("Stage 4/4 assembly: render settings changed, re-rendering...")
        clear_downstream(workdir)
        shutil.rmtree(workdir / "colorized", ignore_errors=True)

    # --- stage 2: narration (audio durations are authoritative) ------------
    if progress is not None:
        progress.stage("video", "narration")
    engine = get_engine(engine_name or config.video.tts_engine, config)
    tts_stage(
        segments, script_path, title, chapter_num, script_model,
        engine, workdir, voice, log,
    )

    # --- stage 3: page assignment + Ken Burns specs -------------------------
    if progress is not None:
        progress.stage("video", "visuals")
    if mode == "cards":
        # Caption cards instead of manga pages: one card per segment, then
        # assembly treats the cards as the page list. This is the only stage
        # 3 that works for books (no pages to pick).
        card_paths = _card_paths(workdir, len(segments))
        if all(s.pages and s.motion for s in segments) and all(
            p.exists() for p in card_paths
        ):
            log("Stage 3/4 visuals: cached")
        else:
            log("Stage 3/4 visuals: rendering caption cards...")
            render_cards(
                segments, title, workdir,
                size=_parse_resolution(config.video.resolution),
                cover=works.find_cover(series["id"]),
            )
            for i, seg in enumerate(segments):
                seg.pages = [i + 1]
                seg.motion = "zoom_in"
            clear_downstream(workdir)  # new visuals invalidate clips/out
            save_script(script_path, segments, title, chapter_num, script_model)
    elif panel_first_enabled:
        # Segments were born with their panel spans at grouping time; there
        # is nothing to assign (and stage 3.5 finds regions already set).
        log("Stage 3/4 visuals: panel-first (spans attached at grouping)")
    elif all(s.pages and s.motion for s in segments):
        log("Stage 3/4 visuals: cached")
    else:
        if translated_enabled:
            page_paths = _page_paths()
        else:
            def _client_factory():
                nonlocal client
                if client is None:
                    from entertainment_harness.sources import get_client

                    client = get_client(series["source"])
                return client

            page_paths = chapter_pages(
                config, _client_factory, series["id"],
                chapter["id"], manga_dir, log,
            )
        vision = get_vision_model(config, profile)
        if vision.warning:
            log(f"Warning: {vision.warning}")
        log(f"Stage 3/4 visuals: page picking with {model_tag(vision.info)}...")
        vision.adapter.ensure(vision.info.name)
        assign_pages(
            vision.adapter, vision.info.name, segments, page_paths,
            workdir, title, chapter_num, log,
        )
        clear_downstream(workdir)  # new page/motion picks invalidate clips/out
        save_script(script_path, segments, title, chapter_num, script_model)

    # --- stage 3.5: region grounding (scroll/panels/animate anchored renders)
    # Ground segments to panel regions (panels.json via Phase 2's extraction)
    # with a verifying judge; anchors are stamped on the segments and cached
    # per chapter (grounding.json). Ken Burns and slideshow skip this entirely.
    if mode in ("scroll", "panels", "animate", "sequence"):
        if progress is not None:
            progress.stage("video", "grounding")
        if all(s.regions for s in segments):
            log("Stage 3.5/4 grounding: cached")
        else:
            vision = get_vision_model(config, profile)
            if vision.warning:
                log(f"Warning: {vision.warning}")
            log(f"Stage 3.5/4 grounding with {model_tag(vision.info)}...")
            vision.adapter.ensure(vision.info.name)
            ground_segments(
                vision.adapter, vision.info.name, segments, _page_paths(),
                series["id"], chapter["id"], title, chapter_num, log=log,
            )
            clear_downstream(workdir)  # new anchors invalidate clips/out
            save_script(script_path, segments, title, chapter_num, script_model)

    # Motion is a render-time choice, not script data: restamp it from the
    # mode so cached script.json files render identically regardless of what
    # motion was persisted by an earlier run.
    for seg in segments:
        if mode in ("scroll", "slideshow", "panels", "animate", "sequence"):
            seg.motion = mode
        else:
            seg.motion = MOTIONS[seg.index % len(MOTIONS)]

    # --- stage 4: assembly ---------------------------------------------------
    if progress is not None:
        progress.stage("video", "assembly")
    stats = pacing_stats(segments)
    log(
        f"  pacing: {stats['segments']} beats, ~{stats['words']} words, "
        f"{stats['visual_cuts']} cuts, {stats['avg_seg_s']:.1f}s avg beat, "
        f"{stats['avg_s_per_visual']:.1f}s per visual"
    )

    rendered = False
    gen_provider_name = "local"  # stills assembly unless motion mode says otherwise
    credits = None
    if credits_enabled:
        meta = works.read_work_metadata(series["id"])
        credits = Credits(
            title=title,
            source=series["source"],
            chapter_label=f"Chapter {chapter_num:g}",
            author=meta.author if meta is not None else None,
        )
    if out_path.exists():
        log("Stage 4/4 assembly: cached")
        duration = sum(s.duration_s + s.pause_after_s for s in segments)
        if state.get("credits"):
            duration += CREDITS_SECONDS
    else:
        if mode == "cards":
            render_paths = _card_paths(workdir, len(segments))
        else:
            page_source_dir = translated_dir if translated_enabled else manga_dir
            if not page_source_dir.exists():
                raise VideoError(
                    f"Pages for chapter {chapter_num:g} are not cached — "
                    "run 'eh recap' first."
                )
            page_paths = _page_paths()
            render_paths = page_paths
            if colorize_enabled:
                from entertainment_harness.video.colorize import get_colorizer

                used = sorted({p for s in segments for p in (s.pages or [])})
                log(f"Stage 3.5/4 colorize: {config.colorize.provider} on"
                    f" {len(used)} pages...")
                colorizer = get_colorizer(config.colorize.provider, config)
                mapping = colorizer.colorize_pages(
                    [page_paths[i - 1] for i in used], workdir / "colorized", log
                )
                render_paths = [mapping.get(p, p) for p in page_paths]
        log("Stage 4/4 assembly: rendering...")
        if mode == "motion":
            gen = get_video_gen_provider(config.video_gen.provider, config)
            gen_provider_name = gen.name
            if gen.animated:
                # Generated clips may run shorter/longer than the narration;
                # segment durations are re-stamped from the actual clips
                # (short.py's pattern), so cues follow the video.
                log(f"Stage 4/4 assembly: generating clips with {gen.name}...")
                clips: list[Path] = []
                for seg in segments:
                    clip = gen.generate_segment(
                        first_page_image(render_paths, seg), seg,
                        seg.duration_s, workdir,
                    )
                    seg.duration_s = video_duration(clip)
                    clips.append(clip)
                    log(f"  segment {seg.index:02d}: generated"
                        f" ({seg.duration_s:.1f}s)")
                out_path, duration = mux_clips(segments, clips, workdir, log,
                                               credits=credits)
            else:
                log(f"Stage 4/4 assembly: provider {gen.name} makes stills,"
                    " not video — rendering with the stills assembly...")
                out_path, duration = assemble(
                    segments, render_paths, workdir,
                    _parse_resolution(config.video.resolution), log,
                    credits=credits,
                )
        elif mode == "animate":
            animator = get_frame_animator(config.frames.provider, config)
            if animator.animated:
                gen_provider_name = animator.name
                log(f"Stage 4/4 assembly: animating panels with"
                    f" {animator.name} frames...")
            else:
                log(f"Stage 4/4 assembly: [frames] provider"
                    f" {animator.name} generates no frames — rendering as"
                    " panels mode...")
            out_path, duration = assemble(
                segments, render_paths, workdir,
                _parse_resolution(config.video.resolution), log,
                frame_animator=animator,
                credits=credits,
            )
        elif mode == "sequence":
            animator = get_frame_animator(config.sequence.provider, config)
            if animator.animated:
                gen_provider_name = animator.name
                log(f"Stage 4/4 assembly: frame-by-frame sequences with"
                    f" {animator.name} at {config.sequence.fps:g} fps...")
                critic = None
                if config.sequence.critic:
                    try:
                        vision = get_vision_model(config, profile)
                        if vision.warning:
                            log(f"Warning: {vision.warning}")
                        vision.adapter.ensure(vision.info.name)
                        from entertainment_harness.video.sequence import (
                            drift_critic,
                        )
                        critic = drift_critic(vision.adapter, vision.info.name)
                        log(f"  drift critic: {model_tag(vision.info)}")
                    except Exception as exc:
                        log(f"  drift critic unavailable ({exc}) —"
                            " continuing without it")
            else:
                log(f"Stage 4/4 assembly: [sequence] provider"
                    f" {animator.name} generates no frames — rendering as"
                    " panels mode...")
                critic = None
            out_path, duration = assemble(
                segments, render_paths, workdir,
                _parse_resolution(config.video.resolution), log,
                frame_animator=animator,
                sequence_interp_fps=config.sequence.interp_fps,
                sequence_critic=critic,
                credits=credits,
            )
        else:
            out_path, duration = assemble(
                segments, render_paths, workdir,
                _parse_resolution(config.video.resolution), log,
                credits=credits,
            )
        state = dict(wanted)
        state_path.write_text(json.dumps(state))
        rendered = True

    # --- stage 5: optional compression (from the master out.mp4) ------------
    final_path = out_path
    if preset is not None:
        if progress is not None:
            progress.stage("video", "compress")
        compressed_path = workdir / f"out-{preset}.mp4"
        if compressed_path.exists() and state.get("compress") == preset:
            log(f"Stage 5/5 compress ({preset}): cached")
        else:
            encode_src = out_path
            if not encode_src.exists():
                # Master was pruned; fall back to the newest compressed copy.
                copies = sorted(
                    workdir.glob("out-*.mp4"),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                if not copies:
                    raise VideoError(
                        f"Cannot compress {workdir}: no master or compressed copy"
                    )
                encode_src = copies[0]
                log(f"Stage 5/5 compress ({preset}): master gone,"
                    f" re-encoding from {encode_src.name} (generation loss)")
            log(f"Stage 5/5 compress ({preset}): encoding...")
            compress_video(encode_src, preset, workdir)
            state["compress"] = preset
            state_path.write_text(json.dumps(state))
        final_path = compressed_path
        if not config.video.keep_master and out_path.exists():
            out_path.unlink()
            log("  pruned master (keep_master = false)")

    if final_path.exists():
        prune_intermediates(workdir, log)

    _supersede_other_kind(conn, series["id"], chapter["id"], chapter_num, kind, log)
    if rendered or preset is not None:
        _record_video(
            conn, series["id"], chapter["id"], chapter_num, kind,
            final_path, duration, engine.name, script_model, rendered,
            video_gen_provider=gen_provider_name,
            compress=preset,
        )

    log(f"Done: {final_path} ({duration:.0f}s)")
    return final_path
