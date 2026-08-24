"""Whole-work vertical short-form ("TikTok") videos: 'eh tiktok'.

One 1080x1920, ~60-90 s narrated summary video per work. Reuses the chapter
pipeline's stages (script -> tts -> visuals -> assemble); the differences are
the script (short, hook-first, written from the rolling series context), the
page list (flattened across all chapters; generated caption cards for books),
faster cut pacing, and the vertical frame. Same staged-caching discipline:
script.json -> seg-NN.wav -> clips -> out.mp4, downstream artifacts cleared
when an upstream stage re-runs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.config import Config
from entertainment_harness.db import utcnow
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelAdapter
from entertainment_harness.models.registry import get_text_model, get_vision_model
from entertainment_harness.pipelines.recap import model_tag
from entertainment_harness.video.assemble import (
    assemble,
    first_page_image,
    mux_clips,
    video_duration,
)
from entertainment_harness.video.cards import render_cards
from entertainment_harness.video.gen import get_provider as get_video_gen_provider
from entertainment_harness.video.pipeline import (
    clear_downstream,
    log_word_count,
    tts_stage,
)
from entertainment_harness.video.script import (
    Segment,
    VideoError,
    load_script,
    parse_script,
    save_script,
)
from entertainment_harness.video.tts import get_engine
from entertainment_harness.video.visuals import assign_pages

RESOLUTION = (1080, 1920)  # 9:16 vertical
MIN_PAGE_SECONDS = 1.5  # faster cuts than the landscape recap (2.5 s)

SHORT_SCRIPT_PROMPT = """You are writing the narration script for a 60-90 second vertical short-form video (TikTok-style) summarizing the story "{title}".

Story summary:
---
{summary}
---

Requirements:
- 8 to 12 segments; 150-230 words in total.
- Segment 1 MUST be a single-sentence hook that makes the viewer want to hear the rest (a question or bold claim about the story — no clickbait lies).
- Every later segment is 15-30 words of punchy spoken English, present tense, no headers, no stage directions.
- Faithful to the summary: no new events, names, or details.
- Tag each segment with a short "moment" label naming the story beat (e.g. "Shin hunts the boar").
{steering}
Output ONLY a JSON array, no other text:
[{{"text": "...", "moment": "..."}}, ...]"""


def generate_short_script(
    adapter: ModelAdapter,
    model: str,
    summary: str,
    title: str,
    steering_prompt: str = "",
) -> list[Segment]:
    steering = (
        f"\nSteering direction for the video: {steering_prompt}\n"
        if steering_prompt
        else ""
    )
    prompt = SHORT_SCRIPT_PROMPT.format(
        title=title, summary=summary, steering=steering
    )
    return parse_script(adapter.generate(model, prompt))


def _flatten_pages(conn: sqlite3.Connection, series_id: str) -> list[Path]:
    """All cached chapter pages, ordered by chapter then page — one flat
    list for the visuals stage (contact sheets span chapters naturally)."""
    rows = conn.execute(
        "SELECT id FROM chapters WHERE series_id = ? ORDER BY chapter_num",
        (series_id,),
    ).fetchall()
    paths: list[Path] = []
    for row in rows:
        chapter_dir = works.source_dir(series_id, row["id"])
        if chapter_dir.is_dir():
            paths.extend(p for p in sorted(chapter_dir.iterdir()) if p.is_file())
    return paths


def build_short(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    config: Config,
    profile: HardwareProfile,
    voice: str | None = None,
    engine_name: str | None = None,
    video_gen_provider: str | None = None,
    steering_prompt: str | None = None,
    log=lambda m: None,
) -> Path:
    """Build (or reuse) the whole-work short-form video. Returns out.mp4."""
    kind = series["kind"] if "kind" in series.keys() else "manga"
    title = series["title"]
    workdir = works.tiktok_dir(series["id"])
    script_path = workdir / "script.json"
    out_path = workdir / "out.mp4"
    chosen_gen = video_gen_provider or config.video_gen.provider
    chosen_steering = steering_prompt if steering_prompt is not None else config.video.steering_prompt

    context = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = ?",
        (series["id"],),
    ).fetchone()
    recapped = conn.execute(
        "SELECT COUNT(*) AS n FROM recaps r JOIN chapters c ON r.chapter_id = c.id"
        " WHERE c.series_id = ?",
        (series["id"],),
    ).fetchone()["n"]
    has_online = conn.execute(
        "SELECT 1 FROM online_summaries WHERE series_id = ?",
        (series["id"],),
    ).fetchone() is not None
    if context is None or (recapped == 0 and not has_online):
        raise VideoError(
            f"{title!r} has no recaps or online summary yet — run"
            " 'eh recap --all' first or use 'eh tiktok <title>'."
        )

    # --- stage 1: short script ---------------------------------------------
    if script_path.exists():
        segments, script_model = load_script(script_path)
        log(f"Stage 1/4 script: cached ({len(segments)} segments)")
    else:
        clear_downstream(workdir, narration=True)
        text = get_text_model(config, profile)
        if text.warning:
            log(f"Warning: {text.warning}")
        if recapped:
            coverage = f"{recapped} recap(s)"
        elif has_online:
            coverage = "online summary"
        else:
            coverage = "unknown"
        log(f"Stage 1/4 script: writing with {model_tag(text.info)}"
            f" (summary covers {coverage})...")
        text.adapter.ensure(text.info.name)
        segments = generate_short_script(
            text.adapter,
            text.info.name,
            context["rolling_summary"],
            title,
            steering_prompt=chosen_steering,
        )
        script_model = model_tag(text.info)
        save_script(script_path, segments, title, 0, script_model)
        log_word_count(segments, log)

    # --- stage 2: narration (audio durations are authoritative) ------------
    engine = get_engine(engine_name or config.video.tts_engine, config)
    tts_stage(
        segments, script_path, title, 0, script_model,
        engine, workdir, voice, log,
    )

    # --- stage 3: visuals ----------------------------------------------------
    if kind == "book":
        card_paths = [
            workdir / "cards" / f"page-{i + 1:03d}.png" for i in range(len(segments))
        ]
        if all(s.pages and s.motion for s in segments) and all(
            p.exists() for p in card_paths
        ):
            log("Stage 3/4 visuals: cached")
        else:
            log("Stage 3/4 visuals: rendering caption cards...")
            render_cards(
                segments, title, workdir, size=RESOLUTION,
                cover=works.find_cover(series["id"]),
            )
            for i, seg in enumerate(segments):
                seg.pages = [i + 1]
                seg.motion = "zoom_in"
            clear_downstream(workdir)
            save_script(script_path, segments, title, 0, script_model)
        render_paths = card_paths
    else:
        page_paths = _flatten_pages(conn, series["id"])
        if not page_paths:
            raise VideoError(
                f"No pages cached for {title!r} — run 'eh recap --all' first"
                " (or restore pages with 'eh store pull')."
            )
        if all(s.pages and s.motion for s in segments):
            log("Stage 3/4 visuals: cached")
        else:
            vision = get_vision_model(config, profile)
            if vision.warning:
                log(f"Warning: {vision.warning}")
            log(f"Stage 3/4 visuals: page picking with {model_tag(vision.info)}"
                f" ({len(page_paths)} pages)...")
            vision.adapter.ensure(vision.info.name)
            assign_pages(
                vision.adapter, vision.info.name, segments, page_paths,
                workdir, title, 0, log,
                where=f'the story "{title}"',
            )
            clear_downstream(workdir)  # new page/motion picks invalidate clips/out
            save_script(script_path, segments, title, 0, script_model)
        render_paths = page_paths

    # --- stage 4: assembly ---------------------------------------------------
    if out_path.exists():
        log("Stage 4/4 assembly: cached")
        duration = sum(s.duration_s for s in segments)
    else:
        gen = get_video_gen_provider(chosen_gen, config)
        log(f"Stage 4/4 assembly: rendering (1080x1920) with {gen.name}...")
        if not gen.animated:
            out_path, duration = assemble(
                segments, render_paths, workdir, RESOLUTION, log,
                min_page_seconds=MIN_PAGE_SECONDS,
            )
        else:
            clips: list[Path] = []
            for seg in segments:
                image = first_page_image(render_paths, seg)
                clip = gen.generate_segment(image, seg, seg.duration_s, workdir)
                seg.duration_s = video_duration(clip)
                clips.append(clip)
                log(f"  segment {seg.index:02d}: generated ({seg.duration_s:.1f}s)")
            out_path, duration = mux_clips(segments, clips, workdir, log)
        state_path = workdir / "render_state.json"
        state_path.write_text(
            json.dumps({"format": "tiktok", "video_gen": gen.name})
        )
        now = utcnow()
        existing = conn.execute(
            "SELECT id FROM videos WHERE series_id = ?"
            " AND from_chapter IS NULL AND to_chapter IS NULL",
            (series["id"],),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
                " duration_s, created_at, tts_engine, model, video_gen_provider)"
                " VALUES (?, NULL, NULL, ?, ?, ?, ?, ?, ?)",
                (series["id"], str(out_path), duration, now, engine.name,
                 script_model, gen.name),
            )
        else:
            conn.execute(
                "UPDATE videos SET path = ?, duration_s = ?, created_at = ?,"
                " tts_engine = ?, model = ?, video_gen_provider = ? WHERE id = ?",
                (str(out_path), duration, now, engine.name, script_model, gen.name,
                 existing["id"]),
            )
        conn.commit()
        works.write_video_metadata(
            series["id"],
            None,
            works.VideoMetadata(
                kind="tiktok",
                duration_s=duration,
                model=script_model,
                tts_engine=engine.name,
                created_at=now,
                video_gen_provider=gen.name,
            ),
        )

    log(f"Done: {out_path} ({duration:.0f}s)")
    return out_path
