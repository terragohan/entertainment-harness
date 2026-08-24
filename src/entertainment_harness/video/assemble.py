"""Stage 4: ffmpeg assembly.

Each segment's assigned pages are rendered into clips that together exactly
fill its narration length (Ken Burns motion: vertical pan or slow zoom-in;
scroll mode: the segment's pages are stacked into a strip the viewport
descends), clips are concatenated, narration is muxed, and the script is
embedded as a mov_text subtitle track (this ffmpeg build has no
libass/drawtext, so soft subs it is).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from entertainment_harness.video.script import (
    Segment,
    VideoConfigError,
    VideoError,
)

FPS = 30
AUDIO_BITRATE = "160k"
MIN_PAGE_SECONDS = 2.5  # never flip pages faster than this
LOUDNESS_TARGET = "I=-16:TP=-1.5:LRA=11"  # EBU R128 narration finishing

STRIP_GUTTER_PX = 24  # gap between stacked pages in scroll strips
STRIP_BACKGROUND = (64, 64, 64)  # neutral gray: gutters + side padding

GLIDE_S = 0.7  # viewport transition slice between anchored regions

XFADE_S = 0.5  # dissolve duration between the pages of one slideshow segment


def _slot_s(seg: Segment) -> float:
    """On-screen time for a segment: its narration plus the pause after it
    (the last page holds through the pause while the subtitle cue ends)."""
    return seg.duration_s + seg.pause_after_s


def video_duration(path: Path) -> float:
    """Return the duration of a video file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VideoError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return float(json.loads(result.stdout)["format"]["duration"])


def _segment_pages(seg: Segment, min_page_seconds: float = MIN_PAGE_SECONDS) -> list[int]:
    """Pages shown during a segment: its assigned pages, capped to a readable
    pace, evenly spaced so the picks span the segment's whole range (first-
    page-only would camp the video on the early chapter)."""
    pages = sorted(set(seg.pages)) or [1]
    max_pages = max(1, int(_slot_s(seg) // min_page_seconds))
    if len(pages) <= max_pages:
        return pages
    if max_pages == 1:
        return [pages[0]]
    step = (len(pages) - 1) / (max_pages - 1)
    return [pages[round(i * step)] for i in range(max_pages)]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise VideoError(f"ffmpeg failed ({cmd[0]} …):\n{tail}")


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[Segment], dest: Path) -> None:
    """Subtitle cues from authoritative audio durations. Cues cover speech
    only: the pause after a segment becomes a gap between cues, so subtitles
    never linger on screen over silence."""
    lines = []
    start = 0.0
    for i, seg in enumerate(segments, start=1):
        end = start + seg.duration_s
        lines.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{seg.text}\n")
        start = end + seg.pause_after_s
    dest.write_text("\n".join(lines))


def _gap_wav(workdir: Path, seconds: float, template: Path) -> Path:
    """A silence WAV matching the segment audio's format — concat-demuxer
    inputs must share codec parameters, so the gap clones a segment's."""
    dest = workdir / f"gap-{round(seconds * 1000)}ms.wav"
    if dest.exists():
        return dest
    with wave.open(str(template), "rb") as tpl:
        params = tpl.getparams()
    n_frames = round(seconds * params.framerate)
    with wave.open(str(dest), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(b"\x00" * n_frames * params.sampwidth * params.nchannels)
    return dest


def _clip_filter(page: Path, motion: str, duration: float,
                 width: int, height: int) -> tuple[str, int]:
    """Return (filtergraph, frame_count) for one segment clip."""
    frames = max(1, round(duration * FPS))
    if motion == "static":
        # Slideshow single page: an unmoving center-crop that fills the frame.
        return (
            f"fps={FPS},scale={width}:{height}"
            f":force_original_aspect_ratio=increase,crop={width}:{height}",
            frames,
        )
    if motion == "pan_down":
        with Image.open(page) as img:
            scaled_h = width * img.height / img.width
        if scaled_h > height * 1.05:
            # Tall page: fit width, drift top -> bottom at reading pace.
            return (
                f"fps={FPS},scale={width}:-2,"
                f"crop={width}:{height}:0:'(ih-{height})*t/{duration:.3f}'",
                frames,
            )
        # Page not tall enough to pan — zoom instead.
    return (
        f"fps={FPS},scale={width * 2}:-2,"
        f"zoompan=z='1+0.12*on/{frames}'"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={width}x{height}:fps={FPS}",
        frames,
    )


def strip_layout(pages: list[Path]) -> tuple[int, list[tuple[int, int]]]:
    """build_strip's stacking plan: (strip width, [(y_offset, page height)])
    per page, in strip order. Kept in one place so the hold-and-glide math
    maps region boxes onto the exact pixels the strip image has."""
    dims = []
    for p in pages:
        with Image.open(p) as img:
            dims.append((img.width, img.height))
    width = max(w for w, _ in dims)
    offsets = []
    y = 0
    for _, h in dims:
        offsets.append((y, h))
        y += h + STRIP_GUTTER_PX
    return width, offsets


def build_strip(pages: list[Path], dest: Path) -> Path:
    """Stack pages vertically in page order into one strip image for scroll
    mode: every page is centered/padded to the widest page's width, with a
    neutral-gray gutter between pages. Cached at dest."""
    if dest.exists():
        return dest
    width, offsets = strip_layout(pages)
    height = offsets[-1][0] + offsets[-1][1]
    strip = Image.new("RGB", (width, height), STRIP_BACKGROUND)
    for p, (y, _) in zip(pages, offsets):
        with Image.open(p) as img:
            strip.paste(img.convert("RGB"), ((width - img.width) // 2, y))
    dest.parent.mkdir(parents=True, exist_ok=True)
    strip.save(dest)
    return dest


def first_page_image(page_paths: list[Path], seg: Segment) -> Path:
    """The segment's lead illustration: its first assigned page, clamped to
    the page list (segments always have pages post-stage-3, but a hand-built
    or legacy script may not)."""
    page_num = (seg.pages or [seg.index + 1])[0]
    if page_num < 1 or page_num > len(page_paths):
        page_num = 1
    return page_paths[page_num - 1]


def _panel_crop(page: Path, box: list[float], dest: Path) -> Path:
    """Crop a normalized [x0, y0, x1, y1] region out of a page image for
    panels mode. Degenerate boxes (under 10% of the page on either axis once
    clamped) return the page itself — a sliver crop reads worse than the
    full page."""
    if dest.exists():
        return dest
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in box)
    if x1 - x0 < 0.1 or y1 - y0 < 0.1:
        return page
    with Image.open(page) as img:
        crop = img.convert("RGB").crop(
            (round(x0 * img.width), round(y0 * img.height),
             round(x1 * img.width), round(y1 * img.height))
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest)
    return dest


def build_clip(page: Path, duration: float, motion: str,
               resolution: tuple[int, int], dest: Path) -> None:
    width, height = resolution
    filtergraph, frames = _clip_filter(page, motion, duration, width, height)
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(page),
        "-vf", filtergraph, "-frames:v", str(frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", dest.as_posix(),
    ])


def build_xfade_clip(pages: list[Path], slot: float,
                     resolution: tuple[int, int], dest: Path,
                     fade_s: float = XFADE_S) -> None:
    """One clip for a slideshow segment: static center-crops that dissolve
    into each other. Per-page holds are padded so the xfade chain's total
    length equals the slot exactly — segment boundaries stay hard cuts, which
    keeps narration and subtitles aligned. A one-page segment is a plain
    static clip."""
    width, height = resolution
    n = len(pages)
    if n == 1:
        build_clip(pages[0], slot, "static", resolution, dest)
        return
    fade = min(fade_s, slot / (2 * n))  # a dissolve never swallows a page
    # xfade overlaps eat (n-1)*fade of runtime; padding each hold by that
    # much makes the chain end exactly at the slot.
    per_page = (slot + (n - 1) * fade) / n
    frames = max(1, round(slot * FPS))
    cmd = ["ffmpeg", "-y"]
    for p in pages:
        cmd += ["-loop", "1", "-t", f"{per_page:.3f}", "-i", str(p)]
    prep = ";".join(
        f"[{i}:v]fps={FPS},scale={width}:{height}"
        f":force_original_aspect_ratio=increase,crop={width}:{height},"
        f"setsar=1[v{i}]"
        for i in range(n)
    )
    links = []
    prev = "v0"
    for k in range(1, n):
        offset = k * (per_page - fade)
        out = f"x{k}" if k < n - 1 else "vout"
        links.append(
            f"[{prev}][v{k}]xfade=transition=fade"
            f":duration={fade:.3f}:offset={offset:.3f}[{out}]"
        )
        prev = out
    _run(cmd + [
        "-filter_complex", prep + ";" + ";".join(links),
        "-map", "[vout]", "-frames:v", str(frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", dest.as_posix(),
    ])


def build_sequence_clip(frames: list[Path], slot: float,
                        resolution: tuple[int, int], dest: Path,
                        interp_fps: int = FPS) -> None:
    """One clip for a sequence anchor: the generated frames concatenate at
    slot/N each (frame 0 is the outpainted expansion), then minterpolate
    motion-compensates up to interp_fps for smooth playback. Slot-exact,
    like build_xfade_clip — segment boundaries stay hard cuts, keeping
    narration and subtitles aligned. A one-frame sequence (the critic
    truncated the chain) is a plain static clip."""
    width, height = resolution
    n = len(frames)
    if n == 1:
        build_clip(frames[0], slot, "static", resolution, dest)
        return
    per_frame = slot / n
    out_frames = max(1, round(slot * interp_fps))
    cmd = ["ffmpeg", "-y"]
    for frame in frames:
        cmd += ["-loop", "1", "-t", f"{per_frame:.3f}", "-i", str(frame)]
    prep = ";".join(
        f"[{i}:v]scale={width}:{height}"
        f":force_original_aspect_ratio=increase,crop={width}:{height},"
        f"setsar=1[v{i}]"
        for i in range(n)
    )
    joined = "".join(f"[v{i}]" for i in range(n))
    filtergraph = (
        f"{prep};{joined}concat=n={n}:v=1:a=0[cat];"
        f"[cat]minterpolate=fps={interp_fps}[vout]"
    )
    _run(cmd + [
        "-filter_complex", filtergraph,
        "-map", "[vout]", "-frames:v", str(out_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", dest.as_posix(),
    ])


# --- hold-and-glide (anchored scroll, Phase 3) ------------------------------
#
# A grounded segment renders one strip clip whose viewport y follows
# piecewise-linear keyframes instead of the linear top->bottom descent: hold
# on each assigned region for its share of the slot, glide to the next
# region in a GLIDE_S slice; a "pan" anchor (ungrounded/failed page) glides
# down its whole page during its share, so fallbacks compose per page.


def _segment_anchors(
    seg: Segment, slot: float, page_count: int,
    min_anchor_seconds: float = MIN_PAGE_SECONDS,
) -> list[dict]:
    """Validated, page-ordered anchors from seg.regions, capped to a readable
    pace (evenly spaced picks survive, like _segment_pages). [] when nothing
    usable is stamped — the caller then takes today's linear path."""
    anchors = []
    for a in seg.regions:
        if not isinstance(a, dict):
            continue
        page, kind = a.get("page"), a.get("kind")
        if not isinstance(page, (int, float)) or not 1 <= int(page) <= page_count:
            continue
        if kind == "pan":
            anchors.append({"page": int(page), "kind": "pan"})
        elif kind == "hold":
            box = a.get("box")
            if (isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)):
                anchors.append({"page": int(page), "kind": "hold", "box": box})
    anchors.sort(key=lambda a: a["page"])  # stable: within-page reading order kept
    max_anchors = max(1, int(slot // min_anchor_seconds))
    if len(anchors) <= max_anchors:
        return anchors
    if max_anchors == 1:
        return anchors[:1]
    step = (len(anchors) - 1) / (max_anchors - 1)
    return [anchors[round(i * step)] for i in range(max_anchors)]


def hold_glide_keyframes(
    anchors: list[dict],
    strip_pages: list[int],
    layout: tuple[int, list[tuple[int, int]]],
    out_w: int,
    out_h: int,
    slot: float,
    glide_s: float = GLIDE_S,
) -> list[tuple[float, float]]:
    """(t, y) viewport keyframes in SCALED strip pixels (the strip is scaled
    to out_w before cropping, so all y math happens in that space).

    Each anchor gets an equal share of the slot; consecutive anchors are
    joined by a glide_s transition. A hold anchor pins the viewport so its
    region is vertically centered; a pan anchor glides from its page's top
    to its bottom across the share. Positions are clamped to the strip and
    forced non-decreasing — the viewport never scrolls back up.
    """
    index = {p: i for i, p in enumerate(strip_pages)}
    strip_w, offsets = layout
    scale = out_w / strip_w
    strip_h = (offsets[-1][0] + offsets[-1][1]) * scale
    max_y = max(0.0, strip_h - out_h)
    positions: list[tuple[float, float]] = []
    for a in anchors:
        y_off, page_h = offsets[index[a["page"]]]
        if a["kind"] == "hold":
            box = a["box"]
            center = (y_off + (box[1] + box[3]) / 2 * page_h) * scale
            y = min(max(center - out_h / 2, 0.0), max_y)
            positions.append((y, y))
        else:  # pan: descend the whole page
            top = min(y_off * scale, max_y)
            bottom = min(max((y_off + page_h) * scale - out_h, top), max_y)
            positions.append((top, bottom))
    monotone: list[tuple[float, float]] = []
    prev = 0.0
    for y0, y1 in positions:
        y0, y1 = max(y0, prev), max(y1, max(y0, prev))
        monotone.append((y0, y1))
        prev = y1
    positions = monotone

    n = len(positions)
    glide = glide_s if n > 1 else 0.0
    hold = (slot - glide * (n - 1)) / n
    keyframes = [(0.0, positions[0][0])]
    t = 0.0
    for i, (y0, y1) in enumerate(positions):
        t += hold
        keyframes.append((t, y1))
        if i + 1 < n:
            t += glide
            keyframes.append((t, positions[i + 1][0]))
    return keyframes


def _y_expression(keyframes: list[tuple[float, float]]) -> str:
    """ffmpeg crop y= expression for piecewise-linear keyframes: constant
    during holds, linear interpolation during glides/pans."""
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.1f}"
    expr = f"{keyframes[-1][1]:.1f}"
    for (t0, y0), (t1, y1) in reversed(list(zip(keyframes, keyframes[1:]))):
        if y0 == y1:
            piece = f"{y0:.1f}"
        else:
            slope = (y1 - y0) / (t1 - t0)
            piece = f"{y0:.1f}{slope:+.4f}*(t-{t0:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{piece},{expr})"
    return expr


def build_hold_glide_clip(strip: Path, duration: float,
                          keyframes: list[tuple[float, float]],
                          resolution: tuple[int, int], dest: Path) -> None:
    width, height = resolution
    frames = max(1, round(duration * FPS))
    filtergraph = (
        f"fps={FPS},scale={width}:-2,"
        f"crop={width}:{height}:0:'{_y_expression(keyframes)}'"
    )
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(strip),
        "-vf", filtergraph, "-frames:v", str(frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", dest.as_posix(),
    ])


def _concat_list(paths: list[Path], name: str, workdir: Path) -> Path:
    lst = workdir / name
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths))
    return lst


def mux_clips(
    segments: list[Segment],
    clips: list[Path],
    workdir: Path,
    log=lambda m: None,
) -> tuple[Path, float]:
    """Concatenate pre-rendered clips and mux with narration + subtitles.

    Returns (out.mp4, total_seconds). The caller is responsible for ensuring
    segment durations match the actual clip durations.
    """
    srt = workdir / "subs.srt"
    write_srt(segments, srt)

    no_subs = workdir / "out-nosubs.mp4"
    wavs: list[Path] = []
    for seg in segments:
        seg_wav = workdir / f"seg-{seg.index:02d}.wav"
        wavs.append(seg_wav)
        if seg.pause_after_s > 0:  # breathing room between beats
            wavs.append(_gap_wav(workdir, seg.pause_after_s, seg_wav))
    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(_concat_list(clips, "clips.txt", workdir)),
        "-f", "concat", "-safe", "0",
        "-i", str(_concat_list(wavs, "wavs.txt", workdir)),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-af", f"loudnorm={LOUDNESS_TARGET}", "-ar", "48000",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        str(no_subs),
    ])

    out = workdir / "out.mp4"
    _run([
        "ffmpeg", "-y", "-i", str(no_subs), "-i", str(srt),
        "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng", str(out),
    ])
    no_subs.unlink()
    duration = sum(_slot_s(seg) for seg in segments)
    return out, duration


def assemble(
    segments: list[Segment],
    page_paths: list[Path],
    workdir: Path,
    resolution: tuple[int, int] = (1920, 1080),
    log=lambda m: None,
    min_page_seconds: float = MIN_PAGE_SECONDS,
    frame_animator=None,
    sequence_interp_fps: int = FPS,
    sequence_critic=None,
) -> tuple[Path, float]:
    """Render clips, concat, mux narration + subtitles. Returns (out, seconds).

    `frame_animator` (a video.frames provider) is consulted by "animate"- and
    "sequence"-motion segments; None or a non-animated provider renders the
    panels-mode Ken Burns crop instead. `sequence_critic` (a drift critic,
    video/sequence.py) is passed through to the sequence animator."""
    clips_dir = workdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    for seg in segments:
        slot = _slot_s(seg)
        anchors = (
            _segment_anchors(seg, slot, len(page_paths))
            if seg.motion == "scroll" else []
        )
        if anchors:
            # Grounded hold-and-glide: one strip clip whose viewport holds on
            # each anchored region and glides between them.
            pages = sorted({a["page"] for a in anchors})
            strip = build_strip(
                [page_paths[p - 1] for p in pages],
                clips_dir / f"strip-{seg.index:02d}.png",
            )
            keyframes = hold_glide_keyframes(
                anchors, pages, strip_layout([page_paths[p - 1] for p in pages]),
                resolution[0], resolution[1], slot,
            )
            dest = clips_dir / f"clip-{seg.index:02d}-0.mp4"
            if dest.exists():
                log(f"  clip-{seg.index:02d}-0: cached")
            else:
                build_hold_glide_clip(strip, slot, keyframes, resolution, dest)
                log(f"  clip-{seg.index:02d}-0: rendered (hold-and-glide,"
                    f" {len(anchors)} anchor(s) on {len(pages)} page(s),"
                    f" {slot:.1f}s)")
            clips.append(dest)
            continue
        if seg.motion in ("panels", "animate", "sequence"):
            # Guided view: one clip per grounded anchor. Hold anchors use
            # their panel crop, pan anchors the full page; equal shares of
            # the slot, hard cuts between panels. "animate" stitches frames
            # generated from the crop (xfade chain); "sequence" expands the
            # crop to video size and chains frames temporally (minterpolate
            # smoothed). Without an animated provider both degrade to
            # panels' Ken Burns zoom, and a provider failure on one anchor
            # degrades just that anchor the same way — except a
            # VideoConfigError (broken model/provider config), which aborts
            # the render instead of storming the API per anchor.
            anchors = _segment_anchors(seg, slot, len(page_paths))
            if anchors:
                share = slot / len(anchors)
                animate = (
                    seg.motion == "animate"
                    and frame_animator is not None
                    and frame_animator.animated
                )
                sequence = (
                    seg.motion == "sequence"
                    and frame_animator is not None
                    and frame_animator.animated
                )
                for k, anchor in enumerate(anchors):
                    page = page_paths[anchor["page"] - 1]
                    if anchor["kind"] == "hold":
                        image = _panel_crop(
                            page, anchor["box"],
                            clips_dir / f"panel-{seg.index:02d}-{k}.png",
                        )
                    else:
                        image = page
                    dest = clips_dir / f"clip-{seg.index:02d}-{k}.mp4"
                    if dest.exists():
                        log(f"  clip-{seg.index:02d}-{k}: cached")
                    elif animate or sequence:
                        degraded = False
                        try:
                            if animate:
                                frames = frame_animator.generate_frames(
                                    image, seg, share, workdir, log
                                )
                            else:
                                frames = frame_animator.generate_frames(
                                    image, seg, share, workdir, log,
                                    critic=sequence_critic,
                                )
                        except VideoConfigError:
                            # Misconfiguration (a model that can't generate
                            # images): retrying every anchor would just
                            # storm the API — abort the render.
                            raise
                        except VideoError as exc:
                            # A provider failure on one panel (a 400, a
                            # network blip) must not kill the chapter —
                            # degrade this anchor to the panels Ken Burns
                            # crop, same as a non-animated provider.
                            log(f"  clip-{seg.index:02d}-{k}: frame"
                                f" generation failed ({exc}) — panels"
                                " fallback")
                            frames = []
                            degraded = True
                        if frames and animate:
                            build_xfade_clip(frames, share, resolution, dest)
                            log(f"  clip-{seg.index:02d}-{k}: rendered"
                                f" ({anchor['kind']} on page"
                                f" {anchor['page']}, {len(frames)} generated"
                                f" frames, {share:.1f}s)")
                        elif frames:
                            build_sequence_clip(
                                frames, share, resolution, dest,
                                sequence_interp_fps,
                            )
                            log(f"  clip-{seg.index:02d}-{k}: rendered"
                                f" ({anchor['kind']} on page"
                                f" {anchor['page']}, {len(frames)} sequence"
                                f" frames, {share:.1f}s)")
                        else:
                            if degraded:
                                # A fallback clip is not the provider's
                                # product — keep it out of the canonical
                                # clip-*.mp4 cache so the next run retries
                                # the generator instead of reusing Ken Burns.
                                dest = dest.with_name(dest.stem + ".fallback.mp4")
                            build_clip(image, share, "zoom_in", resolution,
                                       dest)
                            log(f"  clip-{seg.index:02d}-{k}: rendered"
                                f" ({anchor['kind']} on page"
                                f" {anchor['page']}, {share:.1f}s)")
                    else:
                        build_clip(image, share, "zoom_in", resolution, dest)
                        log(f"  clip-{seg.index:02d}-{k}: rendered"
                            f" ({anchor['kind']} on page {anchor['page']},"
                            f" {share:.1f}s)")
                    clips.append(dest)
                continue
            # Ungrounded segment: fall through to the per-page path (the
            # panels/animate motion there degrades to the plain zoom).
        if seg.motion == "slideshow":
            # Static crops dissolving into each other within the segment;
            # one clip per segment, hard cuts between segments.
            pages = _segment_pages(seg, min_page_seconds)
            dest = clips_dir / f"clip-{seg.index:02d}-0.mp4"
            if dest.exists():
                log(f"  clip-{seg.index:02d}-0: cached")
            else:
                build_xfade_clip(
                    [page_paths[p - 1] for p in pages], slot, resolution, dest
                )
                log(f"  clip-{seg.index:02d}-0: rendered (slideshow,"
                    f" {len(pages)} page(s), {slot:.1f}s)")
            clips.append(dest)
            continue
        pages = _segment_pages(seg, min_page_seconds)
        if seg.motion == "scroll" and len(pages) > 1:
            # One clip per segment: the viewport descends the stacked strip
            # via the existing pan_down crop animation.
            strip = build_strip(
                [page_paths[p - 1] for p in pages],
                clips_dir / f"strip-{seg.index:02d}.png",
            )
            dest = clips_dir / f"clip-{seg.index:02d}-0.mp4"
            if dest.exists():
                log(f"  clip-{seg.index:02d}-0: cached")
            else:
                build_clip(strip, _slot_s(seg), "pan_down", resolution, dest)
                log(f"  clip-{seg.index:02d}-0: rendered"
                    f" ({len(pages)}-page strip, {_slot_s(seg):.1f}s)")
            clips.append(dest)
            continue
        per_page = _slot_s(seg) / len(pages)
        # A one-page scroll segment degrades to today's pan_down behavior.
        motion = "pan_down" if seg.motion == "scroll" else seg.motion
        for slot, page_num in enumerate(pages):
            dest = clips_dir / f"clip-{seg.index:02d}-{slot}.mp4"
            page = page_paths[page_num - 1]
            if dest.exists():
                log(f"  clip-{seg.index:02d}-{slot}: cached")
            else:
                build_clip(page, per_page, motion, resolution, dest)
                log(f"  clip-{seg.index:02d}-{slot}: rendered"
                    f" (page {page_num}, {per_page:.1f}s)")
            clips.append(dest)
    return mux_clips(segments, clips, workdir, log)


def pacing_stats(
    segments: list[Segment], min_page_seconds: float = MIN_PAGE_SECONDS
) -> dict:
    """Deterministic pacing summary for segments with stamped durations
    (post-TTS): narration volume, beat lengths, and how often the picture
    changes. Logged at assembly time; also usable to compare a video's
    pacing across pipeline versions."""
    durations = [s.duration_s for s in segments]
    words = sum(len(s.text.split()) for s in segments)
    shown = [p for s in segments for p in _segment_pages(s, min_page_seconds)]
    total = sum(_slot_s(s) for s in segments)
    return {
        "segments": len(segments),
        "words": words,
        "speech_s": round(sum(durations), 1),
        "pauses_s": round(sum(s.pause_after_s for s in segments), 1),
        "total_s": round(total, 1),
        "avg_seg_s": round(statistics.mean(durations), 2) if durations else 0.0,
        "median_seg_s": round(statistics.median(durations), 2) if durations else 0.0,
        "max_seg_s": round(max(durations), 2) if durations else 0.0,
        "visuals_shown": len(shown),
        "visual_cuts": sum(1 for a, b in zip(shown, shown[1:]) if a != b),
        "avg_s_per_visual": round(total / len(shown), 2) if shown else 0.0,
    }


def probe_format(path: Path) -> tuple[str, int, int, str]:
    """(codec, width, height, avg_frame_rate) of a file's first video
    stream — the identity check that decides lossless vs re-encode."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise VideoError(f"ffprobe failed for {path}: {r.stderr.strip()}")
    s = json.loads(r.stdout)["streams"][0]
    return (s["codec_name"], s["width"], s["height"], s["avg_frame_rate"])


def concat_mp4s(
    parts: list[Path],
    dest: Path,
    log: Callable[[str], None] = print,
) -> None:
    """Concatenate chapter mp4s into one file, in order. When every input
    shares codec/resolution/fps (always true for videos rendered with the
    same config) the concat is lossless (stream copy, no re-encode); mixed
    formats fall back to a balanced 720p re-encode. Raises VideoError when
    ffmpeg fails."""
    import tempfile

    formats = {probe_format(p) for p in parts}
    uniform = len(formats) == 1
    if uniform:
        log("all inputs share one format: lossless stream copy")
    else:
        log(f"mixed formats ({len(formats)}): re-encoding at 720p CRF 25")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False
    ) as lst:
        lst.write("".join(f"file '{p.as_posix()}'\n" for p in parts))
        list_path = lst.name
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
        if uniform:
            cmd += ["-c", "copy"]
        else:
            cmd += [
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease"
                       ":force_divisible_by=2",
                "-c:v", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
            ]
        cmd.append(str(dest))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise VideoError(
                "ffmpeg concat failed:\n"
                + "\n".join(r.stderr.strip().splitlines()[-8:])
            )
    finally:
        Path(list_path).unlink(missing_ok=True)
