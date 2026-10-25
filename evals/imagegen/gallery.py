"""manifest.json -> self-contained index.html (inline CSS, no JS, relative
img refs), plus optional per-arm preview clips. Re-renderable offline
from just the run dir — that is what the CLI's `render` command does.

Clips: frame-animation arms go through the assembly's build_sequence_clip at
the arm's interp_fps (assembly-only knob: playback smoothness, never a
generation variable); video-animation arms get an ffmpeg concat slideshow at
seconds_per_frame per frame. Missing ffmpeg or a failed encode is a note
in the gallery — never a crash.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path

_BADGE_CLASS = {
    "ok": "ok",
    "advancing": "ok",
    "drift": "bad",
    "stagnant": "warn",
    "inconclusive": "mute",
}

_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem;
       background: #14161a; color: #e8e8ea; }
h1 { font-size: 1.4rem; margin-bottom: .2rem; }
h2 { font-size: 1.1rem; margin-bottom: .2rem; }
h3 { font-size: .95rem; color: #9aa5b5; margin-bottom: .2rem; }
.meta { color: #99a3b2; font-size: .85rem; }
section.arm { border: 1px solid #2c313a; border-radius: 8px; padding: 1rem;
              margin: 1.2rem 0; background: #1b1e24; }
table { border-collapse: collapse; margin: .5rem 0; font-size: .82rem; }
td, th { border: 1px solid #333a44; padding: .2rem .55rem; text-align: left; }
th { color: #9aa5b5; }
td.diff { background: #3d3417; color: #ffd76a; font-weight: 600; }
.strip { display: flex; flex-wrap: wrap; gap: .6rem; margin: .6rem 0; }
figure { margin: 0; width: 220px; }
figure img { width: 220px; border: 1px solid #333a44; border-radius: 4px;
             display: block; }
figcaption { font-size: .7rem; color: #9aa5b5; margin-top: .25rem; }
.badge { display: inline-block; font-size: .65rem; border-radius: 3px;
         padding: 0 .35rem; margin: .1rem .2rem 0 0; }
.badge.ok { background: #1d3a24; color: #7fd68f; }
.badge.bad { background: #40201f; color: #ff8f8a; }
.badge.warn { background: #40331b; color: #ffc46a; }
.badge.mute { background: #2a2e36; color: #9aa3b2; }
.badge.norefs { background: #40331b; color: #ffc46a; font-size: .72rem; }
.error { background: #3a1d1d; border: 1px solid #6a2b2b; color: #ffb3ad;
         padding: .5rem .7rem; border-radius: 6px; font-size: .85rem;
         white-space: pre-wrap; }
.note { background: #2a2e36; border-radius: 6px; padding: .4rem .7rem;
        font-size: .8rem; color: #c8ccd4; }
code { color: #a8d3ff; }
video { display: block; width: 420px; margin: .5rem 0; border: 1px solid #333a44;
        border-radius: 4px; }
.summary { font-size: .85rem; color: #c8ccd4; }
details { font-size: .8rem; color: #99a3b2; }
"""


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _pct(rate) -> str:
    return "—" if rate is None else f"{rate * 100:.0f}%"


def _badge(verdict: str, note: str | None = None) -> str:
    cls = _BADGE_CLASS.get(verdict, "mute")
    title = f" title='{_e(note)}'" if note else ""
    return f"<span class='badge {cls}'{title}>{_e(verdict)}</span>"


def _resolution(text: str) -> tuple[int, int]:
    w, _, h = str(text).partition("x")
    try:
        return int(w), int(h)
    except ValueError:
        return 1920, 1080


# --- preview clips ------------------------------------------------------------


def _slideshow_clip(frames: list[Path], seconds_per_frame: float,
                    dest: Path) -> None:
    """Animate-mode preview: each frame held seconds_per_frame (ffmpeg
    concat demuxer; the last file repeats so its duration counts)."""
    listing = dest.with_suffix(".concat.txt")
    lines = []
    for frame in frames:
        lines.append(f"file '{frame.resolve().as_posix()}'")
        lines.append(f"duration {seconds_per_frame:.3f}")
    lines.append(f"file '{frames[-1].resolve().as_posix()}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
                str(listing),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", "30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
            ],
            check=True, capture_output=True,
        )
    finally:
        listing.unlink(missing_ok=True)


def _build_clips(run_dir: Path, manifest: dict, notes: list[str], log) -> None:
    mode = manifest["experiment"]["mode"]
    duration = float(manifest["experiment"]["duration"])
    base = manifest.get("base_config", {})
    for arm in manifest.get("arms", []):
        arm_dir = run_dir / "arms" / arm["name"]
        for i, panel in enumerate(arm.get("panels", [])):
            frames = [run_dir / rel for rel in panel.get("frames", [])]
            if not frames:
                continue
            dest = arm_dir / f"clip-{i:02d}.mp4"
            if dest.exists():
                panel["clip"] = dest.relative_to(run_dir).as_posix()
                continue
            try:
                overrides = arm.get("overrides", {})
                if mode == "frame-animation":
                    resolution = _resolution(
                        overrides.get(
                            "resolution",
                            base.get("video", {}).get("resolution", "1920x1080"),
                        )
                    )
                    interp_fps = int(
                        overrides.get(
                            "interp_fps",
                            base.get("frame_animation", {}).get("interp_fps", 30),
                        )
                    )
                    from entertainment_harness.video.assemble import (
                        build_sequence_clip,
                    )

                    build_sequence_clip(frames, duration, resolution, dest,
                                        interp_fps)
                else:
                    spf = float(
                        overrides.get(
                            "seconds_per_frame",
                            base.get("frames", {}).get("seconds_per_frame", 2.0),
                        )
                    )
                    _slideshow_clip(frames, spf, dest)
                panel["clip"] = dest.relative_to(run_dir).as_posix()
                log(f"clip: {panel['clip']}")
            except Exception as exc:
                notes.append(
                    f"clip failed for arm {arm['name']} panel {i + 1}: {exc}"
                )


# --- HTML ---------------------------------------------------------------------


def _overrides_table(mode: str, base: dict, overrides: dict) -> str:
    section = "frames" if mode == "video-animation" else "frame_animation"
    rows = ["<table><tr><th>knob</th><th>arm</th><th>base</th></tr>"]
    for key in sorted(overrides):
        value = overrides[key]
        if key == "resolution":
            base_value = base.get("video", {}).get("resolution")
        else:
            base_value = base.get(section, {}).get(key)
        cls = " class='diff'" if value != base_value else ""
        display = str(value) if value != "" else "(provider default)"
        rows.append(
            f"<tr><td>{_e(key)}</td><td{cls}>{_e(display)}</td>"
            f"<td>{_e(base_value)}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _strip(run_dir: Path, panel: dict) -> str:
    judge = panel.get("judge")
    judge = judge if isinstance(judge, dict) else {}
    verdict_by_frame = {v["frame"]: v for v in judge.get("frames", [])}
    pair_by_new = {p["frames"][1]: p for p in judge.get("pairs", [])}
    out = ["<div class='strip'>"]
    image = panel.get("image")
    if image and (run_dir / image).is_file():
        out.append(
            f"<figure><img src='{_e(image)}' loading='lazy'>"
            f"<figcaption>source panel</figcaption></figure>"
        )
    for i, rel in enumerate(panel.get("frames", [])):
        badges = []
        verdict = verdict_by_frame.get(rel)
        if verdict:
            badges.append(_badge(verdict["verdict"], verdict.get("note")))
        pair = pair_by_new.get(rel)
        if pair:
            title = "; ".join(pair.get("notes", [])) or None
            if pair["drift"] != "ok":
                badges.append(_badge(pair["drift"], title))
            badges.append(_badge(pair["motion"], title))
        out.append(
            f"<figure><img src='{_e(rel)}' loading='lazy'>"
            f"<figcaption>frame {i + 1} {''.join(badges)}</figcaption></figure>"
        )
    out.append("</div>")
    if judge.get("error"):
        out.append(f"<p class='note'>judge scoring failed: {_e(judge['error'])}</p>")
    return "".join(out)


def _panel_block(run_dir: Path, index: int, panel: dict) -> str:
    out = [
        f"<h3>panel {index + 1}: {_e(Path(panel.get('source', '?')).name)}</h3>"
    ]
    if panel.get("error"):
        # A failed panel has no metrics/frames/judge — show the error the
        # way arm-level failures are shown, never an empty strip.
        out.append(f"<p class='error'>panel failed: {_e(panel['error'])}</p>")
        return "".join(out)
    m = panel.get("metrics", {})
    metrics = (
        f"{m.get('produced', '?')}/{m.get('requested', '?')} frames"
        f" · {m.get('generated', '?')} generated / {m.get('cached', '?')} cached"
        f" · {m.get('wall_s', '?')}s wall · ratio {_e(m.get('ratio', '?'))}"
    )
    if m.get("truncated"):
        metrics += f" · {_badge('truncated', 'the drift critic/director cut the chain short')}"
    out.append(f"<p class='summary'>{metrics}</p>")
    clip = panel.get("clip")
    if clip and (run_dir / clip).is_file():
        out.append(f"<video controls preload='metadata' src='{_e(clip)}'></video>")
    out.append(_strip(run_dir, panel))
    return "".join(out)


def _arm_section(run_dir: Path, manifest: dict, arm: dict) -> str:
    mode = manifest["experiment"]["mode"]
    base = manifest.get("base_config", {})
    out = [
        f"<section class='arm' id='arm-{_e(arm['name'])}'>",
        f"<h2>{_e(arm['name'])}"
        f" <span class='meta'>{_e(arm.get('provider', '?'))}</span>",
    ]
    if not arm.get("references_used", True):
        out.append(
            " <span class='badge norefs'>references_used: false — blind"
            " generation, the panel is not conditioning these frames</span>"
        )
    out.append("</h2>")
    out.append(
        f"<p class='meta'>model: <code>{_e(arm.get('model', '?'))}</code></p>"
    )
    out.append(_overrides_table(mode, base, arm.get("overrides", {})))
    if arm.get("error"):
        out.append(f"<p class='error'>arm failed: {_e(arm['error'])}</p>")
    summary = arm.get("judge_summary")
    if summary:
        out.append(
            f"<p class='summary'>judge: faithfulness pass"
            f" {_pct(summary.get('faithfulness_pass_rate'))} · drift incidents"
            f" {summary.get('drift_incidents', 0)} · stagnation"
            f" {_pct(summary.get('stagnation_rate'))}</p>"
        )
    for i, panel in enumerate(arm.get("panels", [])):
        out.append(_panel_block(run_dir, i, panel))
    out.append("</section>")
    return "".join(out)


def _page(run_dir: Path, manifest: dict, notes: list[str]) -> str:
    exp = manifest["experiment"]
    judge = manifest.get("judge", {})
    if judge.get("warning"):
        judge_line = f"judge: ⚠ {_e(judge['warning'])}"
    elif judge.get("model"):
        judge_line = f"judge: <code>{_e(judge['model'])}</code>"
        if judge.get("note"):
            judge_line += f" ({_e(judge['note'])})"
    else:
        judge_line = "judge: disabled"
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{_e(exp['name'])} — frame-gen eval</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{_e(exp['name'])}</h1>",
        f"<p class='meta'>{_e(exp['mode'])} mode"
        f" · {_e(exp['duration'])}s per panel"
        f" · started {_e(manifest.get('started_at', '?'))}"
        f" · {judge_line}"
        + (f" · limit {manifest['limit']}" if manifest.get("limit") else "")
        + "</p>",
    ]
    if exp.get("context"):
        parts.append(f"<p class='meta'>context: {_e(exp['context'])}</p>")
    for note in notes:
        parts.append(f"<p class='note'>{_e(note)}</p>")
    for arm in manifest.get("arms", []):
        parts.append(_arm_section(run_dir, manifest, arm))
    parts.append(
        "<details><summary>base config</summary>"
        f"<pre>{_e(json.dumps(manifest.get('base_config', {}), indent=2))}</pre>"
        "</details>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def render_gallery(run_dir: Path, *, clips: bool = False, log=print) -> Path:
    """Write index.html for a run dir; with clips=True also build/reuse the
    per-arm preview MP4s and record them back into manifest.json. Returns
    the index.html path."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes: list[str] = []
    if clips:
        if shutil.which("ffmpeg") is None:
            notes.append("ffmpeg not found — preview clips skipped.")
        else:
            _build_clips(run_dir, manifest, notes, log)
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
    out = run_dir / "index.html"
    out.write_text(_page(run_dir, manifest, notes), encoding="utf-8")
    return out
