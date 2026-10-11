"""Run an experiment: per-arm config copies drive the shared generation
loops (generate_frames_with / sequence_frames_with), metrics + judge
scores land in manifest.json, and the gallery renders index.html.

Design rules honored here: arms are config overrides on a deep copy of the
loaded Config; the loops are driven, never reimplemented; a failing arm
(missing API key, missing diffusers, provider error, OOM) is recorded and
the run continues; every path in the manifest is relative to the run dir
so the directory stays portable.
"""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
import signal
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from entertainment_harness.config import Config, load_config
from entertainment_harness.hardware import probe
from entertainment_harness.video.frames import (
    frame_count as animate_frame_count,
    generate_frames_with,
    nearest_ratio,
    resolve_image_model,
)
from entertainment_harness.video.script import Segment
from entertainment_harness.video.sequence import (
    drift_critic,
    frame_count as sequence_frame_count,
    sequence_frames_with,
    video_ratio,
)

from evals.imagegen import clients as clients_mod
from evals.imagegen import gallery as gallery_mod
from evals.imagegen import judge as judge_mod
from evals.imagegen.experiment import Arm, Experiment

MANIFEST_VERSION = 1
DEFAULT_OUT = Path("evals/runs")

#: Default per-panel generation deadline (seconds). A loop call is up to
#: max_frames images; one stuck streaming response must never hold a run
#: hostage (httpx's read timeout resets per received chunk, so a trickling
#: endpoint can otherwise hang for hours).
DEFAULT_GEN_TIMEOUT = 900.0


@contextlib.contextmanager
def _generation_deadline(seconds: float | None):
    """Raise TimeoutError inside the block past `seconds` (SIGALRM-based,
    so a slowly-streaming HTTP body cannot defeat it). Runs untimed when
    disabled (None/0), off the main thread, or where setitimer is missing
    (Windows). Any pre-existing alarm and handler are restored on exit."""
    armed = (
        bool(seconds)
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
    )
    if not armed:
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError(
            f"generation exceeded the {seconds:g}s deadline"
        )

    old_handler = signal.signal(signal.SIGALRM, _raise)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def resolve_judge(config: Config, profile, judge_model: str | None = None,
                  log=print):
    """The vision adapter + model id for the drift critic and scoring,
    resolved like the pipeline does (video/pipeline.py): an explicit
    judge_model binds that id on the openai_compat backend directly (no
    config edit, no ollama); otherwise the configured vision role +
    ensure(). Raises on any failure — the caller records it as a judge
    warning and runs unscored/uncriticized."""
    if judge_model:
        from entertainment_harness.models.registry import get_adapter

        adapter = get_adapter("openai_compat", config)
        adapter.ensure(judge_model)
        return adapter, judge_model
    from entertainment_harness.models.registry import get_vision_model

    vision = get_vision_model(config, profile)
    if vision.warning:
        log(f"Warning: {vision.warning}")
    vision.adapter.ensure(vision.info.name)
    return vision.adapter, vision.info.name


def _resolve_model(arm: Arm) -> str:
    """The effective image model id (provider default when undeclared)."""
    if arm.provider == "diffusers":
        return arm.model or clients_mod.DEFAULT_DIFFUSERS_MODEL
    if arm.provider in ("fal", "replicate"):
        return arm.model  # the endpoint/model id; required, the client says so
    return resolve_image_model(arm.provider, arm.model)


def _apply_arm(config: Config, mode: str, arm: Arm) -> None:
    """Apply one arm's overrides to a deep copy of the base Config."""
    target = config.frames if mode == "video-animation" else config.frame_animation
    target.provider = arm.provider
    target.model = arm.model
    for key, value in arm.knobs.items():
        if key == "resolution":
            config.video.resolution = value
        else:
            setattr(target, key, value)


def _arm_overrides(arm: Arm) -> dict:
    return {"provider": arm.provider, "model": arm.model, **arm.knobs}


def _run_panel(experiment: Experiment, arm_config: Config, client,
               model: str, panel: Path, panel_copy: Path, workdir: Path,
               critic, rel, log) -> tuple[dict, list[Path]]:
    """Drive the mode's loop for one panel; return the manifest entry and
    the produced frame paths (absolute)."""
    segment = Segment(index=0, text=experiment.context or "")
    if experiment.mode == "video-animation":
        cfg = arm_config.frames
        requested = animate_frame_count(
            experiment.duration, cfg.seconds_per_frame, cfg.max_frames
        )
        ratio = nearest_ratio(panel)
        frames_dir = workdir / "frames"
    else:
        cfg = arm_config.frame_animation
        requested = sequence_frame_count(
            experiment.duration, cfg.fps, cfg.max_frames
        )
        ratio = video_ratio(arm_config.video.resolution)
        frames_dir = workdir / "sequence"

    before = set(frames_dir.glob("*.png")) if frames_dir.is_dir() else set()
    start = time.monotonic()
    if experiment.mode == "video-animation":
        frames = generate_frames_with(
            client, model, arm_config.frames, panel, segment,
            experiment.duration, workdir, log,
        )
    else:
        frames = sequence_frames_with(
            client, model, arm_config, panel, segment,
            experiment.duration, workdir, log, critic,
        )
    wall = time.monotonic() - start
    after = set(frames_dir.glob("*.png")) if frames_dir.is_dir() else set()

    metrics = {
        "requested": requested,
        "produced": len(frames),
        "generated": len(after - before),  # new files (incl. critic retries)
        "cached": sum(1 for f in frames if f in before),
        "wall_s": round(wall, 3),
        "ratio": ratio,
    }
    if experiment.mode == "frame-animation" and len(frames) < requested:
        metrics["truncated"] = True  # the critic cut the chain short
    entry = {
        "source": str(panel),
        "image": rel(panel_copy),
        "frames": [rel(f) for f in frames],
        "metrics": metrics,
    }
    return entry, frames


def _arm_judge_summary(arm_entry: dict) -> None:
    """Roll per-panel judge aggregates up to the arm (mean pass/stagnation
    rate, total drift incidents)."""
    scored = [
        panel["judge"]["aggregates"]
        for panel in arm_entry["panels"]
        if isinstance(panel.get("judge"), dict) and "aggregates" in panel["judge"]
    ]
    if not scored:
        return

    def mean(key):
        values = [a[key] for a in scored if a[key] is not None]
        return round(sum(values) / len(values), 3) if values else None

    arm_entry["judge_summary"] = {
        "panels": len(scored),
        "faithfulness_pass_rate": mean("faithfulness_pass_rate"),
        "drift_incidents": sum(a["drift_incidents"] for a in scored),
        "stagnation_rate": mean("stagnation_rate"),
    }


def run_experiment(
    experiment: Experiment,
    *,
    out: Path | None = None,
    run_dir: Path | None = None,
    judge: bool | None = None,
    judge_model: str | None = None,
    clips: bool = False,
    limit: int | None = None,
    gen_timeout: float | None = DEFAULT_GEN_TIMEOUT,
    client_factory=None,
    judge_resolver=None,
    log=print,
) -> Path:
    """Execute one experiment -> a self-contained run dir (panels + frames
    + manifest.json + index.html). `judge` overrides the experiment's
    judge flag; `gen_timeout` is a hard deadline per panel loop call
    (None/0 = untimed); `client_factory(provider, config)` and
    `judge_resolver(config, profile) -> (adapter, model_id)` inject fakes
    in tests. Returns the run dir."""
    config = load_config()
    profile = probe(config.hardware.budget_gb)

    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(out) if out is not None else DEFAULT_OUT
        run_dir = run_dir / f"{stamp}-{experiment.name}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    judge_enabled = experiment.judge if judge is None else bool(judge)
    # The drift critic is a generation feature (it changes frames), not
    # scoring — frame-animation arms with critic on need the adapter even
    # under --no-judge. Both share one resolution, mirroring the pipeline.
    critic_wanted = experiment.mode == "frame-animation" and any(
        bool(arm.knobs.get("critic", config.frame_animation.critic))
        for arm in experiment.arms
    )
    judge_section: dict = {"enabled": judge_enabled}
    if critic_wanted and not judge_enabled:
        judge_section["note"] = "critic only (scoring disabled)"
    judge_adapter = None
    judge_id = None
    if judge_enabled or critic_wanted:
        try:
            if judge_resolver is not None:
                judge_adapter, judge_id = judge_resolver(config, profile)
            else:
                judge_adapter, judge_id = resolve_judge(
                    config, profile, judge_model, log=log
                )
            judge_section["model"] = judge_id
        except Exception as exc:
            judge_section["warning"] = f"judge unavailable: {exc}"
            skipped = "scoring and the drift critic" if critic_wanted else "scoring"
            log(f"Warning: judge unavailable ({exc}) — {skipped} skipped")

    # Panels copied into the run dir keep it a portable artifact.
    panels_dir = run_dir / "panels"
    panels_dir.mkdir(exist_ok=True)
    panel_copies = []
    for i, panel in enumerate(experiment.panels):
        copy_path = panels_dir / f"{i:02d}-{panel.name}"
        if not copy_path.exists():
            shutil.copy2(panel, copy_path)
        panel_copies.append(copy_path)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "experiment": {
            "name": experiment.name,
            "mode": experiment.mode,
            "duration": experiment.duration,
            "context": experiment.context,
            "panels": [str(p) for p in experiment.panels],
        },
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_config": {
            "frames": asdict(config.frames),
            "frame_animation": asdict(config.frame_animation),
            "video": {"resolution": config.video.resolution},
        },
        "judge": judge_section,
        "arms": [],
    }
    if limit is not None:
        manifest["limit"] = limit

    rel = lambda p: Path(p).relative_to(run_dir).as_posix()
    pairs_done = 0
    for arm in experiment.arms:
        if limit is not None and pairs_done >= limit:
            break
        arm_entry = {
            "name": arm.name,
            "provider": arm.provider,
            "overrides": _arm_overrides(arm),
            "panels": [],
        }
        try:
            arm_config = copy.deepcopy(config)
            _apply_arm(arm_config, experiment.mode, arm)
            model = _resolve_model(arm)
            arm_entry["model"] = model
            client = (
                client_factory(arm.provider, arm_config)
                if client_factory is not None
                else clients_mod.build_client(arm.provider, arm_config)
            )
            arm_entry["references_used"] = bool(
                getattr(client, "uses_references", True)
            )
            workdir = run_dir / "arms" / arm.name
            critic = None
            if (
                experiment.mode == "frame-animation"
                and arm_config.frame_animation.critic
                and judge_adapter is not None
            ):
                critic = drift_critic(judge_adapter, judge_id)
            for index, panel in enumerate(experiment.panels):
                if limit is not None and pairs_done >= limit:
                    break
                log(f"[{arm.name}] {panel.name} ...")
                pairs_done += 1  # attempted pairs count, failures included
                try:
                    with _generation_deadline(gen_timeout):
                        entry, frames = _run_panel(
                            experiment, arm_config, client, model, panel,
                            panel_copies[index], workdir, critic, rel, log,
                        )
                except Exception as exc:
                    # Per-panel granularity: a transient provider failure
                    # (timeout, dropped connection) must not abort the
                    # arm's remaining panels.
                    entry = {
                        "source": str(panel),
                        "image": rel(panel_copies[index]),
                        "frames": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    log(f"[{arm.name}] {panel.name} failed:"
                        f" {entry['error']} — continuing")
                    arm_entry["panels"].append(entry)
                    continue
                if judge_enabled and judge_adapter is not None:
                    # Scoring runs after generation (and after any caching),
                    # so a re-run of a fully cached arm still gets scored.
                    try:
                        entry["judge"] = judge_mod.score_frames(
                            judge_adapter, judge_id, panel, frames, rel=rel
                        )
                    except Exception as exc:
                        entry["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
                arm_entry["panels"].append(entry)
        except Exception as exc:
            # Backstop for failures outside the panel loop: config copy,
            # model resolution, client construction (e.g. a missing key).
            arm_entry["error"] = f"{type(exc).__name__}: {exc}"
            log(f"[{arm.name}] failed: {arm_entry['error']} — continuing")
        _arm_judge_summary(arm_entry)
        manifest["arms"].append(arm_entry)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log(f"manifest: {run_dir / 'manifest.json'}")
    try:
        log(f"gallery: {gallery_mod.render_gallery(run_dir, clips=clips, log=log)}")
    except Exception as exc:
        log(f"Warning: gallery render failed: {exc}")
    return run_dir
