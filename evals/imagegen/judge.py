"""Vision scoring for eval runs: faithfulness (each frame vs the source
panel), drift and motion (consecutive frame pairs).

Verdict parsing mirrors video/sequence.py's drift_critic: an OK-style
prefix accepts, a "VERDICT: reason" reply carries its reason, and anything
else is recorded as inconclusive — scoring never crashes on a weird reply,
so a half-broken judge model degrades the scores instead of killing the
run. DRIFT_PROMPT is reused verbatim so the eval's drift dimension
predicts what the pipeline's critic would do on the same pair; the
faithfulness and motion dimensions are eval-only (the cadence question
needs them: high fps risks stagnation, low fps risks drift).
"""

from __future__ import annotations

from pathlib import Path

from entertainment_harness.video.sequence import DRIFT_PROMPT

FAITHFULNESS_PROMPT = (
    "These are two images (first: the original manga panel; second: a"
    " generated animation frame derived from it). The frame must keep the"
    " panel's characters, composition, and art style — only motion may"
    " advance. If it does, answer exactly: OK. If characters, composition,"
    " or style changed, answer: DRIFT: <one sentence describing what"
    " changed>"
)

MOTION_PROMPT = (
    "These are two consecutive frames of a manga panel animation (first"
    " image: previous frame; second image: the new frame). Natural motion"
    " — hair, cloth, smoke, effects, expressions — must advance a small"
    " step from the previous frame. If it does, answer exactly: ADVANCING."
    " If the new frame is essentially identical to the previous one,"
    " answer: STAGNANT: <one sentence describing what stayed still>"
)


def _parse(reply: str, positive: str, negative: str) -> dict:
    """OK-prefix-style verdict parsing: the positive verdict -> {"verdict":
    positive}; "NEGATIVE: reason" -> the negative verdict with the reason
    as note; anything else -> inconclusive with the reply as note."""
    text = reply.strip()
    if text.upper().startswith(positive):
        return {"verdict": positive.lower()}
    verdict, _, reason = text.partition(":")
    if verdict.strip().upper() == negative and reason.strip():
        return {"verdict": negative.lower(), "note": reason.strip()}
    return {"verdict": "inconclusive", "note": text[:200] or "(empty reply)"}


def faithfulness(adapter, model: str, panel: Path, frame: Path) -> dict:
    """One frame against the source panel: ok / drift / inconclusive."""
    reply = adapter.generate(model, FAITHFULNESS_PROMPT, images=[panel, frame])
    return _parse(reply, "OK", "DRIFT")


def drift(adapter, model: str, prev: Path, new: Path) -> dict:
    """One consecutive pair against DRIFT_PROMPT (verbatim reuse)."""
    reply = adapter.generate(model, DRIFT_PROMPT, images=[prev, new])
    return _parse(reply, "OK", "DRIFT")


def motion(adapter, model: str, prev: Path, new: Path) -> dict:
    """One consecutive pair: advancing / stagnant / inconclusive."""
    reply = adapter.generate(model, MOTION_PROMPT, images=[prev, new])
    return _parse(reply, "ADVANCING", "STAGNANT")


def score_frames(
    adapter,
    model: str,
    panel: Path,
    frames: list[Path],
    *,
    rel=lambda p: str(p),
) -> dict:
    """Score one arm x panel: per-frame faithfulness, per-pair drift and
    motion, and aggregates (faithfulness pass rate, drift incidents,
    stagnation rate — None when there are no pairs to rate). Adapter
    exceptions propagate; reply-content problems never do."""
    frame_verdicts = [
        {"frame": rel(frame), **faithfulness(adapter, model, panel, frame)}
        for frame in frames
    ]
    pair_verdicts = []
    for i in range(1, len(frames)):
        d = drift(adapter, model, frames[i - 1], frames[i])
        m = motion(adapter, model, frames[i - 1], frames[i])
        entry = {
            "frames": [rel(frames[i - 1]), rel(frames[i])],
            "drift": d["verdict"],
            "motion": m["verdict"],
        }
        notes = [v["note"] for v in (d, m) if "note" in v]
        if notes:
            entry["notes"] = notes
        pair_verdicts.append(entry)

    ok = sum(1 for v in frame_verdicts if v["verdict"] == "ok")
    drifts = sum(1 for v in pair_verdicts if v["drift"] == "drift")
    stagnant = sum(1 for v in pair_verdicts if v["motion"] == "stagnant")
    n, p = len(frame_verdicts), len(pair_verdicts)
    aggregates = {
        "frames": n,
        "faithfulness_ok": ok,
        "faithfulness_pass_rate": round(ok / n, 3) if n else None,
        "pairs": p,
        "drift_incidents": drifts,
        "stagnant_pairs": stagnant,
        "stagnation_rate": round(stagnant / p, 3) if p else None,
    }
    return {
        "frames": frame_verdicts,
        "pairs": pair_verdicts,
        "aggregates": aggregates,
    }
