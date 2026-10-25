"""The director: instruct → generate → assess → revise around the
frame-animation chain loop.

Where the plain sequence loop (video/sequence.py) generates each frame once
(and the optional drift critic gets a single regeneration), the director
wraps the same primitives in a bounded revision cycle driven by judge
verdicts (evals/imagegen/judge.py — the reward signal):

- An instruction bank holds named {expand, chain} template pairs tagged
  with the model ids they apply to, plus revision instructions
  (preserve-composition, motion-push).
- Frame 0 expands the panel; judge.faithfulness vs the panel; a drift
  verdict triggers a preserve-composition expand revision.
- Frame i≥1 chains off the picked previous frame; judge.drift and
  judge.motion vs the previous frame; drift → preserve-composition,
  stagnant → motion-push. (Motion is not assessed on a drifted attempt —
  the revision is already decided.)
- Each frame gets at most MAX_ATTEMPTS generations (1 + 2 revisions); the
  pick is best by (drift-ok, advancing, faithfulness-ok), ties to the
  earliest attempt. A chained frame whose attempts all drift truncates the
  chain, mirroring the critic's semantics; frame 0 is always appended (the
  critic never rejects the expansion either).
- The trajectory — per frame, each attempt's instruction and verdicts, and
  which attempt was picked (null when the frame was rejected) — lands in
  the manifest.

Candidates reuse sequence.py's content-addressed keys, so revised attempts
are distinct files on disk and a re-run re-assesses cached frames instead
of regenerating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from entertainment_harness.config import Config
from entertainment_harness.video.frames import _own_size
from entertainment_harness.video.script import Segment
from entertainment_harness.video.sequence import (
    CHAIN_PROMPT,
    EXPAND_PROMPT,
    _sequence_key,
    frame_count,
    video_ratio,
)

from evals.imagegen import judge as judge_mod

#: Generations per frame: the initial attempt plus up to two revisions.
MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Instruction:
    """One bank entry: prompt templates for the expansion (frame 0) and the
    chained frames, the model-id substrings it applies to, and why it
    exists. A None expand/chain marks a revision that never applies to that
    frame kind (motion-push: frame 0 has no predecessor to stagnate
    against)."""

    name: str
    expand: str | None = None
    chain: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


_BFL_SAFE_CHAIN = (
    "Animate this manga panel, frame {phase} of {count} of a subtle"
    " animation: keep the scene and drawing style, advancing natural"
    " motion a small step — rain, hair, cloth, smoke, and effects."
    " Scene context: {text}"
)

_PRESERVE_COMPOSITION_EXPAND = (
    "Outpaint the manga panel @panel to a full {ratio} video frame. The"
    " previous attempt changed the panel itself ({note}) — keep the panel's"
    " exact composition, characters, and art style, and add only the scene"
    " beyond its borders. Scene context: {text}"
)

_PRESERVE_COMPOSITION_CHAIN = (
    "Draw frame {phase} of {count} of a subtle frame-by-frame animation of"
    " the scene in @prev. The previous attempt drifted ({note}) — keep the"
    " exact same composition, characters, and art style as @prev and the"
    " original panel @panel, changing only what the motion requires."
    " Scene context: {text}"
)

_MOTION_PUSH_CHAIN = (
    "Draw frame {phase} of {count} of a subtle frame-by-frame animation of"
    " the scene in @prev. The previous attempt barely moved ({note}) — push"
    " the natural motion forward one clear step (hair, cloth, smoke,"
    " effects, expressions) while keeping the exact composition,"
    " characters, and art style of @prev and @panel. Scene context: {text}"
)

#: Generation instructions, checked in order; the first tag match wins.
BANK = (
    Instruction(
        name="bfl-safe",
        expand=EXPAND_PROMPT,  # only the chain phrasing was verified live
        chain=_BFL_SAFE_CHAIN,
        tags=("flux-2-pro", "flux-3"),
        note="chain template verified to pass BFL's flaky content checker"
        " 2026-10-11; the expand variant was never verified and stays"
        " default",
    ),
    Instruction(
        name="default",
        expand=EXPAND_PROMPT,
        chain=CHAIN_PROMPT,
        note="verbatim from video/sequence.py",
    ),
)

#: Revision instructions, triggered by judge verdicts; {note} carries the
#: judge's one-sentence reason into the revision prompt.
REVISIONS = {
    "preserve-composition": Instruction(
        name="preserve-composition",
        expand=_PRESERVE_COMPOSITION_EXPAND,
        chain=_PRESERVE_COMPOSITION_CHAIN,
        note="drift verdict — re-anchor on composition and style",
    ),
    "motion-push": Instruction(
        name="motion-push",
        chain=_MOTION_PUSH_CHAIN,
        note="stagnant verdict — push motion while holding composition",
    ),
}

INSTRUCTIONS = {i.name: i for i in BANK} | REVISIONS


def instruction_by_name(name: str) -> Instruction:
    try:
        return INSTRUCTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown instruction {name!r} (known: {sorted(INSTRUCTIONS)})"
        ) from None


def pick_instruction(model: str) -> Instruction:
    """The first bank entry whose tags substring-match the model id, else
    the untagged default."""
    for instruction in BANK:
        if any(tag in model for tag in instruction.tags):
            return instruction
    return next(i for i in BANK if not i.tags)


def revision_for(verdicts: dict) -> Instruction | None:
    """Which revision a frame's verdicts trigger: any drift →
    preserve-composition; else stagnation → motion-push; else accept.
    Inconclusive verdicts trigger nothing — a murky judge degrades to the
    plain loop rather than burning revisions."""
    if any(v.get("verdict") == "drift" for v in verdicts.values()):
        return REVISIONS["preserve-composition"]
    if verdicts.get("motion", {}).get("verdict") == "stagnant":
        return REVISIONS["motion-push"]
    return None


def _rank(verdicts: dict) -> tuple[bool, bool, bool]:
    """Pick ordering: drift-ok, then advancing, then faithfulness-ok.
    Missing dimensions (motion is skipped on drifted attempts) count as
    not-advancing so assessed-and-advancing beats unassessed."""
    return (
        verdicts.get("drift", {}).get("verdict") != "drift",
        verdicts.get("motion", {}).get("verdict") == "advancing",
        verdicts.get("faithfulness", {}).get("verdict") != "drift",
    )


def _render(template: str, *, ratio: str, phase: int, count: int, text: str,
            note: str) -> str:
    """Format a template; unused slots (phase/count on expand, note on
    non-revisions) are ignored by str.format."""
    return template.format(
        ratio=ratio, phase=phase, count=count, text=text, note=note
    )


def _note(verdicts: dict) -> str:
    """The judge's reason for the failing verdict, for the {note} slot."""
    return next(
        (v["note"] for v in verdicts.values() if v.get("note")),
        "the check failed",
    )


def _assess(adapter, model: str, panel: Path, prev: Path | None,
            dest: Path) -> dict:
    """Frame 0: faithfulness vs the panel. Chained frames: drift vs the
    predecessor, plus motion only when the frame didn't drift (a drifted
    attempt's revision is already decided, so the motion call is waste)."""
    if prev is None:
        return {"faithfulness": judge_mod.faithfulness(adapter, model, panel, dest)}
    verdicts = {"drift": judge_mod.drift(adapter, model, prev, dest)}
    if verdicts["drift"]["verdict"] != "drift":
        verdicts["motion"] = judge_mod.motion(adapter, model, prev, dest)
    return verdicts


def direct_sequence(
    client,
    model: str,
    config: Config,
    image: Path,
    segment: Segment,
    duration: float,
    workdir: Path,
    judge_adapter,
    judge_id: str,
    log=lambda m: None,
) -> tuple[list[Path], list[dict]]:
    """The directed chain loop for one panel: like sequence_frames_with,
    but every frame is assessed by the judge and revised per the bank.
    Returns the accepted frames plus the trajectory (one entry per frame
    attempted: attempts with instruction + verdicts, the picked attempt
    index, null when all attempts drifted and the chain truncated)."""
    cfg = config.frame_animation
    count = frame_count(duration, cfg.fps, cfg.max_frames)
    ratio = video_ratio(config.video.resolution)
    frames_dir = workdir / "sequence"
    frames_dir.mkdir(parents=True, exist_ok=True)
    base = pick_instruction(model)
    log(f"    segment {segment.index:02d}: director instructs with"
        f" {base.name!r}")

    frames: list[Path] = []
    trajectory: list[dict] = []
    for i in range(count):
        if i == 0:
            refs = [("panel", image)]
            ref_ratio = _own_size(image)
        else:
            refs = [("prev", frames[-1]), ("panel", image)]
            ref_ratio = _own_size(frames[-1])

        instruction = base
        note = ""
        attempts: list[dict] = []
        attempt_paths: list[Path] = []
        clean: int | None = None
        for _ in range(MAX_ATTEMPTS):
            template = instruction.expand if i == 0 else instruction.chain
            if template is None:
                break  # a revision that doesn't apply to this frame kind
            prompt = _render(
                template, ratio=ratio, phase=i + 1, count=count,
                text=segment.text[:300], note=note,
            )
            dest = frames_dir / (
                f"s-{_sequence_key(image, prompt, model, ratio, i)}-{i}.png"
            )
            if not dest.exists():
                client.text_to_image(
                    prompt, refs, model=model, ratio=ratio,
                    ref_ratio=ref_ratio, dest=dest,
                )
            verdicts = _assess(
                judge_adapter, judge_id, image,
                frames[-1] if i > 0 else None, dest,
            )
            attempts.append({"instruction": instruction.name,
                             "verdicts": verdicts})
            attempt_paths.append(dest)
            revision = revision_for(verdicts)
            if revision is None:
                clean = len(attempts) - 1
                break
            note = _note(verdicts)
            instruction = revision

        if clean is not None:
            pick = clean
        else:
            # Cap hit without a clean accept: best by rank, ties earliest.
            pick = max(
                range(len(attempts)),
                key=lambda k: (_rank(attempts[k]["verdicts"]), -k),
            )
        if (
            clean is None
            and i > 0
            and any(v.get("verdict") == "drift"
                    for v in attempts[pick]["verdicts"].values())
        ):
            trajectory.append({"frame": i, "attempts": attempts,
                               "picked": None})
            log(f"    segment {segment.index:02d}: frame {i} keeps drifting"
                f" — truncating the chain at {len(frames)} frame(s)")
            break
        frames.append(attempt_paths[pick])
        trajectory.append({"frame": i, "attempts": attempts, "picked": pick})

    log(f"    segment {segment.index:02d}: {len(frames)} sequence frame(s)"
        f" ({sum(len(t['attempts']) for t in trajectory)} attempt(s))")
    return frames, trajectory
