"""Judge stage: a model evaluates pipeline text outputs before they are stored.

Verification is easier than generation, so even the small text-role model is a
serviceable judge. It checks faithfulness to the source summaries, cohesion,
continuity with the rolling context, and form (no refusals or meta-commentary
— the failure modes seen live: recaps written from training memory, context
updates that replaced the story-so-far with commentary about the task).

This module only produces verdicts. The retry policy (regenerate with the
issues fed back, bounded by MAX_ATTEMPTS) lives in the caller — see recap.py.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from entertainment_harness.models.base import ModelAdapter

MAX_ATTEMPTS = 3  # total generations per judged artifact (1 initial + 2 retries)

# Thinking levels: how much critiquing a pipeline output gets.
#   low    — no judge at all
#   medium — text judge, 3 attempts (default; the historical behavior)
#   high   — vision-verified judging where pages exist, 5 attempts
THINKING_LEVELS = ("low", "medium", "high")
ATTEMPTS = {"medium": MAX_ATTEMPTS, "high": 5}

VISION_JUDGE_PAGE_CAP = 6  # max pages sent to the vision judge (evenly spaced)
VISION_JUDGE_MAX_PX = 1024  # longest edge of a judge sample page


class JudgeError(Exception):
    pass


@dataclass
class Verdict:
    passed: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class RenderIssue:
    """A structured issue reported by the render-quality judge."""

    bubble_index: int | None  # 0-based index into the bubble list; None = page-level
    problem: str  # e.g. "original_text_visible", "text_overflow", "unreadable"
    directions: list[str] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class RenderVerdict:
    passed: bool
    issues: list[RenderIssue] = field(default_factory=list)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str) -> Verdict:
    """Extract the verdict JSON from raw model output (tolerates fences and
    preamble). Unparseable output is treated as a pass with a note: a broken
    judge must not stall the pipeline."""
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        return Verdict(passed=True, issues=["judge output was not JSON"])
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(passed=True, issues=["judge output was invalid JSON"])
    passed = bool(data.get("pass", data.get("passed", True)))
    issues = [str(i) for i in data.get("issues", []) if str(i).strip()]
    if passed:
        issues = []
    elif not issues:
        issues = ["judge gave no specific issues"]
    return Verdict(passed=passed, issues=issues)


JUDGE_RECAP_PROMPT = """You are a strict evaluator for a recap of chapter {chapter} of the story "{title}".

Story so far BEFORE this chapter (context only):
---
{context}
---

Source material: sequential summaries of this chapter's page/section batches. The recap below may ONLY contain events and names that appear here:
---
{batches}
---

Recap to evaluate:
---
{recap}
---

Evaluate the recap on:
1. Faithfulness: every event and character name in the recap appears in the source material. Outside knowledge of the actual manga/book/anime is a failure.
2. Cohesion: it reads as one coherent narrative — no repetition loops, no contradictions of itself.
3. Continuity: it does not contradict the story so far. (If the chapter is a bonus/crossover chapter whose content is unrelated to the story so far, that is fine as long as the recap is faithful to the source material.)
4. Form: it is a plain-prose recap only. Any refusal, meta-commentary about the task, commentary about the original work's structure, or addressing the reader is a failure.{direction}

For every issue you report, quote the exact phrase from the recap that causes it and name the source-material sentence it contradicts. If you cannot quote it, do not report it.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


JUDGE_CONTEXT_PROMPT = """You are a strict evaluator for an updated "story so far" of the story "{title}".

Previous story so far:
---
{previous}
---

New chapter recap:
---
{recap}
---

Updated story so far to evaluate:
---
{candidate}
---

Evaluate the update on:
1. It is a story summary, not commentary: any refusal, meta-discussion of the task, or remarks about whether the chapter belongs to the story is a failure.
2. Preservation: key facts from the previous story so far (names, relationships, major plot turns) are still present, possibly compressed.
3. Incorporation: the new chapter's events are folded in. Exception: if the chapter recap is clearly an unrelated bonus/crossover chapter, the correct update is the previous story so far kept essentially unchanged — that is a pass.
4. No invented names or events absent from both inputs.

For every issue you report, quote the exact phrase from the update that causes it. If you cannot quote it, do not report it.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


JUDGE_RECAP_VISION_PROMPT = """You are a strict evaluator for a recap of chapter {chapter} of the story "{title}". You are given the chapter's actual pages (a sample of them) as images.

Story so far BEFORE this chapter (context only):
---
{context}
---

Recap to evaluate:
---
{recap}
---

Evaluate the recap on:
1. Fidelity: the events, character names, and dialogue beats in the recap match what is actually on the pages. Events or names not depicted on any page are a failure. Outside knowledge of the actual manga/book/anime is a failure.
2. Cohesion: it reads as one coherent narrative — no repetition loops, no contradictions of itself.
3. Continuity: it does not contradict the story so far. (If the chapter is a bonus/crossover chapter unrelated to the story so far, that is fine as long as the recap matches the pages.)
4. Form: it is a plain-prose recap only. Any refusal, meta-commentary about the task, commentary about the original work's structure, or addressing the reader is a failure.
Note: you see a sample of pages, not necessarily all of them — only fail the recap for content that contradicts a page you can see, never for content you merely did not see.{direction}

For every issue you report, quote the exact phrase from the recap that causes it and name the page that contradicts it. If you cannot quote it, do not report it.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


def sample_pages(pages: list, cap: int = VISION_JUDGE_PAGE_CAP) -> list:
    """Evenly spaced sample of at most `cap` pages for the vision judge."""
    if len(pages) <= cap:
        return list(pages)
    step = len(pages) / cap
    return [pages[int(i * step)] for i in range(cap)]


def _resize_for_judge(src: Path, dest: Path, max_px: int = VISION_JUDGE_MAX_PX) -> None:
    """Shrink a manga page so the vision-judge payload stays small.

    The judge only needs to spot contradictions between the recap and the
    pages, so full resolution is wasteful and can exceed provider payload
    limits (OpenRouter returns 413). JPEG is fine for this purpose.
    """
    with Image.open(src) as im:
        im.thumbnail((max_px, max_px))
        rgb = im.convert("RGB") if im.mode in ("RGBA", "P") else im
        rgb.save(dest, format="JPEG", quality=85)


def _resized_sample(
    pages: list, tmp_dir: Path, cap: int = VISION_JUDGE_PAGE_CAP
) -> list[Path]:
    """Return a capped, resized sample of pages, all inside ``tmp_dir``."""
    sampled = sample_pages(pages, cap)
    resized: list[Path] = []
    for src in sampled:
        dest = tmp_dir / f"{src.stem}.jpg"
        try:
            _resize_for_judge(src, dest)
        except Exception:
            # If a file isn't a readable image, fall back to the original and
            # let the model/provider surface the real error.
            dest = src
        resized.append(dest)
    return resized


JUDGE_TRANSLATION_PROMPT = """You are a strict evaluator for machine-translated manga dialogue. Chapter {chapter} of "{title}", translated from {lang} to {target}.

Below are the speech-bubble texts found on page {page}, each with its printed original and its translation:
---
{pairs}
---

Evaluate the translations on:
1. Language: every "translation" is actually written in {target}. A translation left in {lang} (or any other language) is a failure. Sound effects rendered as short {target} sound effects (e.g. "sigh...", "thud") count as translated.
2. Faithfulness: each translation conveys what its original says — no invented names, events, or dialogue absent from the original, and no outside knowledge of the actual manga/anime. Only meaning matters: do not fault punctuation, capitalization, or stylistic nuance.
3. Form: translations are dialogue/caption text only. Any refusal, meta-commentary about the task, or notes addressed to the reader inside a translation is a failure.
4. Scope: scanlation credits or watermarks must not appear as entries at all — flag any such entry. Pure sound effects ideally are not entries either; flag an SFX entry only when it was left untranslated.{cast}

For every issue you report, write a full sentence that names the entry number, explains the problem, and quotes the exact translation (or original) that causes it. A bare quote with no explanation is not a valid issue. If every entry is acceptable, pass.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


JUDGE_NARRATION_PROMPT = """You are a strict evaluator for a spoken narration of chapter {chapter} of the story "{title}".

Story so far BEFORE this chapter (context only):
---
{context}
---

Source material: sequential narrations of this chapter's page/section batches. The narration below may ONLY contain events and names that appear here:
---
{batches}
---

Narration to evaluate:
---
{narration}
---

Evaluate the narration on:
1. Faithfulness: every event and character name in the narration appears in the source material. Outside knowledge of the actual manga/book/anime is a failure.
2. Completeness: it tells the chapter's events in story order without compressing them away — a narration that skips beats present in the source material is a failure.
3. Cohesion: it reads as one continuous spoken narrative — no repetition loops, no contradictions of itself.
4. Continuity: it does not contradict the story so far. (If the chapter is a bonus/crossover chapter whose content is unrelated to the story so far, that is fine as long as the narration is faithful to the source material.)
5. Form: it is plain spoken prose only. Any headers, lists, stage directions, refusals, meta-commentary about the task, or addressing the reader is a failure.{direction}

For every issue you report, quote the exact phrase from the narration that causes it and name the source-material sentence it contradicts. If you cannot quote it, do not report it.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


JUDGE_NARRATION_VISION_PROMPT = """You are a strict evaluator for a spoken narration of chapter {chapter} of the story "{title}". You are given the chapter's actual pages (a sample of them) as images.

Story so far BEFORE this chapter (context only):
---
{context}
---

Narration to evaluate:
---
{narration}
---

Evaluate the narration on:
1. Fidelity: the events, character names, and dialogue beats in the narration match what is actually on the pages. Events or names not depicted on any page are a failure. Outside knowledge of the actual manga/book/anime is a failure.
2. Cohesion: it reads as one continuous spoken narrative, in story order — no repetition loops, no contradictions of itself.
3. Continuity: it does not contradict the story so far. (If the chapter is a bonus/crossover chapter unrelated to the story so far, that is fine as long as the narration matches the pages.)
4. Form: it is plain spoken prose only. Any headers, lists, stage directions, refusals, meta-commentary about the task, or addressing the reader is a failure.
Note: you see a sample of pages, not necessarily all of them — only fail the narration for content that contradicts a page you can see, never for content you merely did not see.{direction}

For every issue you report, quote the exact phrase from the narration that causes it and name the page that contradicts it. If you cannot quote it, do not report it.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


def _judge_direction_block(instruction: str) -> str:
    """The USER DIRECTION addendum for artifact judge prompts: the mandated-
    omission/mandated-style rule ('' when no steering instruction is in play,
    leaving prompts byte-identical). The evidence-quoting and output rules
    below the block are untouched and still apply."""
    instruction = instruction.strip()
    if not instruction:
        return ""
    return (
        "\n\nUSER DIRECTION — the reader steering this run asked:\n"
        f"{instruction}\n"
        "Omissions and style choices this direction mandates are NOT issues:"
        " content it says to skip is correctly absent, and a voice it asks"
        " for is not a form problem. Where the direction is silent, every"
        " criterion above applies as written — faithfulness to the source"
        " material (or pages) still applies in full."
    )


def _judge_cast_block(cast: str) -> str:
    """The CAST addendum for artifact judge prompts ('' when no registry is
    in play, leaving prompts byte-identical). Registry names are
    pre-approved: the artifact may use them for characters that appear, and
    that is not the outside-knowledge failure the faithfulness criterion
    guards against. `cast` is the writer-side block from
    characters.cast_block; only its list lines are reused."""
    lines = [
        line for line in cast.strip().splitlines() if line.startswith("- ")
    ]
    if not lines:
        return ""
    return (
        "\n\nKnown cast (the work's character registry, pre-approved):\n"
        + "\n".join(lines)
        + "\nThe artifact may use these names for characters that appear —"
        " that is NOT an outside-knowledge or faithfulness issue. Every"
        " other criterion still applies in full."
    )


def judge_artifact(
    adapter: ModelAdapter,
    model: str,
    text: str,
    batches: list[str],
    context: str | None,
    title: str,
    chapter_num: float,
    *,
    kind: str,
    pages: list | None = None,
    instruction: str = "",
    cast: str = "",
) -> Verdict:
    """Judge a chapter artifact (kind "recap" or "narration"). With `pages`
    (image paths), the adapter must be a vision model: the artifact is
    verified against the chapter's actual pages (evenly sampled) instead of
    only against the batch summaries. A steering `instruction` is shown with
    the mandated-omission rule (see _judge_direction_block); a character-
    registry `cast` block pre-approves its names (see _judge_cast_block)."""
    direction = _judge_direction_block(instruction) + _judge_cast_block(cast)
    if kind == "recap":
        prompt_text = JUDGE_RECAP_PROMPT
        prompt_vision = JUDGE_RECAP_VISION_PROMPT
        first = "(This is the first chapter recapped.)"
    else:
        prompt_text = JUDGE_NARRATION_PROMPT
        prompt_vision = JUDGE_NARRATION_VISION_PROMPT
        first = "(This is the first chapter narrated.)"
    if pages:
        prompt = prompt_vision.format(
            chapter=f"{chapter_num:g}",
            title=title,
            context=context or first,
            direction=direction,
            **{kind: text},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            resized = _resized_sample(pages, Path(tmpdir))
            return parse_verdict(
                adapter.generate(model, prompt, images=resized)
            )
    prompt = prompt_text.format(
        chapter=f"{chapter_num:g}",
        title=title,
        context=context or first,
        batches="\n\n".join(batches),
        direction=direction,
        **{kind: text},
    )
    return parse_verdict(adapter.generate(model, prompt))


def judge_recap(
    adapter: ModelAdapter,
    model: str,
    recap: str,
    batches: list[str],
    context: str | None,
    title: str,
    chapter_num: float,
    pages: list | None = None,
    instruction: str = "",
    cast: str = "",
) -> Verdict:
    """Judge a chapter recap; see judge_artifact."""
    return judge_artifact(
        adapter, model, recap, batches, context, title, chapter_num,
        kind="recap", pages=pages, instruction=instruction, cast=cast,
    )


def judge_narration(
    adapter: ModelAdapter,
    model: str,
    narration: str,
    batches: list[str],
    context: str | None,
    title: str,
    chapter_num: float,
    pages: list | None = None,
    instruction: str = "",
    cast: str = "",
) -> Verdict:
    """Judge a chapter narration; see judge_artifact."""
    return judge_artifact(
        adapter, model, narration, batches, context, title, chapter_num,
        kind="narration", pages=pages, instruction=instruction, cast=cast,
    )


def judge_context(
    adapter: ModelAdapter,
    model: str,
    candidate: str,
    previous: str | None,
    recap: str,
    title: str,
) -> Verdict:
    prompt = JUDGE_CONTEXT_PROMPT.format(
        title=title,
        previous=previous or "(This is the first chapter recapped.)",
        recap=recap,
        candidate=candidate,
    )
    return parse_verdict(adapter.generate(model, prompt))


JUDGE_CAST_PROMPT = """You are a strict evaluator for a character-registry update of the story "{title}".

Chapter recap the update was extracted from:
---
{recap}
---

Characters claimed to appear in this chapter:
---
{cast}
---

Evaluate the list:
1. Every listed character actually appears in the recap, named or clearly identifiable from it. An invented or merely presumed character is a failure.
2. Aliases and roles are supported by the recap — no invented details or outside knowledge of the actual manga/book/anime.
3. Form: the list is plain character data, not commentary on the task.

For every issue you report, quote the exact name or phrase that causes it. If you cannot quote it, do not report it.

Output ONLY JSON, no other text:
{{"pass": true}} or {{"pass": false, "issues": ["specific problem", ...]}}"""


def judge_cast(
    adapter: ModelAdapter,
    model: str,
    cast: list[dict],
    recap: str,
    title: str,
) -> Verdict:
    """Judge one chapter's extracted cast list against its artifact — the
    same faithfulness bar as the artifact judges, so registry entries are
    always traceable to a judged chapter."""
    lines = "\n".join(
        f"- {entry['name']}"
        + (f" (aliases: {', '.join(entry['aliases'])})" if entry["aliases"] else "")
        + (f" — {entry['role']}" if entry["role"] else "")
        for entry in cast
    )
    prompt = JUDGE_CAST_PROMPT.format(title=title, recap=recap, cast=lines)
    return parse_verdict(adapter.generate(model, prompt))


def judge_translation(
    adapter: ModelAdapter,
    model: str,
    bubbles,  # list of objects with .original / .translation (translate.Bubble)
    title: str,
    chapter_num: float,
    page: int,
    lang: str,
    target: str,
    *,
    cast: str = "",
) -> Verdict:
    """Judge a page's original/translation pairs. Deliberately text-only, even
    at --thinking high: attaching the page image made the vision judge
    degenerate (failing every entry with bare quotes or pedantic
    punctuation nitpicks — observed live with qwen3-vl-32b). The registry
    `cast` block pre-approves its names (see _judge_cast_block)."""
    pairs = "\n".join(
        f"{i}. original: {b.original}\n   translation: {b.translation}"
        for i, b in enumerate(bubbles, start=1)
    )
    prompt = JUDGE_TRANSLATION_PROMPT.format(
        chapter=f"{chapter_num:g}",
        title=title,
        lang=lang,
        target=target,
        page=page,
        pairs=pairs or "(no bubbles detected on this page)",
        cast=_judge_cast_block(cast),
    )
    return parse_verdict(adapter.generate(model, prompt))


JUDGE_RENDER_PROMPT = """You are evaluating the visual quality of a translated manga page overlay.

Chapter {chapter} of "{title}", page {page}.

Two images are shown:
1. The ORIGINAL page (with source-language text).
2. The RENDERED page (with English overlay boxes on top of the original art).

The rendered page has these numbered overlay boxes:
---
{bubbles}
---

For each box, evaluate:
1. original_text_visible: Is any non-English text from the original still visible around, under, or outside the overlay box? If yes, name the direction(s): left, right, top, bottom.
2. text_overflow: Does the English text touch or exceed the box edge?
3. unreadable: Is the English font too small, too cramped, or too low-contrast to read comfortably?
4. wrong_order: Does this box appear out of manga reading order (right-to-left, top-to-bottom)?

Rules:
- Only report problems you can actually see in the images.
- Ignore scanlation watermarks/credits; do not flag them.
- Sound effects (SFX) left untranslated in the art are fine; do not flag them.

Output ONLY JSON:
{{"pass": true}}
or
{{"pass": false, "issues": [
  {{"bubble_index": 0, "problem": "original_text_visible", "directions": ["left"]}},
  {{"bubble_index": 2, "problem": "text_overflow"}}
]}}

Valid problems: original_text_visible, text_overflow, unreadable, wrong_order."""


def _bubble_descriptions(bubbles) -> str:
    """Textual bubble list for the render judge prompt."""
    lines = []
    for i, b in enumerate(bubbles):
        text = b.translation.replace("\n", " ")
        lines.append(f"{i}. {text}")
    return "\n".join(lines) or "(no overlay boxes)"


def _resize_pair_for_judge(
    original: Path, rendered: Path, tmp_dir: Path, max_px: int = VISION_JUDGE_MAX_PX
) -> tuple[Path, Path]:
    """Resize original and rendered pages for the render judge."""
    orig_dest = tmp_dir / "original.jpg"
    rend_dest = tmp_dir / "rendered.jpg"
    _resize_for_judge(original, orig_dest, max_px)
    _resize_for_judge(rendered, rend_dest, max_px)
    return orig_dest, rend_dest


def parse_render_verdict(raw: str) -> RenderVerdict:
    """Extract a structured render verdict from model output.

    Unparseable output is treated as a pass so a broken judge never stalls
    the pipeline.
    """
    match = _JSON_OBJ_RE.search(raw)
    if not match:
        return RenderVerdict(passed=True, issues=[RenderIssue(None, "judge output was not JSON")])
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return RenderVerdict(passed=True, issues=[RenderIssue(None, "judge output was invalid JSON")])
    passed = bool(data.get("pass", data.get("passed", True)))
    issues: list[RenderIssue] = []
    for entry in data.get("issues", []):
        if not isinstance(entry, dict):
            continue
        idx = entry.get("bubble_index")
        if isinstance(idx, (int, float)):
            idx = int(idx)
        else:
            idx = None
        directions = entry.get("directions") or []
        if isinstance(directions, str):
            directions = [directions]
        issues.append(
            RenderIssue(
                bubble_index=idx,
                problem=str(entry.get("problem") or "").strip(),
                directions=[str(d).strip().lower() for d in directions if str(d).strip()],
                suggestion=str(entry.get("suggestion") or "").strip(),
            )
        )
    return RenderVerdict(passed=passed, issues=issues)


def judge_render(
    adapter: ModelAdapter,
    model: str,
    original_page: Path,
    rendered_page: Path,
    bubbles,  # list of objects with .translation (translate.Bubble)
    title: str,
    chapter_num: float,
    page: int,
) -> RenderVerdict:
    """Judge the visual quality of a rendered overlay page.

    Returns a structured verdict with per-bubble issues so the renderer can
    apply targeted adjustments.
    """
    prompt = JUDGE_RENDER_PROMPT.format(
        chapter=f"{chapter_num:g}",
        title=title,
        page=page,
        bubbles=_bubble_descriptions(bubbles),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        orig, rend = _resize_pair_for_judge(
            original_page, rendered_page, Path(tmpdir)
        )
        raw = adapter.generate(model, prompt, images=[orig, rend])
    return parse_render_verdict(raw)
