"""Recap pipeline: chapter pages -> batch summaries -> chapter artifact ->
rolling series context. Calls models via models/registry only.

One pipeline with a detail knob (DETAIL_LEVELS): `standard` is the classic
recap, `full` is the classic narration (a complete in-order retelling — the
old narrate pipeline folded in here), and gist/brief/detailed are summary
grains in between. The
detail selects the prompt pair and the judge (full keeps the narration judge
with the completeness axis; the other grains use the recap judge); thinking
levels, retry policy, and the rolling-context rules are identical across
grains. Artifacts land in the recaps table with their detail.

Prompt rules come from the Phase 0 live spike: the naive prompt hallucinated
a protagonist name imported from another series. The strict rules below
("only use names printed on the page", "no outside knowledge", "say when
unsure") fixed it; they must stay in every prompt from day one.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from entertainment_harness.library import works
from entertainment_harness.config import Config, data_dir
from entertainment_harness.db import utcnow
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.library.importer import word_chunks
from entertainment_harness.pipelines.judge import (
    ATTEMPTS,
    MAX_ATTEMPTS,
    judge_context,
    judge_narration,
    judge_recap,
)
from entertainment_harness.library import (
    LibraryError,
    advance_progress,
    get_progress,
    parse_chapter_spec,
)
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import (
    get_judge_model,
    get_text_model,
    get_vision_model,
)
from entertainment_harness.sources.mangadex import MangaDexClient

DEFAULT_BATCH_SIZE = 4
BOOK_CHUNK_WORDS = 3000  # words per text-model call on the book path

# Detail grains, ranked lowest to highest. `standard` is the classic recap
# (pre-merge recaps rows map here); `full` is the classic narration (pre-merge
# narrations rows map there). A chapter whose artifact is already at the
# requested grain is skipped; a higher-grain artifact is never downgraded
# unless forced (--all/--chapter).
DETAIL_LEVELS = ("gist", "brief", "standard", "detailed", "full")
DETAIL_RANK = {name: rank for rank, name in enumerate(DETAIL_LEVELS)}


def video_kind_for_detail(detail: str) -> str:
    """The videos.kind a chapter's video renders as: 'full' artifacts are
    narrations (verbatim retelling, video-narration workdir); every lower
    grain is a recap video. One video per chapter (unify-recap-narrate)."""
    return "narration" if detail == "full" else "recap"


def user_direction_block(instruction: str) -> str:
    """The USER DIRECTION prompt block for a steering instruction
    (unify-recap-narrate Phase 5): appended to the rules block of every
    grain's batch/combine prompts. '' (or whitespace) -> '', so prompts
    render byte-identically to the no-instruction case."""
    instruction = instruction.strip()
    if not instruction:
        return ""
    return (
        "\n\nUSER DIRECTION — the reader steering this run asked:\n"
        f"{instruction}\n"
        "Follow it: content it says to skip stays out of the artifact, and a"
        " voice or emphasis it asks for applies. It never overrides the rules"
        " above."
    )

STRICT_RULES = """Rules:
- Only use character names actually printed on these pages; describe unnamed characters by their appearance or role.
- Do not use outside knowledge of any manga or anime.
- If you are unsure what is happening, say so instead of guessing.
- Ignore credits/promotional pages, author notes, table-of-contents pages, and any other front/back matter. Do not mention them in the summary."""

BOOK_RULES = """Rules:
- Only use character names that actually appear in this text; describe unnamed characters by their role.
- Do not use outside knowledge of any book, manga, or anime.
- If you are unsure what is happening, say so instead of guessing.
- Ignore forewords, afterwords, author notes, and any other non-story front/back matter. Do not mention them in the summary."""

BATCH_PROMPT = """You are reading pages {start}-{end} of {total} from chapter {chapter} of the manga "{title}". The pages are in {lang}; write your summary in English.
{context}
Summarize concisely what happens on THESE pages.

{rules}"""

COMBINE_PROMPT = """You are a recap writer. Below are sequential page-batch summaries of chapter {chapter} of the manga "{title}", produced by a page-reading model.

Write one coherent English recap of the chapter: a few paragraphs of plain prose.
- Output ONLY the recap: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. If some batches are repetitive or describe credits/promotional pages, author notes, table-of-contents pages, or other front/back matter, ignore that noise and recap the story events only. Do not mention non-story content in the recap.

{rules}

Batch summaries:
---
{batches}
---"""

GIST_BATCH_PROMPT = """You are reading pages {start}-{end} of {total} from chapter {chapter} of the manga "{title}". The pages are in {lang}; write your summary in English.
{context}
Summarize what happens on THESE pages in 2-4 sentences.

{rules}"""

GIST_COMBINE_PROMPT = """You are a recap writer. Below are sequential page-batch summaries of chapter {chapter} of the manga "{title}", produced by a page-reading model.

Write one gist of the chapter: what happened, in 2-4 sentences of plain prose.
- Output ONLY the gist: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. If some batches are repetitive or describe credits/promotional pages, author notes, table-of-contents pages, or other front/back matter, ignore that noise and cover the story events only. Do not mention non-story content in the gist.

{rules}

Batch summaries:
---
{batches}
---"""

BRIEF_BATCH_PROMPT = """You are reading pages {start}-{end} of {total} from chapter {chapter} of the manga "{title}". The pages are in {lang}; write your summary in English.
{context}
Summarize what happens on THESE pages in one tight paragraph.

{rules}"""

BRIEF_COMBINE_PROMPT = """You are a recap writer. Below are sequential page-batch summaries of chapter {chapter} of the manga "{title}", produced by a page-reading model.

Write one brief summary of the chapter: a single tight paragraph of plain prose.
- Output ONLY the summary: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. If some batches are repetitive or describe credits/promotional pages, author notes, table-of-contents pages, or other front/back matter, ignore that noise and summarize the story events only. Do not mention non-story content in the summary.

{rules}

Batch summaries:
---
{batches}
---"""

DETAILED_BATCH_PROMPT = """You are reading pages {start}-{end} of {total} from chapter {chapter} of the manga "{title}". The pages are in {lang}; write your summary in English.
{context}
Summarize what happens on THESE pages, covering every notable beat — events, reveals, turning points — in compressed prose.

{rules}"""

DETAILED_COMBINE_PROMPT = """You are a recap writer. Below are sequential page-batch summaries of chapter {chapter} of the manga "{title}", produced by a page-reading model.

Write one detailed recap of the chapter: an expanded summary covering every notable beat — events, reveals, turning points — in compressed plain prose. Do not compress beats away, but do not retell the chapter blow-by-blow either.
- Output ONLY the recap: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. If some batches are repetitive or describe credits/promotional pages, author notes, table-of-contents pages, or other front/back matter, ignore that noise and recap the story events only. Do not mention non-story content in the recap.

{rules}

Batch summaries:
---
{batches}
---"""

NARRATION_BATCH_PROMPT = """You are reading pages {start}-{end} of {total} from chapter {chapter} of the manga "{title}". The pages are in {lang}; write your narration in English.
{context}
Narrate in detail what happens on THESE pages, in story order: every event and reveal, and the substance of each dialogue exchange (who says what, in paraphrase). Write plain prose that could be read aloud — do not compress events away.

{rules}"""

NARRATION_COMBINE_PROMPT = """You are a narration writer. Below are sequential page-batch narrations of chapter {chapter} of the manga "{title}", produced by a page-reading model.

Weave them into one continuous English narration of the whole chapter: present tense, plain spoken prose that could be read aloud, events in story order.
- Full coverage: keep every story beat from the batch narrations; do not compress the chapter into a summary. Smooth out repetition between batches instead of repeating events.
- Output ONLY the narration: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. If some batches are repetitive or describe credits/promotional pages, author notes, table-of-contents pages, or other front/back matter, ignore that noise and narrate the story events only. Do not mention non-story content in the narration.

{rules}

Batch narrations:
---
{batches}
---"""

CONTEXT_PROMPT = """Here is the current "story so far" for "{title}":
---
{previous}
---
Here is the recap of chapter {chapter}:
---
{recap}
---
Write an updated "story so far" in English that incorporates the new chapter. Keep it under 300 words: compress older events instead of dropping key facts (names, relationships, major plot turns). Never invent names or events not present in the inputs above. Never refuse or comment on the task; the inputs are always provided above, even if imperfect. If the new chapter recap is unrelated to the story so far (a bonus/crossover chapter from another series), output the previous story-so-far unchanged. Output ONLY the updated story-so-far text: no preamble, no notes, no word counts, no headers."""


# Meta-commentary / refusal markers in a context update. Combined with a
# length-collapse check before the old context is overwritten — a real update
# adds a chapter's events, it never shrinks the summary to a fraction while
# talking about the task itself.
_CONTEXT_META_RE = re.compile(
    r"cannot|unable|does not belong|unrelated|no events|remains unchanged"
    r"|different (manga|series|story)",
    re.IGNORECASE,
)


def _suspicious_context(new: str, previous: str | None) -> bool:
    if not previous:
        return False
    return bool(_CONTEXT_META_RE.search(new)) and len(new.split()) < 0.6 * len(
        previous.split()
    )

BOOK_BATCH_PROMPT = """You are reading excerpt {start} of {total} from the book "{title}" (the excerpt below is all that matters — part/section numbering is an artifact of splitting the file for processing).
{context}
Summarize concisely what happens in THIS excerpt.

{rules}

Excerpt text:
---
{chunk}
---"""

BOOK_COMBINE_PROMPT = """You are a recap writer. Below are sequential summaries of consecutive excerpts from the book "{title}" (part {chapter} of the imported text), produced by a text-reading model.

Write one coherent English recap of the whole excerpt sequence: a few paragraphs of plain prose.
- Output ONLY the recap: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. The part/section numbering is an artifact of splitting the file for processing — do not comment on it or on the original work's structure. Ignore forewords, afterwords, author notes, and any other non-story front/back matter; do not mention them in the recap.

{rules}

Excerpt summaries:
---
{batches}
---"""

BOOK_GIST_BATCH_PROMPT = """You are reading excerpt {start} of {total} from the book "{title}" (the excerpt below is all that matters — part/section numbering is an artifact of splitting the file for processing).
{context}
Summarize what happens in THIS excerpt in 2-4 sentences.

{rules}

Excerpt text:
---
{chunk}
---"""

BOOK_GIST_COMBINE_PROMPT = """You are a recap writer. Below are sequential summaries of consecutive excerpts from the book "{title}" (part {chapter} of the imported text), produced by a text-reading model.

Write one gist of the whole excerpt sequence: what happened, in 2-4 sentences of plain prose.
- Output ONLY the gist: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. The part/section numbering is an artifact of splitting the file for processing — do not comment on it or on the original work's structure. Ignore forewords, afterwords, author notes, and any other non-story front/back matter; do not mention them in the gist.

{rules}

Excerpt summaries:
---
{batches}
---"""

BOOK_BRIEF_BATCH_PROMPT = """You are reading excerpt {start} of {total} from the book "{title}" (the excerpt below is all that matters — part/section numbering is an artifact of splitting the file for processing).
{context}
Summarize what happens in THIS excerpt in one tight paragraph.

{rules}

Excerpt text:
---
{chunk}
---"""

BOOK_BRIEF_COMBINE_PROMPT = """You are a recap writer. Below are sequential summaries of consecutive excerpts from the book "{title}" (part {chapter} of the imported text), produced by a text-reading model.

Write one brief summary of the whole excerpt sequence: a single tight paragraph of plain prose.
- Output ONLY the summary: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. The part/section numbering is an artifact of splitting the file for processing — do not comment on it or on the original work's structure. Ignore forewords, afterwords, author notes, and any other non-story front/back matter; do not mention them in the summary.

{rules}

Excerpt summaries:
---
{batches}
---"""

BOOK_DETAILED_BATCH_PROMPT = """You are reading excerpt {start} of {total} from the book "{title}" (the excerpt below is all that matters — part/section numbering is an artifact of splitting the file for processing).
{context}
Summarize what happens in THIS excerpt, covering every notable beat — events, reveals, turning points — in compressed prose.

{rules}

Excerpt text:
---
{chunk}
---"""

BOOK_DETAILED_COMBINE_PROMPT = """You are a recap writer. Below are sequential summaries of consecutive excerpts from the book "{title}" (part {chapter} of the imported text), produced by a text-reading model.

Write one detailed recap of the whole excerpt sequence: an expanded summary covering every notable beat — events, reveals, turning points — in compressed plain prose. Do not compress beats away, but do not retell the text blow-by-blow either.
- Output ONLY the recap: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. The part/section numbering is an artifact of splitting the file for processing — do not comment on it or on the original work's structure. Ignore forewords, afterwords, author notes, and any other non-story front/back matter; do not mention them in the recap.

{rules}

Excerpt summaries:
---
{batches}
---"""

BOOK_NARRATION_BATCH_PROMPT = """You are reading excerpt {start} of {total} from the book "{title}" (the excerpt below is all that matters — part/section numbering is an artifact of splitting the file for processing).
{context}
Narrate in detail what happens in THIS excerpt, in story order: every event and reveal, and the substance of each dialogue exchange. Write plain prose that could be read aloud — do not compress events away.

{rules}

Excerpt text:
---
{chunk}
---"""

BOOK_NARRATION_COMBINE_PROMPT = """You are a narration writer. Below are sequential narrations of consecutive excerpts from the book "{title}" (part {chapter} of the imported text), produced by a text-reading model.

Weave them into one continuous English narration of the whole excerpt sequence: present tense, plain spoken prose that could be read aloud, events in story order.
- Full coverage: keep every story beat; do not compress the text into a summary. Smooth out repetition between excerpts instead of repeating events.
- Output ONLY the narration: no headers, no lists, no tables, no preamble, no commentary, never address the reader.
- Never refuse. The part/section numbering is an artifact of splitting the file for processing — do not comment on it or on the original work's structure. Ignore forewords, afterwords, author notes, and any other non-story front/back matter; do not mention them in the narration.

{rules}

Excerpt narrations:
---
{batches}
---"""

# Prompt pair per detail grain: (batch prompt, combine prompt). `standard` is
# the classic recap pair, `full` the classic narration pair; gist/brief/
# detailed are summary variants of the recap prompt.
PROMPTS = {
    "gist": (GIST_BATCH_PROMPT, GIST_COMBINE_PROMPT),
    "brief": (BRIEF_BATCH_PROMPT, BRIEF_COMBINE_PROMPT),
    "standard": (BATCH_PROMPT, COMBINE_PROMPT),
    "detailed": (DETAILED_BATCH_PROMPT, DETAILED_COMBINE_PROMPT),
    "full": (NARRATION_BATCH_PROMPT, NARRATION_COMBINE_PROMPT),
}
BOOK_PROMPTS = {
    "gist": (BOOK_GIST_BATCH_PROMPT, BOOK_GIST_COMBINE_PROMPT),
    "brief": (BOOK_BRIEF_BATCH_PROMPT, BOOK_BRIEF_COMBINE_PROMPT),
    "standard": (BOOK_BATCH_PROMPT, BOOK_COMBINE_PROMPT),
    "detailed": (BOOK_DETAILED_BATCH_PROMPT, BOOK_DETAILED_COMBINE_PROMPT),
    "full": (BOOK_NARRATION_BATCH_PROMPT, BOOK_NARRATION_COMBINE_PROMPT),
}


def model_tag(info: ModelInfo) -> str:
    """Full model tag incl. quant, for recap attribution."""
    return f"{info.name} ({info.quant})" if info.quant else info.name


def _source_dir(series_id: str, chapter_id: str) -> Path:
    """Prefer the new works layout, but fall back to the legacy manga cache."""
    new_dir = works.source_dir(series_id, chapter_id)
    old_dir = data_dir() / "manga" / series_id / chapter_id
    if new_dir.is_dir() or not old_dir.is_dir():
        return new_dir
    return old_dir


def _translated_dir(series_id: str, chapter_id: str) -> Path:
    """Prefer the new works layout, but fall back to the legacy translated dir."""
    new_dir = works.translated_dir(series_id, chapter_id)
    old_dir = data_dir() / "manga" / series_id / chapter_id / "translated"
    if new_dir.is_dir() or not old_dir.is_dir():
        return new_dir
    return old_dir


def _book_part_path(series: sqlite3.Row, chapter: sqlite3.Row) -> Path:
    """Prefer the new works layout, but fall back to the legacy books dir."""
    num = int(chapter["chapter_num"])
    new_path = works.source_dir(series["id"], chapter["id"]) / f"ch-{num:03d}.txt"
    old_path = data_dir() / "books" / series["id"] / f"ch-{num:03d}.txt"
    if new_path.exists() or not old_path.exists():
        return new_path
    return old_path


# SQL rank of recaps.detail, mirroring DETAIL_RANK (unknown values rank as
# 'standard'; a missing recaps row ranks as NULL via the LEFT JOIN).
_DETAIL_RANK_SQL = (
    "CASE r.detail WHEN 'gist' THEN 0 WHEN 'brief' THEN 1"
    " WHEN 'standard' THEN 2 WHEN 'detailed' THEN 3 WHEN 'full' THEN 4"
    " ELSE 2 END"
)


def pending_chapters(
    conn: sqlite3.Connection,
    series_id: str,
    langs: list[str] | None = None,
    detail: str = "standard",
    fill_gaps: bool = False,
) -> list[sqlite3.Row]:
    """Chapters pending an artifact at the `detail` grain, two buckets
    (unify-recap-narrate selection rule): (a) chapters after
    progress.last_read_chapter with NO recaps row at any grain — processing
    them advances progress and folds the rolling context; (b) chapters whose
    artifact sits at a lower grain — upgrade candidates, ungated on
    last_read. Chapters already at (or above) the requested grain are
    skipped. Each row carries the chapter's current artifact grain as
    `artifact_detail` (NULL for bucket a) so callers can tell the buckets
    apart. langs=None disables the language filter (used by --translated,
    where the translation step normalizes any source language). Callers cap
    the list (see select_chapters).

    With fill_gaps (recap/narrate --video), a third bucket joins in:
    chapters at or before last_read with NO artifact AND no usable video —
    behind the read frontier, where folding them now would double-fold the
    rolling context, so recap_series generates them standalone (story-so-far
    withheld, no fold, no progress advance) and the video callback builds
    their video. Chapters with a usable video row are dropped from this
    bucket: nothing is missing for them."""
    if detail not in DETAIL_LEVELS:
        raise ValueError(
            f"unknown detail {detail!r}; expected one of: {', '.join(DETAIL_LEVELS)}"
        )
    last_read = get_progress(conn, series_id)
    read = last_read if last_read is not None else float("-inf")
    gap_clause, gap_params = "", ()
    if fill_gaps:
        gap_clause = " OR (r.chapter_id IS NULL AND c.chapter_num <= ?)"
        gap_params = (read,)
    if langs:
        placeholders = ", ".join("?" for _ in langs)
        lang_filter = f" AND c.lang IN ({placeholders})"
        params = ((series_id,) + tuple(langs) + (read, DETAIL_RANK[detail])
                  + gap_params)
    else:
        lang_filter = ""
        params = (series_id, read, DETAIL_RANK[detail]) + gap_params
    rows = conn.execute(
        "SELECT c.*, r.detail AS artifact_detail FROM chapters c"
        " LEFT JOIN recaps r ON r.chapter_id = c.id"
        " WHERE c.series_id = ? AND c.chapter_num IS NOT NULL"
        + lang_filter +
        " AND ((r.chapter_id IS NULL AND c.chapter_num > ?)"
        f" OR (r.chapter_id IS NOT NULL AND {_DETAIL_RANK_SQL} < ?)"
        + gap_clause + ")"
        " ORDER BY c.chapter_num ASC",
        params,
    ).fetchall()
    if not fill_gaps:
        return rows
    # Drop gap-bucket chapters that still have a usable video: wiped rows
    # (or none) count as missing — those are the ones worth filling.
    usable = set()
    for v in conn.execute(
        "SELECT from_chapter, to_chapter, COUNT(*) AS n,"
        " SUM(wiped_at IS NOT NULL) AS w FROM videos WHERE series_id = ?"
        " GROUP BY from_chapter, to_chapter",
        (series_id,),
    ):
        if v["from_chapter"] == v["to_chapter"] and v["n"] > v["w"]:
            usable.add(v["from_chapter"])
    return [
        r for r in rows
        if not (r["artifact_detail"] is None and r["chapter_num"] <= read
                and r["chapter_num"] in usable)
    ]


def get_context(conn: sqlite3.Connection, series_id: str) -> str | None:
    row = conn.execute(
        "SELECT rolling_summary FROM series_context WHERE series_id = ?", (series_id,)
    ).fetchone()
    return row["rolling_summary"] if row else None


def get_context_through(conn: sqlite3.Connection, series_id: str) -> float | None:
    """The highest chapter number folded into the rolling story-so-far."""
    row = conn.execute(
        "SELECT through_chapter FROM series_context WHERE series_id = ?", (series_id,)
    ).fetchone()
    return row["through_chapter"] if row else None


def setup_models(
    config: Config,
    profile: HardwareProfile,
    is_book: bool,
    thinking: str,
    *,
    book_role: str,
    text_role: str,
    judge_role: str,
    log: Callable[[str], None],
):
    """Resolve text/vision/judge models for a recap-style pipeline, log the
    selection, and ensure adapters. Returns (text, vision, judge,
    max_attempts); vision and judge are None when not in play. The *_role
    strings only feed the log lines."""
    text = get_text_model(config, profile)
    vision = None if is_book else get_vision_model(config, profile)
    for selection in (vision, text):
        if selection is not None and selection.warning:
            log(f"Warning: {selection.warning}")
    if is_book:
        log(f"Using {model_tag(text.info)} for {book_role} (book).")
    else:
        log(f"Using {model_tag(vision.info)} for page reading.")
        log(f"Using {model_tag(text.info)} for {text_role}.")
        vision.adapter.ensure(vision.info.name)
    text.adapter.ensure(text.info.name)

    judge = None
    if thinking != "low":
        judge = get_judge_model(config, profile)
        if judge.warning:
            log(f"Warning: {judge.warning}")
        if thinking == "high" and not is_book:
            log(f"Using {model_tag(vision.info)} for {judge_role} judging"
                f" (vision-verified), {model_tag(judge.info)} for context judging.")
        else:
            log(f"Using {model_tag(judge.info)} for judging.")
        judge.adapter.ensure(judge.info.name)
    max_attempts = ATTEMPTS.get(thinking, MAX_ATTEMPTS)
    return text, vision, judge, max_attempts


def select_chapters(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    config: Config,
    *,
    detail: str,
    chapter_num: float | None,
    chapters_spec: str | None = None,
    all_chapters: bool,
    max_chapters: int,
    translated: bool,
    fill_gaps: bool = False,
    verb: str,
    log: Callable[[str], None],
) -> list[sqlite3.Row]:
    """Pick the chapters to process: exactly chapter_num (--chapter), the
    synced chapters inside chapters_spec (--chapters, e.g. '1-3,4,6-10'),
    every synced chapter (--all, processed or not), or the pending ones
    (default: the grain-aware pending set, see pending_chapters). The list is
    capped at `max_chapters`. `detail` is the requested artifact grain (see
    DETAIL_LEVELS); `verb` ("recap"/"narrate") only feeds the log lines.
    `fill_gaps` adds the --video gap bucket (see pending_chapters). Returns
    [] (after logging why) when there is nothing to do."""
    langs = config.library.langs
    if translated or not langs:
        lang_filter = ""
        lang_params: tuple = ()
    else:
        placeholders = ", ".join("?" for _ in langs)
        lang_filter = f" AND c.lang IN ({placeholders})"
        lang_params = tuple(langs)
    if chapter_num is not None:
        row = conn.execute(
            "SELECT c.*, r.detail AS artifact_detail FROM chapters c"
            " LEFT JOIN recaps r ON r.chapter_id = c.id"
            " WHERE c.series_id = ? AND c.chapter_num = ?" + lang_filter,
            (series["id"], chapter_num) + lang_params,
        ).fetchone()
        if row is None:
            log(f"Chapter {chapter_num:g} is not synced for this series."
                if translated else
                f"Chapter {chapter_num:g} ({', '.join(langs)}) is not synced for this series.")
            return []
        return [row]
    if chapters_spec is not None:
        # --chapters: exactly the synced chapters inside the spec, in order.
        ranges = parse_chapter_spec(chapters_spec)
        spec_filter = " OR ".join(
            "(c.chapter_num >= ? AND c.chapter_num <= ?)" for _ in ranges
        )
        todo = conn.execute(
            "SELECT c.*, r.detail AS artifact_detail FROM chapters c"
            " LEFT JOIN recaps r ON r.chapter_id = c.id"
            " WHERE c.series_id = ? AND c.chapter_num IS NOT NULL AND ("
            + spec_filter + ")" + lang_filter
            + " ORDER BY c.chapter_num ASC",
            (series["id"],)
            + tuple(bound for rng in ranges for bound in rng)
            + lang_params,
        ).fetchall()
        if not todo:
            log(f"No synced chapters match '--chapters {chapters_spec}'.")
        return list(todo)
    if all_chapters:
        # --all: every synced chapter, processed or not, in order.
        todo = conn.execute(
            "SELECT c.*, r.detail AS artifact_detail FROM chapters c"
            " LEFT JOIN recaps r ON r.chapter_id = c.id"
            " WHERE c.series_id = ? AND c.chapter_num IS NOT NULL" + lang_filter
            + " ORDER BY c.chapter_num ASC",
            (series["id"],) + lang_params,
        ).fetchall()
    else:
        # Default: the grain-aware pending set, in order.
        todo = pending_chapters(
            conn, series["id"],
            langs=None if translated else langs, detail=detail,
            fill_gaps=fill_gaps,
        )
    if len(todo) > max_chapters:
        log(f"{len(todo)} chapters to {verb}; capping at {max_chapters}"
            " (--max-chapters). Re-run to continue.")
        todo = todo[:max_chapters]
    if not todo:
        log(f"Nothing to {verb}: no synced chapters."
            if all_chapters else
            f"Nothing to {verb}: no chapters pending at '{detail}' detail.")
    return list(todo)


def chapter_needs_work(
    conn: sqlite3.Connection,
    series_id: str,
    chapter: sqlite3.Row,
    *,
    detail: str,
    want_video: bool,
) -> bool:
    """False only when the chapter is fully done: a recap at exactly `detail`
    (select_chapters' LEFT JOIN exposes it as `artifact_detail`) and, when
    `want_video`, a non-wiped single-chapter video row. Used by skip_done
    runs to leave finished chapters untouched inside a forced scope."""
    if chapter["artifact_detail"] != detail:
        return True
    if not want_video:
        return False
    video = conn.execute(
        "SELECT 1 FROM videos WHERE series_id = ? AND from_chapter = ?"
        " AND to_chapter = ? AND wiped_at IS NULL LIMIT 1",
        (series_id, chapter["chapter_num"], chapter["chapter_num"]),
    ).fetchone()
    return video is None


def resolve_pages(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    chapter: sqlite3.Row,
    config: Config,
    profile: HardwareProfile,
    client_factory: Callable[[], MangaDexClient],
    client: MangaDexClient | None,
    *,
    translated: bool,
    thinking: str,
    progress,
    log: Callable[[str], None],
) -> tuple[list[Path], str]:
    """Resolve the pages a manga chapter is read from: the translated overlay
    when --translated applies (translating first if needed), else the cached /
    downloaded originals. Returns (pages, page_lang)."""
    langs = config.library.langs
    target_lang = langs[0] if langs else "en"
    chapter_num = chapter["chapter_num"]
    dest = _source_dir(series["id"], chapter["id"])
    page_lang = chapter["lang"] or "?"
    if translated and chapter["lang"] not in langs:
        translated_dir = _translated_dir(series["id"], chapter["id"])
        if not (
            translated_dir.is_dir() and any(translated_dir.glob("*.png"))
        ):
            log(f"Translating chapter {chapter_num:g} into {target_lang} first...")
            from entertainment_harness.pipelines.translate import translate_chapters

            translate_chapters(
                conn, series, config, profile,
                chapter_num=chapter_num, client=client,
                thinking=thinking, progress=progress, log=log,
            )
        else:
            log(f"Chapter {chapter_num:g} translation: cached"
                f" ({translated_dir})")
        if translated_dir.is_dir() and any(translated_dir.glob("*.png")):
            return sorted(translated_dir.glob("*.png")), target_lang
        # Re-evaluate after translation in case a legacy translator wrote the old path.
        translated_dir = _translated_dir(series["id"], chapter["id"])
        if translated_dir.is_dir() and any(translated_dir.glob("*.png")):
            return sorted(translated_dir.glob("*.png")), target_lang
        # translate skipped it (already target lang) — originals
    return (
        chapter_pages(config, client_factory, series["id"], chapter["id"], dest, log),
        page_lang,
    )


def chapter_pages(
    config: Config,
    client_factory: Callable[[], MangaDexClient],
    series_id: str,
    chapter_id: str,
    dest: Path,
    log: Callable[[str], None],
) -> list[Path]:
    """Local cache -> HF store -> source, in that order. The source client is
    only constructed when a download is actually needed (imported works have
    no source client at all)."""
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def _image_files(path: Path) -> list[Path]:
        if not path.is_dir():
            return []
        return sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    cached = _image_files(dest)
    if cached:
        log(f"  pages: cached locally ({dest})")
        return cached
    log(f"  pages: downloading from the source -> {dest}")
    return [
        p for p in client_factory().download_pages(chapter_id, dest)
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]


def book_chapter(
    series: sqlite3.Row,
    chapter: sqlite3.Row,
    text,
    context_block: str,
    log: Callable[[str], None],
    *,
    batch_prompt: str,
    combine_prompt: str,
    verb: str,
    instruction: str = "",
    cast: str = "",
) -> tuple[list[str], Callable[[str], str]]:
    """Process one stored text part with the text-role model (no vision).

    Returns (section outputs, generate) where generate(feedback) produces the
    part artifact — with the judge's feedback appended when retrying. The
    prompt pair and the log verb differ per pipeline (recap/narration).
    """
    rules = BOOK_RULES + cast + user_direction_block(instruction)
    part_path = _book_part_path(series, chapter)
    if not part_path.exists():
        raise LibraryError(
            f"Text for part {chapter['chapter_num']:g} is missing: {part_path}"
        )
    chunks = word_chunks(part_path.read_text(), BOOK_CHUNK_WORDS)
    log(f"{verb} part {chapter['chapter_num']:g} ({len(chunks)} sections)...")
    outputs: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = batch_prompt.format(
            start=i,
            total=len(chunks),
            chapter=f"{chapter['chapter_num']:g}",
            title=series["title"],
            context=context_block,
            rules=rules,
            chunk=chunk,
        )
        log(f"  section {i} of {len(chunks)}...")
        outputs.append(text.adapter.generate(text.info.name, prompt))

    def generate(feedback: str = "") -> str:
        if len(outputs) == 1 and not feedback:
            return outputs[0]
        combine = combine_prompt.format(
            chapter=f"{chapter['chapter_num']:g}",
            title=series["title"],
            rules=rules,
            batches="\n\n".join(outputs),
        )
        return text.adapter.generate(text.info.name, combine + feedback)

    return outputs, generate


def _recap_book_chapter(
    series: sqlite3.Row,
    chapter: sqlite3.Row,
    text,
    context_block: str,
    log: Callable[[str], None],
    *,
    detail: str = "standard",
    instruction: str = "",
    cast: str = "",
) -> tuple[list[str], Callable[[str], str]]:
    """Summarize one stored text part at the given detail grain; see
    book_chapter."""
    batch_prompt, combine_prompt = BOOK_PROMPTS[detail]
    return book_chapter(
        series, chapter, text, context_block, log,
        batch_prompt=batch_prompt, combine_prompt=combine_prompt,
        verb="Narrating" if detail == "full" else "Recapping",
        instruction=instruction, cast=cast,
    )


def judge_loop(
    generate: Callable[[str], str],
    judge_fn,
    log: Callable[[str], None],
    label: str,
    max_attempts: int = MAX_ATTEMPTS,
):
    """generate(feedback) -> text; judge_fn(text) -> Verdict. Regenerate with
    the judge's issues fed back until a pass, bounded by max_attempts. Returns
    (text, verdict); the caller decides what a final failure means. On
    persistent failure the FIRST attempt is returned: retries can over-comply
    with a mistaken critique (feedback poisoning), so the unpoisoned first
    version is the safest fallback."""
    feedback = ""
    text_out = ""
    first_text = ""
    for attempt in range(1, max_attempts + 1):
        text_out = generate(feedback)
        if attempt == 1:
            first_text = text_out
        verdict = judge_fn(text_out)
        if verdict.passed:
            return text_out, verdict
        log(
            f"  judge rejected {label} (attempt {attempt}/{max_attempts}):"
            f" {'; '.join(verdict.issues)}"
        )
        feedback = (
            "\n\nA previous attempt was rejected by the evaluator for these"
            " reasons:\n"
            + "\n".join(f"- {i}" for i in verdict.issues)
            + "\nFix these problems in the new version."
        )
    log(
        f"  warning: {label} failed the judge {max_attempts} times;"
        " keeping the first version."
    )
    return first_text, verdict


def manga_batches(
    vision,
    pages: list[Path],
    batch_size: int,
    context_block: str,
    series: sqlite3.Row,
    chapter_num: float,
    page_lang: str,
    *,
    batch_prompt: str,
    combine_prompt: str,
    stage: str,
    progress,
    log: Callable[[str], None],
    instruction: str = "",
    cast: str = "",
) -> tuple[list[str], Callable[[str], str]]:
    """Read a manga chapter's pages in batches with the vision model.

    Returns (batch outputs, generate) where generate(feedback) produces the
    chapter artifact: the single batch verbatim, a feedback-guided re-read of
    the pages, or a combine of the batch outputs. The prompt pair and the
    progress stage name differ per pipeline (recap/narration). `cast` is the
    character-registry block (pipelines/characters.cast_block), appended to
    the rules so recurring characters can be named.
    """
    rules = STRICT_RULES + cast + user_direction_block(instruction)
    batch_outputs: list[str] = []
    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        prompt = batch_prompt.format(
            start=start + 1,
            end=start + len(batch),
            total=len(pages),
            chapter=f"{chapter_num:g}",
            title=series["title"],
            lang=page_lang,
            context=context_block,
            rules=rules,
        )
        log(f"  pages {start + 1}-{start + len(batch)} of {len(pages)}...")
        if progress is not None:
            progress.stage(
                stage,
                f"pages {start + 1}-{start + len(batch)} of {len(pages)}",
            )
        batch_outputs.append(
            vision.adapter.generate(vision.info.name, prompt, images=batch)
        )

    def generate(feedback: str = "") -> str:
        if len(batch_outputs) == 1 and not feedback:
            return batch_outputs[0]
        if len(batch_outputs) == 1:
            # Re-read the single batch's pages with the feedback.
            retry_prompt = (
                batch_prompt.format(
                    start=1,
                    end=len(pages),
                    total=len(pages),
                    chapter=f"{chapter_num:g}",
                    title=series["title"],
                    lang=page_lang,
                    context=context_block,
                    rules=rules,
                )
                + feedback
            )
            return vision.adapter.generate(
                vision.info.name, retry_prompt, images=pages
            )
        # Combine with the vision-role model: it reads the pages itself and
        # its instruction following holds up on noisy batch summaries —
        # the small text-role model goes conversational (refusals, meta
        # commentary) when batches contain repetition loops or scanlation
        # credits pages. Text role stays for rolling-context compression.
        combine = combine_prompt.format(
            chapter=f"{chapter_num:g}",
            title=series["title"],
            rules=rules,
            batches="\n\n".join(batch_outputs),
        )
        return vision.adapter.generate(
            vision.info.name, combine + feedback
        )

    return batch_outputs, generate


def judged_artifact(
    generate: Callable[[str], str],
    batches: list[str],
    rolling: str | None,
    title: str,
    chapter_num: float,
    *,
    judge,
    vision,
    vision_verify: bool,
    pages: list[Path] | None,
    judge_fn,
    label: str,
    max_attempts: int,
    log: Callable[[str], None],
    instruction: str = "",
    cast: str = "",
) -> str:
    """Run generate() through the judge loop and return the kept text.

    With vision_verify (thinking "high" on manga), the artifact is verified
    against the chapter pages by the vision model itself (a real fidelity
    check); books have no pages, so they keep the text judge. With no judge
    (thinking "low"), the first generation is returned unjudged. The steering
    instruction is shown to the judge so mandated omissions/style are not
    flagged (see judge.judge_artifact).
    """
    if judge is None:
        return generate("")
    if vision_verify:
        text_out, _verdict = judge_loop(
            generate,
            lambda t: judge_fn(
                vision.adapter, vision.info.name, t, batches, rolling,
                title, chapter_num, pages=pages, instruction=instruction,
                cast=cast,
            ),
            log,
            label,
            max_attempts,
        )
        return text_out
    text_out, _verdict = judge_loop(
        generate,
        lambda t: judge_fn(
            judge.adapter, judge.info.name, t, batches, rolling,
            title, chapter_num, instruction=instruction, cast=cast,
        ),
        log,
        label,
        max_attempts,
    )
    return text_out


def update_rolling_context(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    chapter_num: float,
    rolling: str | None,
    text,
    judge,
    artifact: str,
    *,
    first_chapter_note: str,
    max_attempts: int,
    advance: bool = True,
    log: Callable[[str], None],
) -> None:
    """Fold a chapter's artifact (recap/narration text) into the rolling
    story-so-far, judged when a judge is active. A failing or suspicious
    update never destroys the accumulated context: the old one is kept.
    Upserts series_context and commits. With advance=True (the default — a
    chapter read for the first time) reading progress advances too;
    advance=False is for detail upgrades of already-covered chapters, which
    must not touch progress. first_chapter_note is the placeholder used when
    there is no prior context."""
    def generate_context(feedback: str = "") -> str:
        context_prompt = CONTEXT_PROMPT.format(
            title=series["title"],
            previous=rolling or first_chapter_note,
            chapter=f"{chapter_num:g}",
            recap=artifact,
        )
        return text.adapter.generate(text.info.name, context_prompt + feedback)

    if judge is not None:
        new_context, verdict = judge_loop(
            generate_context,
            lambda c: judge_context(
                judge.adapter, judge.info.name, c, rolling, artifact,
                series["title"],
            ),
            log,
            "context update",
            max_attempts,
        )
        if not verdict.passed:
            # Never let a failing update destroy the accumulated context.
            log("  keeping the previous story-so-far.")
            new_context = rolling
    else:
        new_context = generate_context("")
    if _suspicious_context(new_context, rolling):
        log(
            f"  context update for chapter {chapter_num:g} looked like"
            " meta-commentary (unrelated bonus chapter?); keeping the"
            " previous story-so-far."
        )
        new_context = rolling
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES (?, ?, ?)"
        " ON CONFLICT(series_id) DO UPDATE SET rolling_summary ="
        " excluded.rolling_summary, through_chapter = excluded.through_chapter",
        (series["id"], new_context, chapter_num),
    )
    if advance:
        advance_progress(conn, series["id"], chapter_num)
    conn.commit()
    meta = works.read_work_metadata(series["id"])
    if meta is not None:
        meta.context = works.WorkContext(
            rolling_summary=new_context or "",
            through_chapter=chapter_num,
        )
        works.write_work_metadata(meta)


def recap_series(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    config: Config,
    profile: HardwareProfile,
    all_chapters: bool = False,
    chapter_num: float | None = None,
    chapters_spec: str | None = None,
    max_chapters: int = 500,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client: MangaDexClient | None = None,
    translated: bool = False,
    thinking: str = "medium",
    detail: str = "standard",
    instruction: str = "",
    on_recap: Callable[[str], None] | None = None,
    progress=None,
    fill_gaps: bool = False,
    skip_done: bool = False,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Recap all pending chapters at the `detail` grain (default; capped at
    max_chapters).

    detail is one of DETAIL_LEVELS (gist < brief < standard < detailed <
    full): it selects the prompt pair (standard = the classic recap, full =
    the classic narration retelling) and the judge (full keeps the narration
    judge with the completeness axis; the other grains use the recap judge).
    The artifact lands in recaps with its detail.

    Default selection is grain-aware (see pending_chapters), two buckets:
    (a) chapters after last_read with no artifact at any grain — these
    advance reading progress and fold the rolling series context; (b)
    chapters with a lower-grain artifact — upgrades, ungated on last_read,
    stored at the requested grain without touching progress, with the context
    re-folded only for chapters not already covered by
    series_context.through_chapter. Chapters already at (or above) the
    requested grain are skipped; a higher-grain artifact is never downgraded
    unless forced.

    With fill_gaps (the CLI sets it for --video), selection gains a third
    bucket: chapters at or before last_read with no artifact and no usable
    video (see pending_chapters). The on_recap callback still fires for
    them, so the caller builds the chapter's video right after.

    Explicit selection — chapter_num (--chapter), chapters_spec (--chapters,
    e.g. '1-3,4,6-10'), or all_chapters (--all) — is forced per chapter:
    existing artifacts are overwritten and rolling context and progress are
    left untouched (an out-of-order rewrite would corrupt them). With
    skip_done=True (the desktop UI's "process unfinished" toggle), that
    forcing is relaxed: chapters already recapped at this grain — and already
    videod, when on_recap is given — are dropped from the selection instead
    of overwritten, so a range run only processes what is actually missing.
    A chapter that survives the filter on its video alone (artifact already
    at this grain, no video) is not re-recapped: on_recap fires directly and
    the loop moves on, so a failed video never costs a re-narration.

    Standalone rule, all modes: a chapter with NO artifact at or before
    last_read is generated without the rolling story-so-far — prepending a
    summary folded beyond the chapter would leak future events into the
    artifact, and folding it would double-fold the context tape. Nothing is
    folded and progress is untouched; the artifact is stored with
    recaps.standalone = 1 so 'eh show' can say how it was made.

    With translated=True, chapters in any language are eligible: each chapter
    is translated first ('eh translate' overlay pages) when it is not already
    in the target language, and the recap reads the translated pages. Recaps
    and rolling context land in the same tables either way, so both modes form
    one continuous story-so-far chain.

    thinking controls critiquing depth: "low" skips the judge, "medium" judges
    text outputs (3 attempts), "high" additionally verifies recaps against the
    chapter pages themselves via the vision model (5 attempts). Thinking
    levels behave identically across grains.

    instruction is a free-form steering direction (content to skip, voice,
    emphasis): injected as a USER DIRECTION block into the batch/combine
    prompts of every grain and shown to the artifact judge (mandated
    omissions/style are not issues there). It is recorded on the artifact
    (recaps.instruction + recap.json) for attribution; changing it never
    invalidates existing artifacts — re-runs stay explicit via
    all_chapters/chapter_num/chapters_spec.

    on_recap, if given, is called with the chapter id right after each
    chapter's artifact is committed (used by the CLI to render that chapter's
    video before moving on to the next chapter). progress, if given, receives
    stage() updates for the live UI (see ui.py).

    Returns the ids of chapters recapped.
    """
    if detail not in DETAIL_LEVELS:
        raise ValueError(
            f"unknown detail {detail!r}; expected one of: {', '.join(DETAIL_LEVELS)}"
        )
    instruction = instruction.strip()
    is_full = detail == "full"
    noun = "narration" if is_full else "recap"  # artifact noun for log lines
    verb = "narrate" if is_full else "recap"
    batch_prompt, combine_prompt = PROMPTS[detail]
    judge_fn = judge_narration if is_full else judge_recap
    first_chapter_note = (
        "(This is the first chapter narrated.)"
        if is_full else
        "(This is the first chapter recapped.)"
    )
    is_book = series["kind"] == "book" if "kind" in series.keys() else False
    _client = client

    def client_factory():
        nonlocal _client
        if _client is None:
            from entertainment_harness.sources import get_client

            _client = get_client(series["source"])
        return _client

    text, vision, judge, max_attempts = setup_models(
        config, profile, is_book, thinking,
        book_role="narration" if is_full else "text summarization",
        text_role="context" if is_full else "summaries/context",
        judge_role="narration" if is_full else "recap", log=log,
    )

    forced = (chapter_num is not None or all_chapters
              or chapters_spec is not None)
    todo = select_chapters(
        conn, series, config, detail=detail, chapter_num=chapter_num,
        chapters_spec=chapters_spec, all_chapters=all_chapters,
        max_chapters=max_chapters,
        translated=translated, fill_gaps=fill_gaps, verb=verb, log=log,
    )
    if skip_done and todo:
        # Forced scopes normally overwrite; skip_done keeps the finished
        # chapters (recap at this grain + video when on_recap is given).
        n_all = len(todo)
        todo = [
            c for c in todo
            if chapter_needs_work(
                conn, series["id"], c, detail=detail,
                want_video=on_recap is not None,
            )
        ]
        if n_all > len(todo):
            log(
                f"Skipping {n_all - len(todo)} chapter(s) already recapped"
                + (" and videod." if on_recap is not None else " at this grain.")
            )
    if not todo:
        return []
    last_read = get_progress(conn, series["id"])
    if fill_gaps and not forced:
        n_gaps = sum(
            1 for c in todo
            if c["artifact_detail"] is None
            and last_read is not None
            and c["chapter_num"] <= last_read
        )
        if n_gaps:
            log(
                f"Filling {n_gaps} chapter(s) behind the read frontier"
                f" (last read {last_read:g}): standalone artifacts, story-so-far"
                " withheld, so their videos can be built."
            )
    if progress is not None:
        progress.start(len(todo))

    done: list[str] = []
    for chapter in todo:
        chapter_num = chapter["chapter_num"]
        if progress is not None:
            progress.chapter_start(chapter_num)
        if (
            skip_done
            and on_recap is not None
            and chapter["artifact_detail"] == detail
        ):
            # skip_done kept this chapter only for its missing video — the
            # artifact is already at the requested grain. Re-generating it
            # would overwrite a good artifact (and flush the cached script
            # and TTS downstream of it) for nothing; render and move on.
            log(f"Chapter {chapter_num:g} {noun} already stored; video only.")
            done.append(chapter["id"])
            on_recap(chapter["id"])
            if progress is not None:
                progress.chapter_done()
            continue
        # Standalone rule (any mode): a chapter with no artifact at or before
        # last_read is generated without the rolling story-so-far — prepending
        # a summary folded beyond it would leak future events into the
        # artifact, and folding it would double-fold the context tape.
        # Bucket-(a) rows (past the frontier) never match; forced rows with an
        # existing artifact keep the long-standing prepend behavior.
        rolling = get_context(conn, series["id"])
        standalone = (
            rolling is not None
            and last_read is not None
            and chapter["artifact_detail"] is None
            and chapter_num <= last_read
        )
        if standalone:
            rolling = None
        context_block = (
            f'\nStory so far (context only, do not repeat it):\n"{rolling}"\n'
            if rolling
            else ""
        )
        # Character bible: the work's registry rides the rules slot of every
        # prompt (and the artifact judge) so recurring characters can be
        # named — per-work knowledge, injected for standalone chapters too.
        cast = ""
        if config.pipeline.characters:
            from entertainment_harness.pipelines.characters import (
                load_cast_block,
            )
            cast = load_cast_block(conn, series["id"])

        if is_book:
            batches, generate = _recap_book_chapter(
                series, chapter, text, context_block, log, detail=detail,
                instruction=instruction, cast=cast,
            )
            attribution = model_tag(text.info)
        else:
            pages, page_lang = resolve_pages(
                conn, series, chapter, config, profile, client_factory, _client,
                translated=translated, thinking=thinking, progress=progress,
                log=log,
            )
            if not chapter["pages"]:
                # sources without per-chapter page counts (weebcentral) learn it here
                conn.execute(
                    "UPDATE chapters SET pages = ? WHERE id = ?",
                    (len(pages), chapter["id"]),
                )
                conn.commit()
            log(f"{'Narrating' if is_full else 'Recapping'} chapter"
                f" {chapter_num:g} ({len(pages)} pages)...")
            if progress is not None:
                progress.stage(verb, f"{len(pages)} pages")

            batches, generate = manga_batches(
                vision, pages, batch_size, context_block, series, chapter_num,
                page_lang, batch_prompt=batch_prompt,
                combine_prompt=combine_prompt, stage=verb,
                progress=progress, log=log, instruction=instruction,
                cast=cast,
            )

            attribution = model_tag(vision.info)

        # On high, manga recaps are verified against the chapter pages by the
        # vision model itself (real fidelity check); books have no pages, so
        # they keep the text judge with the higher attempt budget.
        vision_verify = thinking == "high" and not is_book
        artifact_text = judged_artifact(
            generate, batches, rolling, series["title"], chapter_num,
            judge=judge, vision=vision, vision_verify=vision_verify,
            pages=None if is_book else pages, judge_fn=judge_fn,
            label=f"chapter {chapter_num:g} {noun}",
            max_attempts=max_attempts, log=log, instruction=instruction,
            cast=cast,
        )

        now = utcnow()
        conn.execute(
            "INSERT INTO recaps (chapter_id, summary, created_at, model, detail,"
            " instruction, standalone)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(chapter_id) DO UPDATE SET summary = excluded.summary,"
            " created_at = excluded.created_at, model = excluded.model,"
            " detail = excluded.detail, instruction = excluded.instruction,"
            " standalone = excluded.standalone",
            (chapter["id"], artifact_text, now, attribution, detail, instruction,
             1 if standalone else 0),
        )
        works.write_recap(
            series["id"],
            chapter["id"],
            works.RecapMetadata(
                summary=artifact_text, model=attribution, created_at=now,
                detail=detail, instruction=instruction, standalone=standalone,
            ),
        )

        if config.pipeline.characters:
            # Character bible: merge this chapter's cast into the registry so
            # later chapters can use real names (pipelines/characters.py).
            # Runs on every stored artifact — upserts are idempotent and the
            # registry is a set, not a chronology. Never fails the chapter.
            from entertainment_harness.pipelines.characters import (
                update_characters,
            )
            try:
                update_characters(
                    conn, series, chapter_num, text, judge, artifact_text,
                    max_attempts=max_attempts, log=log,
                )
            except Exception as exc:
                log(f"  cast update skipped: {exc}")

        if standalone:
            # Artifact generated without story-so-far (gap fill or an
            # explicit selection behind the read frontier) and nothing is
            # folded — the rolling context already covers this chapter, and
            # re-folding it would corrupt the tape.
            conn.commit()
            done.append(chapter["id"])
            log(
                f"Chapter {chapter_num:g} {noun} stored (model:"
                f" {attribution}); standalone (no story-so-far context),"
                " context and progress unchanged."
            )
            if on_recap is not None:
                on_recap(chapter["id"])
            if progress is not None:
                progress.chapter_done()
            continue

        if forced:
            # Forced re-recap: leave rolling context and progress alone.
            conn.commit()
            done.append(chapter["id"])
            log(
                f"Chapter {chapter_num:g} {noun} stored (model:"
                f" {attribution}); context and progress unchanged."
            )
            if on_recap is not None:
                on_recap(chapter["id"])
            if progress is not None:
                progress.chapter_done()
            continue

        if chapter["artifact_detail"] is not None:
            # Bucket (b): upgrade from a lower-grain artifact. Progress is
            # never touched; the rolling context is re-folded only when the
            # chapter is not already covered by it.
            through = get_context_through(conn, series["id"])
            if through is None or chapter_num > through:
                update_rolling_context(
                    conn, series, chapter_num, rolling, text, judge,
                    artifact_text, first_chapter_note=first_chapter_note,
                    max_attempts=max_attempts, advance=False, log=log,
                )
                log(
                    f"Chapter {chapter_num:g} {noun} stored (model:"
                    f" {attribution}); progress unchanged."
                )
            else:
                conn.commit()
                log(
                    f"Chapter {chapter_num:g} {noun} stored (model:"
                    f" {attribution}); context and progress unchanged."
                )
            done.append(chapter["id"])
            if on_recap is not None:
                on_recap(chapter["id"])
            if progress is not None:
                progress.chapter_done()
            continue

        # Bucket (a): no artifact at any grain — fold the rolling context and
        # advance reading progress.
        update_rolling_context(
            conn, series, chapter_num, rolling, text, judge, artifact_text,
            first_chapter_note=first_chapter_note,
            max_attempts=max_attempts, log=log,
        )
        done.append(chapter["id"])
        log(f"Chapter {chapter_num:g} {noun} stored (model: {attribution}).")
        if on_recap is not None:
            on_recap(chapter["id"])
        if progress is not None:
            progress.chapter_done()
    return done
