"""Character registry (character-bible initiative): learn the work's cast
from judged chapter artifacts so later chapters can use real names.

The structural gap this closes: STRICT_RULES forbids names not printed on the
current pages, and the 300-word rolling story-so-far compresses names out —
so without a per-work registry the model can never call a recurring
character by name. The registry is a set, not a chronology: extraction runs
on every stored artifact (idempotent upserts), never folds tape, and a
failing or unparseable update keeps the previous registry — the same
never-destroy-accumulated-state guard as `_suspicious_context` for the
rolling context.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Callable

from entertainment_harness import db
from entertainment_harness.pipelines.judge import Verdict, judge_cast
from entertainment_harness.pipelines.recap import judge_loop

CAST_MAX_ENTRIES = 15  # registry rows injected into prompts
CAST_MAX_CHARS = 700  # total CAST block budget

CHARACTERS_PROMPT = """You are tracking the cast of "{title}". Current character registry (canonical names; may be empty):
---
{cast}
---
Recap of chapter {chapter}:
---
{recap}
---
List the characters who appear in THIS chapter, as a JSON array:
[{{"name": "...", "aliases": ["..."], "role": "one line, under 12 words"}}]
Rules:
- Only characters who appear in this chapter.
- Use the registry's canonical name for characters already in it; add any new alias this chapter reveals.
- For new characters, use the name as printed in the recap; characters the recap leaves unnamed stay out of the list.
- Keep roles factual and short; refine a registry role only when this chapter changes it.
- Never invent characters, names, or details that are not in the recap.
- Output ONLY the JSON array: no preamble, no notes. Output [] when no characters appear."""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_cast(raw: str) -> list[dict]:
    """Extract the cast JSON array from raw model output (tolerates fences
    and preamble). Raises ValueError on unusable output — the caller turns
    that into a failed verdict so the judge loop retries."""
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        raise ValueError("no JSON array in output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("cast output is not a JSON array")
    entries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        aliases = [
            str(a).strip() for a in item.get("aliases") or [] if str(a).strip()
        ]
        role = str(item.get("role") or "").strip()
        entries.append({"name": name, "aliases": aliases, "role": role})
    return entries


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _keys(name: str, aliases: list[str]) -> set[str]:
    return {_norm(name), *(_norm(a) for a in aliases)}


def _find(rows: list[sqlite3.Row], name: str, aliases: list[str]):
    """Entity resolution: an incoming entry matches a stored row when their
    name/alias sets intersect (case-insensitive)."""
    candidate = _keys(name, aliases)
    for row in rows:
        if candidate & _keys(row["name"], json.loads(row["aliases"])):
            return row
    return None


def merge_cast(
    conn: sqlite3.Connection,
    series_id: str,
    entries: list[dict],
    chapter_num: float,
) -> None:
    """Fold one chapter's extracted cast into the registry. Deterministic —
    the model proposes, this code disposes: canonical names never flip-flop
    (a differing incoming name becomes an alias), aliases union, the latest
    non-empty role wins, and user-locked (edited) rows only advance their
    last-seen chapter."""
    for entry in entries:
        rows = db.get_characters(conn, series_id)
        match = _find(rows, entry["name"], entry["aliases"])
        if match is None:
            db.upsert_character(
                conn, series_id, entry["name"],
                aliases=entry["aliases"], role=entry["role"],
                chapter_num=chapter_num,
            )
        elif match["edited"]:
            db.upsert_character(
                conn, series_id, match["name"], chapter_num=chapter_num
            )
        else:
            aliases = list(entry["aliases"])
            if _norm(entry["name"]) != _norm(match["name"]):
                aliases = [entry["name"], *aliases]
            db.upsert_character(
                conn, series_id, match["name"],
                aliases=aliases, role=entry["role"], chapter_num=chapter_num,
            )


def cast_entries(rows: list[sqlite3.Row]) -> list[dict]:
    return [
        {"name": r["name"], "aliases": json.loads(r["aliases"]),
         "role": r["role"]}
        for r in rows
    ]


def cast_block(
    cast: list[dict],
    amendment: str = (
        "You MAY use these names when that character appears (they"
        ' override "only printed names" for these characters only); never'
        " apply them to other characters."
    ),
) -> str:
    """The CAST prompt block for the rules slot of every artifact prompt.
    Empty registry -> '' (prompts render byte-identically to the no-registry
    case — the user_direction_block discipline). The block carries its own
    scoped amendment to the name rule, co-located with the list so it can't
    leak to other characters; each prompt family passes the amendment worded
    for its own rules."""
    if not cast:
        return ""
    lines: list[str] = []
    for entry in cast[:CAST_MAX_ENTRIES]:
        alias = f" (also {', '.join(entry['aliases'])})" if entry["aliases"] else ""
        role = f" — {entry['role']}" if entry["role"] else ""
        line = f"- {entry['name']}{alias}{role}"
        # stay inside the character budget without cutting a line in half
        if sum(len(l) + 1 for l in lines) + len(line) > CAST_MAX_CHARS - 200:
            break
        lines.append(line)
    if not lines:
        return ""
    return (
        "\n\nCAST — characters known from previous chapters:\n"
        + "\n".join(lines)
        + f"\n{amendment}"
    )


def cast_script_block(cast: list[dict]) -> str:
    """CAST block for SCRIPT_PROMPT (video beats): the script's name rule is
    "no new names beyond the recap", so the amendment authorizes registry
    names for recap characters the recap leaves unnamed."""
    return cast_block(
        cast,
        amendment=(
            "You MAY use these names for characters that appear in the"
            " recap, even where the recap leaves them unnamed; never apply"
            " them to other characters."
        ),
    )


def cast_translation_block(cast: list[dict]) -> str:
    """CAST block for TRANSLATE_PROMPT: canonical spellings, so the
    translation romanizes a recurring character the same way every chapter."""
    return cast_block(
        cast,
        amendment=(
            "When the original text names one of these characters (under"
            " any listed alias), use the canonical spelling in the"
            " translation; never apply these names to other characters."
        ),
    )


def load_cast_block(
    conn: sqlite3.Connection,
    series_id: str,
    block: Callable[[list[dict]], str] = cast_block,
) -> str:
    """The prompt-ready CAST block for the work's current registry ("" when
    empty or when the registry table has nothing for the series). `block`
    selects the prompt-family wording (recap default, script, translation)."""
    return block(cast_entries(db.get_characters(conn, series_id)))


def update_characters(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    chapter_num: float,
    text,
    judge,
    artifact: str,
    *,
    max_attempts: int,
    log: Callable[[str], None],
) -> None:
    """Extract this chapter's cast from its stored artifact and merge it into
    the registry. A cast update is an enhancement, never a chapter-blocking
    stage: persistent judge failure or unusable output keeps the previous
    registry."""
    current = cast_entries(db.get_characters(conn, series["id"]))

    def generate(feedback: str = "") -> str:
        prompt = CHARACTERS_PROMPT.format(
            title=series["title"],
            chapter=f"{chapter_num:g}",
            cast=json.dumps(current, ensure_ascii=False) if current else "(empty)",
            recap=artifact,
        )
        return text.adapter.generate(text.info.name, prompt + feedback)

    parsed: list[dict] = []

    def judge_fn(output: str) -> Verdict:
        nonlocal parsed
        try:
            parsed = parse_cast(output)
        except ValueError as exc:
            return Verdict(passed=False, issues=[f"cast output unusable: {exc}"])
        if judge is None or not parsed:
            return Verdict(passed=True, issues=[])
        return judge_cast(
            judge.adapter, judge.info.name, parsed, artifact, series["title"]
        )

    _, verdict = judge_loop(generate, judge_fn, log, "cast update", max_attempts)
    if not verdict.passed:
        log("  keeping the previous character registry.")
        return
    merge_cast(conn, series["id"], parsed, chapter_num)
    conn.commit()
    if parsed:
        log(
            f"  cast: merged {len(parsed)} character(s) from chapter"
            f" {chapter_num:g}."
        )


def rebuild_cast(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    config,
    profile,
    *,
    thinking: str = "medium",
    log: Callable[[str], None] = print,
) -> int:
    """Sequential cast re-extraction over every stored artifact — the fold
    behind `eh cast --rebuild` and the work view's Rebuild button. The same
    update_characters the pipeline runs per chapter, folded oldest-first so
    aliases merge the way they did live. REPLACES the whole registry,
    user-edited rows included. Returns the rebuilt registry's size; raises
    ValueError on a bad thinking level or when no artifacts exist."""
    from entertainment_harness.models.registry import (
        get_judge_model,
        get_text_model,
        model_tag,
    )
    from entertainment_harness.pipelines.judge import ATTEMPTS, THINKING_LEVELS
    from entertainment_harness.pipelines.recap import MAX_ATTEMPTS

    if thinking not in THINKING_LEVELS:
        raise ValueError(
            f"thinking must be one of: {', '.join(THINKING_LEVELS)}"
        )
    chapters = conn.execute(
        "SELECT c.chapter_num, r.summary FROM recaps r"
        " JOIN chapters c ON r.chapter_id = c.id"
        " WHERE c.series_id = ? ORDER BY c.chapter_num",
        (series["id"],),
    ).fetchall()
    if not chapters:
        raise ValueError("No recaps yet — nothing to rebuild from.")
    text = get_text_model(config, profile)
    log(f"Using {model_tag(text.info)} for cast extraction.")
    text.adapter.ensure(text.info.name)
    judge = None
    if thinking != "low":
        judge = get_judge_model(config, profile)
        log(f"Using {model_tag(judge.info)} for cast judging.")
        judge.adapter.ensure(judge.info.name)
    conn.execute("DELETE FROM characters WHERE series_id = ?", (series["id"],))
    conn.commit()
    for chapter in chapters:
        log(f"Chapter {chapter['chapter_num']:g}...")
        update_characters(
            conn, series, chapter["chapter_num"], text, judge,
            chapter["summary"],
            max_attempts=ATTEMPTS.get(thinking, MAX_ATTEMPTS),
            log=log,
        )
    return conn.execute(
        "SELECT COUNT(*) AS n FROM characters WHERE series_id = ?",
        (series["id"],),
    ).fetchone()["n"]
