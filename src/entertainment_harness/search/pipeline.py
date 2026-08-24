"""Search the web for summaries, extract them, and synthesize a rolling summary."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

from entertainment_harness.config import Config
from entertainment_harness.db import utcnow
from entertainment_harness.models.base import ModelAdapter
from entertainment_harness.search import get_provider
from entertainment_harness.search.extractor import gather_sources

_QUERIES = [
    '{title} recap blog',
    '{title} reddit discussion',
    '{title} summary youtube',
    '{title} explained',
]

_SYNTHESIS_PROMPT = """You are writing a concise "story so far" summary for the work "{title}" based ONLY on the online sources below. Do not use outside knowledge.

Sources:
{sources}

Requirements:
- 250-400 words of plain English prose.
- Faithful to the sources: no new events, names, or details not present above.
- If sources disagree, note the uncertainty briefly or use the most consistent account.
- Present tense.
{steering}

Output ONLY the summary text, no headers, no lists, no commentary."""


def _search_all(provider, title: str, max_per_query: int = 5) -> list:
    seen: set[str] = set()
    results: list = []
    for template in _QUERIES:
        query = template.format(title=title)
        try:
            for r in provider.search(query, max_results=max_per_query):
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    results.append(r)
        except Exception:
            continue
    return results


def synthesize_online_summary(
    adapter: ModelAdapter,
    model: str,
    title: str,
    config: Config,
    steering_prompt: str = "",
    log: Callable[[str], None] = lambda m: None,
) -> tuple[str, list[dict]]:
    """Return (summary, sources_json_list)."""
    provider = get_provider(config.search.provider, config)
    log(f"Searching the web for summaries of {title!r} via {provider.name}...")
    results = _search_all(provider, title, max_per_query=5)
    log(f"  {len(results)} unique results; extracting text...")
    sources = gather_sources(results, max_sources=config.search.max_sources)
    log(f"  extracted {len(sources)} usable sources")

    if not sources:
        raise RuntimeError(
            f"No usable online summaries found for {title!r}. "
            "Try a different title or check your network connection."
        )

    source_block = "\n\n---\n\n".join(
        f"Source: {s.title}\nURL: {s.url}\nType: {s.source_type}\n\n{s.text}"
        for s in sources
    )
    steering = (
        f"Additional direction: {steering_prompt}" if steering_prompt else ""
    )
    prompt = _SYNTHESIS_PROMPT.format(
        title=title,
        sources=source_block,
        steering=steering,
    )
    summary = adapter.generate(model, prompt).strip()
    sources_json = [
        {"title": s.title, "url": s.url, "source_type": s.source_type}
        for s in sources
    ]
    return summary, sources_json


def store_online_summary(
    conn: sqlite3.Connection,
    series_id: str,
    provider_name: str,
    query: str,
    summary: str,
    sources_json: list[dict],
    steering_prompt: str = "",
) -> None:
    now = utcnow()
    conn.execute(
        "INSERT INTO online_summaries (series_id, provider, query, summary,"
        " sources_json, steering_prompt, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(series_id) DO UPDATE SET"
        " provider=excluded.provider, query=excluded.query,"
        " summary=excluded.summary, sources_json=excluded.sources_json,"
        " steering_prompt=excluded.steering_prompt, created_at=excluded.created_at",
        (
            series_id,
            provider_name,
            query,
            summary,
            json.dumps(sources_json),
            steering_prompt,
            now,
        ),
    )
    # Also keep series_context in sync so build_short can consume it unchanged.
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES (?, ?, ?)"
        " ON CONFLICT(series_id) DO UPDATE SET"
        " rolling_summary=excluded.rolling_summary",
        (series_id, summary, None),
    )
    conn.commit()


def summarize_and_store(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    config: Config,
    adapter: ModelAdapter,
    model: str,
    steering_prompt: str = "",
    log: Callable[[str], None] = lambda m: None,
) -> str:
    summary, sources = synthesize_online_summary(
        adapter, model, series["title"], config, steering_prompt, log
    )
    store_online_summary(
        conn,
        series["id"],
        config.search.provider,
        series["title"],
        summary,
        sources,
        steering_prompt,
    )
    log(f"Stored online summary ({len(summary.split())} words)")
    return summary
