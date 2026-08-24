"""DuckDuckGo text search provider (scraping via duckduckgo-search)."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from entertainment_harness.config import Config
from entertainment_harness.search import SearchResult

# Lazy import of ddgs so the module loads even if the dependency is missing.


_SOURCE_PATTERNS = [
    (re.compile(r"^https?://(www\.|old\.)?reddit\.com/"), "reddit"),
    (re.compile(r"^https?://(www\.)?youtube\.com/"), "youtube"),
    (re.compile(r"^https?://youtu\.be/"), "youtube"),
]

_BLOG_HINTS = re.compile(
    r"blog|wordpress|medium\.com|substack\.com|tumblr|ghost\.io|blogger"
)


def _classify(url: str) -> str:
    for pat, kind in _SOURCE_PATTERNS:
        if pat.match(url):
            return kind
    netloc = urlparse(url).netloc.lower()
    if _BLOG_HINTS.search(netloc):
        return "blog"
    return "unknown"


def _source_priority(source_type: str) -> int:
    return {"reddit": 0, "blog": 1, "youtube": 2, "unknown": 3}.get(source_type, 3)


class DuckDuckGoSearchProvider:
    name = "duckduckgo"

    def __init__(self, config: Config | None = None) -> None:
        pass  # contract: (config, **overrides); no [search.duckduckgo] section yet

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results, region="wt-wt")
        results: list[SearchResult] = []
        for item in raw or []:
            url = item.get("href", "")
            kind = _classify(url)
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("body", ""),
                    source_type=kind,
                )
            )
        results.sort(key=lambda r: (_source_priority(r.source_type), r.title))
        return results
