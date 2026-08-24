"""Web search providers for online recaps/summaries.

Providers register in the shared plugin registry (plugins.py) — built-ins via
lazy dotted strings, third-party providers via the
entertainment_harness.search entry-point group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from entertainment_harness.config import Config
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginRegistry


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_type: str  # "blog", "reddit", "youtube", "unknown"


class SearchProvider(Protocol):
    name: str
    capabilities: frozenset[str]  # what the provider offers (none declared yet)

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...


REGISTRY = PluginRegistry("search", ENTRY_POINT_GROUPS["search"])
REGISTRY.register(
    "duckduckgo", "entertainment_harness.search.duckduckgo:DuckDuckGoSearchProvider"
)


def get_provider(name: str, config: Config | None = None) -> SearchProvider:
    return REGISTRY.create(name, config)
