"""Content sources. One small protocol; MangaDex (official API) and
Weeb Central (English scanlation aggregator, HTML scraping).

New sources register in REGISTRY below (or via the
entertainment_harness.sources entry-point group — see plugins.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from entertainment_harness.config import Config, load_config
from entertainment_harness.plugins import ENTRY_POINT_GROUPS, PluginError, PluginRegistry
from entertainment_harness.sources.mangadex import Chapter, Series

__all__ = [
    "Chapter", "Series", "Source", "builtin_source_names", "get_client",
    "is_source_enabled", "looks_like_id",
]

REGISTRY = PluginRegistry("sources", ENTRY_POINT_GROUPS["sources"])
REGISTRY.register("mangadex", "entertainment_harness.sources.mangadex:MangaDexClient")
REGISTRY.register(
    "weebcentral", "entertainment_harness.sources.weebcentral:WeebCentralClient"
)


class Source(Protocol):
    capabilities: frozenset[str]  # what the source offers (none declared yet)

    def search(self, title: str) -> list[Series]: ...
    def get_series(self, source_id: str) -> Series: ...
    def chapters(self, manga_id: str, langs: list[str] | None = None) -> list[Chapter]: ...
    def download_pages(self, chapter_id: str, dest_dir: Path) -> list[Path]: ...

    @staticmethod
    def looks_like_id(query: str) -> bool:
        """Whether a CLI argument is a series id rather than a title to search."""
        ...


def builtin_source_names() -> list[str]:
    """Built-in (non-entry-point) source names, in registration order."""
    return REGISTRY.builtin_names()


def is_source_enabled(name: str, config: Config) -> bool:
    """Whether a registered source is usable per [sources].enabled.

    Only built-in sources are gated: a built-in name absent from
    [sources].enabled is disabled. Third-party (entry-point) sources are
    always enabled so existing plugin users are unaffected.
    """
    if REGISTRY.is_entry_point(name):
        return True
    return name in config.sources.enabled


def get_client(source: str, config: Config | None = None) -> Source:
    """Instantiate the client for a stored/configured source name.

    Raises PluginError when the source is a built-in disabled via
    [sources].enabled; unknown names keep the registry's unknown-plugin
    error."""
    if source in REGISTRY.names() and not REGISTRY.is_entry_point(source):
        config = config or load_config()
        if source not in config.sources.enabled:
            raise PluginError(
                f"source {source!r} is disabled in config.toml"
                f" ([sources].enabled); re-enable with `eh sources enable {source}`"
            )
    return REGISTRY.create(source, config)


def looks_like_id(source: str, query: str) -> bool:
    """Whether a CLI argument is a series id rather than a title to search."""
    return REGISTRY.load(source).looks_like_id(query)
