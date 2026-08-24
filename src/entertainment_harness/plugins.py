"""Shared plugin registry for the seven pluggable seams.

One idiom per category (model backends, sources, TTS engines, video
generators, frame animators, page colorizers, search providers): built-ins register under a
name — a class or a lazy "pkg.mod:Class" dotted string that is not imported
until create() — and third-party packages can add plugins via stdlib entry
points:

    [project.entry-points."entertainment_harness.tts"]
    elevenlabs = "my_pkg.tts:ElevenLabsEngine"

All plugins share the constructor contract
``__init__(self, config: Config | None = None, **explicit_overrides)``:
explicit kwargs win, else the plugin's `[<section>.<plugin>]` config.toml
table (merged automatically by create() from Config.raw), else
``load_config()``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entertainment_harness.config import Config


class PluginError(ValueError):
    """Unknown plugin name; the message lists the available plugins."""


# Plugin category -> entry-point group third-party packages can register into.
ENTRY_POINT_GROUPS: dict[str, str] = {
    "model_backend": "entertainment_harness.model_backends",
    "sources": "entertainment_harness.sources",
    "tts": "entertainment_harness.tts",
    "video_gen": "entertainment_harness.video_gen",
    "frames": "entertainment_harness.frames",
    "colorize": "entertainment_harness.colorize",
    "search": "entertainment_harness.search",
}

# Plugin category -> config.toml section holding per-plugin tables:
# [<section>.<plugin>] flows into the constructor at create() time (see
# create()), so a plugin with settings needs no core config.py changes.
CATEGORY_CONFIG_SECTIONS: dict[str, str] = {
    "model_backend": "models",
    "sources": "sources",
    "tts": "tts",
    "video_gen": "video_gen",
    "frames": "frames",
    "colorize": "colorize",
    "search": "search",
}


def _accepted_kwargs(cls: type) -> tuple[set[str], bool]:
    """Constructor introspection: (named params beyond `config`, accepts
    **kwargs). Used to filter a plugin's config section to keys the
    constructor actually takes — unknown keys are ignored, matching how
    load_config treats unknown TOML keys everywhere else."""
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set(), True
    names: set[str] = set()
    var_kw = False
    for param in list(sig.parameters.values())[1:]:  # skip self
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            var_kw = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            if param.name != "config":
                names.add(param.name)
    return names, var_kw


class PluginRegistry:
    """Name -> plugin class registry with lazy dotted imports and entry-point
    discovery (once per process, on first names()/create())."""

    def __init__(self, category: str, entry_point_group: str) -> None:
        self.category = category
        self.entry_point_group = entry_point_group
        self._plugins: dict[str, type | str] = {}
        self._entry_point_names: set[str] = set()
        self._entry_points_loaded = False

    def register(self, name: str, cls_or_dotted: type | str) -> None:
        """Register a plugin class, or a lazy "pkg.mod:Class" dotted string."""
        self._plugins[name] = cls_or_dotted

    def load_entry_points(self) -> None:
        """Discover third-party plugins via importlib.metadata (once).

        A broken entry point is skipped so built-ins stay usable; built-ins
        win over an entry point of the same name.
        """
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for ep in importlib.metadata.entry_points(group=self.entry_point_group):
            if ep.name in self._plugins:
                continue
            try:
                self._plugins[ep.name] = ep.load()
            except Exception:
                continue
            self._entry_point_names.add(ep.name)

    def names(self) -> list[str]:
        """Built-in + discovered entry-point plugin names, sorted."""
        self.load_entry_points()
        return sorted(self._plugins)

    def names_with(self, capability: str) -> list[str]:
        """Names whose plugin class declares `capability` in its
        `capabilities` frozenset ("who can generate images?"). Loads every
        plugin (lazy imports fire); plugins without the attribute count as
        capability-less, so older entry-point plugins keep working."""
        return [
            name
            for name in self.names()
            if capability in getattr(self.load(name), "capabilities", frozenset())
        ]

    def is_entry_point(self, name: str) -> bool:
        """Whether name came from a third-party entry point (for display)."""
        self.load_entry_points()
        return name in self._entry_point_names

    def builtin_names(self) -> list[str]:
        """Registered names that did not come from entry points, in
        registration order."""
        self.load_entry_points()
        return [n for n in self._plugins if n not in self._entry_point_names]

    def load(self, name: str) -> type:
        """Resolve a name to its plugin class, importing lazily if needed."""
        self.load_entry_points()
        try:
            plugin = self._plugins[name]
        except KeyError:
            available = ", ".join(sorted(self._plugins))
            raise PluginError(
                f"Unknown {self.category} plugin {name!r}; available: {available}"
            ) from None
        if isinstance(plugin, str):
            module_name, _, attr = plugin.partition(":")
            plugin = getattr(importlib.import_module(module_name), attr)
            self._plugins[name] = plugin
        return plugin

    def create(self, name: str, config: Config | None = None, **overrides):
        """Instantiate a plugin with the uniform (config, **overrides) contract.

        When config carries a raw TOML view (load_config sets Config.raw),
        the plugin's `[<section>.<name>]` table is merged in: keys the
        constructor names (or every key when it takes **kwargs) become
        constructor kwargs; explicit overrides win; unknown keys are
        ignored. This is what lets a plugin have settings without any core
        config.py changes.
        """
        cls = self.load(name)
        merged = dict(overrides)
        raw = getattr(config, "raw", None) if config is not None else None
        if raw:
            section = CATEGORY_CONFIG_SECTIONS.get(self.category, self.category)
            table = raw.get(section, {})
            table = table.get(name, {}) if isinstance(table, dict) else {}
            if isinstance(table, dict) and table:
                accepted, var_kw = _accepted_kwargs(cls)
                for key, value in table.items():
                    if key not in overrides and (var_kw or key in accepted):
                        merged[key] = value
        return cls(config, **merged)
