"""Plugin conformance kit.

The same checks `eh plugins --check` and `eh plugin check` run, importable
so third-party plugins can assert conformance in their own test suites:

    from entertainment_harness.testing import check_plugin

    def test_conformance():
        assert check_plugin(MyEngine, category="tts") == []

Every check returns problem strings (empty = conforming); nothing raises, so
a broken plugin reports all its problems at once.
"""

from __future__ import annotations

import inspect

# Category -> members a plugin class must provide (attribute or method).
# Sources are identified by their registration key, so `name` is not required
# of them (the Source protocol doesn't declare it).
CATEGORY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "model_backend": (
        "name", "remote", "supports", "ensure", "remove", "list_available",
        "list_remote", "generate",
    ),
    "sources": (
        "search", "get_series", "chapters", "download_pages", "looks_like_id",
    ),
    "tts": ("name", "default_voice", "synthesize"),
    "video_gen": ("name", "animated", "generate_segment"),
    "frames": ("name", "animated", "generate_frames"),
    "colorize": ("name", "colorize_page", "colorize_pages"),
    "search": ("name", "search"),
}


def check_constructor(cls: type) -> list[str]:
    """The shared contract: __init__(self, config=..., <defaults or **kwargs>).

    create() calls ``cls(config, **overrides)``, so `config` must be an
    accepted parameter and every other named parameter must have a default.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError) as exc:
        return [f"cannot introspect __init__: {exc}"]
    problems = []
    params = list(sig.parameters.values())[1:]  # skip self
    names = [
        p.name for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY)
    ]
    if "config" not in names:
        problems.append("__init__ must accept a `config` parameter")
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name != "config" and p.default is p.empty:
            problems.append(
                f"__init__ parameter {p.name!r} has no default — create()"
                " only ever passes `config` plus configured overrides"
            )
    return problems


def check_plugin(cls: type, category: str) -> list[str]:
    """All conformance problems for a plugin class in a category."""
    if category not in CATEGORY_REQUIREMENTS:
        known = ", ".join(sorted(CATEGORY_REQUIREMENTS))
        return [f"unknown category {category!r}; known: {known}"]
    problems = check_constructor(cls)
    capabilities = getattr(cls, "capabilities", frozenset())
    if not isinstance(capabilities, frozenset):
        problems.append(
            "class attribute `capabilities` must be a frozenset[str]"
            f" (got {type(capabilities).__name__})"
        )
    for member in CATEGORY_REQUIREMENTS[category]:
        if not hasattr(cls, member):
            problems.append(f"{category} plugins must provide `{member}`")
    return problems
