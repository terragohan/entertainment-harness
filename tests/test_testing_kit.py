"""Plugin conformance kit (testing.py) + `eh plugins --check` / `eh plugin check`."""

from __future__ import annotations

from typer.testing import CliRunner

from entertainment_harness.cli import app
from entertainment_harness.config import Config
from entertainment_harness.testing import check_plugin

runner = CliRunner()


class _ConformingTTS:
    name = "conforming"
    default_voice = "x"
    capabilities = frozenset({"tts"})

    def __init__(self, config: Config | None = None, voice: str = "x") -> None:
        pass

    def synthesize(self, text, voice, dest) -> None:
        pass


class _NoDefault:
    def __init__(self, config: Config | None, required: str) -> None:
        pass


class _BadCapabilities:
    name = "bad"
    capabilities = ["tts"]  # list, not frozenset

    def __init__(self, config: Config | None = None) -> None:
        pass

    def synthesize(self, text, voice, dest) -> None:
        pass


class _MissingMembers:
    def __init__(self, config: Config | None = None) -> None:
        pass


def test_check_plugin_conforming_class_passes():
    assert check_plugin(_ConformingTTS, category="tts") == []


def test_check_plugin_flags_required_constructor_params():
    problems = check_plugin(_NoDefault, category="tts")
    assert any("'required' has no default" in p for p in problems)


def test_check_plugin_flags_missing_config_param():
    class NoConfig:
        def __init__(self, other=None) -> None:
            pass

    problems = check_plugin(NoConfig, category="tts")
    assert any("config" in p for p in problems)


def test_check_plugin_flags_non_frozenset_capabilities():
    problems = check_plugin(_BadCapabilities, category="tts")
    assert any("frozenset" in p for p in problems)


def test_check_plugin_flags_missing_category_members():
    problems = check_plugin(_MissingMembers, category="video_gen")
    assert any("`animated`" in p for p in problems)
    assert any("`generate_segment`" in p for p in problems)


def test_check_plugin_unknown_category_is_a_problem():
    (problem,) = check_plugin(_ConformingTTS, category="nope")
    assert "unknown category" in problem


def test_all_builtin_plugins_conform():
    from entertainment_harness.cli.plugins import _registries

    problems = []
    for label, registry in _registries():
        for name in registry.names():
            cls = registry.load(name)
            for problem in check_plugin(cls, registry.category):
                problems.append(f"{label}/{name}: {problem}")
    assert problems == []


def test_eh_plugins_lists_frames_and_capabilities():
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert "frame animators" in result.output
    assert "image-gen" in result.output
    assert "voice-clone" in result.output


def test_eh_plugins_check_passes_on_builtins():
    result = runner.invoke(app, ["plugins", "--check"])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_eh_plugin_check_conforming_class():
    result = runner.invoke(
        app,
        ["plugin", "check", "entertainment_harness.video.tts:SayEngine",
         "--category", "tts"],
    )
    assert result.exit_code == 0
    assert "conforms" in result.output


def test_eh_plugin_check_bad_target_and_import():
    result = runner.invoke(app, ["plugin", "check", "no-colon", "--category", "tts"])
    assert result.exit_code == 1
    result = runner.invoke(
        app, ["plugin", "check", "no.such.mod:Cls", "--category", "tts"]
    )
    assert result.exit_code == 1
