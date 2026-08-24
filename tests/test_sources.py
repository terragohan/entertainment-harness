"""Source gating via [sources].enabled and the `eh sources` commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from entertainment_harness.cli import app
from entertainment_harness.config import Config, load_config
from entertainment_harness.plugins import PluginError
from entertainment_harness.sources import REGISTRY, get_client, is_source_enabled
from entertainment_harness.sources.mangadex import MangaDexClient
from entertainment_harness.sources.weebcentral import WeebCentralClient

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    return tmp_path


def _write_enabled(data_dir, enabled: list[str]) -> None:
    items = ", ".join(f'"{n}"' for n in enabled)
    (data_dir / "config.toml").write_text(f"[sources]\nenabled = [{items}]\n")


class _FakeEntryPointSource:
    def __init__(self, config=None, **overrides) -> None:
        self.config = config


@pytest.fixture
def fake_entry_point(monkeypatch):
    """Register a third-party (entry-point) source named 'fakesrc'."""
    monkeypatch.setattr(
        REGISTRY, "_plugins", {**REGISTRY._plugins, "fakesrc": _FakeEntryPointSource}
    )
    monkeypatch.setattr(
        REGISTRY, "_entry_point_names", {*REGISTRY._entry_point_names, "fakesrc"}
    )
    return "fakesrc"


# --- gating ------------------------------------------------------------------


def test_disabled_builtin_raises_clear_error(data_dir):
    _write_enabled(data_dir, ["mangadex"])
    with pytest.raises(PluginError, match="disabled in config.toml"):
        get_client("weebcentral")
    with pytest.raises(PluginError, match="eh sources enable weebcentral"):
        get_client("weebcentral", load_config())
    assert isinstance(get_client("mangadex"), MangaDexClient)


def test_explicit_config_wins_over_file(data_dir):
    _write_enabled(data_dir, ["mangadex"])
    assert isinstance(get_client("weebcentral", Config()), WeebCentralClient)


def test_absent_sources_section_keeps_all_builtins_enabled(data_dir):
    assert isinstance(get_client("weebcentral"), WeebCentralClient)


def test_unknown_source_keeps_unknown_plugin_error(data_dir):
    _write_enabled(data_dir, ["mangadex"])
    with pytest.raises(PluginError, match="Unknown sources plugin"):
        get_client("nope")


def test_entry_point_sources_are_not_gated(data_dir, fake_entry_point):
    _write_enabled(data_dir, ["mangadex"])
    assert is_source_enabled("fakesrc", Config()) is True
    assert isinstance(get_client("fakesrc"), _FakeEntryPointSource)


def test_search_disabled_source_errors_cleanly(data_dir):
    _write_enabled(data_dir, ["mangadex"])
    result = runner.invoke(app, ["search", "anything", "--source", "weebcentral"])
    assert result.exit_code == 1
    assert "disabled in config.toml" in result.output
    # the [sources] reference must survive rich markup rendering
    assert "[sources].enabled" in result.output


def test_plugins_shows_source_state(data_dir):
    _write_enabled(data_dir, ["mangadex"])
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    assert "disabled" in result.output
    assert "enabled" in result.output


# --- eh sources ---------------------------------------------------------------


def test_sources_lists_registered_with_state(data_dir):
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    assert "mangadex" in result.output
    assert "weebcentral" in result.output
    assert "built-in" in result.output
    assert "enabled" in result.output


def test_sources_disable_writes_config_and_gates(data_dir):
    result = runner.invoke(app, ["sources", "disable", "weebcentral"])
    assert result.exit_code == 0, result.output
    assert load_config().sources.enabled == ["mangadex"]
    result = runner.invoke(app, ["sources"])
    assert "disabled" in result.output
    with pytest.raises(PluginError):
        get_client("weebcentral")


def test_sources_disable_is_idempotent(data_dir):
    assert runner.invoke(app, ["sources", "disable", "weebcentral"]).exit_code == 0
    result = runner.invoke(app, ["sources", "disable", "weebcentral"])
    assert result.exit_code == 0
    assert "already disabled" in result.output


def test_sources_enable_restores_builtin_order(data_dir):
    assert runner.invoke(app, ["sources", "disable", "mangadex"]).exit_code == 0
    result = runner.invoke(app, ["sources", "enable", "mangadex"])
    assert result.exit_code == 0, result.output
    assert load_config().sources.enabled == ["mangadex", "weebcentral"]


def test_sources_enable_is_idempotent(data_dir):
    result = runner.invoke(app, ["sources", "enable", "mangadex"])
    assert result.exit_code == 0
    assert "already enabled" in result.output


def test_sources_enable_unknown_rejected(data_dir):
    result = runner.invoke(app, ["sources", "enable", "nope"])
    assert result.exit_code == 1
    assert "Unknown source" in result.output
    assert not (data_dir / "config.toml").exists()


def test_sources_disable_entry_point_rejected(data_dir, fake_entry_point):
    result = runner.invoke(app, ["sources", "disable", "fakesrc"])
    assert result.exit_code == 1
    assert "third-party" in result.output
    assert not (data_dir / "config.toml").exists()


def test_sources_reset_restores_defaults(data_dir):
    assert runner.invoke(app, ["sources", "disable", "weebcentral"]).exit_code == 0
    result = runner.invoke(app, ["sources", "reset"])
    assert result.exit_code == 0, result.output
    assert load_config().sources.enabled == ["mangadex", "weebcentral"]
