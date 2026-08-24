"""config.toml [sources] parsing and the save_config writer."""

from __future__ import annotations

import pytest

from entertainment_harness.config import (
    DEFAULT_ENABLED_SOURCES,
    Config,
    ConfigWriteError,
    load_config,
    save_config,
)


def test_sources_absent_uses_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[video]\nvoice = "af_heart"\n')
    assert load_config(path).sources.enabled == DEFAULT_ENABLED_SOURCES


def test_sources_missing_file_uses_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.sources.enabled == ["mangadex", "weebcentral"]


def test_sources_empty_table_uses_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[sources]\n")
    assert load_config(path).sources.enabled == DEFAULT_ENABLED_SOURCES


def test_sources_enabled_list(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[sources]\nenabled = ["mangadex"]\n')
    assert load_config(path).sources.enabled == ["mangadex"]


def test_sources_default_list_is_per_instance():
    a, b = Config(), Config()
    a.sources.enabled.append("x")
    assert b.sources.enabled == DEFAULT_ENABLED_SOURCES


# --- save_config writer ------------------------------------------------------


def test_save_config_creates_file_and_parents(tmp_path):
    path = tmp_path / "sub" / "config.toml"
    save_config(path, {"sources": {"enabled": ["weebcentral"]}})
    assert load_config(path).sources.enabled == ["weebcentral"]


def test_save_config_preserves_comments_and_unrelated_content(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "# hand-edited config\n"
        "[video]\n"
        'voice = "af_heart"  # my favorite voice\n'
        "\n"
        "[library]\n"
        'langs = ["en", "ja"]\n'
    )
    save_config(path, {"sources": {"enabled": ["mangadex"]}})
    text = path.read_text()
    assert "# hand-edited config" in text
    assert 'voice = "af_heart"  # my favorite voice' in text
    cfg = load_config(path)
    assert cfg.sources.enabled == ["mangadex"]
    assert cfg.video.voice == "af_heart"
    assert cfg.library.langs == ["en", "ja"]


def test_save_config_merges_nested_tables(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[models.vision]\nbackend = "ollama"\nmodel = "old"\n')
    save_config(path, {"models": {"vision": {"model": "new"}}})
    cfg = load_config(path)
    assert cfg.models.vision.model == "new"
    assert cfg.models.vision.backend == "ollama"


def test_save_config_result_roundtrips_through_load_config(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, {
        "sources": {"enabled": ["weebcentral"]},
        "video": {"colorize": True},
        "preflight": {"min_free_disk_gb": 10.5},
    })
    cfg = load_config(path)
    assert cfg.sources.enabled == ["weebcentral"]
    assert cfg.video.colorize is True
    assert cfg.preflight.min_free_disk_gb == 10.5


@pytest.mark.parametrize("updates", [
    {"bogus": {"x": 1}},                            # unknown section
    {"sources": {"enabledd": []}},                  # unknown key
    {"sources": {"enabled": "mangadex"}},           # list, not string
    {"sources": {"enabled": [1, 2]}},               # list of strings only
    {"video": {"keep_master": "yes"}},              # bool, not string
    {"hardware": {"budget_gb": "sixteen"}},         # number, not string
    {"sources": "mangadex"},                        # section update must be a table
])
def test_save_config_rejects_invalid_updates_without_touching_file(tmp_path, updates):
    path = tmp_path / "config.toml"
    original = '[video]\nvoice = "af_heart"\n'
    path.write_text(original)
    with pytest.raises(ConfigWriteError):
        save_config(path, updates)
    assert path.read_text() == original


def test_save_config_atomic_no_temp_leftovers(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, {"sources": {"enabled": ["mangadex"]}})
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]


def test_save_config_failed_replace_leaves_original_and_cleans_up(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.toml"
    original = '[video]\nvoice = "af_heart"\n'
    path.write_text(original)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("entertainment_harness.config.os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        save_config(path, {"sources": {"enabled": []}})
    assert path.read_text() == original
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"]
