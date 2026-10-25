"""config.toml [sources] parsing and the save_config writer."""

from __future__ import annotations

import pytest

from entertainment_harness.config import (
    DEFAULT_ENABLED_SOURCES,
    Config,
    ConfigError,
    ConfigWriteError,
    apply_cloud_profile,
    load_config,
    parse_config,
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


def test_parse_config_rejects_the_renamed_sequence_section():
    with pytest.raises(ConfigError, match="frame_animation"):
        parse_config({"sequence": {"fps": 2.0}})


def test_cloud_absent_uses_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.cloud.enabled is False
    assert cfg.cloud.worker_url == ""
    assert cfg.cloud.max_job_seconds == 1800


def test_cloud_section_parse(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[cloud]\n"
        "enabled = true\n"
        'worker_url = "https://w.example.workers.dev"\n'
        'database_id = "db1"\n'
    )
    cfg = load_config(path)
    assert cfg.cloud.enabled is True
    assert cfg.cloud.worker_url == "https://w.example.workers.dev"
    assert cfg.cloud.database_id == "db1"


def test_cloud_token_parse_and_resolution(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[cloud]\ntoken = "cfg-tok"\n')
    cfg = load_config(path)
    assert cfg.cloud.token == "cfg-tok"
    # The config value wins over the env var.
    monkeypatch.setenv("EH_CLOUD_TOKEN", "env-tok")
    assert cfg.cloud.resolved_token == "cfg-tok"
    # An empty config token falls back to the env var.
    cfg.cloud.token = ""
    assert cfg.cloud.resolved_token == "env-tok"
    # Neither set → no token.
    monkeypatch.delenv("EH_CLOUD_TOKEN")
    assert cfg.cloud.resolved_token == ""


def test_save_config_accepts_cloud_token(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[cloud]\nenabled = true\n")
    save_config(path, {"cloud": {"token": "tok"}})
    assert load_config(path).cloud.token == "tok"


def test_cloud_profile_falls_back_to_kokoro_without_fish_key(monkeypatch):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    cfg = apply_cloud_profile(Config())
    assert cfg.video.tts_engine == "kokoro"


def test_cloud_profile_keeps_fish_when_key_present(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "sk-fish-test")
    cfg = apply_cloud_profile(Config())
    assert cfg.video.tts_engine == "fish"


def test_cloud_profile_leaves_non_fish_engines_alone(monkeypatch):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    cfg = Config()
    cfg.video.tts_engine = "say"
    assert apply_cloud_profile(cfg).video.tts_engine == "say"
