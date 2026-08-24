"""Tests for scanlation job config and resolution."""

from __future__ import annotations

from entertainment_harness.config import (
    Config,
    ScanlationJobConfig,
    StageConfig,
    load_config,
)
from entertainment_harness.pipelines.translate import (
    _active_scanlation_job,
    _synthesize_scanlation_job_from_roles,
)


def test_load_config_parses_scanlation_job(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[scanlation]\n'
        'job = "lfm"\n\n'
        '[jobs.scanlation.lfm]\n\n'
        '[jobs.scanlation.lfm.extract]\n'
        'backend = "lfm"\n'
        'model = "LiquidAI/LFM2.5-VL-3B"\n\n'
        '[jobs.scanlation.lfm.translate]\n'
        'backend = "ollama"\n'
        'model = "qwen3:4b"\n\n'
        '[jobs.scanlation.lfm.render]\n'
        'backend = "openai_compat"\n'
        'model = "openai/gpt-4o"\n'
    )
    cfg = load_config(path)
    assert cfg.scanlation.job == "lfm"
    assert "lfm" in cfg.jobs.scanlation
    job = cfg.jobs.scanlation["lfm"]
    assert job.extract == StageConfig(backend="lfm", model="LiquidAI/LFM2.5-VL-3B")
    assert job.translate == StageConfig(backend="ollama", model="qwen3:4b")
    # judge omitted -> defaults to translate stage
    assert job.judge_stage == job.translate
    assert job.render == StageConfig(backend="openai_compat", model="openai/gpt-4o")


def test_load_config_supports_default_job_name(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[jobs.scanlation]\n'
        'default = "vision_ollama"\n\n'
        '[jobs.scanlation.vision_ollama]\n\n'
        '[jobs.scanlation.vision_ollama.extract]\n'
        'backend = "ollama"\n'
        'model = "qwen3-vl:8b-instruct"\n'
    )
    cfg = load_config(path)
    assert cfg.scanlation.job == "vision_ollama"
    assert "vision_ollama" in cfg.jobs.scanlation


def test_load_config_preserves_stage_extra_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[jobs.scanlation.lfm.extract]\n'
        'backend = "lfm"\n'
        'model = "LiquidAI/LFM2.5-VL-3B"\n'
        'include_text = true\n'
        'device = "cpu"\n'
    )
    cfg = load_config(path)
    stage = cfg.jobs.scanlation["lfm"].extract
    assert stage.extra == {"include_text": True, "device": "cpu"}


def test_active_job_returns_explicit_job():
    cfg = Config()
    cfg.scanlation.job = "openai_compat"
    cfg.jobs.scanlation["openai_compat"] = ScanlationJobConfig(
        extract=StageConfig(backend="openai_compat", model="openai/gpt-4o"),
        translate=StageConfig(backend="openai_compat", model="openai/gpt-4o-mini"),
        render=StageConfig(backend="openai_compat", model="openai/gpt-4o"),
    )
    job = _active_scanlation_job(cfg)
    assert job.extract.backend == "openai_compat"


def test_active_job_falls_back_to_first_defined_job():
    cfg = Config()
    cfg.jobs.scanlation["vision_ollama"] = ScanlationJobConfig(
        extract=StageConfig(backend="ollama", model="qwen3-vl:8b-instruct"),
        translate=StageConfig(backend="ollama", model="qwen3:4b"),
        render=StageConfig(backend="ollama", model="qwen3-vl:8b-instruct"),
    )
    job = _active_scanlation_job(cfg)
    assert job.extract.model == "qwen3-vl:8b-instruct"


def test_synthesize_job_uses_lfm_when_configured():
    cfg = Config()
    cfg.models.lfm_model = "LiquidAI/LFM2.5-VL-3B"
    job = _synthesize_scanlation_job_from_roles(cfg)
    assert job.extract == StageConfig(backend="lfm", model="LiquidAI/LFM2.5-VL-3B")
    # translate/judge fall back to the default text role
    assert job.translate.backend == "ollama"
    assert job.translate.model == "qwen3:4b"
    assert job.render.backend == "ollama"
    assert job.render.model == "qwen3-vl:8b-instruct"


def test_synthesize_job_uses_vision_role_without_lfm():
    cfg = Config()
    job = _synthesize_scanlation_job_from_roles(cfg)
    assert job.extract.backend == "ollama"
    assert job.extract.model == "qwen3-vl:8b-instruct"
