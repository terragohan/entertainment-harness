"""`eh cloud` CLI tests against a respx-mocked Worker. No network."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from typer.testing import CliRunner

from entertainment_harness import db
from entertainment_harness.cli import app
from entertainment_harness.library import works
from entertainment_harness.store import r2 as store_r2
from entertainment_harness.store import staging

runner = CliRunner()

BASE = "https://w.example.workers.dev"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text(f'[cloud]\nworker_url = "{BASE}"\n')
    monkeypatch.setenv("EH_CLOUD_TOKEN", "tok")
    # submit() resolves the work against the local index before posting.
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.commit()
    conn.close()
    return tmp_path


def _job(**over):
    job = {
        "id": "abc123",
        "kind": "recap",
        "work": "s1",
        "status": "queued",
        "options": {},
        "current": None,
        "error": None,
        "stop_requested": 0,
        "created_at": "2026-10-15T00:00:00",
        "updated_at": "2026-10-15T00:00:00",
    }
    job.update(over)
    return job


# --- configuration --------------------------------------------------------


def test_missing_worker_url_exits_with_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EH_CLOUD_TOKEN", "tok")
    result = runner.invoke(app, ["cloud", "status"])
    assert result.exit_code == 1
    assert "worker_url" in result.output


def test_missing_token_exits_with_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text(f'[cloud]\nworker_url = "{BASE}"\n')
    monkeypatch.delenv("EH_CLOUD_TOKEN", raising=False)
    result = runner.invoke(app, ["cloud", "status"])
    assert result.exit_code == 1
    assert "EH_CLOUD_TOKEN" in result.output


@respx.mock
def test_config_token_used_when_env_unset(tmp_path, monkeypatch):
    """[cloud] token from config.toml authenticates without the env var."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        f'[cloud]\nworker_url = "{BASE}"\ntoken = "cfg-tok"\n'
    )
    monkeypatch.delenv("EH_CLOUD_TOKEN", raising=False)
    route = respx.get(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )
    result = runner.invoke(app, ["cloud", "status"])
    assert result.exit_code == 0
    assert route.calls[0].request.headers["Authorization"] == "Bearer cfg-tok"


# --- submit ---------------------------------------------------------------


@respx.mock
def test_submit_posts_spec_with_bearer(env):
    route = respx.post(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    result = runner.invoke(
        app, ["cloud", "submit", "s1", "--chapter", "1", "--video"]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "work": "s1",
        "kind": "recap",
        "options": {"chapter": 1.0, "video": True},
    }
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"
    assert "abc123" in result.output


@respx.mock
def test_submit_options_json_merges_flags_win(env):
    route = respx.post(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    result = runner.invoke(
        app,
        ["cloud", "submit", "s1", "--chapter", "2",
         "--options", '{"chapter": 9, "detail": "brief"}'],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(route.calls[0].request.content)
    assert body["options"] == {"chapter": 2.0, "detail": "brief"}


def test_submit_rejects_bad_options_json(env):
    result = runner.invoke(app, ["cloud", "submit", "s1", "--options", "{"])
    assert result.exit_code == 1
    assert "not valid JSON" in result.output


def test_submit_unknown_work_fails_before_http(env):
    """A typo'd work must fail locally — not after a container boot."""
    with respx.mock(assert_all_called=False) as router:
        result = runner.invoke(app, ["cloud", "submit", "nope"])
    assert result.exit_code == 1
    assert "No series matches" in result.output
    assert router.calls.call_count == 0


@respx.mock
def test_submit_stages_up_local_work(env, monkeypatch):
    """With a local work dir and R2 configured, submit pushes the work first."""
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    staged = []
    monkeypatch.setattr(store_r2, "is_ready", lambda config: True)
    monkeypatch.setattr(
        staging, "stage_up",
        lambda config, sid, log=print: staged.append(sid) or 3,
    )
    route = respx.post(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    result = runner.invoke(app, ["cloud", "submit", "s1"])
    assert result.exit_code == 0, result.output
    assert staged == ["s1"]
    assert route.called


@respx.mock
def test_submit_without_r2_warns_and_submits(env, monkeypatch):
    """R2 unconfigured → a warning, and the submit still goes through (the
    cloud may already hold the work from an earlier job)."""
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    for var in ("R2_BUCKET", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    respx.post(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    result = runner.invoke(app, ["cloud", "submit", "s1"])
    assert result.exit_code == 0, result.output
    assert "without stage-up" in result.output


def test_submit_stage_up_failure_aborts(env, monkeypatch):
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    monkeypatch.setattr(store_r2, "is_ready", lambda config: True)

    def fail(config, sid, log=print):
        from entertainment_harness.store import StoreError

        raise StoreError("upload boom")

    monkeypatch.setattr(staging, "stage_up", fail)
    with respx.mock(assert_all_called=False) as router:
        result = runner.invoke(app, ["cloud", "submit", "s1"])
    assert result.exit_code == 1
    assert "Stage-up failed" in result.output
    assert router.calls.call_count == 0


@respx.mock
def test_submit_conflict_prints_worker_error(env):
    respx.post(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(
            409, json={"error": "job zzz is already active for this work"}
        )
    )
    result = runner.invoke(app, ["cloud", "submit", "s1"])
    assert result.exit_code == 1
    assert "already active" in result.output


@respx.mock
def test_unauthorized_has_friendly_message(env):
    respx.post(f"{BASE}/api/jobs").mock(return_value=httpx.Response(401))
    result = runner.invoke(app, ["cloud", "submit", "s1"])
    assert result.exit_code == 1
    assert "Unauthorized" in result.output


# --- status / stop / logs ---------------------------------------------------


@respx.mock
def test_status_lists_jobs(env):
    respx.get(f"{BASE}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [
            _job(status="running", work="Paper Moons"),
            _job(id="old1", status="done"),
        ]})
    )
    result = runner.invoke(app, ["cloud", "status"])
    assert result.exit_code == 0, result.output
    assert "running" in result.output and "Paper Moons" in result.output
    assert "old1" in result.output


@respx.mock
def test_status_single_job_detail(env):
    respx.get(f"{BASE}/api/jobs/abc123").mock(
        return_value=httpx.Response(200, json={"job": _job(
            status="running",
            current={"event": "chapter-start", "chapter": 2, "stage": None,
                     "detail": None},
        )})
    )
    result = runner.invoke(app, ["cloud", "status", "abc123"])
    assert result.exit_code == 0, result.output
    assert "running" in result.output
    assert "chapter=2" in result.output


@respx.mock
def test_stop_posts_and_explains(env):
    respx.post(f"{BASE}/api/jobs/abc123/stop").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    result = runner.invoke(app, ["cloud", "stop", "abc123"])
    assert result.exit_code == 0, result.output
    assert "next chapter boundary" in result.output


@respx.mock
def test_logs_pretty_prints_events(env):
    lines = "\n".join([
        json.dumps({"event": "log", "ts": "2026-10-15T01:02:03",
                    "message": "D1 snapshot staged (0 row(s))."}),
        json.dumps({"event": "chapter-done", "ts": "2026-10-15T01:05:03",
                    "chapter": 1.0}),
    ])
    respx.get(f"{BASE}/api/jobs/abc123/events").mock(
        return_value=httpx.Response(
            200, text=lines, headers={"content-type": "application/x-ndjson"}
        )
    )
    result = runner.invoke(app, ["cloud", "logs", "abc123"])
    assert result.exit_code == 0, result.output
    assert "D1 snapshot staged" in result.output
    assert "chapter-done" in result.output
    assert "chapter 1" in result.output


@respx.mock
def test_logs_missing_job(env):
    respx.get(f"{BASE}/api/jobs/nope/events").mock(
        return_value=httpx.Response(404, json={"error": "no events for job nope"})
    )
    result = runner.invoke(app, ["cloud", "logs", "nope"])
    assert result.exit_code == 1
    assert "no events" in result.output
