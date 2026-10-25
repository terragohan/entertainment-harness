"""`eh serve` remote mode ([cloud] enabled): /api/runs* proxy to the cloud
Worker (respx-mocked; no network). See server/remote.py."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from entertainment_harness import db, library
from entertainment_harness.library import works
from entertainment_harness.server import auto, remote
from entertainment_harness.server.app import create_app
from entertainment_harness.store import StoreError

runner_base = "https://w.example.workers.dev"


@pytest.fixture(autouse=True)
def _no_auto_timers(monkeypatch):
    """The supervisor's timers must never fire in tests: with a remote
    manager they would make real HTTP calls once respx is torn down."""
    monkeypatch.setattr(auto, "_ERROR_RECHECK_S", 3600.0)
    monkeypatch.setattr(auto.AutoSupervisor, "_start_timer", lambda self: None)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Series s1 (Test Manga, ch-1/ch-2) in a scratch data dir, cloud mode on."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        f'[cloud]\nenabled = true\nworker_url = "{runner_base}"\n'
    )
    monkeypatch.setenv("EH_CLOUD_TOKEN", "tok")
    # stage-up checks R2 readiness from the env too — keep tests hermetic.
    for var in ("R2_BUCKET", "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    works.write_chapter_metadata("s1", works.ChapterMetadata(
        id="ch-1", chapter_num=1.0, title="Ch 1", lang="en",
        pages=9, published_at=None, fetched_at="now",
    ))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'Ch 1', 'en', 9, 'now')"
    )
    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture
def client(data):
    return TestClient(create_app())


def _job(**over):
    job = {
        "id": "j1",
        "kind": "recap",
        "work": "s1",
        "status": "queued",
        "options": {"video": True},
        "current": None,
        "error": None,
        "stop_requested": 0,
        "created_at": "2026-10-15T00:00:00",
        "updated_at": "2026-10-15T00:00:00",
    }
    job.update(over)
    return job


# --- run control proxy ------------------------------------------------------


@respx.mock
def test_start_run_submits_cloud_job(client):
    route = respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    resp = client.post("/api/runs", json={"work": "s1", "video": True})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"] == "j1"
    assert body["status"] == "queued"
    assert body["title"] == "Test Manga"  # injected from the local index
    assert body["options"]["video"] is True  # normalized RunOptions shape
    assert "skip_done" in body["options"]
    sent = json.loads(route.calls[0].request.content)
    assert sent["work"] == "s1" and sent["kind"] == "recap"
    assert sent["options"]["video"] is True
    assert route.calls[0].request.headers["authorization"] == "Bearer tok"


@respx.mock
def test_start_run_conflict_maps_to_409(client):
    respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(
            409, json={"error": "job j1 is already active for this work"}
        )
    )
    resp = client.post("/api/runs", json={"work": "s1"})
    assert resp.status_code == 409


# --- stage-up on submit -------------------------------------------------------


@respx.mock
def test_start_run_stages_up_before_submit(client, monkeypatch):
    """A local work dir + configured R2 → the work is pushed before the job
    is queued, so the container can stage it back in."""
    staged = []
    monkeypatch.setattr(remote.store_r2, "is_ready", lambda config: True)
    monkeypatch.setattr(
        remote.staging, "stage_up",
        lambda config, sid, log=print: staged.append(sid) or 4,
    )
    route = respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    resp = client.post("/api/runs", json={"work": "s1"})
    assert resp.status_code == 201, resp.text
    assert staged == ["s1"]
    assert route.called


@respx.mock
def test_start_run_skips_stage_up_without_r2(client, monkeypatch):
    """R2 unconfigured → submit proceeds with a stderr note (the cloud may
    already hold the work from an earlier job)."""
    monkeypatch.setattr(remote.store_r2, "is_ready", lambda config: False)

    def boom(config, sid, log=print):
        raise AssertionError("stage_up must not run when R2 is unconfigured")

    monkeypatch.setattr(remote.staging, "stage_up", boom)
    respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    resp = client.post("/api/runs", json={"work": "s1"})
    assert resp.status_code == 201, resp.text


@respx.mock
def test_start_run_stage_up_failure_aborts_submit(client, monkeypatch):
    """A failed push aborts the submit with 502 — a half-staged work would
    fail in the container anyway."""
    monkeypatch.setattr(remote.store_r2, "is_ready", lambda config: True)

    def fail(config, sid, log=print):
        raise StoreError("upload boom")

    monkeypatch.setattr(remote.staging, "stage_up", fail)
    route = respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()})
    )
    resp = client.post("/api/runs", json={"work": "s1"})
    assert resp.status_code == 502
    assert "stage-up failed" in resp.json()["detail"]
    assert not route.called


@respx.mock
def test_list_runs_maps_ledger(client):
    respx.get(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [
            _job(status="running"),
            _job(id="j0", status="done", work="gone"),  # unknown work: id as title
        ]})
    )
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["id"] for r in runs] == ["j1", "j0"]
    assert runs[0]["title"] == "Test Manga"
    assert runs[1]["title"] == "gone"


@respx.mock
def test_get_run_404(client):
    respx.get(f"{runner_base}/api/jobs/nope").mock(
        return_value=httpx.Response(404, json={"error": "no job nope"})
    )
    assert client.get("/api/runs/nope").status_code == 404


@respx.mock
def test_stop_run(client):
    respx.get(f"{runner_base}/api/jobs/j1").mock(
        return_value=httpx.Response(200, json={"job": _job(status="running")})
    )
    stop = respx.post(f"{runner_base}/api/jobs/j1/stop").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    resp = client.post("/api/runs/j1/stop")
    assert resp.status_code == 200
    assert stop.called


@respx.mock
def test_stop_terminal_run_is_noop(client):
    """Worker 409 (job not active) mirrors the local idempotent stop."""
    respx.get(f"{runner_base}/api/jobs/j1").mock(
        return_value=httpx.Response(200, json={"job": _job(status="done")})
    )
    respx.post(f"{runner_base}/api/jobs/j1/stop").mock(
        return_value=httpx.Response(409, json={"error": "job j1 is not active"})
    )
    resp = client.post("/api/runs/j1/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"


@respx.mock
def test_worker_unreachable_is_502(client):
    respx.get(f"{runner_base}/api/jobs").mock(
        side_effect=httpx.ConnectError("down")
    )
    resp = client.get("/api/runs")
    assert resp.status_code == 502
    assert "unreachable" in resp.json()["detail"]


# --- SSE bridge ---------------------------------------------------------------


@respx.mock
def test_events_stream_replays_and_closes_on_terminal_event(client):
    events = "\n".join([
        json.dumps({"event": "run-start", "run_id": "j1", "chapters": 1}),
        json.dumps({"event": "chapter-done", "run_id": "j1", "chapter": 1.0}),
        json.dumps({"event": "run-done", "run_id": "j1", "chapters": 1}),
    ])
    respx.get(f"{runner_base}/api/jobs/j1").mock(
        return_value=httpx.Response(200, json={"job": _job(status="done")})
    )
    respx.get(f"{runner_base}/api/jobs/j1/events").mock(
        return_value=httpx.Response(200, text=events)
    )
    with client.stream("GET", "/api/runs/j1/events") as resp:
        body = "".join(resp.iter_text())
    assert resp.status_code == 200
    assert "event: run-start" in body
    assert "event: chapter-done" in body
    assert "event: run-done" in body


@respx.mock
def test_events_stream_synthesizes_close_on_crash(client):
    """Ledger terminal but the events file never got a terminal event (hard
    crash): the stream still closes with a synthesized frame."""
    respx.get(f"{runner_base}/api/jobs/j1").mock(
        return_value=httpx.Response(200, json={
            "job": _job(status="error", error="container exited with code 137"),
        })
    )
    respx.get(f"{runner_base}/api/jobs/j1/events").mock(
        return_value=httpx.Response(
            200, text=json.dumps({"event": "run-start", "run_id": "j1"}) + "\n"
        )
    )
    with client.stream("GET", "/api/runs/j1/events") as resp:
        body = "".join(resp.iter_text())
    assert "event: run-start" in body
    assert "event: run-error" in body
    assert "container exited with code 137" in body


@respx.mock
def test_events_stream_404_unknown_job(client):
    respx.get(f"{runner_base}/api/jobs/nope").mock(
        return_value=httpx.Response(404, json={"error": "no job nope"})
    )
    assert client.get("/api/runs/nope/events").status_code == 404


# --- auto toggle --------------------------------------------------------------


@respx.mock
def test_auto_toggle_submits_cloud_job(client):
    route = respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()}))
    respx.get(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )
    resp = client.put("/api/works/s1/auto", json={"enabled": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["enabled"] is True
    assert body["run"]["id"] == "j1"
    sent = json.loads(route.calls[0].request.content)
    assert sent["work"] == "s1"
    assert sent["options"]["video"] is True  # auto runs always build videos


@respx.mock
def test_auto_reconcile_disarms_after_terminal_remote_run(client):
    """A finished cloud job disarms the toggle instead of re-submitting
    against the (stale) local index."""
    respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()}))
    respx.get(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )
    resp = client.put("/api/works/s1/auto", json={"enabled": True})
    assert resp.json()["enabled"] is True
    supervisor = client.app.state.auto
    assert supervisor.is_enabled("s1")

    # The job finishes; the next reconcile sees the terminal ledger row.
    respx.get(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [_job(status="done")]})
    )
    supervisor.reconcile()
    assert not supervisor.is_enabled("s1")
    # ...and no new job was submitted.
    assert respx.post(f"{runner_base}/api/jobs").call_count == 1


@respx.mock
def test_auto_queued_job_counts_as_active(client):
    """A queued cloud job blocks a duplicate submission."""
    respx.post(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(202, json={"job": _job()}))
    respx.get(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )
    client.put("/api/works/s1/auto", json={"enabled": True})
    respx.get(f"{runner_base}/api/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": [_job(status="queued")]})
    )
    client.app.state.auto.reconcile()
    assert respx.post(f"{runner_base}/api/jobs").call_count == 1


def test_remote_mode_off_by_default(data, monkeypatch):
    """No [cloud] enabled → the local RunManager is used."""
    (data / "config.toml").write_text("")
    app = create_app()
    assert type(app.state.runs).__name__ == "RunManager"


def test_remote_mode_selected(client):
    assert type(client.app.state.runs).__name__ == "RemoteRunManager"


def test_missing_token_errors_actionably_without_network(data, monkeypatch):
    """An empty EH_CLOUD_TOKEN must surface as guidance, not an httpx
    'Illegal header value' garble — and must not attempt any request."""
    monkeypatch.delenv("EH_CLOUD_TOKEN", raising=False)
    client = TestClient(create_app())
    with respx.mock(assert_all_called=False) as router:
        resp = client.get("/api/runs")
    assert resp.status_code == 502
    assert "EH_CLOUD_TOKEN" in resp.json()["detail"]
    assert router.calls.call_count == 0


def test_config_token_bearer_applied_live(data, monkeypatch):
    """[cloud] token from config.toml is sent as the bearer, and a later
    settings save is picked up per request — no manager rebuild/restart."""
    monkeypatch.delenv("EH_CLOUD_TOKEN", raising=False)
    (data / "config.toml").write_text(
        f'[cloud]\nenabled = true\nworker_url = "{runner_base}"\n'
        'token = "cfg-tok"\n'
    )
    client = TestClient(create_app())
    with respx.mock:
        route = respx.get(f"{runner_base}/api/jobs").mock(
            return_value=httpx.Response(200, json={"jobs": []})
        )
        assert client.get("/api/runs").status_code == 200
        assert (
            route.calls[0].request.headers["Authorization"]
            == "Bearer cfg-tok"
        )
        (data / "config.toml").write_text(
            f'[cloud]\nenabled = true\nworker_url = "{runner_base}"\n'
            'token = "cfg-tok-2"\n'
        )
        assert client.get("/api/runs").status_code == 200
        assert (
            route.calls[1].request.headers["Authorization"]
            == "Bearer cfg-tok-2"
        )
