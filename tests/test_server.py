"""`eh serve` HTTP API: health, library, video streaming (Range), config and
sources editing, and background runs with SSE progress. Runs are exercised
with a stubbed recap pipeline/build_video — no real TTS/ffmpeg/models."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from entertainment_harness import db
from entertainment_harness.config import load_config
from entertainment_harness.library import works
from entertainment_harness.server.app import create_app


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Series s1 (Test Manga, ch-1/ch-2) in a scratch data dir."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    for cid, num, pages in (("ch-1", 1.0, 9), ("ch-2", 2.0, 4)):
        works.write_chapter_metadata("s1", works.ChapterMetadata(
            id=cid, chapter_num=num, title=f"Ch {num:g}", lang="en",
            pages=pages, published_at=None, fetched_at="now",
        ))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, ?, 'en', ?, 'now')",
        [("ch-1", 1.0, "Ch 1", 9), ("ch-2", 2.0, "Ch 2", 4)],
    )
    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture
def client(data):
    return TestClient(create_app())


@pytest.fixture
def video(client):
    """A rendered recap video for ch-1: DB row + mp4 in the works layout.
    Returns (client, path, payload)."""
    payload = bytes(range(256)) * 4  # 1024 bytes
    path = works.video_file_path("s1", "ch-1", "recap")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    works.write_video_metadata("s1", "ch-1", works.VideoMetadata(
        kind="recap", duration_s=12.5, model=None, tts_engine=None,
        created_at="now",
    ))
    conn = db.connect()
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
        " VALUES ('ch-1', 'recap text', 'now', 'standard')"
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 12.5, 'now', 'recap')",
        (str(path),),
    )
    conn.commit()
    conn.close()
    return client, path, payload


def _wait_for_status(client, run_id, want=("done", "error"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in want:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach {want}: {run}")


def _parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.strip().split("\n\n"):
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if event and data:
            payload = json.loads(data)
            assert payload["event"] == event
            events.append(payload)
    return events


# --- health / library --------------------------------------------------------


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_library_shape(client):
    works_list = client.get("/api/library").json()["works"]
    assert len(works_list) == 1
    work = works_list[0]
    assert work["id"] == "s1"
    assert work["title"] == "Test Manga"
    assert work["source"] == "mangadex"
    assert [c["chapter_num"] for c in work["chapters"]] == [1.0, 2.0]
    ch1 = work["chapters"][0]
    assert ch1["pages"] == 9
    assert ch1["has_recap"] is False
    assert ch1["has_video"] is False
    assert ch1["video"] is None
    assert ch1["stream"] is None


def test_library_chapter_video_status(video):
    client, _, _ = video
    ch1 = client.get("/api/library").json()["works"][0]["chapters"][0]
    assert ch1["has_recap"] is True
    assert ch1["detail"] == "standard"
    assert ch1["has_video"] is True
    assert ch1["video"] == {"kind": "recap", "duration_s": 12.5}
    assert ch1["stream"] == "/api/videos/s1/1/stream"


def test_library_dedupes_multiple_video_rows(client):
    """A chapter keeps exactly one library row no matter how many videos
    rows it has (one per kind is legal); the usable video wins, and a
    chapter whose only row is wiped shows as missing — once."""
    conn = db.connect()
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
        " VALUES ('ch-1', 'recap text', 'now', 'standard')"
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind, wiped_at)"
        " VALUES ('s1', 1.0, 1.0, ?, 1.0, 'now', 'recap', 'later')",
        (str(works.video_file_path("s1", "ch-1", "recap")),),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 2.0, 'now', 'narration')",
        (str(works.video_file_path("s1", "ch-1", "narration")),),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind, wiped_at)"
        " VALUES ('s1', 2.0, 2.0, ?, 3.0, 'now', 'recap', 'later')",
        (str(works.video_file_path("s1", "ch-2", "recap")),),
    )
    conn.commit()
    conn.close()
    work = client.get("/api/library").json()["works"][0]
    assert len(work["chapters"]) == 2
    ch1 = [c for c in work["chapters"] if c["chapter_num"] == 1.0]
    assert len(ch1) == 1
    assert ch1[0]["has_video"] is True
    assert ch1[0]["video"] == {"kind": "narration", "duration_s": 2.0}
    ch2 = [c for c in work["chapters"] if c["chapter_num"] == 2.0]
    assert len(ch2) == 1
    assert ch2[0]["has_video"] is False
    assert ch2[0]["video"] is None


def test_stream_prefers_usable_video(client):
    """With several videos rows for one chapter, the stream serves the
    usable one — a wiped row must not shadow it with a 404."""
    good = works.video_file_path("s1", "ch-1", "narration")
    good.parent.mkdir(parents=True, exist_ok=True)
    good.write_bytes(b"fresh-narration")
    conn = db.connect()
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind, wiped_at)"
        " VALUES ('s1', 1.0, 1.0, ?, 1.0, 'now', 'recap', 'later')",
        (str(works.video_file_path("s1", "ch-1", "recap")),),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 2.0, 'now', 'narration')",
        (str(good),),
    )
    conn.commit()
    conn.close()
    resp = client.get("/api/videos/s1/1/stream")
    assert resp.status_code == 200
    assert resp.content == b"fresh-narration"


# --- video streaming ---------------------------------------------------------


def test_stream_full(video):
    client, _, payload = video
    resp = client.get("/api/videos/s1/1/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.headers["accept-ranges"] == "bytes"
    assert int(resp.headers["content-length"]) == len(payload)
    assert resp.content == payload


def test_stream_range(video):
    client, _, payload = video
    resp = client.get("/api/videos/s1/1/stream", headers={"Range": "bytes=0-99"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 0-99/{len(payload)}"
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.content == payload[:100]


def test_stream_range_suffix(video):
    client, _, payload = video
    resp = client.get("/api/videos/s1/1/stream", headers={"Range": "bytes=-10"})
    assert resp.status_code == 206
    assert resp.content == payload[-10:]


def test_stream_range_not_satisfiable(video):
    client, _, _ = video
    resp = client.get(
        "/api/videos/s1/1/stream", headers={"Range": "bytes=99999-"}
    )
    assert resp.status_code == 416


def test_stream_missing_chapter_video_404(client):
    assert client.get("/api/videos/s1/2/stream").status_code == 404


def test_stream_unknown_work_404(client):
    assert client.get("/api/videos/nope/1/stream").status_code == 404


def test_stream_traversal_rejected(video):
    client, _, payload = video
    # ".." path segments never route (and ids only ever come from DB rows,
    # never raw path joins): every traversal attempt is a 404, not a file.
    resp = client.get("/api/videos/..%2F..%2F..%2Fetc/1/stream")
    assert resp.status_code == 404
    assert resp.content != payload


# --- video export ------------------------------------------------------------


def _wait_for_export(client, job_id, want=("done", "error"), timeout=30.0):
    deadline = time.monotonic() + timeout
    job = None
    while time.monotonic() < deadline:
        jobs = client.get("/api/exports").json()["exports"]
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["status"] in want:
            return job
        time.sleep(0.1)
    raise AssertionError(f"export {job_id} did not reach {want}: {job}")


def test_export_no_playable_videos_errors(client, data, monkeypatch, tmp_path):
    """Nothing playable → the job still returns 202 but ends in a clear
    error state."""
    monkeypatch.setenv("EH_EXPORT_DIR", str(tmp_path / "exports"))
    resp = client.post("/api/works/s1/export")
    assert resp.status_code == 202
    job = _wait_for_export(client, resp.json()["id"], want=("error",))
    assert "no playable local videos" in job["error"]


def test_export_unknown_work_404(client):
    assert client.post("/api/works/nope/export").status_code == 404


def test_export_single_video_copies(client, video, monkeypatch, tmp_path):
    """One playable chapter → straight copy into the export dir."""
    client, path, payload = video
    monkeypatch.setenv("EH_EXPORT_DIR", str(tmp_path / "exports"))
    resp = client.post("/api/works/s1/export")
    assert resp.status_code == 202
    job = _wait_for_export(client, resp.json()["id"])
    assert job["status"] == "done"
    assert job["total"] == 1
    assert job["skipped"] == 1  # ch-2 has no video
    dest = Path(job["dest"])
    assert dest.read_bytes() == payload
    assert dest.name == "test-manga-ch1.mp4"


def test_export_concat_videos(client, data, monkeypatch, tmp_path):
    """Two playable chapters → one assembled mp4 (lossless concat), named
    for the chapter span."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not installed")
    from entertainment_harness.video.assemble import video_duration

    monkeypatch.setenv("EH_EXPORT_DIR", str(tmp_path / "exports"))
    conn = db.connect()
    for cid, num, color in (("ch-1", 1.0, "red"), ("ch-2", 2.0, "blue")):
        path = works.video_file_path("s1", cid, "narration")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"color=c={color}:s=640x360:d=0.5:r=30",
             "-pix_fmt", "yuv420p", str(path)],
            capture_output=True, check=True,
        )
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES ('s1', ?, ?, ?, 0.5, 'now', 'narration')",
            (num, num, str(path)),
        )
    conn.commit()
    conn.close()
    resp = client.post("/api/works/s1/export")
    assert resp.status_code == 202
    job = _wait_for_export(client, resp.json()["id"])
    assert job["status"] == "done"
    assert job["total"] == 2
    assert job["skipped"] == 0
    dest = Path(job["dest"])
    assert dest.name == "test-manga-ch1-2.mp4"
    assert 0.9 < video_duration(dest) < 1.5


def test_export_conflict_409(client, data, monkeypatch, tmp_path):
    """One active export per work: a concurrent POST gets a 409, and the
    blocked job completes (and surfaces its error) once released."""
    import entertainment_harness.video.assemble as assemble_mod

    monkeypatch.setenv("EH_EXPORT_DIR", str(tmp_path / "exports"))
    started = threading.Event()
    release = threading.Event()

    def blocking_concat(parts, dest, log=print):
        started.set()
        assert release.wait(10)
        raise RuntimeError("released")

    monkeypatch.setattr(assemble_mod, "concat_mp4s", blocking_concat)
    conn = db.connect()
    for cid, num in (("ch-1", 1.0), ("ch-2", 2.0)):
        path = works.video_file_path("s1", cid, "recap")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES ('s1', ?, ?, ?, 0.5, 'now', 'recap')",
            (num, num, str(path)),
        )
    conn.commit()
    conn.close()

    resp1 = client.post("/api/works/s1/export")
    assert resp1.status_code == 202
    assert started.wait(5)
    resp2 = client.post("/api/works/s1/export")
    assert resp2.status_code == 409
    release.set()
    job = _wait_for_export(client, resp1.json()["id"], want=("error",))
    assert "released" in job["error"]


# --- config ------------------------------------------------------------------


def test_config_get_defaults(client):
    cfg = client.get("/api/config").json()
    assert cfg["video"]["voice"] == "af_heart"
    assert cfg["sources"]["enabled"] == ["mangadex", "weebcentral"]
    # `raw` (the parsed config.toml for plugin merges) is not exposed
    assert "raw" not in cfg


def test_config_put_roundtrip(client, data):
    resp = client.put("/api/config", json={"video": {"voice": "af_sky"}})
    assert resp.status_code == 200
    assert resp.json()["video"]["voice"] == "af_sky"
    assert client.get("/api/config").json()["video"]["voice"] == "af_sky"
    # actually written to config.toml, not just in memory
    assert 'voice = "af_sky"' in (data / "config.toml").read_text()


def test_config_put_invalid_422(client, data):
    resp = client.put("/api/config", json={"video": {"voice": 123}})
    assert resp.status_code == 422
    resp = client.put("/api/config", json={"nope": {"x": 1}})
    assert resp.status_code == 422
    # the file was never created by the failed updates
    assert not (data / "config.toml").exists()


# --- sources -----------------------------------------------------------------


def test_sources_get(client):
    sources = {
        s["name"]: s for s in client.get("/api/sources").json()["sources"]
    }
    assert sources["mangadex"] == {
        "name": "mangadex", "builtin": True, "enabled": True,
    }
    assert sources["weebcentral"]["enabled"] is True


def test_sources_put_and_reset(client):
    resp = client.put("/api/sources", json={"enabled": ["mangadex"]})
    assert resp.status_code == 200
    sources = {
        s["name"]: s for s in resp.json()["sources"]
    }
    assert sources["mangadex"]["enabled"] is True
    assert sources["weebcentral"]["enabled"] is False
    assert client.get("/api/config").json()["sources"]["enabled"] == ["mangadex"]

    resp = client.post("/api/sources/reset")
    assert resp.status_code == 200
    assert all(s["enabled"] for s in resp.json()["sources"])
    assert client.get("/api/config").json()["sources"]["enabled"] == [
        "mangadex", "weebcentral",
    ]


def test_sources_put_unknown_422(client):
    resp = client.put("/api/sources", json={"enabled": ["mangadex", "nope"]})
    assert resp.status_code == 422


# --- runs --------------------------------------------------------------------


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub recap_series + build_video: drives the progress interface and
    materializes a recap row + recap-kind video for ch-1, no real models."""
    import entertainment_harness.pipelines.recap as recap_mod
    import entertainment_harness.video.pipeline as video_mod

    def fake_recap_series(conn, series, config, profile, **kw):
        progress = kw.get("progress")
        on_recap = kw.get("on_recap")
        progress.start(1)
        progress.chapter_start(1.0)
        progress.stage("recap", "9 pages")
        progress.log("Recapping chapter 1 (9 pages)...")
        conn.execute(
            "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
            " VALUES ('ch-1', 'recap text', 'now', 'standard')"
        )
        conn.commit()
        if on_recap is not None:
            on_recap("ch-1")
        progress.chapter_done()
        return ["ch-1"]

    def fake_build_video(conn, row, chapter_row, config, profile, **kw):
        out = works.video_file_path(row["id"], chapter_row["id"], "recap")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-mp4")
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES (?, ?, ?, ?, 1.0, 'now', 'recap')",
            (row["id"], chapter_row["chapter_num"],
             chapter_row["chapter_num"], str(out)),
        )
        conn.commit()
        return out

    monkeypatch.setattr(recap_mod, "recap_series", fake_recap_series)
    monkeypatch.setattr(video_mod, "build_video", fake_build_video)


def test_run_events_sequence(client, stub_pipeline):
    resp = client.post("/api/runs", json={
        "work": "s1", "video": True, "panel_first": False,
        "skip_preflight": True,
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "running"
    assert body["work"] == "s1"
    assert body["title"] == "Test Manga"
    run_id = body["id"]

    run = _wait_for_status(client, run_id)
    assert run["status"] == "done"
    assert run["error"] is None

    resp = client.get(f"/api/runs/{run_id}/events")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    types = [e["event"] for e in events]
    assert types[0] == "run-start"
    assert types[-1] == "run-done"
    # ordering of the chapter lifecycle within the run
    assert types.index("chapter-start") < types.index("stage")
    assert types.index("stage") < types.index("video-ready")
    assert types.index("video-ready") < types.index("chapter-done")
    assert all(e["run_id"] == run_id and e["work"] == "s1" for e in events)

    start = events[0]
    assert start["chapters"] == 1
    ready = next(e for e in events if e["event"] == "video-ready")
    assert ready["chapter"] == 1.0
    assert ready["kind"] == "recap"
    assert ready["duration_s"] == 1.0
    assert ready["stream"] == "/api/videos/s1/1/stream"
    done = events[-1]
    assert done["chapters"] == 1

    # the run is listed, and the video it "rendered" is streamable
    runs = client.get("/api/runs").json()["runs"]
    assert [r["id"] for r in runs] == [run_id]
    assert client.get("/api/videos/s1/1/stream").status_code == 200


def test_run_current_stage_in_api(client, monkeypatch):
    """`current` mirrors the latest non-log event for polling clients, and
    clears at the terminal state."""
    import entertainment_harness.pipelines.recap as recap_mod

    started = threading.Event()
    release = threading.Event()

    def blocking_recap_series(conn, series, config, profile, **kw):
        progress = kw.get("progress")
        progress.start(2)
        progress.chapter_start(3.0)
        progress.stage("recap", "pages 1-4 of 9")
        progress.log("Recapping chapter 3 (9 pages)...")  # must not move it
        started.set()
        release.wait(10)
        return []

    monkeypatch.setattr(recap_mod, "recap_series", blocking_recap_series)
    try:
        resp = client.post(
            "/api/runs", json={"work": "s1", "skip_preflight": True}
        )
        assert resp.status_code == 201
        assert "current" in resp.json()
        run_id = resp.json()["id"]
        assert started.wait(5)

        want = {
            "event": "stage", "chapter": 3.0,
            "stage": "recap", "detail": "pages 1-4 of 9",
        }
        run = client.get(f"/api/runs/{run_id}").json()
        assert run["current"] == want
        listed = client.get("/api/runs").json()["runs"]
        assert listed[0]["id"] == run_id
        assert listed[0]["current"] == want
    finally:
        release.set()

    run = _wait_for_status(client, run_id)
    assert run["status"] == "done"
    assert run["current"] is None


def test_run_stop_cancels(client, monkeypatch):
    """POST /api/runs/{id}/stop unwinds the run cooperatively: the terminal
    event is run-cancelled, status is 'cancelled', done chapters are
    reported, and a second stop is a harmless no-op."""
    import entertainment_harness.pipelines.recap as recap_mod

    started = threading.Event()
    release = threading.Event()

    def blocking_recap_series(conn, series, config, profile, **kw):
        progress = kw.get("progress")
        progress.start(2)
        progress.chapter_start(1.0)
        progress.stage("recap", "pages 1-4 of 9")
        started.set()
        release.wait(10)
        # After the stop flag is set, the next structural call raises.
        progress.chapter_done()
        return ["ch-1"]

    monkeypatch.setattr(recap_mod, "recap_series", blocking_recap_series)
    try:
        resp = client.post(
            "/api/runs", json={"work": "s1", "skip_preflight": True}
        )
        assert resp.status_code == 201
        run_id = resp.json()["id"]
        assert started.wait(5)

        stop = client.post(f"/api/runs/{run_id}/stop")
        assert stop.status_code == 200
        assert stop.json()["status"] == "running"  # unwinds asynchronously
        release.set()
    finally:
        release.set()

    run = _wait_for_status(client, run_id, want=("cancelled",))
    assert run["status"] == "cancelled"
    assert run["error"] is None
    assert run["current"] is None

    events = _parse_sse(client.get(f"/api/runs/{run_id}/events").text)
    assert events[-1]["event"] == "run-cancelled"
    assert events[-1]["chapters"] == 0  # chapter-done never fired

    # Idempotent: stopping a finished run returns it unchanged.
    again = client.post(f"/api/runs/{run_id}/stop")
    assert again.status_code == 200
    assert again.json()["status"] == "cancelled"


def test_run_stop_unknown_404(client):
    assert client.post("/api/runs/nope/stop").status_code == 404


def test_run_error_clears_current(client, monkeypatch):
    import entertainment_harness.pipelines.recap as recap_mod

    def failing_recap_series(conn, series, config, profile, **kw):
        kw.get("progress").start(1)
        raise RuntimeError("kaput")

    monkeypatch.setattr(recap_mod, "recap_series", failing_recap_series)
    resp = client.post(
        "/api/runs", json={"work": "s1", "skip_preflight": True}
    )
    run_id = resp.json()["id"]

    run = _wait_for_status(client, run_id)
    assert run["status"] == "error"
    assert run["current"] is None


def test_run_conflict_409(client, monkeypatch):
    import entertainment_harness.pipelines.recap as recap_mod

    started = threading.Event()
    release = threading.Event()

    def blocking_recap_series(conn, series, config, profile, **kw):
        started.set()
        release.wait(10)
        return []

    monkeypatch.setattr(recap_mod, "recap_series", blocking_recap_series)
    try:
        resp = client.post(
            "/api/runs", json={"work": "s1", "skip_preflight": True}
        )
        assert resp.status_code == 201
        run_id = resp.json()["id"]
        assert started.wait(5)

        resp = client.post(
            "/api/runs", json={"work": "s1", "skip_preflight": True}
        )
        assert resp.status_code == 409
    finally:
        release.set()
    assert _wait_for_status(client, run_id)["status"] == "done"


def test_run_error_status_and_event(client, monkeypatch):
    import entertainment_harness.pipelines.recap as recap_mod

    def failing_recap_series(conn, series, config, profile, **kw):
        raise RuntimeError("kaput")

    monkeypatch.setattr(recap_mod, "recap_series", failing_recap_series)
    resp = client.post(
        "/api/runs", json={"work": "s1", "skip_preflight": True}
    )
    assert resp.status_code == 201
    run_id = resp.json()["id"]

    run = _wait_for_status(client, run_id)
    assert run["status"] == "error"
    assert "kaput" in run["error"]

    events = _parse_sse(client.get(f"/api/runs/{run_id}/events").text)
    assert events[-1]["event"] == "run-error"
    assert "kaput" in events[-1]["message"]


def test_run_unknown_work_404(client):
    resp = client.post("/api/runs", json={"work": "nope"})
    assert resp.status_code == 404


def test_run_invalid_options_422(client):
    assert client.post(
        "/api/runs", json={"work": "s1", "detail": "weird"}
    ).status_code == 422
    assert client.post(
        "/api/runs", json={"work": "s1", "thinking": "galaxy"}
    ).status_code == 422
    assert client.post(
        "/api/runs", json={"work": "s1", "max_chapters": 0}
    ).status_code == 422
    assert client.post(
        "/api/runs", json={"work": "s1", "chapters": "3-1"}
    ).status_code == 422


def test_chapter_needs_work(data):
    """chapter_needs_work: a chapter is done only with a recap at the
    requested grain and (when video is wanted) a non-wiped video."""
    import entertainment_harness.pipelines.recap as recap_mod

    conn = db.connect()
    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()

    def row_for(num):
        # Re-select so artifact_detail reflects the latest recaps row.
        rows = recap_mod.select_chapters(
            conn, series, load_config(), detail="standard",
            chapter_num=None, chapters_spec=f"{num:g}", all_chapters=False,
            max_chapters=500, translated=False,
            verb="recap", log=lambda m: None,
        )
        return rows[0]

    def needs(num, want_video, detail="standard"):
        return recap_mod.chapter_needs_work(
            conn, "s1", row_for(num), detail=detail,
            want_video=want_video,
        )

    # Neither chapter has a recap yet.
    assert needs(1.0, False) is True
    assert needs(1.0, True) is True
    # Recap ch-1 at the requested grain.
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
        " VALUES ('ch-1', 'recap text', 'now', 'standard')"
    )
    conn.commit()
    assert needs(1.0, False) is False  # recapped, no video requested
    assert needs(1.0, True) is True    # recapped, but no video yet
    path = works.video_file_path("s1", "ch-1", "recap")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mp4")
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 1.0, 'now', 'recap')",
        (str(path),),
    )
    conn.commit()
    assert needs(1.0, True) is False   # recap + video: done
    # A different grain is still "not recapped" at the requested detail.
    assert needs(1.0, True, detail="full") is True
    # A wiped video needs work again.
    conn.execute("UPDATE videos SET wiped_at = 'now' WHERE series_id = 's1'")
    conn.commit()
    assert needs(1.0, True) is True
    conn.close()


def test_run_skip_done_passed_through(client, monkeypatch):
    """The toggle's skip_done flag reaches recap_series and the pre-flight
    selection filter (the desktop UI's 'Process unfinished' switch)."""
    import entertainment_harness.pipelines.recap as recap_mod

    seen: dict = {}

    def fake_recap_series(conn, series, config, profile, **kw):
        seen["skip_done"] = kw.get("skip_done")
        seen["chapters_spec"] = kw.get("chapters_spec")
        kw.get("progress").start(1)
        return ["ch-1"]

    monkeypatch.setattr(recap_mod, "recap_series", fake_recap_series)
    resp = client.post("/api/runs", json={
        "work": "s1", "chapters": "1-2", "skip_done": True,
        "skip_preflight": True,
    })
    assert resp.status_code == 201
    run = _wait_for_status(client, resp.json()["id"])
    assert run["status"] == "done"
    assert run["options"]["skip_done"] is True
    assert seen["skip_done"] is True
    assert seen["chapters_spec"] == "1-2"


def test_run_chapters_spec_selects_exact_range(client, monkeypatch):
    """`chapters: "2"` selects exactly ch-2 — not everything from ch 0 — and
    the spec reaches recap_series (the desktop UI's from–to range path)."""
    import entertainment_harness.pipelines.recap as recap_mod

    selected: list[float] = []
    real_select = recap_mod.select_chapters

    def spy_select(conn, series, config, **kw):
        rows = real_select(conn, series, config, **kw)
        selected.extend(r["chapter_num"] for r in rows)
        return rows

    seen: dict = {}

    def fake_recap_series(conn, series, config, profile, **kw):
        seen["chapters_spec"] = kw.get("chapters_spec")
        seen["all_chapters"] = kw.get("all_chapters")
        kw.get("progress").start(1)
        return ["ch-2"]

    monkeypatch.setattr(recap_mod, "select_chapters", spy_select)
    monkeypatch.setattr(recap_mod, "recap_series", fake_recap_series)

    resp = client.post("/api/runs", json={
        "work": "s1", "chapters": "2", "skip_preflight": True,
    })
    assert resp.status_code == 201
    run_id = resp.json()["id"]
    run = _wait_for_status(client, run_id)
    assert run["status"] == "done"
    assert run["options"]["chapters"] == "2"
    assert selected == [2.0]
    assert seen["chapters_spec"] == "2"
    assert seen["all_chapters"] is False


def test_run_fill_gaps_follows_video_flag(client, monkeypatch):
    """`video: true` runs pass fill_gaps to select_chapters and
    recap_series (the CLI's `--video` gap bucket), so chapters behind the
    read frontier with no artifact and no usable video get processed too;
    text-only runs leave the frontier alone."""
    import entertainment_harness.pipelines.recap as recap_mod

    seen: dict = {}

    def spy_select(conn, series, config, **kw):
        seen["select_fill_gaps"] = kw.get("fill_gaps")
        return []

    def fake_recap_series(conn, series, config, profile, **kw):
        seen["recap_fill_gaps"] = kw.get("fill_gaps")
        kw.get("progress").start(0)
        return []

    monkeypatch.setattr(recap_mod, "select_chapters", spy_select)
    monkeypatch.setattr(recap_mod, "recap_series", fake_recap_series)

    resp = client.post("/api/runs", json={
        "work": "s1", "video": True, "skip_preflight": True,
    })
    assert resp.status_code == 201
    assert _wait_for_status(client, resp.json()["id"])["status"] == "done"
    assert seen["select_fill_gaps"] is True
    assert seen["recap_fill_gaps"] is True

    resp = client.post("/api/runs", json={"work": "s1", "skip_preflight": True})
    assert resp.status_code == 201
    assert _wait_for_status(client, resp.json()["id"])["status"] == "done"
    assert seen["select_fill_gaps"] is False
    assert seen["recap_fill_gaps"] is False


def test_run_events_unknown_run_404(client):
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.get("/api/runs/nope").status_code == 404


# --- auto-process toggle -----------------------------------------------------


@pytest.fixture
def all_done(client):
    """Both s1 chapters recapped at 'standard' with non-wiped videos —
    nothing pending, nothing missing a video."""
    payload = b"mp4-bytes"
    for cid, num in (("ch-1", 1.0), ("ch-2", 2.0)):
        path = works.video_file_path("s1", cid, "recap")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        works.write_video_metadata("s1", cid, works.VideoMetadata(
            kind="recap", duration_s=9.0, model=None, tts_engine=None,
            created_at="now",
        ))
        conn = db.connect()
        conn.execute(
            "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
            " VALUES (?, 'recap text', 'now', 'standard')", (cid,),
        )
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES ('s1', ?, ?, ?, 9.0, 'now', 'recap')",
            (num, num, str(path)),
        )
        conn.commit()
        conn.close()
    return client


def test_auto_enable_starts_run_and_persists(client, stub_pipeline, data):
    """Enabling the toggle on a work with unfinished chapters starts a run
    immediately and persists the toggle + modifiers to auto.json."""
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "detail": "brief",
        "instruction": "auto marker", "skip_preflight": True,
        "video_mode": "scroll",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["work"] == "s1"
    assert body["enabled"] is True
    assert body["run"] is not None
    run_id = body["run"]["id"]
    # Supervisor runs are scope-less pending selection + video, with the
    # stored modifiers.
    assert body["run"]["options"]["video"] is True
    assert body["run"]["options"]["detail"] == "brief"
    assert body["run"]["options"]["instruction"] == "auto marker"
    assert body["run"]["options"]["skip_preflight"] is True
    assert body["run"]["options"]["video_mode"] == "scroll"
    assert body["run"]["options"]["chapter"] is None
    assert body["run"]["options"]["chapters"] is None
    assert body["run"]["options"]["all_chapters"] is False

    run = _wait_for_status(client, run_id)
    assert run["status"] == "done"

    # The library payload carries the toggle state, and the state file
    # holds exactly what was enabled with.
    work = client.get("/api/library").json()["works"][0]
    assert work["auto"] is True
    assert json.loads((data / "auto.json").read_text()) == {
        "s1": {
            "enabled": True,
            "options": {
                "detail": "brief",
                "instruction": "auto marker",
                "skip_preflight": True,
                "video_mode": "scroll",
                "chapters": None,
                "all_chapters": False,
            },
        }
    }

    # Disabling persists too and flips the library payload back.
    resp = client.put("/api/works/s1/auto", json={"enabled": False})
    assert resp.json() == {"work": "s1", "enabled": False, "run": None}
    work = client.get("/api/library").json()["works"][0]
    assert work["auto"] is False
    assert json.loads((data / "auto.json").read_text())["s1"]["enabled"] is False


def test_auto_chapters_scope_reaches_run_options(client, data, monkeypatch):
    """An optional chapters scope on the toggle is stored with it and
    reaches the run it starts; scoped runs are skip_done, so only the
    unfinished chapters inside the range are processed."""
    import entertainment_harness.pipelines.recap as recap_mod

    seen: dict = {}

    def fake_recap_series(conn, series, config, profile, **kw):
        seen["chapters_spec"] = kw.get("chapters_spec")
        seen["all_chapters"] = kw.get("all_chapters")
        seen["skip_done"] = kw.get("skip_done")
        kw.get("progress").start(0)
        return []

    monkeypatch.setattr(recap_mod, "recap_series", fake_recap_series)
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "chapters": "1-2", "skip_preflight": True,
    })
    assert resp.status_code == 200
    run_id = resp.json()["run"]["id"]
    run = _wait_for_status(client, run_id)
    assert run["options"]["chapters"] == "1-2"
    assert run["options"]["all_chapters"] is False
    assert run["options"]["skip_done"] is True
    assert seen["chapters_spec"] == "1-2"
    assert seen["all_chapters"] is False
    assert seen["skip_done"] is True
    assert json.loads((data / "auto.json").read_text())["s1"][
        "options"
    ]["chapters"] == "1-2"


def test_auto_all_chapters_reaches_run_options(client, data, monkeypatch):
    """The all-chapters scope widens the toggle's runs to every synced
    chapter (still skip_done — only unfinished ones are processed)."""
    import entertainment_harness.pipelines.recap as recap_mod

    seen: dict = {}

    def fake_recap_series(conn, series, config, profile, **kw):
        seen["all_chapters"] = kw.get("all_chapters")
        seen["chapters_spec"] = kw.get("chapters_spec")
        seen["skip_done"] = kw.get("skip_done")
        kw.get("progress").start(0)
        return []

    monkeypatch.setattr(recap_mod, "recap_series", fake_recap_series)
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "all_chapters": True, "skip_preflight": True,
    })
    assert resp.status_code == 200
    run_id = resp.json()["run"]["id"]
    run = _wait_for_status(client, run_id)
    assert run["options"]["all_chapters"] is True
    assert run["options"]["skip_done"] is True
    assert seen["all_chapters"] is True
    assert seen["chapters_spec"] is None
    assert seen["skip_done"] is True


def test_auto_scope_invalid_422(client):
    """A malformed chapters spec — or one combined with all_chapters — is
    rejected up front."""
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "chapters": "3-1",
    })
    assert resp.status_code == 422
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "chapters": "1-2", "all_chapters": True,
    })
    assert resp.status_code == 422


def test_auto_video_mode_invalid_422(client):
    """video_mode is validated like detail: an unknown style is rejected up
    front rather than failing mid-run."""
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "video_mode": "diagonal",
    })
    assert resp.status_code == 422


def test_auto_enable_nothing_unfinished(all_done, data):
    """A fully-done work has nothing to process: enabling persists the
    toggle but starts no run."""
    resp = all_done.put("/api/works/s1/auto", json={"enabled": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["run"] is None
    assert all_done.get("/api/runs").json()["runs"] == []
    work = all_done.get("/api/library").json()["works"][0]
    assert work["auto"] is True


def test_auto_no_resume_after_restart(client, data, stub_pipeline):
    """A work enabled before a restart does NOT auto-resume: loaded state is
    disarmed, so even an unfinished work stays untouched until the user
    flips the toggle on again (this is what stops a crash-looping run from
    re-arming itself on every launch)."""
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "skip_preflight": True,
    })
    run_id = resp.json()["run"]["id"]
    assert _wait_for_status(client, run_id)["status"] == "done"

    restarted = TestClient(create_app())
    restarted.app.state.auto.reconcile()  # no armed works → nothing starts
    assert restarted.get("/api/runs").json()["runs"] == []
    work = restarted.get("/api/library").json()["works"][0]
    assert work["auto"] is False

    resp = restarted.put("/api/works/s1/auto", json={
        "enabled": True, "skip_preflight": True,
    })
    assert resp.json()["run"] is not None  # re-toggling re-arms + starts


def test_auto_state_survives_restart(all_done, data):
    """auto.json is loaded at app creation, so the toggle and its modifiers
    survive a backend restart — but loaded state is disarmed: the library
    payload reads off and no run starts until the user flips it on again."""
    assert all_done.put("/api/works/s1/auto", json={"enabled": True}).json()[
        "enabled"
    ] is True
    restarted = TestClient(create_app())
    work = restarted.get("/api/library").json()["works"][0]
    assert work["auto"] is False
    assert restarted.get("/api/runs").json()["runs"] == []
    # Re-toggling on re-arms it and starts a run again (modifiers kept).
    restarted.app.state.auto.reconcile()
    assert restarted.get("/api/runs").json()["runs"] == []
    resp = restarted.put("/api/works/s1/auto", json={"enabled": True})
    assert resp.json() == {"work": "s1", "enabled": True, "run": None}


def test_auto_corrupt_state_tolerated(all_done, data):
    """A corrupt auto.json means empty state, not a crash."""
    (data / "auto.json").write_text("{not json")
    restarted = TestClient(create_app())
    work = restarted.get("/api/library").json()["works"][0]
    assert work["auto"] is False


def test_auto_disable_stops_active_run(client, monkeypatch):
    """Flipping the toggle off cooperatively stops the run it started."""
    import entertainment_harness.pipelines.recap as recap_mod

    started = threading.Event()
    release = threading.Event()

    def blocking_recap_series(conn, series, config, profile, **kw):
        kw.get("progress").start(1)
        kw.get("progress").chapter_start(1.0)
        started.set()
        release.wait(10)
        # After the stop flag is set, the next structural call raises.
        kw.get("progress").chapter_done()
        return ["ch-1"]

    monkeypatch.setattr(recap_mod, "recap_series", blocking_recap_series)
    try:
        resp = client.put("/api/works/s1/auto", json={
            "enabled": True, "skip_preflight": True,
        })
        run_id = resp.json()["run"]["id"]
        assert started.wait(5)

        resp = client.put("/api/works/s1/auto", json={"enabled": False})
        assert resp.json()["enabled"] is False
    finally:
        release.set()

    run = _wait_for_status(client, run_id, want=("cancelled",))
    assert run["status"] == "cancelled"
    work = client.get("/api/library").json()["works"][0]
    assert work["auto"] is False


def test_auto_error_disables_toggle(client, monkeypatch):
    """A supervisor-started run that errors turns the toggle off (via the
    one-shot recheck), so a broken config can't loop forever."""
    import entertainment_harness.pipelines.recap as recap_mod

    def failing_recap_series(conn, series, config, profile, **kw):
        raise RuntimeError("kaput")

    monkeypatch.setattr(recap_mod, "recap_series", failing_recap_series)
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "skip_preflight": True,
    })
    run_id = resp.json()["run"]["id"]
    assert _wait_for_status(client, run_id)["status"] == "error"

    # The one-shot recheck (not the 30 s poll) turns the toggle off.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        work = client.get("/api/library").json()["works"][0]
        if work["auto"] is False:
            break
        time.sleep(0.1)
    assert work["auto"] is False
    # Reconciling again does not start another run for the disabled work.
    client.app.state.auto.reconcile()
    assert client.get("/api/runs").json()["runs"][0]["status"] == "error"


def test_has_unfinished(data):
    """has_unfinished mirrors a scope-less run's work: pending recaps at
    the grain, or recapped chapters missing a playable video."""
    from entertainment_harness.server.auto import has_unfinished

    conn = db.connect()
    try:
        config = load_config()
        # No recaps at all → both chapters pending.
        assert has_unfinished(conn, "s1", config) is True
        # Recap both chapters at the grain → nothing pending, but the
        # missing videos make it unfinished.
        conn.executemany(
            "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
            " VALUES (?, 'recap text', 'now', 'standard')",
            [("ch-1",), ("ch-2",)],
        )
        conn.commit()
        assert has_unfinished(conn, "s1", config) is True
        # Videos for both → fully done.
        for cid, num in (("ch-1", 1.0), ("ch-2", 2.0)):
            path = works.video_file_path("s1", cid, "recap")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mp4")
            conn.execute(
                "INSERT INTO videos (series_id, from_chapter, to_chapter,"
                " path, duration_s, created_at, kind)"
                " VALUES ('s1', ?, ?, ?, 1.0, 'now', 'recap')",
                (num, num, str(path)),
            )
        conn.commit()
        assert has_unfinished(conn, "s1", config) is False
        # A wiped video needs work again.
        conn.execute("UPDATE videos SET wiped_at = 'now' WHERE series_id = 's1'")
        conn.commit()
        assert has_unfinished(conn, "s1", config) is True
    finally:
        conn.close()


def test_has_unfinished_gap_behind_frontier(data):
    """A chapter at/before the read frontier with no recap and no usable
    video is unfinished too (the `--video` gap bucket) — otherwise a
    library whose frontier outran its artifacts would stall the auto
    toggle with "nothing to do"."""
    from entertainment_harness.server.auto import has_unfinished

    conn = db.connect()
    try:
        config = load_config()
        conn.execute(
            "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
            " VALUES ('s1', 2.0, 'now')"
        )
        conn.commit()
        # Both chapters sit behind the frontier with no artifact and no
        # video: neither the plain pending set nor the video backfill
        # sees them.
        assert has_unfinished(conn, "s1", config) is True
        # A usable video drops ch-2 from the gap bucket; ch-1 remains.
        path = works.video_file_path("s1", "ch-2", "recap")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES ('s1', 2.0, 2.0, ?, 1.0, 'now', 'recap')",
            (str(path),),
        )
        conn.commit()
        assert has_unfinished(conn, "s1", config) is True
        # Usable videos for both → nothing a run could do.
        path = works.video_file_path("s1", "ch-1", "recap")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES ('s1', 1.0, 1.0, ?, 1.0, 'now', 'recap')",
            (str(path),),
        )
        conn.commit()
        assert has_unfinished(conn, "s1", config) is False
    finally:
        conn.close()


def test_has_unfinished_scoped(data):
    """With a scope stored on the toggle, "unfinished" is judged inside the
    scope only: done chapters outside it don't keep the toggle busy,
    unfinished ones inside it do (scoped runs are skip_done)."""
    from entertainment_harness.server.auto import has_unfinished

    conn = db.connect()
    try:
        config = load_config()
        scoped = {"options": {"chapters": "2"}}
        # Nothing done: ch-2 in scope is unfinished.
        assert has_unfinished(conn, "s1", config, scoped) is True
        # Finish ch-2 (recap at the grain + usable video): the in-scope
        # work is done even though ch-1 outside the scope has nothing.
        path = works.video_file_path("s1", "ch-2", "recap")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        conn.execute(
            "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
            " VALUES ('ch-2', 'recap text', 'now', 'standard')"
        )
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, kind)"
            " VALUES ('s1', 2.0, 2.0, ?, 1.0, 'now', 'recap')",
            (str(path),),
        )
        conn.commit()
        assert has_unfinished(conn, "s1", config, scoped) is False
        # "all" sees the unfinished ch-1.
        assert has_unfinished(
            conn, "s1", config, {"options": {"all_chapters": True}}
        ) is True
    finally:
        conn.close()


def test_auto_unknown_work_404(client):
    resp = client.put("/api/works/nope/auto", json={"enabled": True})
    assert resp.status_code == 404


def test_auto_invalid_detail_422(client):
    resp = client.put("/api/works/s1/auto", json={
        "enabled": True, "detail": "weird",
    })
    assert resp.status_code == 422


# --- character registry --------------------------------------------------------


def test_characters_empty_by_default(client):
    resp = client.get("/api/works/s1/characters")
    assert resp.status_code == 200
    assert resp.json() == {"characters": []}


def test_characters_put_roundtrip_marks_user_edited(client):
    resp = client.put("/api/works/s1/characters", json={
        "characters": [
            {"name": "Shin", "aliases": ["Shinu", " "], "role": "hunter"},
            {"name": "Merlin"},
        ],
    })
    assert resp.status_code == 200
    got = resp.json()["characters"]
    assert {c["name"] for c in got} == {"Shin", "Merlin"}
    shin = next(c for c in got if c["name"] == "Shin")
    assert shin["aliases"] == ["Shinu"]  # blanks stripped
    assert shin["role"] == "hunter"
    assert shin["origin"] == "user" and shin["edited"] is True
    # persists across connections
    got = client.get("/api/works/s1/characters").json()["characters"]
    assert {c["name"] for c in got} == {"Shin", "Merlin"}


def test_characters_put_replaces_registry(client):
    client.put("/api/works/s1/characters", json={
        "characters": [{"name": "Shin"}],
    })
    resp = client.put("/api/works/s1/characters", json={
        "characters": [{"name": "Maria", "role": "sage"}],
    })
    got = resp.json()["characters"]
    assert [c["name"] for c in got] == ["Maria"]


def test_characters_put_empty_name_422(client):
    resp = client.put("/api/works/s1/characters", json={
        "characters": [{"name": "  "}],
    })
    assert resp.status_code == 422


def test_characters_unknown_work_404(client):
    assert client.get("/api/works/nope/characters").status_code == 404
    resp = client.put("/api/works/nope/characters", json={"characters": []})
    assert resp.status_code == 404


def _seed_recap():
    conn = db.connect()
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, detail)"
        " VALUES ('ch-1', 'Shin hunts the boar.', 'now', 'standard')"
    )
    conn.commit()
    conn.close()


def _wait_for_build(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        builds = client.get("/api/cast-builds").json()["builds"]
        job = next((b for b in builds if b["id"] == job_id), None)
        if job is not None and job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"cast build {job_id} did not finish: {job}")


def test_cast_rebuild_no_recaps_422(client):
    resp = client.post("/api/works/s1/characters/rebuild")
    assert resp.status_code == 422


def test_cast_rebuild_unknown_work_404(client):
    resp = client.post("/api/works/nope/characters/rebuild")
    assert resp.status_code == 404


def test_cast_rebuild_roundtrip(client, data, monkeypatch):
    _seed_recap()
    from entertainment_harness.pipelines import characters as chars_mod

    def fake_rebuild(conn, series, config, profile, *, thinking="medium",
                     log=lambda m: None):
        log("Chapter 1...")
        db.upsert_character(
            conn, series["id"], "Shin", role="hunter", chapter_num=1.0
        )
        conn.commit()
        return 1

    monkeypatch.setattr(chars_mod, "rebuild_cast", fake_rebuild)
    resp = client.post("/api/works/s1/characters/rebuild")
    assert resp.status_code == 202
    job = _wait_for_build(client, resp.json()["id"])
    assert job["status"] == "done"
    assert job["count"] == 1
    assert job["log"] == ["Chapter 1..."]
    got = client.get("/api/works/s1/characters").json()["characters"]
    assert [c["name"] for c in got] == ["Shin"]


def test_cast_rebuild_conflict_409(client, data, monkeypatch):
    _seed_recap()
    import threading

    from entertainment_harness.pipelines import characters as chars_mod

    release = threading.Event()

    def slow_rebuild(conn, series, config, profile, *, thinking="medium",
                     log=lambda m: None):
        release.wait(10)
        return 0

    monkeypatch.setattr(chars_mod, "rebuild_cast", slow_rebuild)
    assert client.post("/api/works/s1/characters/rebuild").status_code == 202
    try:
        resp = client.post("/api/works/s1/characters/rebuild")
        assert resp.status_code == 409
    finally:
        release.set()


def test_cast_rebuild_error_surfaces(client, data, monkeypatch):
    _seed_recap()
    from entertainment_harness.pipelines import characters as chars_mod

    def boom(conn, series, config, profile, *, thinking="medium",
             log=lambda m: None):
        raise RuntimeError("ollama is not running")

    monkeypatch.setattr(chars_mod, "rebuild_cast", boom)
    resp = client.post("/api/works/s1/characters/rebuild")
    assert resp.status_code == 202
    job = _wait_for_build(client, resp.json()["id"])
    assert job["status"] == "error"
    assert "ollama is not running" in job["error"]


# --- eh serve CLI ------------------------------------------------------------


def test_serve_cli_listening_line(tmp_path):
    """`eh serve --port 0` prints 'listening <port>' as its first stdout
    line and serves /api/health on that port."""
    env = {
        **os.environ,
        "EH_DATA_DIR": str(tmp_path),
        "EH_NO_AUTO_MIGRATE": "1",
    }
    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            "from entertainment_harness.cli import app; app()",
            "serve", "--port", "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        ready, _, _ = select.select([proc.stdout], [], [], 60)
        assert ready, "server printed no listening line within 60s"
        line = proc.stdout.readline().strip()
        assert line.startswith("listening "), line
        port = int(line.split()[1])
        assert port > 0
        resp = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=10)
        assert resp.json() == {"ok": True}
    finally:
        proc.terminate()
        proc.wait(timeout=10)
