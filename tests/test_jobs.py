"""Job runner tests (jobs.py): spec parsing, cloud profile, event sink, and
run_job orchestration with the pipeline flow stubbed — no network, no models.
"""

from __future__ import annotations

import json

import pytest

from entertainment_harness import db, jobs
from entertainment_harness.config import load_config
from entertainment_harness.library import works


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("EH_CLOUD", raising=False)
    monkeypatch.delenv("EH_JOB_EVENTS", raising=False)
    for var in ("D1_ACCOUNT_ID", "D1_DATABASE_ID", "D1_API_TOKEN",
                "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# --- spec parsing ---------------------------------------------------------------


def test_spec_from_dict_minimal():
    spec = jobs.JobSpec.from_dict({"work": "s1"})
    assert spec.work == "s1"
    assert spec.kind == "recap"
    assert spec.options == {}
    assert spec.id


def test_spec_rejects_missing_work():
    with pytest.raises(jobs.JobSpecError, match='"work"'):
        jobs.JobSpec.from_dict({})


def test_spec_rejects_unknown_kind():
    with pytest.raises(jobs.JobSpecError, match="Unknown job kind"):
        jobs.JobSpec.from_dict({"work": "s1", "kind": "scanlation"})


def test_spec_rejects_work_inside_options():
    with pytest.raises(jobs.JobSpecError, match="top level"):
        jobs.JobSpec.from_dict({"work": "s1", "options": {"work": "s1"}})


def test_load_spec_inline_json():
    spec = jobs.load_spec('{"work": "s1", "options": {"chapter": 2}}')
    assert spec.options == {"chapter": 2}


def test_load_spec_from_file(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text('{"work": "s1", "id": "abc"}')
    spec = jobs.load_spec(str(path))
    assert spec.id == "abc"


def test_load_spec_invalid_json():
    with pytest.raises(jobs.JobSpecError, match="not valid JSON"):
        jobs.load_spec("{nope")


def test_run_options_rejects_unknown_keys():
    spec = jobs.JobSpec.from_dict({"work": "s1", "options": {"bogus": 1}})
    with pytest.raises(jobs.JobSpecError, match="Unknown job option"):
        jobs._run_options(spec)


def test_run_options_maps_recap_flags():
    spec = jobs.JobSpec.from_dict({
        "work": "s1",
        "options": {"chapter": 2, "video": True, "detail": "full"},
    })
    options = jobs._run_options(spec)
    assert options.work == "s1"
    assert options.chapter == 2
    assert options.video is True
    assert options.detail == "full"


# --- cloud profile ---------------------------------------------------------------


def test_cloud_profile_forces_remote_roles(root, monkeypatch):
    monkeypatch.setenv("EH_CLOUD", "1")
    config = load_config()
    assert config.roles["vision"].backend == "openai_compat"
    assert config.roles["vision"].model == "qwen/qwen3-vl-235b-a22b-instruct"
    assert config.roles["text"].backend == "openai_compat"
    assert config.store.provider == "r2"
    assert config.preflight.enabled is False


def test_cloud_profile_model_env_overrides(root, monkeypatch):
    monkeypatch.setenv("EH_CLOUD", "true")
    monkeypatch.setenv("EH_CLOUD_VISION_MODEL", "google/gemini-3-flash")
    monkeypatch.setenv("EH_CLOUD_TEXT_MODEL", "qwen/qwen3-8b")
    config = load_config()
    assert config.roles["vision"].model == "google/gemini-3-flash"
    assert config.roles["text"].model == "qwen/qwen3-8b"


def test_cloud_profile_off_by_default(root):
    config = load_config()
    assert "vision" not in config.roles
    assert config.store.provider == "hf"
    assert config.preflight.enabled is True


def test_cloud_profile_falls_back_to_openrouter_frames(root, monkeypatch):
    """No Runway key in the container → the default Runway-backed
    frame-animation animator swaps to the OpenRouter endpoint."""
    monkeypatch.setenv("EH_CLOUD", "1")
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    config = load_config()
    assert config.frame_animation.provider == "openrouter-frame-animation"


def test_cloud_profile_keeps_runway_frames_when_keyed(root, monkeypatch):
    monkeypatch.setenv("EH_CLOUD", "1")
    monkeypatch.setenv("RUNWAY_API_KEY", "rk")
    config = load_config()
    assert config.frame_animation.provider == "frame-animation"


def test_cloud_profile_respects_explicit_frame_provider(root, monkeypatch):
    monkeypatch.setenv("EH_CLOUD", "1")
    monkeypatch.delenv("RUNWAY_API_KEY", raising=False)
    (root / "config.toml").write_text(
        '[frame_animation]\nprovider = "acme-animator"\n'
    )
    config = load_config()
    assert config.frame_animation.provider == "acme-animator"


def test_cloud_profile_roles_win_over_legacy_aliases(root, monkeypatch):
    """[roles.*] bindings beat [models.*] aliases (models/roles.py
    precedence), so the profile works even when a config.toml pins local
    models."""
    (root / "config.toml").write_text(
        '[models.vision]\nmodel = "qwen3-vl:8b-instruct"\n'
    )
    monkeypatch.setenv("EH_CLOUD", "1")
    from entertainment_harness.models.roles import role_config

    config = load_config()
    assert role_config("vision", config).backend == "openai_compat"
    # translation/judge fall back to the text role.
    assert role_config("judge", config).backend == "openai_compat"


# --- event sink ------------------------------------------------------------------


def test_job_run_writes_jsonl_events(root, capsys):
    spec = jobs.JobSpec.from_dict({"work": "s1", "id": "j1"})
    run = jobs.JobRun(spec)
    run.emit("run-start", chapters=1)
    run.emit("chapter-done", chapter=1.0)
    run.finish(1)
    lines = (root / "jobs" / "j1" / "events.jsonl").read_text().splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["run-start", "chapter-done", "run-done"]
    assert run.status == "done"
    assert run.current is None
    # stdout carries the same JSONL feed.
    assert json.loads(capsys.readouterr().out.splitlines()[0])["event"] == "run-start"


def test_job_run_env_events_path(root, monkeypatch, tmp_path):
    target = tmp_path / "custom.jsonl"
    monkeypatch.setenv("EH_JOB_EVENTS", str(target))
    run = jobs.JobRun(jobs.JobSpec.from_dict({"work": "s1"}))
    run.emit("log", message="hello")
    assert json.loads(target.read_text().splitlines()[0])["message"] == "hello"


# --- run_job orchestration -------------------------------------------------------


def _stub_flow(monkeypatch, chapters=1, raise_exc=None):
    from entertainment_harness.server import runs as server_runs

    def fake_flow(run, options, progress):
        if raise_exc is not None:
            raise raise_exc
        run.emit("chapter-done", chapter=1.0)
        return chapters

    monkeypatch.setattr(server_runs, "_run_flow", fake_flow)


def test_run_job_happy_path(root, monkeypatch):
    _stub_flow(monkeypatch)
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j2"}))
    assert code == 0
    events = [
        json.loads(line)["event"]
        for line in (root / "jobs" / "j2" / "events.jsonl").read_text().splitlines()
    ]
    assert "run-done" in events
    assert events[0] == "log"  # "Store not configured..." heads-up


def test_run_job_error_exit_code(root, monkeypatch):
    _stub_flow(monkeypatch, raise_exc=RuntimeError("boom"))
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j3"}))
    assert code == 1
    events = [
        json.loads(line)
        for line in (root / "jobs" / "j3" / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "run-error"
    assert events[-1]["message"] == "boom"


def test_run_job_stages_around_the_flow(root, monkeypatch):
    """With a configured store, pull happens before the flow and push after."""
    _stub_flow(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "entertainment_harness.store.staging.pull_work_matching",
        lambda config, sid, keep, log=print: calls.append(("pull", sid))
        or (3, {}),
    )
    monkeypatch.setattr(
        "entertainment_harness.store.staging.push_work",
        lambda config, sid, log=print: calls.append(("push", sid)) or 5,
    )
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("EH_CLOUD", "1")
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j4"}))
    assert code == 0
    # An empty remote listing means the scoped second pull is skipped (the
    # fallback already pulled whole-work, or the store holds nothing).
    assert calls == [("pull", "s1"), ("push", "s1")]


# --- D1 wiring ---------------------------------------------------------------


def _stub_d1(monkeypatch, rows):
    calls = []
    monkeypatch.setenv("D1_ACCOUNT_ID", "acc")
    monkeypatch.setenv("D1_DATABASE_ID", "db")
    monkeypatch.setenv("D1_API_TOKEN", "tok")
    monkeypatch.setattr(
        "entertainment_harness.store.d1.snapshot_pull",
        lambda cfg, dest, baseline: calls.append(("pull",)) or rows,
    )
    monkeypatch.setattr(
        "entertainment_harness.store.d1.snapshot_flush",
        lambda cfg, local, baseline: calls.append(("flush",)) or {},
    )
    monkeypatch.setattr(
        "entertainment_harness.store.d1.ledger_update",
        lambda cfg, job_id, **kw: calls.append(("ledger", kw.get("status"))),
    )
    monkeypatch.setattr(
        "entertainment_harness.store.d1.ledger_stop_requested",
        lambda cfg, job_id: False,
    )
    monkeypatch.setattr(
        "entertainment_harness.library.index.index_works",
        lambda root=None: calls.append(("index",)) or 0,
    )
    return calls


def test_run_job_d1_snapshot_skips_index(root, monkeypatch):
    _stub_flow(monkeypatch)
    calls = _stub_d1(monkeypatch, rows=5)
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j5"}))
    assert code == 0
    # snapshot in, terminal flush + ledger status out; no filesystem rescan
    # over a populated snapshot (index_works clears every table).
    assert calls == [("pull",), ("flush",), ("ledger", "done")]


def test_run_job_d1_empty_snapshot_bootstraps_index(root, monkeypatch):
    _stub_flow(monkeypatch)
    calls = _stub_d1(monkeypatch, rows=0)
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j6"}))
    assert code == 0
    assert calls == [("pull",), ("index",), ("flush",), ("ledger", "done")]


def test_run_job_d1_snapshot_indexes_missing_work(root, monkeypatch):
    """A work first submitted after D1 already holds rows is absent from the
    snapshot: index the staged work additively so the flow can resolve it.
    index_works stays skipped (it clears every table)."""
    _stub_flow(monkeypatch)
    calls = _stub_d1(monkeypatch, rows=5)
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j9"}))
    assert code == 0
    assert ("index",) not in calls
    conn = db.connect()
    row = conn.execute(
        "SELECT title FROM series WHERE id = 's1'"
    ).fetchone()
    conn.close()
    assert row["title"] == "Test Manga"


def test_run_job_d1_snapshot_reindexes_known_work(root, monkeypatch):
    """A known work is re-indexed anyway: index_work upserts, so a partial
    snapshot — series row flushed by a job whose later ops failed, leaving
    no chapters behind (cloud job 8f6b05faaec1, 2026-10-19) — heals on the
    next run instead of finding zero chapters."""
    _stub_flow(monkeypatch)
    calls = _stub_d1(monkeypatch, rows=5)
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.commit()
    conn.close()
    works.write_work_metadata(works.WorkMetadata(
        id="s1", title="Test Manga", source="mangadex", source_id="s1",
        added_at="now",
    ))
    works.write_chapter_metadata("s1", works.ChapterMetadata(
        id="c1", chapter_num=1.0, title="Ch 1", lang="en", pages=9,
        published_at=None, fetched_at="now",
    ))
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j10"}))
    assert code == 0
    assert ("index",) not in calls
    conn = db.connect()
    row = conn.execute(
        "SELECT title FROM chapters WHERE id = 'c1'"
    ).fetchone()
    conn.close()
    assert row["title"] == "Ch 1"


# --- scoped stage-in ----------------------------------------------------------


def test_register_store_videos(root):
    """Rows are synthesized for store-held mp4s (never pulled), from the
    pulled video.json metadata; existing rows are left alone."""
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, ?, 'en', 9, 'now')",
        [("ch-1", 1.0, "Ch 1"), ("ch-2", 2.0, "Ch 2")],
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, kind, video_gen_provider)"
        " VALUES ('s1', 2.0, 2.0, '/x.mp4', 3.0, 'now', 'recap', 'local')"
    )
    conn.commit()
    conn.close()
    works.write_video_metadata("s1", "ch-1", works.VideoMetadata(
        kind="recap", duration_s=12.5, model="m", tts_engine="t",
        created_at="then",
    ))
    remote = {
        "works/s1/chapters/ch-1/video-recap/out.mp4": 9999,
        "works/s1/chapters/ch-1/video-recap/video.json": 100,
        "works/s1/chapters/ch-1/source/001.png": 100,
        "works/s1/chapters/ch-2/video-recap/out.mp4": 8888,  # row exists
        "works/s1/chapters/ch-2/video-narration/out.mp4": 7777,
        "works/s1/tiktok/out.mp4": 6666,  # not a chapter video: ignored
    }
    added = jobs._register_store_videos("s1", remote)
    assert added == 2  # ch-1 recap + ch-2 narration; ch-2 recap already known
    conn = db.connect()
    rows = conn.execute(
        "SELECT from_chapter, kind, duration_s, tts_engine FROM videos"
        " WHERE series_id = 's1' ORDER BY from_chapter, kind"
    ).fetchall()
    conn.close()
    assert [(r["from_chapter"], r["kind"]) for r in rows] == [
        (1.0, "recap"), (2.0, "narration"), (2.0, "recap"),
    ]
    assert rows[0]["duration_s"] == 12.5  # from the pulled video.json
    assert rows[0]["tts_engine"] == "t"
    # Idempotent.
    assert jobs._register_store_videos("s1", remote) == 0


def test_run_job_scopes_stage_in_to_selected_chapters(root, monkeypatch):
    """Stage-in pulls metadata first, then only the selected chapters'
    pages — mp4s and out-of-scope sources never download."""
    _stub_flow(monkeypatch)
    _stub_d1(monkeypatch, rows=5)
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("EH_CLOUD", "1")

    listing = {
        "works/s1/work.json": 10,
        "works/s1/chapters/ch-1/chapter.json": 10,
        "works/s1/chapters/ch-2/chapter.json": 10,
        "works/s1/chapters/ch-2/video-recap/video.json": 10,
        "works/s1/chapters/ch-1/source/001.png": 100,
        "works/s1/chapters/ch-2/source/001.png": 100,
        "works/s1/chapters/ch-2/video-recap/out.mp4": 9999,
    }
    contents = {
        "works/s1/work.json": {
            "id": "s1", "title": "Test Manga", "source": "mangadex",
            "source_id": "s1", "added_at": "now",
        },
        "works/s1/chapters/ch-1/chapter.json": {
            "id": "ch-1", "chapter_num": 1, "title": None, "lang": "en",
            "pages": 1, "published_at": None, "fetched_at": "now",
        },
        "works/s1/chapters/ch-2/chapter.json": {
            "id": "ch-2", "chapter_num": 2, "title": None, "lang": "en",
            "pages": 1, "published_at": None, "fetched_at": "now",
        },
        "works/s1/chapters/ch-2/video-recap/video.json": {
            "kind": "recap", "duration_s": 7.5, "model": "m",
            "tts_engine": "t", "created_at": "then",
        },
    }
    pulls: list[list[str]] = []

    def fake_pull(config, sid, keep, log=print):
        kept = sorted(
            k for k in listing if keep(k.removeprefix(f"works/{sid}/"))
        )
        pulls.append(kept)
        for key in kept:
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            if key in contents:
                dest.write_text(json.dumps(contents[key]))
            else:
                dest.write_bytes(b"page")
        works.refresh_paths()
        return len(kept), listing

    monkeypatch.setattr(
        "entertainment_harness.store.staging.pull_work_matching", fake_pull
    )
    monkeypatch.setattr(
        "entertainment_harness.store.staging.push_work",
        lambda config, sid, log=print: 0,
    )
    code = jobs.run_job(jobs.JobSpec.from_dict({
        "work": "s1", "id": "j11", "options": {"chapter": 1},
    }))
    assert code == 0
    # Phase 1: metadata only — json files, no pages, no mp4s.
    assert pulls[0] == sorted(k for k in listing if k.endswith(".json"))
    # Phase 2: just the selected chapter's pages.
    assert pulls[1] == ["works/s1/chapters/ch-1/source/001.png"]
    # The store-held ch-2 video was registered without pulling the mp4.
    conn = db.connect()
    row = conn.execute(
        "SELECT duration_s FROM videos WHERE series_id = 's1'"
        " AND from_chapter = 2.0 AND kind = 'recap'"
    ).fetchone()
    conn.close()
    assert row["duration_s"] == 7.5


def test_run_job_backfill_scope_includes_videoless_recapped(root, monkeypatch):
    """With video on, chapters that have a recap but no video anywhere
    (neither D1 nor the store) join the stage-in scope for the backfill."""
    _stub_flow(monkeypatch)
    _stub_d1(monkeypatch, rows=5)
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("EH_CLOUD", "1")

    listing = {
        "works/s1/work.json": 10,
        "works/s1/chapters/ch-1/chapter.json": 10,
        "works/s1/chapters/ch-1/recap.json": 10,
        "works/s1/chapters/ch-1/source/001.png": 100,
        "works/s1/chapters/ch-2/chapter.json": 10,
        "works/s1/chapters/ch-2/recap.json": 10,
        "works/s1/chapters/ch-2/source/001.png": 100,
        "works/s1/chapters/ch-2/video-recap/out.mp4": 9999,
    }
    contents = {
        "works/s1/work.json": {
            "id": "s1", "title": "Test Manga", "source": "mangadex",
            "source_id": "s1", "added_at": "now",
        },
        "works/s1/chapters/ch-1/chapter.json": {
            "id": "ch-1", "chapter_num": 1, "title": None, "lang": "en",
            "pages": 1, "published_at": None, "fetched_at": "now",
        },
        "works/s1/chapters/ch-1/recap.json": {
            "summary": "s", "model": "m", "created_at": "now",
        },
        "works/s1/chapters/ch-2/chapter.json": {
            "id": "ch-2", "chapter_num": 2, "title": None, "lang": "en",
            "pages": 1, "published_at": None, "fetched_at": "now",
        },
        "works/s1/chapters/ch-2/recap.json": {
            "summary": "s2", "model": "m", "created_at": "now",
        },
    }
    pulls: list[list[str]] = []

    def fake_pull(config, sid, keep, log=print):
        kept = sorted(
            k for k in listing if keep(k.removeprefix(f"works/{sid}/"))
        )
        pulls.append(kept)
        for key in kept:
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            if key in contents:
                dest.write_text(json.dumps(contents[key]))
            else:
                dest.write_bytes(b"page")
        works.refresh_paths()
        return len(kept), listing

    monkeypatch.setattr(
        "entertainment_harness.store.staging.pull_work_matching", fake_pull
    )
    monkeypatch.setattr(
        "entertainment_harness.store.staging.push_work",
        lambda config, sid, log=print: 0,
    )
    # --chapter 2 --video: ch-2 is selected; ch-1 (recapped, no video) is
    # the backfill set.
    code = jobs.run_job(jobs.JobSpec.from_dict({
        "work": "s1", "id": "j12", "options": {"chapter": 2, "video": True},
    }))
    assert code == 0
    assert pulls[1] == [
        "works/s1/chapters/ch-1/source/001.png",
        "works/s1/chapters/ch-2/source/001.png",
    ]


def test_run_job_early_stop_skips_flow(root, monkeypatch):
    calls = _stub_d1(monkeypatch, rows=0)
    monkeypatch.setattr(
        "entertainment_harness.store.d1.ledger_stop_requested",
        lambda cfg, job_id: True,
    )
    flow_ran = []
    from entertainment_harness.server import runs as server_runs

    monkeypatch.setattr(
        server_runs, "_run_flow", lambda *args: flow_ran.append(1)
    )
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j8"}))
    assert code == 0
    assert not flow_ran
    assert calls == [("pull",), ("flush",), ("ledger", "cancelled")]


def test_run_job_without_d1_env_skips_snapshot(root, monkeypatch):
    _stub_flow(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "entertainment_harness.store.d1.snapshot_pull",
        lambda *args: calls.append(1),
    )
    code = jobs.run_job(jobs.JobSpec.from_dict({"work": "s1", "id": "j7"}))
    assert code == 0
    assert not calls


def test_flush_progress_wrapper(root):
    calls = []
    run = jobs.JobRun(jobs.JobSpec.from_dict({"work": "s1"}))
    progress = jobs._FlushOnChapterDone(run, lambda: calls.append(1))
    progress.chapter_start(1.0)
    progress.chapter_done()
    assert calls == [1]
    assert run.current is None or run.current["event"] == "chapter-done"


# --- cloud sink ------------------------------------------------------------------


def _make_sink(root, monkeypatch, run):
    """A _CloudSink with stubbed D1 + R2 calls; returns (sink, captured)."""
    from entertainment_harness.config import load_config
    from entertainment_harness.store import d1

    captured = {"flush": 0, "uploads": [], "ledger": [], "stop": False}
    monkeypatch.setenv("EH_CLOUD", "1")
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setattr(
        d1, "snapshot_flush",
        lambda cfg, local, baseline: captured.__setitem__(
            "flush", captured["flush"] + 1
        ) or {"recaps": {"upserted": 2, "deleted": 0}},
    )
    monkeypatch.setattr(
        "entertainment_harness.store.r2.upload",
        lambda config, files, log=print: captured["uploads"].append(files),
    )
    monkeypatch.setattr(
        d1, "ledger_update",
        lambda cfg, job_id, **kw: captured["ledger"].append(kw),
    )
    monkeypatch.setattr(
        d1, "ledger_stop_requested",
        lambda cfg, job_id: captured["stop"],
    )
    cfg = d1.D1Config(account_id="a", database_id="d", token="t")
    sink = jobs._CloudSink(
        load_config(), run, cfg, root / "harness.db", root / "baseline.db"
    )
    return sink, captured


def test_cloud_sink_boundary_syncs_and_polls_stop(root, monkeypatch):
    run = jobs.JobRun(jobs.JobSpec.from_dict({"work": "s1", "id": "js1"}))
    run.emit("chapter-done", chapter=1.0)
    sink, captured = _make_sink(root, monkeypatch, run)
    sink.boundary()
    assert captured["flush"] == 1
    assert captured["uploads"] == [
        [(run.events_path, "jobs/js1/events.jsonl")]
    ]
    assert captured["ledger"] == [{"current": run.current}]
    assert not run.stop_requested.is_set()

    captured["stop"] = True
    sink.boundary()
    assert run.stop_requested.is_set()


def test_cloud_sink_terminal_writes_status(root, monkeypatch):
    run = jobs.JobRun(jobs.JobSpec.from_dict({"work": "s1", "id": "js2"}))
    run.fail("boom")
    sink, captured = _make_sink(root, monkeypatch, run)
    sink.terminal()
    assert captured["ledger"] == [{"status": "error", "error": "boom"}]
    assert captured["uploads"] == [
        [(run.events_path, "jobs/js2/events.jsonl")]
    ]


def test_cloud_sink_push_events_skipped_without_store(root, monkeypatch):
    from entertainment_harness.config import load_config
    from entertainment_harness.store import d1

    run = jobs.JobRun(jobs.JobSpec.from_dict({"work": "s1", "id": "js3"}))
    monkeypatch.setattr(
        d1, "ledger_update", lambda cfg, job_id, **kw: None
    )
    sink = jobs._CloudSink(
        load_config(), run,
        d1.D1Config(account_id="a", database_id="d", token="t"),
        root / "harness.db", root / "baseline.db",
    )
    sink.push_events()  # no store configured: a no-op, not an error
