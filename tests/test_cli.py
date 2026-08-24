"""CLI surface tests: validation and command registry. No models, no network."""

from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from entertainment_harness.cli import app

runner = CliRunner()


def test_max_chapters_rejects_zero():
    result = runner.invoke(app, ["recap", "anything", "--max-chapters", "0"])
    assert result.exit_code == 1
    assert "--max-chapters must be at least 1" in result.output


def test_thinking_rejects_bad_value():
    result = runner.invoke(app, ["recap", "anything", "--thinking", "bogus"])
    assert result.exit_code == 1
    assert "low, medium, high" in result.output


def test_video_and_translate_commands_removed():
    for cmd in ("video", "translate", "store"):
        result = runner.invoke(app, [cmd])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_recap_help_shows_thinking_and_video():
    result = runner.invoke(app, ["recap", "--help"])
    assert result.exit_code == 0
    for flag in ("--thinking", "--video", "--translated", "--detail",
                 "--instruction", "--max-chapters", "--colorize",
                 "--compress", "--video-mode", "--panel-first",
                 "--no-panel-first"):
        assert flag in result.output


def test_compress_rejects_bad_value():
    result = runner.invoke(app, ["recap", "anything", "--compress", "4k"])
    assert result.exit_code == 1
    assert "hd, balanced, small" in result.output


def test_video_mode_rejects_bad_value():
    result = runner.invoke(app, ["recap", "anything", "--video-mode", "diagonal"])
    assert result.exit_code == 1
    assert "kenburns, scroll" in result.output


def test_concat_rejects_bad_kind():
    result = runner.invoke(app, ["concat", "anything", "--kind", "bogus"])
    assert result.exit_code == 1
    assert "--kind must be" in result.output


def test_concat_unknown_series_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["concat", "ghost"])
    assert result.exit_code == 1
    assert "No series matches" in result.output


# --- eh wipe -----------------------------------------------------------------


def test_chapter_spec_grammar():
    from entertainment_harness import library

    ranges = library.parse_chapter_spec("1-3, 4, 6-10, 9.5")
    for hit in (1, 2.5, 3, 4, 7, 10, 9.5):
        assert library.chapter_in_spec(hit, ranges), hit
    for miss in (0, 3.5, 5, 5.5, 10.5):
        assert not library.chapter_in_spec(miss, ranges), miss
    for bad in ("", "a", "3-1", "1-", "-2"):
        try:
            library.parse_chapter_spec(bad)
        except library.LibraryError:
            pass
        else:
            raise AssertionError(f"{bad!r} should fail")


def _setup_wipe_series(tmp_path, monkeypatch):
    """Series s1 with chapters 1-3, each holding narration (+ch1 recap) videos."""
    from entertainment_harness import db
    from entertainment_harness.library import works

    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Wipe Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, NULL, 'en', 5, 'now')",
        [("c1", 1.0), ("c2", 2.0), ("c3", 3.0)],
    )
    conn.executemany(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " created_at, kind) VALUES ('s1', ?, ?, 'p', 'now', ?)",
        [(n, n, k) for n in (1.0, 2.0, 3.0) for k in ("narration",)]
        + [(1.0, 1.0, "recap")],
    )
    conn.commit()
    conn.close()
    works.write_work_metadata(
        works.WorkMetadata(id="s1", title="Wipe Manga", source="mangadex",
                           source_id="s1", added_at="now")
    )
    for cid, num in (("c1", 1.0), ("c2", 2.0), ("c3", 3.0)):
        works.write_chapter_metadata(
            "s1", works.ChapterMetadata(id=cid, chapter_num=num, title=None,
                                        lang="en", pages=5, published_at=None,
                                        fetched_at="now")
        )
        for kind in (("narration", "recap") if num == 1.0 else ("narration",)):
            wdir = works.chapter_dir("s1", cid) / f"video-{kind}"
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "out.mp4").write_bytes(b"x" * 1000)
            (wdir / "out-balanced.mp4").write_bytes(b"y" * 500)
            works.write_video_metadata(
                "s1", cid,
                works.VideoMetadata(kind=kind, duration_s=10.0, model="m",
                                    tts_engine="t", created_at="now",
                                    compress="balanced"),
            )


def test_wipe_deletes_files_and_marks(tmp_path, monkeypatch):
    _setup_wipe_series(tmp_path, monkeypatch)
    from entertainment_harness import db
    from entertainment_harness.library import works

    result = runner.invoke(app, ["wipe", "s1", "--chapters", "1-2"])
    assert result.exit_code == 0, result.output
    assert "Wiped 3 video(s)" in result.output  # narration ch1-2 + recap ch1
    for cid, num in (("c1", 1.0), ("c2", 2.0)):
        for kind in (("narration", "recap") if num == 1.0 else ("narration",)):
            wdir = works.chapter_dir("s1", cid) / f"video-{kind}"
            assert not list(wdir.glob("out*.mp4"))
            meta = works.read_video_metadata("s1", cid, kind)
            assert meta.wiped_at is not None
    # ch-3 untouched, no wiped mark
    wdir3 = works.chapter_dir("s1", "c3") / "video-narration"
    assert (wdir3 / "out.mp4").exists()
    assert works.read_video_metadata("s1", "c3", "narration").wiped_at is None
    conn = db.connect()
    rows = conn.execute(
        "SELECT from_chapter, kind, wiped_at FROM videos ORDER BY from_chapter"
    ).fetchall()
    conn.close()
    assert sorted(
        (r["from_chapter"], r["kind"]) for r in rows if r["wiped_at"]
    ) == [(1.0, "narration"), (1.0, "recap"), (2.0, "narration")]


def test_wipe_kind_limits_scope(tmp_path, monkeypatch):
    _setup_wipe_series(tmp_path, monkeypatch)
    from entertainment_harness.library import works

    result = runner.invoke(app, ["wipe", "s1", "--chapters", "1",
                                 "--kind", "narration"])
    assert result.exit_code == 0, result.output
    assert "Wiped 1 video(s)" in result.output
    recap_dir = works.chapter_dir("s1", "c1") / "video-recap"
    assert (recap_dir / "out.mp4").exists()


def test_wipe_rejects_bad_kind_and_spec(tmp_path, monkeypatch):
    _setup_wipe_series(tmp_path, monkeypatch)
    result = runner.invoke(app, ["wipe", "s1", "--chapters", "1", "--kind", "x"])
    assert result.exit_code == 1
    assert "--kind must be" in result.output
    result = runner.invoke(app, ["wipe", "s1", "--chapters", "3-1"])
    assert result.exit_code == 1
    assert "Reversed chapter range" in result.output


def test_play_reports_wiped(tmp_path, monkeypatch):
    _setup_wipe_series(tmp_path, monkeypatch)
    result = runner.invoke(app, ["wipe", "s1", "--chapters", "1"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["play", "s1", "--chapter", "1", "--narration"])
    assert result.exit_code == 1
    assert "wiped" in result.output


def test_backfill_includes_wiped_chapters(tmp_path, monkeypatch):
    """One video per chapter: a chapter counts as missing only when it has
    no videos row of EITHER kind, or all of its rows are wiped — wiped
    chapters are re-rendered by 'eh recap --video'."""
    _setup_wipe_series(tmp_path, monkeypatch)
    from entertainment_harness import db
    from entertainment_harness.cli import _chapters_missing_video

    conn = db.connect()
    conn.executemany(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES (?, 's', 'now', 'm')",
        [("c1",), ("c2",), ("c3",)],
    )
    conn.commit()
    missing = _chapters_missing_video(
        conn, "s1", table="recaps", translated=False, langs=["en"],
    )
    # Every chapter has a healthy (non-wiped) video row of some kind.
    assert missing == []

    result = runner.invoke(app, ["wipe", "s1", "--chapters", "2-3"])
    assert result.exit_code == 0
    missing = _chapters_missing_video(
        conn, "s1", table="recaps", translated=False, langs=["en"],
    )
    assert [r["id"] for r in missing] == ["c2", "c3"]

    # Wiping only c1's narration kind leaves its recap-kind row, which still
    # satisfies the chapter under the one-video-per-chapter rule.
    result = runner.invoke(app, ["wipe", "s1", "--chapters", "1",
                                 "--kind", "narration"])
    assert result.exit_code == 0
    missing = _chapters_missing_video(
        conn, "s1", table="recaps", translated=False, langs=["en"],
    )
    assert [r["id"] for r in missing] == ["c2", "c3"]
    conn.close()


def test_make_chapter_video_derives_kind_from_artifact_detail(tmp_path, monkeypatch):
    """The --video callback renders the kind implied by the artifact's detail
    (full -> narration) regardless of which command ran, and clears that
    kind's stale row/caches before the rebuild."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db
    from entertainment_harness.cli import _make_chapter_video
    from entertainment_harness.config import Config
    from entertainment_harness.library import works

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
        " VALUES ('ch-1', 's', 'now', 'm', 'full')"
    )
    # A stale narration-kind video from the previous artifact.
    stale_dir = works.video_dir_for_kind("s1", "ch-1", "narration")
    stale_dir.mkdir(parents=True)
    (stale_dir / "script.json").write_text("{}")
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " created_at, kind) VALUES ('s1', 1.0, 1.0, 'p', 'now', 'narration')"
    )
    conn.commit()
    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()

    calls: list[dict] = []

    def fake_build(conn, row, chapter_row, config, profile, **kwargs):
        calls.append(kwargs)
        return stale_dir / "out.mp4"

    monkeypatch.setattr(
        "entertainment_harness.video.pipeline.build_video", fake_build
    )

    class _UI:
        def log(self, msg):
            pass

    make_video = _make_chapter_video(
        conn, series, Config(), None, _UI(),
        voice=None, tts_engine=None, colorize=False, translated=False,
        compress=None, video_mode=None,
    )
    make_video("ch-1")

    # build_video is called without a source override — it derives the kind
    # from the artifact itself.
    assert len(calls) == 1
    assert "source" not in calls[0]
    # the stale same-kind row was deleted for the rebuild
    assert conn.execute("SELECT COUNT(*) AS n FROM videos").fetchone()["n"] == 0
    # the stale workdir was cleared
    assert not (stale_dir / "script.json").exists()
    conn.close()


def test_make_chapter_video_panel_first_forces_narration(tmp_path, monkeypatch):
    """With --panel-first, the callback targets the narration kind even when
    the artifact is a lower grain, and build_video receives the flag."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db
    from entertainment_harness.cli import _make_chapter_video
    from entertainment_harness.config import Config
    from entertainment_harness.library import works

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
        " VALUES ('ch-1', 's', 'now', 'm', 'standard')"
    )
    # A stale narration-kind video from a previous panel-first build.
    stale_dir = works.video_dir_for_kind("s1", "ch-1", "narration")
    stale_dir.mkdir(parents=True)
    (stale_dir / "script.json").write_text("{}")
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " created_at, kind) VALUES ('s1', 1.0, 1.0, 'p', 'now', 'narration')"
    )
    conn.commit()
    series = conn.execute("SELECT * FROM series WHERE id = 's1'").fetchone()

    calls: list[dict] = []

    def fake_build(conn, row, chapter_row, config, profile, **kwargs):
        calls.append(kwargs)
        return stale_dir / "out.mp4"

    monkeypatch.setattr(
        "entertainment_harness.video.pipeline.build_video", fake_build
    )

    class _UI:
        def log(self, msg):
            pass

    make_video = _make_chapter_video(
        conn, series, Config(), None, _UI(),
        voice=None, tts_engine=None, colorize=False, translated=False,
        compress=None, video_mode="scroll", panel_first=True,
    )
    make_video("ch-1")

    assert len(calls) == 1
    assert calls[0]["panel_first"] is True
    # the stale narration-kind row was deleted for the rebuild (kind forced
    # despite the standard-grain artifact)
    assert conn.execute("SELECT COUNT(*) AS n FROM videos").fetchone()["n"] == 0
    assert not (stale_dir / "script.json").exists()
    conn.close()


def test_recap_panel_first_default_and_overrides(tmp_path, monkeypatch):
    """Panel-first is the default video script path. The flag is tri-state:
    no flag resolves to [video] panel_first= (true by default — a false
    config must reach the video callback, regression for the dead config
    key); --panel-first overrides a false config; --no-panel-first opts out
    of the default."""
    _seed_series_only(tmp_path, monkeypatch)
    seen: list[bool] = []

    def fake_make(conn, row, config, profile, ui, **kwargs):
        seen.append(kwargs["panel_first"])

        def make_video(chapter_id):  # no chapters; must never fire
            raise AssertionError("callback fired")

        return make_video

    def fake_run(conn, row, config, profile, **kwargs):
        return []

    monkeypatch.setattr("entertainment_harness.cli._make_chapter_video",
                        fake_make)
    monkeypatch.setattr(
        "entertainment_harness.pipelines.recap.recap_series", fake_run
    )

    # 1. No config file, no flag: default on.
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 0, result.output
    # 2. [video] panel_first = false honored when no flag is passed.
    (tmp_path / "config.toml").write_text("[video]\npanel_first = false\n")
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 0, result.output
    # 3. Explicit --panel-first beats the false config.
    result = runner.invoke(app, ["recap", "s1", "--panel-first"])
    assert result.exit_code == 0, result.output
    # 4. --no-panel-first opts out of the default.
    (tmp_path / "config.toml").unlink()
    result = runner.invoke(app, ["recap", "s1", "--no-panel-first"])
    assert result.exit_code == 0, result.output
    assert seen == [True, False, True, False]

def test_sync_help_shows_force_and_lang():
    result = runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output
    assert "--lang" in result.output


def test_tiktok_help_shows_title_argument():
    result = runner.invoke(app, ["tiktok", "--help"])
    assert result.exit_code == 0
    for flag in ("--video-gen", "--steering-prompt"):
        assert flag in result.output
    assert "--online" not in result.output
    assert "{title}" in result.output


def test_online_summary_help_exists():
    result = runner.invoke(app, ["online-summary", "--help"])
    assert result.exit_code == 0
    assert "--steering-prompt" in result.output


def test_plugins_lists_registries(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))  # isolate from real config
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    for name in ("ollama", "mangadex", "kokoro", "local", "duckduckgo"):
        assert name in result.output


def _seed_playable_video(tmp_path):
    from entertainment_harness import db
    from entertainment_harness.library import works

    video_dir = works.video_recap_dir("s1", "ch-1")
    missing = video_dir / "out.mp4"
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, tts_engine, model, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 60.0, 'now', 'fake', 'm', 'recap')",
        (str(missing),),
    )
    conn.commit()
    conn.close()
    works.write_video_metadata(
        "s1", "ch-1", works.VideoMetadata(
            kind="recap", duration_s=60.0, model="m",
            tts_engine="fake", created_at="now",
        )
    )
    return missing


def test_play_pulls_missing_video_from_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    (tmp_path / "config.toml").write_text('[store]\nrepo = "me/archive"\n')
    missing = _seed_playable_video(tmp_path)

    from entertainment_harness import store

    def fake_pull(config, series_id, chapter_key):
        assert (series_id, chapter_key) == ("s1", "ch-1")
        missing.parent.mkdir(parents=True, exist_ok=True)
        missing.write_bytes(b"video")
        return [missing]

    monkeypatch.setattr(store, "pull_video", fake_pull)
    opened: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: opened.append(cmd))

    result = runner.invoke(app, ["play", "s1"])
    assert result.exit_code == 0
    assert opened and str(missing) in opened[0]
    assert "Restored from the store" in result.output


def test_play_missing_video_without_store_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))  # no store configured
    _seed_playable_video(tmp_path)
    result = runner.invoke(app, ["play", "s1"])
    assert result.exit_code == 1
    assert "Video file is missing" in result.output


def test_recap_help_shows_pipeline_flags():
    result = runner.invoke(app, ["recap", "--help"])
    assert result.exit_code == 0
    for flag in ("--thinking", "--video", "--translated",
                 "--max-chapters", "--colorize", "--compress", "--video-mode",
                 "--panel-first", "--no-panel-first"):
        assert flag in result.output


def test_recap_max_chapters_rejects_zero():
    result = runner.invoke(app, ["recap", "anything", "--max-chapters", "0"])
    assert result.exit_code == 1
    assert "--max-chapters must be at least 1" in result.output


def test_recap_thinking_rejects_bad_value():
    result = runner.invoke(app, ["recap", "anything", "--thinking", "bogus"])
    assert result.exit_code == 1
    assert "low, medium, high" in result.output


def test_recap_compress_rejects_bad_value():
    result = runner.invoke(app, ["recap", "anything", "--compress", "4k"])
    assert result.exit_code == 1
    assert "hd, balanced, small" in result.output


def test_recap_video_mode_rejects_bad_value():
    result = runner.invoke(
        app, ["recap", "anything", "--video-mode", "diagonal"]
    )
    assert result.exit_code == 1
    assert "kenburns, scroll" in result.output


def test_show_narration_prints_narrations(tmp_path, monkeypatch):
    """Legacy narrations rows fold into recaps on open, so 'eh show' prints
    them as detail='full' artifacts (with or without the deprecated flag)."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO narrations (chapter_id, text, created_at, model)"
        " VALUES ('ch-1', 'Full narration text.', 'now', 'm')"
    )
    conn.commit()
    conn.close()
    result = runner.invoke(app, ["show", "s1", "--narration"])
    assert result.exit_code == 0
    assert "Full narration text." in result.output
    assert "full" in result.output  # detail shown in the header
    assert "deprecated" in result.output  # --narration pointer


def test_play_narration_selects_narration_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db
    from entertainment_harness.library import works

    recap_path = works.video_recap_dir("s1", "ch-1") / "out.mp4"
    narration_path = works.video_narration_dir("s1", "ch-1") / "out.mp4"
    recap_path.parent.mkdir(parents=True)
    narration_path.parent.mkdir(parents=True)
    recap_path.write_bytes(b"video")
    narration_path.write_bytes(b"narration-video")
    works.write_video_metadata(
        "s1", "ch-1", works.VideoMetadata(
            kind="recap", duration_s=60.0, model="m",
            tts_engine="fake", created_at="now",
        )
    )
    works.write_video_metadata(
        "s1", "ch-1", works.VideoMetadata(
            kind="narration", duration_s=120.0, model="m",
            tts_engine="fake", created_at="now",
        )
    )
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, tts_engine, model, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 60.0, 'now', 'fake', 'm', 'recap')",
        (str(recap_path),),
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " duration_s, created_at, tts_engine, model, kind)"
        " VALUES ('s1', 1.0, 1.0, ?, 120.0, 'now', 'fake', 'm', 'narration')",
        (str(narration_path),),
    )
    conn.commit()
    conn.close()

    opened: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: opened.append(cmd))
    result = runner.invoke(app, ["play", "s1", "--narration"])
    assert result.exit_code == 0
    assert opened and str(narration_path) in opened[0]


def test_recap_help_shows_preflight_flags():
    result = runner.invoke(app, ["recap", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--skip-preflight" in result.output


def test_tiktok_help_shows_preflight_flags():
    result = runner.invoke(app, ["tiktok", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--skip-preflight" in result.output


def test_quantize_help_shows_preflight_flags():
    result = runner.invoke(app, ["quantize", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--skip-preflight" in result.output


def test_recap_dry_run_prints_plan_and_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import cli

    plan_printed = []

    def fake_plan(*args, **kwargs):
        from entertainment_harness.preflight import ResourcePlan

        return ResourcePlan(
            command="recap",
            models=[],
            peak_memory_bytes=None,
            min_free_disk_bytes=None,
            requires_gpu=False,
        )

    def fake_check(plan, snap, cfg):
        return []

    def fake_snapshot(*args, **kwargs):
        from entertainment_harness.hardware import SystemSnapshot

        return SystemSnapshot(
            gpu_backend="metal",
            total_ram_bytes=32_000_000_000,
            free_ram_bytes=16_000_000_000,
            budget_bytes=16_000_000_000,
            free_disk_bytes=50_000_000_000,
            free_vram_bytes=None,
        )

    monkeypatch.setattr(cli, "plan_for_recap", fake_plan)
    monkeypatch.setattr(cli, "check", fake_check)
    monkeypatch.setattr(cli, "snapshot", fake_snapshot)

    from entertainment_harness import db

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["recap", "s1", "--dry-run"])
    assert result.exit_code == 0
    assert "Pre-flight plan" in result.output


def test_recap_preflight_failure_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import cli

    def fake_plan(*args, **kwargs):
        from entertainment_harness.preflight import ResourcePlan

        return ResourcePlan(
            command="recap",
            models=[],
            peak_memory_bytes=None,
            min_free_disk_bytes=100_000_000_000,
            requires_gpu=False,
        )

    def fake_check(plan, snap, cfg):
        return ["Disk need 100.0 GB exceeds free space 50.0 GB."]

    def fake_snapshot(*args, **kwargs):
        from entertainment_harness.hardware import SystemSnapshot

        return SystemSnapshot(
            gpu_backend="metal",
            total_ram_bytes=32_000_000_000,
            free_ram_bytes=16_000_000_000,
            budget_bytes=16_000_000_000,
            free_disk_bytes=50_000_000_000,
            free_vram_bytes=None,
        )

    monkeypatch.setattr(cli, "plan_for_recap", fake_plan)
    monkeypatch.setattr(cli, "check", fake_check)
    monkeypatch.setattr(cli, "snapshot", fake_snapshot)

    from entertainment_harness import db

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 1
    assert "Pre-flight check failed" in result.output
    assert "--skip-preflight" in result.output


# --- detail grain CLI (unify-recap-narrate Phase 2) ---------------------------


def _seed_series_only(tmp_path, monkeypatch):
    """Series s1 with no chapters: pipeline commands reach run_series without
    touching models or preflight."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.commit()
    conn.close()


def test_recap_detail_rejects_bad_value(tmp_path, monkeypatch):
    _seed_series_only(tmp_path, monkeypatch)
    result = runner.invoke(app, ["recap", "s1", "--detail", "bogus"])
    assert result.exit_code == 1
    assert "gist, brief, standard, detailed, full" in result.output


def test_recap_detail_rejects_bad_config_value(tmp_path, monkeypatch):
    _seed_series_only(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text('[pipeline]\ndetail = "bogus"\n')
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 1
    assert "gist, brief, standard, detailed, full" in result.output


def test_recap_detail_config_default_and_flag_precedence(tmp_path, monkeypatch):
    """[pipeline] detail sets the grain; --detail wins over it."""
    _seed_series_only(tmp_path, monkeypatch)
    (tmp_path / "config.toml").write_text('[pipeline]\ndetail = "gist"\n')
    seen = []

    def fake_run(conn, row, config, profile, **kwargs):
        seen.append(kwargs["detail"])
        return []

    monkeypatch.setattr(
        "entertainment_harness.pipelines.recap.recap_series", fake_run
    )
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["recap", "s1", "--detail", "detailed"])
    assert result.exit_code == 0, result.output
    assert seen == ["gist", "detailed"]


def test_recap_instruction_config_default_and_flag_precedence(tmp_path, monkeypatch):
    """[pipeline] instructions steers the run; --instruction wins over it."""
    _seed_series_only(tmp_path, monkeypatch)
    seen = []

    def fake_run(conn, row, config, profile, **kwargs):
        seen.append(kwargs["instruction"])
        return []

    monkeypatch.setattr(
        "entertainment_harness.pipelines.recap.recap_series", fake_run
    )
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 0, result.output
    (tmp_path / "config.toml").write_text(
        '[pipeline]\ninstructions = "skip author notes"\n'
    )
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["recap", "s1", "--instruction", "Gen Z slang"])
    assert result.exit_code == 0, result.output
    assert seen == ["", "skip author notes", "Gen Z slang"]


def _capture_instruction(monkeypatch):
    """Patch recap_series with a fake recording the resolved instruction."""
    seen = []

    def fake_run(conn, row, config, profile, **kwargs):
        seen.append(kwargs["instruction"])
        return []

    monkeypatch.setattr(
        "entertainment_harness.pipelines.recap.recap_series", fake_run
    )
    return seen


def test_recap_instruction_at_file_loads_contents(tmp_path, monkeypatch):
    """--instruction @file reads the direction from the file (relative paths
    resolve from the cwd); inline values pass through unchanged."""
    _seed_series_only(tmp_path, monkeypatch)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "gen-z.txt").write_text("write in Gen Z slang\n")
    seen = _capture_instruction(monkeypatch)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["recap", "s1", "--instruction", "@prompts/gen-z.txt"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["recap", "s1", "-i", "plain inline"])
    assert result.exit_code == 0, result.output
    assert seen == ["write in Gen Z slang", "plain inline"]


def test_recap_instruction_config_at_file(tmp_path, monkeypatch):
    """[pipeline] instructions = "@file" resolves through the same loader."""
    _seed_series_only(tmp_path, monkeypatch)
    directions = tmp_path / "directions.txt"
    directions.write_text("skip the author notes\n")
    (tmp_path / "config.toml").write_text(
        f'[pipeline]\ninstructions = "@{directions}"\n'
    )
    seen = _capture_instruction(monkeypatch)
    result = runner.invoke(app, ["recap", "s1"])
    assert result.exit_code == 0, result.output
    assert seen == ["skip the author notes"]


def test_recap_instruction_at_file_missing_errors(tmp_path, monkeypatch):
    _seed_series_only(tmp_path, monkeypatch)
    result = runner.invoke(app, ["recap", "s1", "-i", "@no/such.txt"])
    assert result.exit_code == 1
    assert "Cannot read instruction file" in result.output
    assert "no/such.txt" in result.output


def test_recap_instruction_empty_file_means_no_instruction(tmp_path, monkeypatch):
    _seed_series_only(tmp_path, monkeypatch)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    seen = _capture_instruction(monkeypatch)
    result = runner.invoke(app, ["recap", "s1", "-i", f"@{empty}"])
    assert result.exit_code == 0, result.output
    assert seen == [""]


def _seed_show_series(tmp_path, monkeypatch):
    """Series s1 with two chapters holding artifacts at different grains."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db

    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, 'C', 'en', 3, 'now')",
        [("ch-1", 1.0), ("ch-2", 2.0)],
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
        " VALUES ('ch-1', 'A quick gist.', 'now', 'm', 'gist')"
    )
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
        " VALUES ('ch-2', 'The full retelling.', 'now', 'm', 'full')"
    )
    conn.commit()
    conn.close()


def test_show_prints_artifact_with_detail_header(tmp_path, monkeypatch):
    _seed_show_series(tmp_path, monkeypatch)
    result = runner.invoke(app, ["show", "s1"])
    assert result.exit_code == 0
    assert "A quick gist." in result.output
    assert "The full retelling." in result.output
    assert "gist" in result.output and "full" in result.output  # headers
    result = runner.invoke(app, ["show", "s1", "--chapter", "2"])
    assert result.exit_code == 0
    assert "The full retelling." in result.output
    assert "A quick gist." not in result.output


def test_show_header_shows_instruction(tmp_path, monkeypatch):
    """A steered artifact shows a truncated instruction in its header;
    unsteered chapters show nothing."""
    _seed_show_series(tmp_path, monkeypatch)
    from entertainment_harness import db

    instruction = (
        "write in Gen Z slang and keep it breezy — "
        "skip the cold-open recap pages entirely"
    )
    assert len(instruction) > 60  # exercises truncation
    conn = db.connect()
    conn.execute(
        "UPDATE recaps SET instruction = ? WHERE chapter_id = 'ch-2'",
        (instruction,),
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["show", "s1"])
    assert result.exit_code == 0, result.output
    assert "Gen Z slang" in result.output
    assert "entirely" not in result.output  # truncated at 60 chars
    chapter_lines = [ln for ln in result.output.splitlines() if "Chapter 1" in ln]
    assert chapter_lines and all("instruction" not in ln for ln in chapter_lines)


def test_list_shows_artifact_detail(tmp_path, monkeypatch):
    _seed_show_series(tmp_path, monkeypatch)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "detail" in result.output  # the new column
    assert "gist" in result.output and "full" in result.output


def _seed_kind_videos(tmp_path, monkeypatch, kinds=("recap", "narration")):
    """Series s1/ch-1 with a videos row + file + metadata per given kind."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    from entertainment_harness import db
    from entertainment_harness.library import works

    paths = {}
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'T', 'mangadex', 'x', 'ongoing', 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1', 's1', 1.0, 'C', 'en', 3, 'now')"
    )
    for kind in kinds:
        path = works.video_dir_for_kind("s1", "ch-1", kind) / "out.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{kind}-video".encode())
        works.write_video_metadata(
            "s1", "ch-1", works.VideoMetadata(
                kind=kind, duration_s=60.0, model="m",
                tts_engine="fake", created_at="now",
            )
        )
        conn.execute(
            "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
            " duration_s, created_at, tts_engine, model, kind)"
            " VALUES ('s1', 1.0, 1.0, ?, 60.0, 'now', 'fake', 'm', ?)",
            (str(path), kind),
        )
        paths[kind] = path
    conn.commit()
    conn.close()
    return paths


def test_play_finds_narration_video_without_flag(tmp_path, monkeypatch):
    """Unified lookup: a chapter with only a narration-kind video plays
    without --narration."""
    paths = _seed_kind_videos(tmp_path, monkeypatch, kinds=("narration",))
    opened: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: opened.append(cmd))
    result = runner.invoke(app, ["play", "s1"])
    assert result.exit_code == 0, result.output
    assert opened and str(paths["narration"]) in opened[0]


def test_play_prefers_narration_when_both_kinds(tmp_path, monkeypatch):
    """Best detail wins: with both kinds present, the narration video opens."""
    paths = _seed_kind_videos(tmp_path, monkeypatch)
    opened: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: opened.append(cmd))
    result = runner.invoke(app, ["play", "s1"])
    assert result.exit_code == 0, result.output
    assert opened and str(paths["narration"]) in opened[0]
    # ...while the deprecated flag still restricts to the narration kind
    result = runner.invoke(app, ["play", "s1", "--narration"])
    assert result.exit_code == 0, result.output
    assert str(paths["narration"]) in opened[-1]


def test_wipe_warns_when_video_has_no_artifact(tmp_path, monkeypatch):
    """A video whose chapter has no recap/narration on file cannot be
    re-rendered from cache — wiping it warns explicitly; chapters with an
    artifact don't warn."""
    _setup_wipe_series(tmp_path, monkeypatch)
    from entertainment_harness import db

    conn = db.connect()
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('c2', 's', 'now', 'm')"
    )
    conn.commit()
    conn.close()
    result = runner.invoke(app, ["wipe", "s1", "--chapters", "1-2"])
    assert result.exit_code == 0, result.output
    assert "Wiped 3 video(s)" in result.output
    assert "ch 1 narration: no recap or narration on file" in result.output
    assert "ch 2 narration: no recap or narration on file" not in result.output


def test_recap_video_help_mentions_gap_fill():
    result = runner.invoke(app, ["recap", "--help"])
    assert result.exit_code == 0
    assert "gap-filled" in result.output


def test_recap_chapters_rejects_combinations(tmp_path, monkeypatch):
    _seed_series_only(tmp_path, monkeypatch)
    result = runner.invoke(app, ["recap", "s1", "--chapter", "1",
                                 "--chapters", "1-2"])
    assert result.exit_code == 1
    assert "cannot be combined" in result.output
    result = runner.invoke(app, ["recap", "s1", "--all", "--chapters", "1-2"])
    assert result.exit_code == 1
    assert "cannot be combined" in result.output


def test_recap_chapters_rejects_bad_spec(tmp_path, monkeypatch):
    _seed_series_only(tmp_path, monkeypatch)
    result = runner.invoke(app, ["recap", "s1", "--chapters", "3-1"])
    assert result.exit_code == 1
    assert "Reversed chapter range" in result.output


def test_recap_chapters_spec_reaches_pipeline_as_string(tmp_path, monkeypatch):
    """Regression: _run_chapter_pipeline used to shadow the `chapters` spec
    string with the selected chapter rows, then pass that list to
    recap_series — `eh recap --chapters 1` crashed in parse_chapter_spec."""
    _seed_series_only(tmp_path, monkeypatch)
    seen: dict = {}

    def fake_run(conn, row, config, profile, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(
        "entertainment_harness.pipelines.recap.recap_series", fake_run
    )
    result = runner.invoke(app, ["recap", "s1", "--chapters", "1-2"])
    assert result.exit_code == 0, result.output
    assert seen["chapters_spec"] == "1-2"


def test_recap_help_lists_chapters():
    result = runner.invoke(app, ["recap", "--help"])
    assert result.exit_code == 0
    assert "--chapters" in result.output
