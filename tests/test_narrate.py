"""Full-detail (narration grain) pipeline tests with a fake in-memory
ModelAdapter.

Mirrors test_recap.py at detail='full': narrations land in the recaps table
with detail='full' (the legacy narrations table is no longer written), and
the rolling context / progress follow the grain-aware two-bucket rule. No
network or Ollama involved.
"""

from __future__ import annotations

import pytest

from entertainment_harness import db
from entertainment_harness.pipelines import recap as recap_mod
from entertainment_harness.config import Config
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.models.base import ModelInfo
from entertainment_harness.models.registry import Selection

from conftest import GB, FakeAdapter, ScriptedJudge


def narrate_series(conn, series, config=None, profile=None, **kwargs):
    """The full-grain recap run these tests exercise (old narrate semantics)."""
    return recap_mod.recap_series(
        conn, series, config or Config(), profile, detail="full", **kwargs
    )


@pytest.fixture
def batch_label() -> str:
    return "narration"


# --- pending selection ---------------------------------------------------------


def test_pending_chapters_full_detail(harness):
    conn, vision, text, client, series, profile = harness
    narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None
    )
    rows = recap_mod.pending_chapters(
        conn, "s1", detail="full"
    )
    assert [r["id"] for r in rows] == ["ch-2"]  # ch-1 already narrated
    with pytest.raises(ValueError, match="unknown detail"):
        recap_mod.pending_chapters(conn, "s1", detail="bogus")


def test_narration_pending_ignores_read_progress(harness):
    """Chapters with a lower-grain artifact stay pending at detail='full'
    even when progress is past them: bucket b (upgrades) is ungated on
    last_read. Chapters with no artifact at all are NOT pending before
    last_read (bucket a is gated)."""
    conn, vision, text, client, series, profile = harness
    conn.executemany(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES (?, 's', 'now', 'm')",
        [("ch-1",), ("ch-2",)],
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-0', 's1', 0.5, 'Ch 0', 'en', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 99.0, 'now')"
    )
    conn.commit()
    # nothing pending at 'standard' (both chapters already have that grain)
    assert recap_mod.pending_chapters(conn, "s1", detail="standard") == []
    rows = recap_mod.pending_chapters(conn, "s1", detail="full")
    assert [r["id"] for r in rows] == ["ch-1", "ch-2"]  # ch-0: no artifact, gated


def test_narrated_chapters_skipped_on_rerun(harness):
    conn, vision, text, client, series, profile = harness
    narrate_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    done = narrate_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    assert done == []


# --- end to end ------------------------------------------------------------------


def test_narrate_all_pending_by_default(harness):
    conn, vision, text, client, series, profile = harness
    done = narrate_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    assert done == ["ch-1", "ch-2"]
    # the wrapper stores full-detail recaps rows, never narrations rows
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM recaps WHERE detail = 'full'"
    ).fetchone()["n"] == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM narrations").fetchone()["n"] == 0


def test_max_chapters_caps_default_run(harness):
    conn, vision, text, client, series, profile = harness
    done = narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]  # ch-2 still unfinished
    assert conn.execute("SELECT COUNT(*) AS n FROM recaps").fetchone()["n"] == 1


def test_narration_stored_with_model_tag_and_callback(harness):
    conn, vision, text, client, series, profile = harness
    events = []
    done = narrate_series(
        conn, series, Config(), profile, all_chapters=True, client=client,
        on_recap=events.append, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert events == ["ch-1", "ch-2"]  # fired once per committed narration
    row = conn.execute(
        "SELECT r.* FROM recaps r JOIN chapters c ON r.chapter_id = c.id"
        " WHERE c.chapter_num = 1"
    ).fetchone()
    assert row["detail"] == "full"
    assert row["model"] == "fake-vision:8b (Q8_0)"
    assert row["summary"]
    assert vision.ensured == ["fake-vision:8b"]
    assert text.ensured == ["fake-text:4b"]


# --- context / progress policy -----------------------------------------------------


def test_narration_with_existing_recap_skips_context_and_progress(harness):
    """A chapter that was already recapped had its events folded into the
    story-so-far: narrating it (a bucket b upgrade to detail='full') must not
    rewrite the context nor touch reading progress."""
    conn, vision, text, client, series, profile = harness
    ctx = "Shin trains under Merlin and learns magic. " * 10
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model)"
        " VALUES ('ch-1', 'Shin hunts.', 'now', 'm')"
    )
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES ('s1', ?, 1.0)",
        (ctx,),
    )
    conn.commit()

    done = narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None
    )
    assert done == ["ch-1"]
    row = conn.execute(
        "SELECT summary, detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()
    assert row["detail"] == "full"
    assert row["summary"] != "Shin hunts."  # upgraded, not skipped
    row = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    assert (row["rolling_summary"], row["through_chapter"]) == (ctx, 1.0)
    # no context generation was attempted at all
    assert not [c for c in text.calls if "story so far" in c["prompt"].lower()]
    # progress is never touched for bucket b upgrades
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone() is None


def test_narration_without_recap_runs_full_context_loop(harness):
    conn, vision, text, client, series, profile = harness
    narrate_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    ctx = conn.execute(
        "SELECT * FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0
    assert ctx["rolling_summary"]
    assert any("story so far" in c["prompt"].lower() for c in text.calls)
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_forced_renarration_preserves_context_and_progress(harness):
    """--chapter re-narrates one chapter (overwriting the stored narration)
    without rewinding the rolling context or reading progress."""
    conn, vision, text, client, series, profile = harness
    narrate_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    calls_before = len(vision.calls)

    done = narrate_series(
        conn, series, Config(), profile, chapter_num=1.0,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    assert len(vision.calls) > calls_before  # pages actually re-read
    # overwritten in place, not duplicated
    assert conn.execute("SELECT COUNT(*) AS n FROM recaps").fetchone()["n"] == 2
    ctx = conn.execute(
        "SELECT through_chapter FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0  # not rewound to 1.0
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_all_renarrates_every_synced_chapter(harness):
    conn, vision, text, client, series, profile = harness
    narrate_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    calls_after_first = len(vision.calls)

    done = narrate_series(
        conn, series, Config(), profile, all_chapters=True,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]  # both re-narrated, in chapter order
    assert len(vision.calls) > calls_after_first
    # re-narrations leave rolling context and progress alone
    ctx = conn.execute(
        "SELECT through_chapter FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


# --- judge / thinking levels ---------------------------------------------------


FAIL_KAZUMA = '{"pass": false, "issues": ["invented the name Kazuma"]}'


def test_thinking_low_skips_judge(harness, monkeypatch):
    conn, vision, text, client, series, profile = harness

    def no_judge(config, profile):
        raise AssertionError("judge must not be resolved with thinking='low'")

    monkeypatch.setattr(recap_mod, "get_judge_model", no_judge)
    done = narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, thinking="low",
        log=lambda m: None,
    )
    assert done == ["ch-1"]


def test_judge_rejection_retries_with_feedback(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    patch_judge(ScriptedJudge(verdicts=[FAIL_KAZUMA]))
    done = narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None
    )
    assert done == ["ch-1"]
    combines = [c for c in vision.calls if not c["images"]]
    assert len(combines) == 2  # initial combine + one retry
    assert "Kazuma" in combines[1]["prompt"]  # issues fed back
    assert "rejected by the evaluator" in combines[1]["prompt"]
    assert conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["summary"]


def test_judge_persistent_narration_failure_keeps_first(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    patch_judge(
        ScriptedJudge(verdicts=[FAIL_KAZUMA] * recap_mod.MAX_ATTEMPTS),
    )
    done = narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None
    )
    assert done == ["ch-1"]  # stored anyway, with a warning
    combines = [c for c in vision.calls if not c["images"]]
    assert len(combines) == recap_mod.MAX_ATTEMPTS  # bounded loop
    # on persistent failure the FIRST attempt is stored, not the last
    stored = conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["summary"]
    assert stored == "[text output 4]"  # 3 batch calls, then the first combine


def test_thinking_high_vision_verifies_narration(harness, monkeypatch, patch_judge):
    """On high, the narration is judged by the VISION adapter with page
    images, and the attempt budget is 5."""
    conn, vision, text, client, series, profile = harness

    class ScriptedVision(FakeAdapter):
        def __init__(self, verdicts):
            super().__init__("fake-vision:8b")
            self.verdicts = list(verdicts)

        def generate(self, model, prompt, images=None):
            self.calls.append(
                {"model": model, "prompt": prompt, "images": images or []}
            )
            if "strict evaluator" in prompt:  # judge prompt
                return (
                    self.verdicts.pop(0) if self.verdicts else '{"pass": true}'
                )
            if images:
                return f"[batch narration of {len(images)} pages]"
            return f"[text output {len(self.calls)}]"

    vision = ScriptedVision([FAIL_KAZUMA] * recap_mod.ATTEMPTS["high"])
    info_v = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    monkeypatch.setattr(
        recap_mod, "get_vision_model",
        lambda config, profile: Selection(adapter=vision, info=info_v),
    )
    patch_judge(ScriptedJudge())  # context judging: pass
    done = narrate_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, thinking="high",
        log=lambda m: None,
    )
    assert done == ["ch-1"]
    judge_calls = [c for c in vision.calls if "strict evaluator" in c["prompt"]]
    assert len(judge_calls) == recap_mod.ATTEMPTS["high"]  # bounded at 5
    assert all(c["images"] for c in judge_calls)  # pages attached
    # first attempt kept after persistent failure
    stored = conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["summary"]
    assert stored == "[text output 4]"


# --- book path (imported text; no vision model) -------------------------------


@pytest.fixture
def book_harness(tmp_path, monkeypatch, patch_judge):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at, kind)"
        " VALUES ('b1', 'Test Book', 'import', 'test-book', 'imported', 'now',"
        " 'book')"
    )
    conn.executemany(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 'b1', ?, ?, 'en', NULL, 'now')",
        [("b1-0001", 1.0, "Part 1"), ("b1-0002", 2.0, "Part 2")],
    )
    conn.commit()
    book_dir = tmp_path / "books" / "b1"
    book_dir.mkdir(parents=True)
    (book_dir / "ch-001.txt").write_text(("alpha " * 25 + "\n\n") * 8)
    (book_dir / "ch-002.txt").write_text(("omega " * 25 + "\n\n") * 8)

    text = FakeAdapter("fake-text:4b")

    def no_vision(config, profile):
        raise AssertionError("book narrations must not resolve a vision model")

    monkeypatch.setattr(recap_mod, "get_vision_model", no_vision)
    info_t = ModelInfo("fake-text:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        recap_mod, "get_text_model",
        lambda config, profile: Selection(adapter=text, info=info_t),
    )
    patch_judge(ScriptedJudge())
    profile = HardwareProfile("fake chip", 24 * GB, 16 * GB, "cpu")
    series = conn.execute("SELECT * FROM series WHERE id = 'b1'").fetchone()
    yield conn, text, series, profile
    conn.close()


def test_book_narration_uses_text_model_only(book_harness):
    conn, text, series, profile = book_harness
    done = narrate_series(
        conn, series, Config(), profile, log=lambda m: None
    )
    assert done == ["b1-0001", "b1-0002"]
    rows = conn.execute(
        "SELECT summary, model, detail FROM recaps ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["model"] == "fake-text:4b (Q8_0)" for r in rows)
    assert all(r["detail"] == "full" for r in rows)
    # single 200-word chunk per part -> no combine call; context update only
    # (cast-extraction calls — pipelines/characters.py — are filtered out)
    part_narrations = [c for c in text.calls if "reading excerpt" in c["prompt"]]
    context_updates = [
        c for c in text.calls if 'updated "story so far"' in c["prompt"]
    ]
    assert len(part_narrations) == 2 and len(context_updates) == 2
    ctx = conn.execute(
        "SELECT * FROM series_context WHERE series_id = 'b1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0


# --- translated narration path (translate first, narrate the overlays) --------


@pytest.fixture
def pt_chapter(harness):
    """Add an unread pt-br chapter (ch 3) to the standard harness, with
    ch-1/ch-2 already narrated (full-detail recaps rows) so ch-3 is the next
    pending narration."""
    conn, *_ = harness
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-3', 's1', 3.0, 'Ch 3', 'pt-br', 2, 'now')"
    )
    conn.executemany(
        "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
        " VALUES (?, 'n', 'now', 'm', 'full')",
        [("ch-1",), ("ch-2",)],
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 2.0, 'now')"
    )
    conn.commit()
    return harness


def test_narrate_translated_translates_first(pt_chapter, monkeypatch):
    conn, vision, text, client, series, profile = pt_chapter
    translated_calls = []

    def fake_translate(conn, series, config, profile, chapter_num=None, **kwargs):
        translated_calls.append(chapter_num)
        from entertainment_harness.config import data_dir

        dest = data_dir() / "manga" / "s1" / "ch-3" / "translated"
        dest.mkdir(parents=True, exist_ok=True)
        for i in range(1, 3):
            (dest / f"page-{i:03d}.png").write_bytes(b"")

    monkeypatch.setattr(
        "entertainment_harness.pipelines.translate.translate_chapters", fake_translate
    )
    done = narrate_series(
        conn, series, Config(), profile, translated=True, log=lambda m: None
    )
    assert done == ["ch-3"]
    assert translated_calls == [3.0]  # translation ran before narrating
    image_calls = [c for c in vision.calls if c["images"]]
    assert all(
        "translated" in p.parts for c in image_calls for p in c["images"]
    )
    assert conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-3'"
        " AND detail = 'full'"
    ).fetchone()["summary"]
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 3.0


# --- gap fill (narrate --video) ---------------------------------------------------


def test_fill_gaps_narrates_standalone(harness):
    """The narrate alias honors fill_gaps too: full-grain standalone
    artifacts (story-so-far withheld), context untouched, no progress
    advance."""
    conn, vision, text, client, series, profile = harness
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 2.0, 'now')"
    )
    # progress never exists without a folded context row in practice; seed
    # one so the standalone marker (withheld story-so-far) applies
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary,"
        " through_chapter) VALUES ('s1', 'story through 2', 2.0)"
    )
    conn.commit()
    done = narrate_series(
        conn, series, Config(), profile, fill_gaps=True,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    rows = conn.execute(
        "SELECT detail, standalone FROM recaps ORDER BY chapter_id"
    ).fetchall()
    assert [(r["detail"], r["standalone"]) for r in rows] == [
        ("full", 1), ("full", 1),
    ]
    ctx = conn.execute(
        "SELECT rolling_summary FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["rolling_summary"] == "story through 2"
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0
