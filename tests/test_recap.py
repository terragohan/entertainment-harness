"""Recap pipeline tests with a fake in-memory ModelAdapter.

Verifies batching, strict prompt rules, rolling-context update, progress
advance, and model attribution. No network or Ollama involved.
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


def test_recaps_all_unfinished_by_default(harness):
    conn, vision, text, client, series, profile = harness
    done = recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    assert done == ["ch-1", "ch-2"]
    # every unfinished chapter recapped
    assert conn.execute("SELECT COUNT(*) AS n FROM recaps").fetchone()["n"] == 2


def test_on_recap_fires_per_chapter_interleaved(harness):
    conn, vision, text, client, series, profile = harness
    events = []

    def on_recap(chapter_id: str) -> None:
        # when the callback fires, only chapters up to this one are recapped
        recapped = {
            r["chapter_id"]
            for r in conn.execute("SELECT chapter_id FROM recaps")
        }
        events.append((chapter_id, sorted(recapped)))

    done = recap_mod.recap_series(
        conn, series, Config(), profile, all_chapters=True, client=client,
        on_recap=on_recap, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert events == [
        ("ch-1", ["ch-1"]),            # video can render before ch-2 starts
        ("ch-2", ["ch-1", "ch-2"]),
    ]


def test_skip_done_video_only_chapters_are_not_renarrated(harness):
    """A chapter whose artifact is already at the requested grain survives
    the skip_done filter only when its video is missing; it must get its
    on_recap callback WITHOUT regenerating the artifact (a failed video
    must never cost a re-narration)."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    calls_after_first = len(vision.calls)
    before = {
        r["chapter_id"]: r["summary"]
        for r in conn.execute("SELECT chapter_id, summary FROM recaps")
    }
    fired: list[str] = []
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1-2", skip_done=True,
        client=client, on_recap=fired.append, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert fired == ["ch-1", "ch-2"]          # videos rendered, in order
    assert len(vision.calls) == calls_after_first  # no pages re-read
    after = {
        r["chapter_id"]: r["summary"]
        for r in conn.execute("SELECT chapter_id, summary FROM recaps")
    }
    assert after == before                     # artifacts untouched


def test_skip_done_video_only_without_skip_done_still_forces(harness):
    """The same scoped run WITHOUT skip_done keeps the explicit-overwrite
    semantics: artifacts at the grain are regenerated."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    calls_after_first = len(vision.calls)
    fired: list[str] = []
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1-2",
        client=client, on_recap=fired.append, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert fired == ["ch-1", "ch-2"]
    assert len(vision.calls) > calls_after_first  # pages actually re-read


def test_skip_done_fully_done_chapters_fire_no_callback(harness):
    """Artifact at the grain + a playable video row = nothing to do: the
    chapter is dropped from the selection entirely."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    conn.execute(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " created_at) VALUES ('s1', 1.0, 1.0, 'ch1.mp4', 'now')"
    )
    conn.commit()
    fired: list[str] = []
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1-2", skip_done=True,
        client=client, on_recap=fired.append, log=lambda m: None,
    )
    assert done == ["ch-2"]   # ch-1 is fully done
    assert fired == ["ch-2"]


def test_all_rerecaps_every_synced_chapter(harness):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    calls_after_first = len(vision.calls)
    ctx_before = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()

    done = recap_mod.recap_series(
        conn, series, Config(), profile, all_chapters=True,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]  # both re-recapped, in chapter order
    assert len(vision.calls) > calls_after_first  # pages actually re-read
    # re-recaps leave rolling context and progress alone
    ctx_after = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    assert (ctx_after["rolling_summary"], ctx_after["through_chapter"]) == (
        ctx_before["rolling_summary"], ctx_before["through_chapter"],
    )
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_non_english_chapters_are_skipped(harness):
    conn, vision, text, client, series, profile = harness
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-1-pt', 's1', 1.5, 'Ch 1.5', 'pt-br', 5, 'now')"
    )
    conn.commit()
    client.pages_per_chapter["ch-1-pt"] = 5
    done = recap_mod.recap_series(
        conn, series, Config(), profile, all_chapters=True, client=client,
        log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]  # the pt-br chapter is never picked up


def test_batching_and_strict_rules(harness):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    vision_calls = vision.calls
    image_calls = [c for c in vision_calls if c["images"]]
    assert [len(c["images"]) for c in image_calls] == [4, 4, 1, 4]  # ch-1: 9 pages, ch-2: 4
    for call in image_calls:
        assert "Only use character names actually printed" in call["prompt"]
        assert "outside knowledge" in call["prompt"]
        assert "say so" in call["prompt"]
    # ch-1's 3 batch summaries -> combined by the vision model (text-only call);
    # ch-2 is a single batch and is used verbatim (no combine call)
    combine_call = [c for c in vision.calls if not c["images"]][0]
    assert not combine_call["images"]
    assert "[batch summary of 4 pages]" in combine_call["prompt"]
    text_prompts = [c["prompt"] for c in text.calls]
    assert any("story so far" in p.lower() for p in text_prompts)


def test_progress_and_attribution(harness):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    row = conn.execute(
        "SELECT r.* FROM recaps r JOIN chapters c ON r.chapter_id = c.id"
        " WHERE c.chapter_num = 1"
    ).fetchone()
    assert row["model"] == "fake-vision:8b (Q8_0)"
    assert row["summary"]
    progress = conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()
    assert progress["last_read_chapter"] == 2.0
    assert vision.ensured == ["fake-vision:8b"]
    assert text.ensured == ["fake-text:4b"]


def test_rolling_context_feeds_next_chapter(harness):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    assert client.downloaded == ["ch-1", "ch-2"]

    ctx = conn.execute(
        "SELECT * FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0
    assert ctx["rolling_summary"]

    # the ch-2 batch prompts must carry the rolling summary from ch-1
    ch2_prompts = [
        c["prompt"]
        for c in vision.calls
        if c["images"] and "chapter 2 of the manga" in c["prompt"]
    ]
    assert len(ch2_prompts) == 1  # ch-2 has 4 pages -> single batch
    assert "Story so far" in ch2_prompts[0]
    # ch-2 context update must include the previous rolling summary
    context_updates = [
        c["prompt"] for c in text.calls if 'updated "story so far"' in c["prompt"]
    ]
    assert len(context_updates) == 2  # one fold per chapter
    assert "story so far" in context_updates[-1].lower()

    progress = conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()
    assert progress["last_read_chapter"] == 2.0


def test_nothing_pending(harness, capsys):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    done = recap_mod.recap_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    assert done == []


def test_max_chapters_caps_default_run(harness):
    conn, vision, text, client, series, profile = harness
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]  # ch-2 still unfinished
    assert conn.execute("SELECT COUNT(*) AS n FROM recaps").fetchone()["n"] == 1


def test_forced_rerecap_preserves_context_and_progress(harness):
    """--chapter re-recaps one chapter (e.g. after a model swap) without
    rewinding the rolling context or reading progress."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile,
        client=client, log=lambda m: None,
    )
    calls_before = len(vision.calls)

    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapter_num=1.0,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    assert len(vision.calls) > calls_before  # pages actually re-read

    ctx = conn.execute(
        "SELECT * FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0  # not rewound to 1.0
    progress = conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()
    assert progress["last_read_chapter"] == 2.0
    assert conn.execute("SELECT COUNT(*) AS n FROM recaps").fetchone()["n"] == 2


def test_forced_rerecap_unknown_chapter(harness):
    conn, vision, text, client, series, profile = harness
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapter_num=99.0,
        client=client, log=lambda m: None,
    )
    assert done == []
    assert not vision.calls


# --- book path (imported text works; no vision model) ------------------------


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
        raise AssertionError("book recaps must not resolve a vision model")

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


def test_book_recap_uses_text_model_only(book_harness):
    conn, text, series, profile = book_harness
    done = recap_mod.recap_series(
        conn, series, Config(), profile, log=lambda m: None
    )
    assert done == ["b1-0001", "b1-0002"]
    rows = conn.execute(
        "SELECT summary, model FROM recaps ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["model"] == "fake-text:4b (Q8_0)" for r in rows)
    # single 200-word chunk per part -> no combine call; context update only
    # (cast-extraction calls — pipelines/characters.py — are filtered out)
    part_summaries = [c for c in text.calls if "reading excerpt" in c["prompt"]]
    context_updates = [
        c for c in text.calls if 'updated "story so far"' in c["prompt"]
    ]
    assert len(part_summaries) == 2 and len(context_updates) == 2
    ctx = conn.execute(
        "SELECT * FROM series_context WHERE series_id = 'b1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0


def test_book_recap_chunks_long_parts(book_harness, monkeypatch):
    conn, text, series, profile = book_harness
    monkeypatch.setattr(recap_mod, "BOOK_CHUNK_WORDS", 50)  # 200 words -> 4 chunks
    recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, log=lambda m: None
    )
    prompts = [c["prompt"] for c in text.calls]
    batch_prompts = [p for p in prompts if "reading excerpt" in p]
    combine_prompts = [p for p in prompts if "Excerpt summaries" in p]
    assert len(batch_prompts) == 4
    assert len(combine_prompts) == 1
    assert "Do not use outside knowledge" in batch_prompts[0]
    assert "alpha alpha" in batch_prompts[0]  # the excerpt text is included


def test_imported_comic_needs_no_source_client(tmp_path, monkeypatch, patch_judge):
    """Imported comics have pages on disk and source='import' — the source
    client must never be constructed (get_client('import') would raise)."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at, kind)"
        " VALUES ('c1', 'Test Comic', 'import', 'test-comic', 'imported', 'now',"
        " 'comic')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('c1-0001', 'c1', 1.0, 'Issue 1', 'en', 2, 'now')"
    )
    conn.commit()
    page_dir = tmp_path / "manga" / "c1" / "c1-0001"
    page_dir.mkdir(parents=True)
    (page_dir / "page-001.jpg").write_bytes(b"")
    (page_dir / "page-002.jpg").write_bytes(b"")

    vision = FakeAdapter("fake-vision:8b")
    text = FakeAdapter("fake-text:4b")
    info_v = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    info_t = ModelInfo("fake-text:4b", "fake", 4.0, "Q8_0", 3 * GB)
    monkeypatch.setattr(
        recap_mod, "get_vision_model",
        lambda c, p: Selection(adapter=vision, info=info_v),
    )
    monkeypatch.setattr(
        recap_mod, "get_text_model",
        lambda c, p: Selection(adapter=text, info=info_t),
    )
    patch_judge(ScriptedJudge())
    profile = HardwareProfile("fake chip", 24 * GB, 16 * GB, "cpu")
    series = conn.execute("SELECT * FROM series WHERE id = 'c1'").fetchone()
    done = recap_mod.recap_series(
        conn, series, Config(), profile, log=lambda m: None  # client=None
    )
    assert done == ["c1-0001"]
    assert any(c["images"] for c in vision.calls)  # pages actually read
    conn.close()


# --- translated recap path (translate first, recap the English overlays) ------


@pytest.fixture
def pt_chapter(harness):
    """Add an unread pt-br chapter (ch 3) to the standard harness."""
    conn, *_ = harness
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-3', 's1', 3.0, 'Ch 3', 'pt-br', 2, 'now')"
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 2.0, 'now')"
    )
    conn.commit()
    return harness


def _seed_translated(tmp_path, series_id="s1", chapter_id="ch-3", pages=2):
    dest = tmp_path / "manga" / series_id / chapter_id / "translated"
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(1, pages + 1):
        (dest / f"page-{i:03d}.png").write_bytes(b"")
    return dest


def test_pending_chapters_lang_none_ignores_language(pt_chapter):
    conn, *_ = pt_chapter
    rows = recap_mod.pending_chapters(conn, "s1", langs=None)
    assert [r["id"] for r in rows] == ["ch-3"]
    rows_en = recap_mod.pending_chapters(conn, "s1", langs=["en"])
    assert rows_en == []  # ch-3 is pt-br


def test_recap_translated_reads_translated_pages(pt_chapter, tmp_path):
    conn, vision, text, client, series, profile = pt_chapter
    _seed_translated(tmp_path)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, translated=True, log=lambda m: None
    )
    assert done == ["ch-3"]
    image_calls = [c for c in vision.calls if c["images"]]
    assert image_calls
    assert all(
        "translated" in p.parts for c in image_calls for p in c["images"]
    )
    assert all("pages are in en" in c["prompt"] for c in image_calls)
    # never downloaded originals
    assert client.downloaded == []
    # recap + progress + context updated through the normal chain
    assert conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-3'"
    ).fetchone()["summary"]
    progress = conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()
    assert progress["last_read_chapter"] == 3.0


def test_recap_translated_translates_first(pt_chapter, monkeypatch):
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
    done = recap_mod.recap_series(
        conn, series, Config(), profile, translated=True, log=lambda m: None
    )
    assert done == ["ch-3"]
    assert translated_calls == [3.0]
    image_calls = [c for c in vision.calls if c["images"]]
    assert all(
        "translated" in p.parts for c in image_calls for p in c["images"]
    )


def test_recap_translated_forced_chapter_ignores_lang(harness):
    conn, vision, text, client, series, profile = harness
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-9', 's1', 9.0, 'Ch 9', 'es-la', 2, 'now')"
    )
    conn.commit()
    # without --translated an es-la chapter is invisible to forced recap
    assert recap_mod.recap_series(
        conn, series, Config(), profile, chapter_num=9.0, log=lambda m: None
    ) == []
    from entertainment_harness.config import data_dir

    dest = data_dir() / "manga" / "s1" / "ch-9" / "translated"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "page-001.png").write_bytes(b"")
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapter_num=9.0, translated=True,
        log=lambda m: None,
    )
    assert done == ["ch-9"]


def test_recap_translated_skips_chapters_in_preferred_langs(harness, monkeypatch):
    """With langs=["en", "es-la"], an es-la chapter is recapped without
    translation; a pt-br chapter is translated to the first preferred lang."""
    conn, vision, text, client, series, profile = harness
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-es', 's1', 10.0, 'Ch 10', 'es-la', 1, 'now')"
    )
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-pt', 's1', 11.0, 'Ch 11', 'pt-br', 1, 'now')"
    )
    conn.commit()

    translated_calls = []

    def fake_translate(conn, series, config, profile, chapter_num=None, **kwargs):
        translated_calls.append(chapter_num)
        from entertainment_harness.config import data_dir

        dest = data_dir() / "manga" / "s1" / "ch-pt" / "translated"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "page-001.png").write_bytes(b"")

    monkeypatch.setattr(
        "entertainment_harness.pipelines.translate.translate_chapters", fake_translate
    )

    client.pages_per_chapter.update({"ch-es": 1, "ch-pt": 1})
    config = Config()
    config.library.langs = ["en", "es-la"]
    done = recap_mod.recap_series(
        conn, series, config, profile, translated=True, all_chapters=True,
        client=client, log=lambda m: None,
    )
    assert sorted(done) == ["ch-1", "ch-2", "ch-es", "ch-pt"]
    # es-la is in the preferred list, so it was not translated
    assert translated_calls == [11.0]
    # pt-br recap was fed the target language
    image_calls = [c for c in vision.calls if c["images"]]
    pt_calls = [c for c in image_calls if "ch-pt" in str(c["images"][0])]
    assert all("pages are in en" in c["prompt"] for c in pt_calls)


# --- rolling-context guard -----------------------------------------------------


def test_suspicious_context_detection():
    previous = "Shin trains under Merlin. " * 30  # ~180 words
    meta = ("The provided recap is from a different manga series and cannot be"
            " incorporated. The story so far remains unchanged.")
    assert recap_mod._suspicious_context(meta, previous)
    # a genuine update that incorporates new events is fine even if shorter
    genuine = "Shin trains under Merlin and joins the academy. " * 5
    assert not recap_mod._suspicious_context(genuine, previous)
    # no previous context -> nothing to protect
    assert not recap_mod._suspicious_context(meta, None)


def test_meta_context_update_keeps_previous(harness):
    """A crossover/bonus chapter whose context update comes back as meta-
    commentary must not destroy the accumulated story-so-far."""
    conn, vision, text, client, series, profile = harness
    long_ctx = "Shin trains under Merlin and learns magic. " * 20  # 160 words
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES ('s1', ?, 1.0)",
        (long_ctx,),
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 1.0, 'now')"
    )
    conn.commit()

    # ch-2's context update comes back as meta-commentary
    text.generate = lambda model, prompt, images=None: (
        "The provided chapter recap is unrelated and cannot be incorporated."
        if "story so far" in prompt
        else "[text output]"
    )
    client.pages_per_chapter["ch-2"] = 4
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    ctx_after = conn.execute(
        "SELECT rolling_summary FROM series_context WHERE series_id = 's1'"
    ).fetchone()["rolling_summary"]
    assert ctx_after == long_ctx  # previous context preserved
    # ...but the chapter recap and progress still advance normally
    assert conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-2'"
    ).fetchone()["summary"]
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


# --- judge / retry loop ---------------------------------------------------------


FAIL_KAZUMA = '{"pass": false, "issues": ["invented the name Kazuma"]}'
FAIL_META = '{"pass": false, "issues": ["meta-commentary about the task"]}'


def test_judge_rejection_retries_with_feedback(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    patch_judge(ScriptedJudge(verdicts=[FAIL_KAZUMA]))
    done = recap_mod.recap_series(
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


def test_judge_persistent_recap_failure_keeps_first(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    patch_judge(
        ScriptedJudge(verdicts=[FAIL_KAZUMA] * recap_mod.MAX_ATTEMPTS),
    )
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None
    )
    assert done == ["ch-1"]  # stored anyway, with a warning
    combines = [c for c in vision.calls if not c["images"]]
    assert len(combines) == recap_mod.MAX_ATTEMPTS  # bounded loop
    # on persistent failure the FIRST attempt is stored, not the last:
    # retries risk over-complying with a mistaken critique
    stored = conn.execute(
        "SELECT summary FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["summary"]
    assert stored == "[text output 4]"  # 3 batch calls, then the first combine


def test_judge_context_failure_keeps_previous(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    long_ctx = "Shin trains under Merlin and learns magic. " * 20
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary, through_chapter)"
        " VALUES ('s1', ?, 1.0)",
        (long_ctx,),
    )
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', 1.0, 'now')"
    )
    conn.commit()
    patch_judge(
        ScriptedJudge(context_verdicts=[FAIL_META] * recap_mod.MAX_ATTEMPTS),
    )
    done = recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    assert done == ["ch-2"]  # recap + progress still advance
    assert conn.execute(
        "SELECT rolling_summary FROM series_context WHERE series_id = 's1'"
    ).fetchone()["rolling_summary"] == long_ctx
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0
    context_gens = [c for c in text.calls if "story so far" in c["prompt"]]
    assert len(context_gens) == recap_mod.MAX_ATTEMPTS


def test_judge_passes_first_try_no_retries(harness):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    combines = [c for c in vision.calls if not c["images"]]
    assert len(combines) == 1  # no feedback retry


def test_thinking_low_skips_judge(harness, monkeypatch):
    conn, vision, text, client, series, profile = harness

    def no_judge(config, profile):
        raise AssertionError("judge must not be resolved with thinking='low'")

    monkeypatch.setattr(recap_mod, "get_judge_model", no_judge)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, thinking="low",
        log=lambda m: None,
    )
    assert done == ["ch-1"]


def test_thinking_high_vision_verifies_recap(harness, monkeypatch, patch_judge):
    """On high, the recap is judged by the VISION adapter with page images,
    and the attempt budget is 5."""
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
                return f"[batch summary of {len(images)} pages]"
            return f"[text output {len(self.calls)}]"

    vision = ScriptedVision([FAIL_KAZUMA] * recap_mod.ATTEMPTS["high"])
    info_v = ModelInfo("fake-vision:8b", "fake", 8.0, "Q8_0", 5 * GB)
    monkeypatch.setattr(
        recap_mod, "get_vision_model",
        lambda config, profile: Selection(adapter=vision, info=info_v),
    )
    patch_judge(ScriptedJudge())  # context judging: pass
    done = recap_mod.recap_series(
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


# --- detail grains (unify-recap-narrate) ---------------------------------------


def test_detail_rank_ordering():
    assert recap_mod.DETAIL_LEVELS == (
        "gist", "brief", "standard", "detailed", "full",
    )
    assert [recap_mod.DETAIL_RANK[d] for d in recap_mod.DETAIL_LEVELS] == [0, 1, 2, 3, 4]


def test_recap_stores_standard_detail_by_default(harness):
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None,
    )
    row = conn.execute(
        "SELECT detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()
    assert row["detail"] == "standard"


def test_invalid_detail_raises(harness):
    conn, vision, text, client, series, profile = harness
    with pytest.raises(ValueError, match="unknown detail"):
        recap_mod.recap_series(
            conn, series, Config(), profile, detail="bogus",
            client=client, log=lambda m: None,
        )
    with pytest.raises(ValueError, match="unknown detail"):
        recap_mod.pending_chapters(conn, "s1", detail="bogus")


# --- grain-aware selection -----------------------------------------------------


def _seed_artifact(conn, chapter_id, detail, summary="s"):
    conn.execute(
        "INSERT INTO recaps (chapter_id, summary, created_at, model, detail)"
        " VALUES (?, ?, 'now', 'm', ?)",
        (chapter_id, summary, detail),
    )
    conn.commit()


def test_pending_skips_at_grain_and_never_downgrades(harness):
    """A chapter already AT the requested grain is skipped; one at a HIGHER
    grain is never downgraded by default; one at a LOWER grain is an upgrade
    candidate (ungated on last_read)."""
    conn, *_ = harness
    _seed_artifact(conn, "ch-1", "detailed")
    _seed_artifact(conn, "ch-2", "full")

    assert recap_mod.pending_chapters(conn, "s1", detail="standard") == []
    assert recap_mod.pending_chapters(conn, "s1", detail="detailed") == []
    assert recap_mod.pending_chapters(conn, "s1", detail="gist") == []
    rows = recap_mod.pending_chapters(conn, "s1", detail="full")
    assert [r["id"] for r in rows] == ["ch-1"]  # the lower-grain upgrade
    assert rows[0]["artifact_detail"] == "detailed"


def test_lower_grain_run_is_bucket_a_and_advances(harness):
    """Chapters with no artifact at any grain (bucket a) advance progress and
    fold the rolling context at any requested grain, as recaps always did."""
    conn, vision, text, client, series, profile = harness
    done = recap_mod.recap_series(
        conn, series, Config(), profile, detail="gist",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM recaps WHERE detail = 'gist'"
    ).fetchone()["n"] == 2
    ctx = conn.execute(
        "SELECT through_chapter FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 2.0
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_bucket_b_upgrade_leaves_covered_context_and_progress_alone(harness):
    """Upgrading chapters already folded into the story-so-far stores the new
    grain without touching context or progress."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(  # both chapters at standard, context through 2.0
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    ctx_before = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    calls_before = len(text.calls)

    done = recap_mod.recap_series(
        conn, series, Config(), profile, detail="detailed",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM recaps WHERE detail = 'detailed'"
    ).fetchone()["n"] == 2
    ctx_after = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    assert (ctx_after["rolling_summary"], ctx_after["through_chapter"]) == (
        ctx_before["rolling_summary"], ctx_before["through_chapter"],
    )
    # no context regeneration was attempted for either chapter
    assert not [
        c for c in text.calls[calls_before:] if "story so far" in c["prompt"].lower()
    ]
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_bucket_b_upgrade_refolds_uncovered_context_without_progress(harness):
    """A lower-grain chapter NOT covered by series_context.through_chapter
    gets the context re-folded on upgrade — but progress is still untouched."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(  # ch-1 only: context through 1.0, progress 1.0
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None,
    )
    recap_mod.recap_series(  # forced ch-2 at standard: context/progress untouched
        conn, series, Config(), profile, chapter_num=2.0,
        client=client, log=lambda m: None,
    )
    ctx_before = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    assert ctx_before["through_chapter"] == 1.0  # ch-2 not covered

    done = recap_mod.recap_series(
        conn, series, Config(), profile, detail="detailed",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]  # both are lower-grain upgrades
    ctx_after = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    # ch-2 was not covered -> its events are folded in now
    assert ctx_after["through_chapter"] == 2.0
    assert ctx_after["rolling_summary"] != ctx_before["rolling_summary"]
    # ...but progress must not move for bucket b
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 1.0


def test_forced_rerun_overwrites_at_requested_grain(harness):
    """--chapter forces a re-run at the requested grain, even below the
    chapter's current grain, without touching context/progress."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, detail="full",
        client=client, log=lambda m: None,
    )
    assert conn.execute(
        "SELECT detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["detail"] == "full"

    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapter_num=1.0, detail="standard",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    assert conn.execute(
        "SELECT detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["detail"] == "standard"
    assert conn.execute(
        "SELECT through_chapter FROM series_context WHERE series_id = 's1'"
    ).fetchone()["through_chapter"] == 2.0
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


# --- prompt / judge selection per grain -----------------------------------------


def _chapter_judge_prompts(judge):
    """ScriptedJudge prompts for the chapter artifact (not context updates)."""
    return [
        c["prompt"] for c in judge.calls
        if "strict evaluator" in c["prompt"]
        and "Updated story so far" not in c["prompt"]
    ]


def test_gist_grain_uses_gist_prompts_and_recap_judge(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    judge = ScriptedJudge()
    patch_judge(judge)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, detail="gist",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    image_prompts = [c["prompt"] for c in vision.calls if c["images"]]
    assert all("in 2-4 sentences" in p for p in image_prompts)
    combine_prompts = [c["prompt"] for c in vision.calls if not c["images"]]
    assert any("gist of the chapter" in p for p in combine_prompts)
    # non-full grains use the recap judge
    judge_prompts = _chapter_judge_prompts(judge)
    assert judge_prompts
    assert all("for a recap of chapter" in p for p in judge_prompts)
    assert conn.execute(
        "SELECT detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["detail"] == "gist"


def test_full_grain_uses_narration_prompts_and_judge(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    judge = ScriptedJudge()
    patch_judge(judge)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, detail="full",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    image_prompts = [c["prompt"] for c in vision.calls if c["images"]]
    assert all("Narrate in detail" in p for p in image_prompts)
    combine_prompts = [c["prompt"] for c in vision.calls if not c["images"]]
    assert any("continuous English narration" in p for p in combine_prompts)
    # full keeps the narration judge with the completeness axis
    judge_prompts = _chapter_judge_prompts(judge)
    assert judge_prompts
    assert all("for a spoken narration of chapter" in p for p in judge_prompts)
    assert all("Completeness" in p for p in judge_prompts)
    assert conn.execute(
        "SELECT detail FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["detail"] == "full"


def test_standard_grain_uses_recap_prompts_and_judge(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    judge = ScriptedJudge()
    patch_judge(judge)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    image_prompts = [c["prompt"] for c in vision.calls if c["images"]]
    assert all("Summarize concisely" in p for p in image_prompts)
    judge_prompts = _chapter_judge_prompts(judge)
    assert judge_prompts
    assert all("for a recap of chapter" in p for p in judge_prompts)


# --- steering instructions (Phase 5) --------------------------------------------

GEN_Z = "write in Gen Z slang"


def test_user_direction_block_empty_renders_nothing():
    assert recap_mod.user_direction_block("") == ""
    assert recap_mod.user_direction_block("   \n ") == ""
    block = recap_mod.user_direction_block(" write in Gen Z slang ")
    assert "USER DIRECTION" in block and "write in Gen Z slang" in block


def test_instruction_reaches_batch_combine_and_judge(harness, patch_judge):
    """The steering instruction lands as a USER DIRECTION block in the batch
    and combine prompts, and in the artifact judge's prompt together with
    the mandated-omission rule."""
    conn, vision, text, client, series, profile = harness
    judge = ScriptedJudge()
    patch_judge(judge)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, client=client,
        instruction=GEN_Z, log=lambda m: None,
    )
    assert done == ["ch-1"]
    batch_prompts = [c["prompt"] for c in vision.calls if c["images"]]
    combine_prompts = [c["prompt"] for c in vision.calls if not c["images"]]
    assert batch_prompts and combine_prompts  # ch-1: 9 pages -> batches + combine
    for prompt in batch_prompts + combine_prompts:
        assert "USER DIRECTION" in prompt
        assert GEN_Z in prompt
    judge_prompts = _chapter_judge_prompts(judge)
    assert judge_prompts
    for prompt in judge_prompts:
        assert "USER DIRECTION" in prompt
        assert GEN_Z in prompt
        assert "NOT issues" in prompt  # mandated omissions/style are tolerated


def test_instruction_reaches_full_grain_prompts(harness, patch_judge):
    """Grain-independent: the narration (full) prompt pair and judge get the
    block too."""
    conn, vision, text, client, series, profile = harness
    judge = ScriptedJudge()
    patch_judge(judge)
    recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, detail="full",
        client=client, instruction="skip chapter-opening recap pages",
        log=lambda m: None,
    )
    prompts = [c["prompt"] for c in vision.calls]
    assert prompts
    assert all("USER DIRECTION" in p for p in prompts)
    assert all("skip chapter-opening recap pages" in p for p in prompts)
    judge_prompts = _chapter_judge_prompts(judge)
    assert judge_prompts
    assert all("spoken narration" in p for p in judge_prompts)  # narration judge
    assert all("NOT issues" in p for p in judge_prompts)


def test_no_instruction_leaves_prompts_untouched(harness, patch_judge):
    """With no instruction there is no USER DIRECTION block anywhere —
    prompts render exactly as before the feature."""
    conn, vision, text, client, series, profile = harness
    judge = ScriptedJudge()
    patch_judge(judge)
    recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, client=client,
        log=lambda m: None,
    )
    for call in vision.calls + text.calls + judge.calls:
        assert "USER DIRECTION" not in call["prompt"]
    assert conn.execute(
        "SELECT instruction FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["instruction"] == ""


def test_instruction_stored_on_artifact(harness):
    """Attribution: the instruction lands on the recaps row and recap.json."""
    from entertainment_harness.library import works

    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, client=client,
        instruction=GEN_Z, log=lambda m: None,
    )
    row = conn.execute(
        "SELECT detail, instruction FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()
    assert row["instruction"] == GEN_Z
    assert row["detail"] == "standard"
    meta = works.read_recap("s1", "ch-1")
    assert meta is not None
    assert meta.instruction == GEN_Z
    assert meta.detail == "standard"


def test_instruction_does_not_reselect_completed_chapters(harness):
    """Changing the instruction never invalidates an artifact: a chapter
    already at the requested grain is skipped even with a new instruction."""
    conn, vision, text, client, series, profile = harness
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None,
    )
    calls_after_first = len(vision.calls)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, client=client,
        instruction="write in Gen Z slang", log=lambda m: None,
    )
    assert done == []  # nothing pending at 'standard' — instruction is no trigger
    assert len(vision.calls) == calls_after_first


# --- gap fill (recap --video) ---------------------------------------------------


def _seed_progress(conn, last_read: float) -> None:
    conn.execute(
        "INSERT INTO progress (series_id, last_read_chapter, updated_at)"
        " VALUES ('s1', ?, 'now')",
        (last_read,),
    )
    conn.commit()


def test_no_gap_fill_by_default(harness):
    """Without fill_gaps, chapters at or before last_read with no artifact
    stay orphaned — the frontier gate is unchanged."""
    conn, vision, text, client, series, profile = harness
    _seed_progress(conn, 2.0)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    assert done == []
    assert conn.execute("SELECT COUNT(*) AS n FROM recaps").fetchone()["n"] == 0


def test_pending_fill_gaps_selects_artifactless_behind_frontier(harness):
    conn, *_ = harness
    _seed_progress(conn, 2.0)
    assert recap_mod.pending_chapters(conn, "s1", detail="full") == []
    rows = recap_mod.pending_chapters(
        conn, "s1", detail="full", fill_gaps=True
    )
    assert [r["id"] for r in rows] == ["ch-1", "ch-2"]
    assert all(r["artifact_detail"] is None for r in rows)


def test_pending_fill_gaps_skips_chapters_with_usable_video(harness):
    """A usable video row means nothing is missing for that chapter; a fully
    wiped row counts as missing again."""
    conn, *_ = harness
    _seed_progress(conn, 2.0)
    conn.executemany(
        "INSERT INTO videos (series_id, from_chapter, to_chapter, path,"
        " created_at, kind) VALUES ('s1', ?, ?, 'p', 'now', 'recap')",
        [(1.0, 1.0), (2.0, 2.0)],
    )
    conn.commit()
    assert recap_mod.pending_chapters(
        conn, "s1", detail="full", fill_gaps=True
    ) == []
    conn.execute(
        "UPDATE videos SET wiped_at = 'now' WHERE from_chapter = 1.0"
    )
    conn.commit()
    rows = recap_mod.pending_chapters(
        conn, "s1", detail="full", fill_gaps=True
    )
    assert [r["id"] for r in rows] == ["ch-1"]


def test_pending_fill_gaps_keeps_buckets_a_and_b(harness):
    """fill_gaps only ADDS the gap bucket: lower-grain upgrades (bucket b)
    are still selected, and so are artifactless chapters past the frontier
    (bucket a)."""
    conn, *_ = harness
    _seed_artifact(conn, "ch-1", "standard")
    _seed_progress(conn, 1.0)  # ch-2: no artifact at 2.0 > 1.0 -> bucket a
    rows = recap_mod.pending_chapters(
        conn, "s1", detail="full", fill_gaps=True
    )
    assert [r["id"] for r in rows] == ["ch-1", "ch-2"]
    assert rows[0]["artifact_detail"] == "standard"  # upgrade, not a gap
    assert rows[1]["artifact_detail"] is None  # bucket a, not a gap


def test_fill_gaps_run_generates_standalone_artifacts(harness):
    """Behind the read frontier with no videos: artifacts are stored
    standalone (story-so-far withheld), nothing is folded, progress is
    untouched, and the video callback fires per chapter."""
    from entertainment_harness.library import works

    conn, vision, text, client, series, profile = harness
    _seed_progress(conn, 2.0)
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary,"
        " through_chapter) VALUES ('s1', 'story through 2', 2.0)"
    )
    conn.commit()
    events = []
    done = recap_mod.recap_series(
        conn, series, Config(), profile, fill_gaps=True,
        client=client, on_recap=events.append, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    assert events == ["ch-1", "ch-2"]
    rows = conn.execute(
        "SELECT standalone FROM recaps ORDER BY chapter_id"
    ).fetchall()
    assert [r["standalone"] for r in rows] == [1, 1]
    assert works.read_recap("s1", "ch-1").standalone is True
    # the rolling summary was withheld: no prompt saw it
    assert all(
        "story through 2" not in c["prompt"]
        for c in vision.calls + text.calls
    )
    ctx = conn.execute(
        "SELECT rolling_summary, through_chapter FROM series_context"
        " WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["rolling_summary"] == "story through 2"
    assert ctx["through_chapter"] == 2.0
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_fill_gaps_mixed_with_pending_frontier(harness):
    """One run: gap chapters first (standalone), then frontier chapters as
    bucket a (fold + advance) — in chapter order."""
    conn, vision, text, client, series, profile = harness
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES ('ch-3', 's1', 3.0, 'Ch 3', 'en', 2, 'now')"
    )
    _seed_progress(conn, 1.0)
    # context accompanies progress in practice; ch-1 is a gap only when
    # there is story-so-far to withhold from it
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary,"
        " through_chapter) VALUES ('s1', 'story through 1', 1.0)"
    )
    conn.commit()
    client.pages_per_chapter["ch-3"] = 2
    done = recap_mod.recap_series(
        conn, series, Config(), profile, fill_gaps=True,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2", "ch-3"]
    standalone = {
        r["chapter_id"]: r["standalone"]
        for r in conn.execute("SELECT chapter_id, standalone FROM recaps")
    }
    assert standalone == {"ch-1": 1, "ch-2": 0, "ch-3": 0}
    ctx = conn.execute(
        "SELECT through_chapter FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["through_chapter"] == 3.0
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 3.0


# --- explicit chapter selection (--chapters) -------------------------------------


def _seed_side_chapter(conn, cid, num, pages=3):
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " fetched_at) VALUES (?, 's1', ?, 'Side', 'en', ?, 'now')",
        (cid, num, pages),
    )
    conn.commit()


def _seed_context(conn, text="story through 2", through=2.0):
    conn.execute(
        "INSERT INTO series_context (series_id, rolling_summary,"
        " through_chapter) VALUES ('s1', ?, ?)",
        (text, through),
    )
    conn.commit()


def test_chapters_spec_selects_exact_range(harness):
    """--chapters selects exactly the synced chapters inside the spec, in
    chapter order; side chapters match ranges by containment."""
    conn, vision, text, client, series, profile = harness
    _seed_side_chapter(conn, "ch-15", 1.5)
    client.pages_per_chapter["ch-15"] = 3
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1.2-1.8",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-15"]  # containment: 1.5 in 1.2-1.8, 1.0 and 2.0 out
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1-1.5",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-15"]  # ascending; ch-2 outside the spec


def test_chapters_spec_behind_frontier_is_standalone(harness):
    """Explicit selection behind the read frontier with no artifact gets the
    same protection as gap fill: generated standalone."""
    conn, vision, text, client, series, profile = harness
    _seed_progress(conn, 2.0)
    _seed_context(conn)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1-2",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1", "ch-2"]
    rows = conn.execute(
        "SELECT standalone FROM recaps ORDER BY chapter_id"
    ).fetchall()
    assert [r["standalone"] for r in rows] == [1, 1]
    assert all(
        "story through 2" not in c["prompt"]
        for c in vision.calls + text.calls
    )
    # forced: context and progress untouched
    ctx = conn.execute(
        "SELECT rolling_summary FROM series_context WHERE series_id = 's1'"
    ).fetchone()
    assert ctx["rolling_summary"] == "story through 2"
    assert conn.execute(
        "SELECT last_read_chapter FROM progress WHERE series_id = 's1'"
    ).fetchone()["last_read_chapter"] == 2.0


def test_chapters_spec_with_artifact_prepends_context(harness):
    """Chapters that already have an artifact keep the classic forced
    behavior: current story-so-far prepended, nothing folded, no
    standalone marker."""
    conn, vision, text, client, series, profile = harness
    _seed_artifact(conn, "ch-1", "standard")
    _seed_context(conn, "old story", 1.0)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapters_spec="1",
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    assert conn.execute(
        "SELECT standalone FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["standalone"] == 0
    assert any(
        "old story" in c["prompt"]
        for c in vision.calls + text.calls
    )
    assert conn.execute(
        "SELECT rolling_summary FROM series_context WHERE series_id = 's1'"
    ).fetchone()["rolling_summary"] == "old story"


def test_chapter_num_behind_frontier_is_standalone(harness):
    """--chapter gets the same standalone treatment: a no-artifact chapter
    behind the read frontier is generated without story-so-far (prepending
    it would leak future events into the artifact)."""
    conn, vision, text, client, series, profile = harness
    _seed_progress(conn, 2.0)
    _seed_context(conn)
    done = recap_mod.recap_series(
        conn, series, Config(), profile, chapter_num=1.0,
        client=client, log=lambda m: None,
    )
    assert done == ["ch-1"]
    assert conn.execute(
        "SELECT standalone FROM recaps WHERE chapter_id = 'ch-1'"
    ).fetchone()["standalone"] == 1


def test_chapters_spec_bad_spec_raises(harness):
    conn, vision, text, client, series, profile = harness
    from entertainment_harness.library import LibraryError

    with pytest.raises(LibraryError, match="Reversed chapter range"):
        recap_mod.recap_series(
            conn, series, Config(), profile, chapters_spec="3-1",
            client=client, log=lambda m: None,
        )
