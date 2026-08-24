"""Character-bible tests: registry persistence, extraction parsing, the
deterministic merge (entity resolution, alias union, edited-lock, no
canonical flip-flop), the CAST prompt block, and the recap_series fold hook.
"""

from __future__ import annotations

import json

import pytest

from entertainment_harness import db
from entertainment_harness.config import Config
from entertainment_harness.pipelines import characters as chars
from entertainment_harness.pipelines import recap as recap_mod

from conftest import FakeAdapter, ScriptedJudge  # noqa: F401  (fixture deps)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    connection = db.connect()
    connection.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    connection.commit()
    yield connection
    connection.close()


def _entry(name, aliases=(), role=""):
    return {"name": name, "aliases": list(aliases), "role": role}


# --- parse_cast ---------------------------------------------------------------


def test_parse_cast_tolerates_fences_and_preamble():
    raw = 'Sure! Here is the list:\n```json\n[{"name": "Shin", "aliases": ["Shinu"], "role": "hunter"}]\n```'
    assert chars.parse_cast(raw) == [_entry("Shin", ["Shinu"], "hunter")]


def test_parse_cast_skips_malformed_items():
    raw = '[{"name": "A"}, 42, {"noname": "x"}, {"name": "  ", "role": "y"}]'
    assert chars.parse_cast(raw) == [_entry("A")]


def test_parse_cast_raises_on_garbage():
    with pytest.raises(ValueError):
        chars.parse_cast("I cannot help with that.")
    with pytest.raises(ValueError):
        chars.parse_cast('{"name": "not a list"}')


# --- cast_block ---------------------------------------------------------------


def test_cast_block_empty_registry_is_byte_identical():
    assert chars.cast_block([]) == ""


def test_cast_block_formats_names_aliases_roles():
    block = chars.cast_block([
        _entry("Shin", ["Shinu"], "young hunter"),
        _entry("Merlin"),
    ])
    assert "CAST — characters known from previous chapters:" in block
    assert "- Shin (also Shinu) — young hunter" in block
    assert "- Merlin" in block
    assert 'override "only printed names"' in block


def test_cast_block_caps_entries():
    cast = [_entry(f"Char{i:02d}", role="x") for i in range(30)]
    block = chars.cast_block(cast)
    assert "- Char14" in block
    assert "- Char15" not in block


# --- merge_cast ---------------------------------------------------------------


def test_merge_inserts_new_characters(conn):
    chars.merge_cast(conn, "s1", [_entry("Shin", role="hunter")], 1.0)
    rows = db.get_characters(conn, "s1")
    assert [r["name"] for r in rows] == ["Shin"]
    assert rows[0]["first_seen"] == 1.0 and rows[0]["last_seen"] == 1.0
    assert rows[0]["origin"] == "observed" and rows[0]["edited"] == 0


def test_merge_matches_case_insensitively_and_advances_last_seen(conn):
    chars.merge_cast(conn, "s1", [_entry("Shin")], 1.0)
    chars.merge_cast(conn, "s1", [_entry("SHIN")], 3.0)
    rows = db.get_characters(conn, "s1")
    assert len(rows) == 1
    assert rows[0]["first_seen"] == 1.0 and rows[0]["last_seen"] == 3.0


def test_merge_matches_on_alias_and_never_flip_flops_canonical(conn):
    chars.merge_cast(conn, "s1", [_entry("Shin", ["Shinu"], "hunter")], 1.0)
    # a later chapter's extractor prefers the alias as the canonical name
    chars.merge_cast(conn, "s1", [_entry("Shinu", role="hero hunter")], 2.0)
    rows = db.get_characters(conn, "s1")
    assert len(rows) == 1
    assert rows[0]["name"] == "Shin"  # canonical name stays
    assert set(json.loads(rows[0]["aliases"])) == {"Shinu"}
    assert rows[0]["role"] == "hero hunter"  # latest non-empty role wins


def test_merge_edited_rows_only_advance_last_seen(conn):
    db.replace_characters(conn, "s1", [
        {"name": "Canon Name", "aliases": ["Nick"], "role": "user role"},
    ])
    chars.merge_cast(
        conn, "s1", [_entry("Nick", ["Other"], "extractor role")], 5.0
    )
    rows = db.get_characters(conn, "s1")
    assert len(rows) == 1
    assert rows[0]["name"] == "Canon Name"
    assert rows[0]["role"] == "user role"  # locked
    assert set(json.loads(rows[0]["aliases"])) == {"Nick"}  # locked
    assert rows[0]["last_seen"] == 5.0  # the sighting still advances


# --- db helpers ----------------------------------------------------------------


def test_replace_characters_marks_everything_user_edited(conn):
    chars.merge_cast(conn, "s1", [_entry("Auto", role="x")], 1.0)
    db.replace_characters(conn, "s1", [
        {"name": "Manual", "aliases": ["M"], "role": "y"},
    ])
    rows = db.get_characters(conn, "s1")
    assert [r["name"] for r in rows] == ["Manual"]
    assert rows[0]["origin"] == "user" and rows[0]["edited"] == 1


# --- prompt + judge injection (Phase 2) ---------------------------------------


def test_cast_reaches_later_chapter_prompts(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    _cast_aware(text, '[{"name": "Shin", "aliases": ["Shinu"],'
                      ' "role": "hunter"}]')
    judge = ScriptedJudge()
    patch_judge(judge)  # capture the artifact judge's prompts
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    ch1_prompts = [
        c["prompt"] for c in vision.calls
        if "chapter 1 of the manga" in c["prompt"]
    ]
    ch2_prompts = [
        c["prompt"] for c in vision.calls
        if "chapter 2 of the manga" in c["prompt"]
    ]
    assert not any("CAST — characters" in p for p in ch1_prompts)
    assert any("CAST — characters" in p for p in ch2_prompts)
    assert any("- Shin (also Shinu) — hunter" in p for p in ch2_prompts)
    # the artifact judge saw the same registry, pre-approved
    assert any(
        "pre-approved" in c["prompt"] and "- Shin" in c["prompt"]
        for c in judge.calls
    )


def test_cast_empty_registry_keeps_prompts_clean(harness):
    conn, vision, text, client, series, profile = harness
    # unparseable cast output -> registry stays empty -> no CAST anywhere
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    assert not any(
        "CAST — characters" in c["prompt"]
        for c in vision.calls + text.calls
    )


def test_judge_cast_block_renders_pre_approved_addendum():
    from entertainment_harness.pipelines.judge import _judge_cast_block

    cast = chars.cast_block([_entry("Shin", ["Shinu"], "hunter")])
    block = _judge_cast_block(cast)
    assert "- Shin (also Shinu) — hunter" in block
    assert "pre-approved" in block
    assert _judge_cast_block("") == ""


def test_judge_recap_prompt_includes_cast_addendum():
    from entertainment_harness.pipelines.judge import judge_recap

    adapter = FakeAdapter("fake-judge:4b")
    cast = chars.cast_block([_entry("Shin", role="hunter")])
    judge_recap(
        adapter, "fake-judge:4b", "Shin hunts.", ["Shin hunts."], None,
        "Test Manga", 1.0, cast=cast,
    )
    prompt = adapter.calls[-1]["prompt"]
    assert "pre-approved" in prompt
    assert "- Shin — hunter" in prompt


def _cast_aware(text: FakeAdapter, payload: str) -> None:
    """Make the shared fake answer cast-extraction prompts with real JSON."""
    original = FakeAdapter.generate.__get__(text)

    def generate(model, prompt, images=None):
        if "tracking the cast" in prompt:
            text.calls.append({"model": model, "prompt": prompt, "images": []})
            return payload
        return original(model, prompt, images)

    text.generate = generate


def test_recap_populates_registry_after_each_chapter(harness):
    conn, vision, text, client, series, profile = harness
    _cast_aware(text, '[{"name": "Shin", "aliases": [], "role": "hunter"}]')
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    rows = db.get_characters(conn, "s1")
    assert [r["name"] for r in rows] == ["Shin"]
    assert (rows[0]["first_seen"], rows[0]["last_seen"]) == (1.0, 2.0)


def test_recap_cast_disabled_leaves_registry_empty(harness):
    conn, vision, text, client, series, profile = harness
    config = Config()
    config.pipeline.characters = False
    recap_mod.recap_series(
        conn, series, config, profile, client=client, log=lambda m: None
    )
    assert db.get_characters(conn, "s1") == []
    assert not any("tracking the cast" in c["prompt"] for c in text.calls)


def test_recap_keeps_registry_on_unparseable_extraction(harness):
    conn, vision, text, client, series, profile = harness
    # the stock fake's "[text output N]" is not cast JSON: every attempt
    # fails parsing and the registry must survive untouched
    recap_mod.recap_series(
        conn, series, Config(), profile, client=client, log=lambda m: None
    )
    assert db.get_characters(conn, "s1") == []


def test_recap_keeps_registry_when_cast_judge_fails(harness, patch_judge):
    conn, vision, text, client, series, profile = harness
    _cast_aware(text, '[{"name": "Invented", "aliases": [], "role": "ghost"}]')

    class CastFailingJudge(ScriptedJudge):
        def generate(self, model, prompt, images=None):
            if "character-registry update" in prompt:
                self.calls.append({"model": model, "prompt": prompt, "images": []})
                return ('{"pass": false, "issues":'
                        ' ["Invented is not in the chapter"]}')
            return super().generate(model, prompt, images)

    patch_judge(CastFailingJudge())
    recap_mod.recap_series(
        conn, series, Config(), profile, max_chapters=1, client=client,
        log=lambda m: None,
    )
    # the failing cast update kept the registry empty...
    assert db.get_characters(conn, "s1") == []
    # ...but the chapter itself was still recapped
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM recaps"
    ).fetchone()["n"] == 1


# --- translation + video script injection (Phase 3) ---------------------------


class _JsonAdapter(FakeAdapter):
    """FakeAdapter that returns one canned payload for every call."""

    def __init__(self, payload: str) -> None:
        super().__init__("fake-text:8b")
        self.payload = payload

    def generate(self, model, prompt, images=None):
        self.calls.append({"model": model, "prompt": prompt, "images": []})
        return self.payload


def test_cast_script_block_amendment():
    block = chars.cast_script_block([_entry("Shin", ["Shinu"], "hunter")])
    assert "- Shin (also Shinu) — hunter" in block
    assert "even where the recap leaves them unnamed" in block
    assert chars.cast_script_block([]) == ""


def test_cast_translation_block_amendment():
    block = chars.cast_translation_block([_entry("Shin", ["Shinu"], "hunter")])
    assert "- Shin (also Shinu) — hunter" in block
    assert "canonical spelling" in block
    assert chars.cast_translation_block([]) == ""


def _translate_args():
    return {
        "page": 1, "total": 9, "chapter": "2",
        "title": "Test Manga", "lang": "ja", "target": "en",
    }


def test_translate_prompt_carries_canonical_spellings():
    from entertainment_harness.pipelines import translate as translate_mod

    adapter = _JsonAdapter('[{"translation": "Shin draws his bow."}]')
    bubbles = [translate_mod.Bubble(box=[0, 0, 1, 1], original="弓を引く",
                                    translation="")]
    cast = chars.cast_translation_block([_entry("Shin", ["Shinu"], "hunter")])
    out = translate_mod._translate_bubbles(
        adapter, "fake-text:8b", bubbles, _translate_args(), cast=cast
    )
    prompt = adapter.calls[-1]["prompt"]
    assert "CAST — characters" in prompt
    assert "- Shin (also Shinu) — hunter" in prompt
    assert "canonical spelling" in prompt
    assert out[0].translation == "Shin draws his bow."


def test_translate_prompt_clean_without_cast():
    from entertainment_harness.pipelines import translate as translate_mod

    adapter = _JsonAdapter('[{"translation": "He draws his bow."}]')
    bubbles = [translate_mod.Bubble(box=[0, 0, 1, 1], original="弓を引く",
                                    translation="")]
    translate_mod._translate_bubbles(
        adapter, "fake-text:8b", bubbles, _translate_args()
    )
    prompt = adapter.calls[-1]["prompt"]
    assert "CAST" not in prompt
    expected = translate_mod.TRANSLATE_PROMPT.format(
        originals="1. 弓を引く", cast="", **_translate_args()
    )
    assert prompt == expected


def test_judge_translation_prompt_includes_cast_addendum():
    from entertainment_harness.pipelines.judge import judge_translation
    from entertainment_harness.pipelines.translate import Bubble

    adapter = FakeAdapter("fake-judge:4b")
    cast = chars.cast_block([_entry("Shin", ["Shinu"], "hunter")])
    bubbles = [Bubble(box=[0, 0, 1, 1], original="シヌ", translation="Shin")]
    judge_translation(
        adapter, "fake-judge:4b", bubbles, "Test Manga", 2.0, 1, "ja", "en",
        cast=cast,
    )
    prompt = adapter.calls[-1]["prompt"]
    assert "pre-approved" in prompt
    assert "- Shin (also Shinu) — hunter" in prompt


def test_generate_script_prompt_includes_cast():
    from entertainment_harness.video import script as script_mod

    adapter = _JsonAdapter('[{"text": "Shin hunts the boar.", "moment": "hunt"}]')
    cast = chars.cast_script_block([_entry("Shin", ["Shinu"], "hunter")])
    script_mod.generate_script(
        adapter, "fake-text:8b", "The boy hunts.", "Test Manga", 2.0, cast=cast
    )
    prompt = adapter.calls[-1]["prompt"]
    assert "CAST — characters" in prompt
    assert "- Shin (also Shinu) — hunter" in prompt
    assert "even where the recap leaves them unnamed" in prompt


def test_generate_script_prompt_clean_without_cast():
    from entertainment_harness.video import script as script_mod

    adapter = _JsonAdapter('[{"text": "The boy hunts the boar.", "moment": "hunt"}]')
    script_mod.generate_script(
        adapter, "fake-text:8b", "The boy hunts.", "Test Manga", 2.0
    )
    assert "CAST" not in adapter.calls[-1]["prompt"]
