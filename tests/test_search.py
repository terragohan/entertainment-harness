"""Tests for the online search summary pipeline.

All external network calls are mocked: DuckDuckGo search results, httpx page
fetches, and YouTube transcript API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entertainment_harness import db
from entertainment_harness.config import Config
from entertainment_harness.search import SearchResult
from entertainment_harness.search import duckduckgo, extractor, pipeline


class FakeAdapter:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[str] = []
        self.index = 0

    def generate(self, model: str, prompt: str, images=None) -> str:
        self.calls.append(prompt)
        if self.index < len(self.responses):
            resp = self.responses[self.index]
            self.index += 1
            return resp
        return "summary"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    conn.execute(
        "INSERT INTO series (id, title, source, source_id, status, added_at)"
        " VALUES ('s1', 'Test Manga', 'mangadex', 's1', 'ongoing', 'now')"
    )
    conn.commit()
    yield conn
    conn.close()


def test_classify_recognizes_sources():
    assert duckduckgo._classify("https://www.reddit.com/r/manga/comments/x/") == "reddit"
    assert duckduckgo._classify("https://youtu.be/abc123") == "youtube"
    assert duckduckgo._classify("https://myblog.wordpress.com/post") == "blog"
    assert duckduckgo._classify("https://example.com/") == "unknown"


def test_source_priority_sorts_results():
    results = [
        SearchResult("yt", "https://youtu.be/1", "snip", "youtube"),
        SearchResult("rd", "https://reddit.com/r/x", "snip", "reddit"),
        SearchResult("bl", "https://blog.com/x", "snip", "blog"),
    ]
    sorted_results = sorted(results, key=lambda r: duckduckgo._source_priority(r.source_type))
    assert [r.source_type for r in sorted_results] == ["reddit", "blog", "youtube"]


def test_extract_youtube_video_id():
    assert extractor._youtube_video_id("https://www.youtube.com/watch?v=ABC123") == "ABC123"
    assert extractor._youtube_video_id("https://youtu.be/ABC123") == "ABC123"
    assert extractor._youtube_video_id("https://example.com/") is None


def test_strip_html_tags():
    raw = "<html><body><p>Hello  <b>world</b>!</p></body></html>"
    text = extractor._strip_html_tags(raw)
    assert "Hello" in text
    assert "world" in text
    assert "<" not in text


def test_gather_sources_filters_short_text(monkeypatch, tmp_path):
    def fake_extract_text(url, source_type=None):
        return "This is a long enough summary text. " * 10

    monkeypatch.setattr(extractor, "extract_text", fake_extract_text)
    results = [
        SearchResult("title", "https://blog.com/a", "snippet", "blog"),
        SearchResult("short", "https://x.com/b", "tiny", "unknown"),
    ]
    # Patch fake_extract_text for short to return tiny text
    def fake_extract_text2(url, source_type=None):
        if "x.com" in url:
            return "hi"
        return "This is a long enough summary text. " * 10

    monkeypatch.setattr(extractor, "extract_text", fake_extract_text2)
    sources = extractor.gather_sources(results, max_sources=2)
    assert len(sources) == 1
    assert sources[0].source_type == "blog"


def test_synthesize_online_summary_builds_prompt_and_returns_sources(monkeypatch):
    results = [
        SearchResult("Reddit post", "https://reddit.com/r/x", "discussion", "reddit"),
    ]

    class FakeProvider:
        name = "duckduckgo"
        def search(self, query, max_results=10):
            return results

    monkeypatch.setattr(pipeline, "get_provider", lambda name, config=None: FakeProvider())
    monkeypatch.setattr(
        pipeline, "gather_sources",
        lambda results, max_sources: [
            extractor.ExtractedSource("Reddit post", "https://reddit.com/r/x", "reddit", "Shin trains." * 20)
        ],
    )

    adapter = FakeAdapter(["Online summary text."])
    summary, sources = pipeline.synthesize_online_summary(
        adapter, "fake", "Test Manga", Config()
    )
    assert summary == "Online summary text."
    assert sources[0]["url"] == "https://reddit.com/r/x"
    assert "Test Manga" in adapter.calls[0]
    assert "Shin trains" in adapter.calls[0]


def test_store_online_summary_updates_context_and_table(conn):
    pipeline.store_online_summary(
        conn, "s1", "duckduckgo", "Test Manga",
        "Shin trains.", [{"title": "t", "url": "u", "source_type": "reddit"}],
        steering_prompt="focus on action",
    )
    row = conn.execute("SELECT * FROM online_summaries WHERE series_id = 's1'").fetchone()
    assert row["provider"] == "duckduckgo"
    assert row["summary"] == "Shin trains."
    assert json.loads(row["sources_json"])[0]["url"] == "u"
    ctx = conn.execute("SELECT rolling_summary FROM series_context WHERE series_id = 's1'").fetchone()
    assert ctx["rolling_summary"] == "Shin trains."


def test_duckduckgo_provider_uses_ddgs(monkeypatch):
    """The provider must import DDGS from the renamed `ddgs` package (the old
    `duckduckgo_search` package returns 0 results)."""
    import sys
    import types

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results=10, region=None):
            return [
                {"href": "https://reddit.com/r/x", "title": "t", "body": "b"},
            ]

    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)
    results = duckduckgo.DuckDuckGoSearchProvider().search("query")
    assert len(results) == 1
    assert results[0].url == "https://reddit.com/r/x"
    assert results[0].source_type == "reddit"


def test_extract_youtube_uses_instance_api(monkeypatch):
    """youtube-transcript-api >= 1.0 removed the static get_transcript();
    extraction must use the instance .fetch() API and snippet .text attrs."""
    import sys
    import types
    from dataclasses import dataclass

    @dataclass
    class FakeSnippet:
        text: str

    class FakeYouTubeTranscriptApi:
        def fetch(self, video_id):
            assert video_id == "abc123"
            return [FakeSnippet("hello"), FakeSnippet("world")]

    fake_module = types.ModuleType("youtube_transcript_api")
    fake_module.YouTubeTranscriptApi = FakeYouTubeTranscriptApi
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)
    text = extractor.extract_text("https://www.youtube.com/watch?v=abc123")
    assert text == "hello world"
