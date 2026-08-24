"""MangaDex client tests with respx-mocked responses."""

from __future__ import annotations

import io

import respx
from httpx import Response
from PIL import Image

from entertainment_harness.sources.mangadex import MangaDexClient

BASE = "https://api.mangadex.org"

SEARCH_RESPONSE = {
    "result": "ok",
    "total": 1,
    "data": [
        {
            "id": "835feda4-2db0-4753-8249-4575a3ceffe2",
            "attributes": {
                "title": {"en": "Kenja no Mago"},
                "altTitles": [{"ja": "賢者の孫"}, {"en": "Wise Man's Grandchild"}],
                "year": 2016,
                "status": "ongoing",
            },
        }
    ],
}


def _chapter(idx: int, lang: str = "pt-br") -> dict:
    return {
        "id": f"chapter-{idx}",
        "attributes": {
            "chapter": str(idx),
            "title": f"Chapter {idx}",
            "translatedLanguage": lang,
            "pages": 16,
            "publishAt": "2026-01-01T00:00:00+00:00",
        },
    }


@respx.mock
def test_search_parses_series():
    route = respx.get(f"{BASE}/manga").mock(
        return_value=Response(200, json=SEARCH_RESPONSE)
    )
    client = MangaDexClient()
    results = client.search("Kenja no Mago")
    assert route.called
    request = route.calls.last.request
    assert request.url.params["title"] == "Kenja no Mago"
    assert "entertainment-harness" in request.headers["User-Agent"]

    assert len(results) == 1
    series = results[0]
    assert series.id == "835feda4-2db0-4753-8249-4575a3ceffe2"
    assert series.title == "Kenja no Mago"
    assert "Wise Man's Grandchild" in series.alt_titles
    assert series.year == 2016
    assert series.status == "ongoing"


@respx.mock
def test_chapters_paginates_until_total():
    manga_id = "835feda4-2db0-4753-8249-4575a3ceffe2"
    page1 = {
        "result": "ok",
        "total": 3,
        "limit": 2,
        "offset": 0,
        "data": [_chapter(0), _chapter(1)],
    }
    page2 = {
        "result": "ok",
        "total": 3,
        "limit": 2,
        "offset": 2,
        "data": [_chapter(2)],
    }

    def respond(request):
        offset = int(request.url.params.get("offset", "0"))
        assert request.url.params["order[chapter]"] == "asc"
        return Response(200, json=page1 if offset == 0 else page2)

    respx.get(f"{BASE}/manga/{manga_id}/feed").mock(side_effect=respond)
    client = MangaDexClient()
    chapters = client.chapters(manga_id)
    assert [c.chapter for c in chapters] == ["0", "1", "2"]
    assert chapters[0].lang == "pt-br"
    assert chapters[0].pages == 16
    assert chapters[0].published_at == "2026-01-01T00:00:00+00:00"


@respx.mock
def test_chapters_lang_filter_is_sent():
    manga_id = "abc"
    route = respx.get(f"{BASE}/manga/{manga_id}/feed").mock(
        return_value=Response(
            200, json={"result": "ok", "total": 1, "data": [_chapter(0)]}
        )
    )
    client = MangaDexClient()
    chapters = client.chapters(manga_id, langs=["pt-br"])
    assert len(chapters) == 1
    assert route.calls.last.request.url.params["translatedLanguage[]"] == "pt-br"


@respx.mock
def test_retries_on_429_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "entertainment_harness.sources.mangadex.time.sleep", sleeps.append
    )
    route = respx.get(f"{BASE}/manga").mock(
        side_effect=[
            Response(429, headers={"Retry-After": "0"}),
            Response(200, json=SEARCH_RESPONSE),
        ]
    )
    client = MangaDexClient()
    results = client.search("Kenja no Mago")
    assert len(results) == 1
    assert route.call_count == 2
    assert sleeps  # backoff happened


AT_HOME_RESPONSE = {
    "result": "ok",
    "baseUrl": "https://uploads.mangadex.org",
    "chapter": {"hash": "abc123", "data": ["p1.jpg", "p2.png"], "dataSaver": []},
}


@respx.mock
def test_page_urls_from_at_home():
    respx.get(f"{BASE}/at-home/server/ch-1").mock(
        return_value=Response(200, json=AT_HOME_RESPONSE)
    )
    client = MangaDexClient()
    urls = client.page_urls("ch-1")
    assert urls == [
        "https://uploads.mangadex.org/data/abc123/p1.jpg",
        "https://uploads.mangadex.org/data/abc123/p2.png",
    ]


@respx.mock
def test_download_pages_caches(tmp_path):
    respx.get(f"{BASE}/at-home/server/ch-1").mock(
        return_value=Response(200, json=AT_HOME_RESPONSE)
    )
    img1 = respx.get("https://uploads.mangadex.org/data/abc123/p1.jpg").mock(
        return_value=Response(200, content=b"jpg-bytes")
    )
    img2 = respx.get("https://uploads.mangadex.org/data/abc123/p2.png").mock(
        return_value=Response(200, content=b"png-bytes")
    )
    client = MangaDexClient()
    dest = tmp_path / "manga" / "series-1" / "ch-1"

    paths = client.download_pages("ch-1", dest)
    assert [p.name for p in paths] == ["page-001.jpg", "page-002.png"]
    assert (dest / "page-001.jpg").read_bytes() == b"jpg-bytes"
    assert (dest / "page-002.png").read_bytes() == b"png-bytes"
    assert img1.call_count == 1 and img2.call_count == 1

    # second run: files exist -> no image re-download
    paths2 = client.download_pages("ch-1", dest)
    assert paths2 == paths
    assert img1.call_count == 1 and img2.call_count == 1


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="JPEG")
    return buf.getvalue()


@respx.mock
def test_download_uses_real_image_format_not_url_extension(tmp_path):
    """Regression: a host may return JPEG bytes for a `.png` filename. The
    saved file must use the real extension so downstream MIME detection is
    correct."""
    at_home = {
        "result": "ok",
        "baseUrl": "https://uploads.mangadex.org",
        "chapter": {"hash": "abc123", "data": ["page.png"], "dataSaver": []},
    }
    respx.get(f"{BASE}/at-home/server/ch-1").mock(
        return_value=Response(200, json=at_home)
    )
    respx.get("https://uploads.mangadex.org/data/abc123/page.png").mock(
        return_value=Response(200, content=_jpeg_bytes())
    )
    dest = tmp_path / "manga" / "series-1" / "ch-1"
    paths = MangaDexClient().download_pages("ch-1", dest)
    assert [p.name for p in paths] == ["page-001.jpg"]
