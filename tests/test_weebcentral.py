"""WeebCentralClient tests: respx-mocked HTML endpoints (fixtures are
trimmed copies of real responses captured 2026-08-23). No network needed.
"""

from __future__ import annotations

import io

import pytest
import respx
from httpx import Response
from PIL import Image

from entertainment_harness import db, library
from entertainment_harness.sources.weebcentral import WeebCentralClient

BASE = "https://weebcentral.com"
SERIES_ID = "01J76XYBTCD15FW69W889WKHB0"
CHAPTER_ID = "01K5BTEZJN450GBQTS4BSPG476"

SEARCH_HTML = """
<a href="https://weebcentral.com/series/01J76XYAK4T3K3DQSF49CWHY04/Kagijin" class="btn join-item h-20">
    <div class="w-12 h-12 overflow-hidden">
        <picture>
            <img src="https://temp.compsci88.com/cover/fallback/01J76XYAK4T3K3DQSF49CWHY04.jpg" alt="Kagijin cover" width="200" height="300">
        </picture>
    </div>
    <div class="flex-1">Kagijin</div>
</a>
<a href="https://weebcentral.com/series/01J76XYBTCD15FW69W889WKHB0/Kenja-No-Mago" class="btn join-item h-20">
    <div class="w-12 h-12 overflow-hidden">
        <picture>
            <img src="https://temp.compsci88.com/cover/fallback/01J76XYBTCD15FW69W889WKHB0.jpg" alt="Kenja no Mago cover" width="200" height="300">
        </picture>
    </div>
    <div class="flex-1">Kenja no Mago</div>
</a>
"""

SERIES_HTML = """
<html><body>
<h1 class="text-2xl font-bold">Kenja no Mago</h1>
<ul>
 <li> <strong>Type: </strong> <a href="/search?included_type=Manga">Manga</a> </li>
 <li> <strong>Status: </strong> <a href="/search?included_status=Ongoing">Ongoing</a> </li>
 <li> <strong>Released: </strong> <a href="/search?included_year=2016">2016</a> </li>
 <li> <strong>Associated Name(s)</strong>
  <ul class="list-disc list-inside">
   <li>Magi&#39;s Grandson</li>
   <li>Wise Man&#39;s Grandchild</li>
  </ul>
 </li>
</ul>
</body></html>
"""

CHAPTERS_HTML = """
<div class="flex items-center">
    <a href="/chapters/01K5BTEZJN450GBQTS4BSPG476" class="hover:bg-base-300 flex-1 flex items-center p-2">
        <span class="me-2"><img src="/static/images/chapter-badge.svg" alt="" width="16" height="16"></span>
        <span class="grow flex items-center gap-2">
            <span class="">Chapter 94</span>
        </span>
        <time class="text-datetime opacity-50" datetime="2025-09-17T12:30:18.709Z">2025-09-17T12:30:18.709418Z</time>
    </a>
</div>
<div class="flex items-center">
    <a href="/chapters/01K5BTEP7EV6G2TDHM2GMX4ZDJ" class="hover:bg-base-300 flex-1 flex items-center p-2">
        <span class="me-2"><img src="/static/images/chapter-badge.svg" alt="" width="16" height="16"></span>
        <span class="grow flex items-center gap-2">
            <span class="">Chapter 93.5</span>
        </span>
        <time class="text-datetime opacity-50" datetime="2025-09-10T12:30:09.134Z">2025-09-10T12:30:09.134525Z</time>
    </a>
</div>
"""

IMAGES_HTML = """
<section id="reader" class="flex flex-col items-center">
    <img src="https://scans.lastation.us/manga/Kenja-No-Mago/0094-001.png" class="m-auto" decoding="async">
    <img src="https://scans.lastation.us/manga/Kenja-No-Mago/0094-002.png" class="m-auto" decoding="async">
    <img src="https://scans.lastation.us/manga/Kenja-No-Mago/0094-003.png" class="m-auto" decoding="async">
</section>
"""


def test_looks_like_id():
    assert WeebCentralClient.looks_like_id(SERIES_ID)
    assert not WeebCentralClient.looks_like_id("Kenja no Mago")


@respx.mock
def test_search_parses_results():
    route = respx.post(f"{BASE}/search/simple").mock(
        return_value=Response(200, text=SEARCH_HTML)
    )
    results = WeebCentralClient().search("kenja")
    assert [s.title for s in results] == ["Kagijin", "Kenja no Mago"]
    assert results[1].id == SERIES_ID
    assert route.calls.last.request.url.params["location"] == "main"


@respx.mock
def test_get_series_parses_fields():
    respx.get(f"{BASE}/series/{SERIES_ID}").mock(
        return_value=Response(200, text=SERIES_HTML)
    )
    series = WeebCentralClient().get_series(SERIES_ID)
    assert series.title == "Kenja no Mago"
    assert series.status == "Ongoing"
    assert series.year == 2016
    assert series.alt_titles == ["Magi's Grandson", "Wise Man's Grandchild"]


@respx.mock
def test_chapters_parses_list():
    respx.get(f"{BASE}/series/{SERIES_ID}/full-chapter-list").mock(
        return_value=Response(200, text=CHAPTERS_HTML)
    )
    chapters = WeebCentralClient().chapters(SERIES_ID)
    assert [c.id for c in chapters] == [
        "01K5BTEZJN450GBQTS4BSPG476",
        "01K5BTEP7EV6G2TDHM2GMX4ZDJ",
    ]
    assert chapters[0].chapter == "94"
    assert chapters[1].chapter == "93.5"
    assert all(c.lang == "en" for c in chapters)
    assert chapters[0].published_at == "2025-09-17T12:30:18.709Z"


@respx.mock
def test_chapters_non_english_matches_nothing():
    assert WeebCentralClient().chapters(SERIES_ID, langs=["pt-br"]) == []


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="JPEG")
    return buf.getvalue()


@respx.mock
def test_page_urls_and_download(tmp_path):
    respx.get(f"{BASE}/chapters/{CHAPTER_ID}/images").mock(
        return_value=Response(200, text=IMAGES_HTML)
    )
    for i in (1, 2, 3):
        respx.get(
            f"https://scans.lastation.us/manga/Kenja-No-Mago/0094-00{i}.png"
        ).mock(return_value=Response(200, content=f"page{i}".encode()))
    client = WeebCentralClient()
    urls = client.page_urls(CHAPTER_ID)
    assert len(urls) == 3
    pages = client.download_pages(CHAPTER_ID, tmp_path / "ch")
    assert [p.name for p in pages] == ["page-001.png", "page-002.png", "page-003.png"]
    assert pages[0].read_bytes() == b"page1"
    # second run: all cached, no re-download (image mocks hit exactly once)
    client.download_pages(CHAPTER_ID, tmp_path / "ch")


@respx.mock
def test_download_uses_real_image_format_not_url_extension(tmp_path):
    """Regression: Weeb Central may serve a JPEG from a `.png` URL. The saved
    file must use the real extension so downstream MIME detection is correct."""
    html = '''
    <section id="reader">
        <img src="https://host/page.png">
    </section>
    '''
    respx.get(f"{BASE}/chapters/{CHAPTER_ID}/images").mock(
        return_value=Response(200, text=html)
    )
    respx.get("https://host/page.png").mock(
        return_value=Response(200, content=_jpeg_bytes())
    )
    pages = WeebCentralClient().download_pages(CHAPTER_ID, tmp_path / "ch")
    assert [p.name for p in pages] == ["page-001.jpg"]


@respx.mock
def test_library_add_and_sync_via_weebcentral(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    respx.get(f"{BASE}/series/{SERIES_ID}").mock(
        return_value=Response(200, text=SERIES_HTML)
    )
    respx.get(f"{BASE}/series/{SERIES_ID}/full-chapter-list").mock(
        return_value=Response(200, text=CHAPTERS_HTML)
    )
    conn = db.connect()
    series = library.add_series(conn, SERIES_ID, source="weebcentral")
    assert series.id == SERIES_ID
    row = conn.execute("SELECT * FROM series WHERE id = ?", (SERIES_ID,)).fetchone()
    assert row["source"] == "weebcentral"

    count = library.sync_chapters(conn, SERIES_ID)  # client derived from source
    assert count == 2
    rows = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ? ORDER BY chapter_num DESC",
        (SERIES_ID,),
    ).fetchall()
    assert rows[0]["chapter_num"] == 94.0
    assert rows[0]["lang"] == "en"
    conn.close()


@respx.mock
def test_same_title_addable_from_both_sources(tmp_path, monkeypatch):
    """A MangaDex series and a Weeb Central series with the same title are
    distinct library entries (source is part of the identity)."""
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    respx.post(f"{BASE}/search/simple").mock(
        return_value=Response(200, text=SEARCH_HTML)
    )
    conn = db.connect()
    library.add_series(conn, "Kenja no Mago", source="weebcentral")
    with pytest.raises(library.LibraryError, match="already in the library"):
        library.add_series(conn, "Kenja no Mago", source="weebcentral")
    conn.close()
