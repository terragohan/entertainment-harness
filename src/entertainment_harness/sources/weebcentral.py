"""WeebCentralClient — weebcentral.com scanlation aggregator (English only).

No official API; this scrapes the site's server-rendered HTML endpoints:

- search:   POST /search/simple?location=main        (form field "text")
- series:   GET  /series/<id>                        (slug optional)
- chapters: GET  /series/<id>/full-chapter-list
- pages:    GET  /chapters/<id>/images?is_prev=False&reading_style=long_strip

Everything hosted here is English scanlation, so Chapter.lang is always "en".
Same polite behavior as the MangaDex client: descriptive User-Agent and
retry/backoff on 429/5xx. HTML scraping is inherently fragile — if the site
restructures, the regexes below are what to revisit.
"""

from __future__ import annotations

import re
import time
from html import unescape
from pathlib import Path

import httpx

from entertainment_harness.config import Config
from entertainment_harness.sources.mangadex import USER_AGENT, Chapter, Series
from entertainment_harness.sources.utils import canonical_image_ext, page_dest_from_url

BASE_URL = "https://weebcentral.com"
MAX_RETRIES = 4

_ID_RE = re.compile(r"^[0-9A-Z]{26}$")  # series/chapter ids are ULIDs
_SEARCH_RE = re.compile(
    r'href="https://weebcentral\.com/series/([0-9A-Z]{26})/[^"]*"[^>]*>'
    r'.*?<img[^>]*alt="([^"]+) cover"',
    re.DOTALL,
)
_CHAPTER_RE = re.compile(
    r'<a href="/chapters/([0-9A-Z]{26})"[^>]*>'
    r'.*?<span class="">([^<]+)</span>'
    r'.*?<time[^>]*datetime="([^"]+)"',
    re.DOTALL,
)
_CHAPTER_NUM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
_IMG_RE = re.compile(r'<img[^>]*src="(https://[^"]+\.(?:png|jpg|jpeg|webp))"')


def _strip_tags(fragment: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _field(html: str, label: str) -> str | None:
    """Value of a "<strong>LABEL: </strong> value</li>" list item."""
    match = re.search(
        rf"<strong>{re.escape(label)}:\s*</strong>\s*(.*?)</li>", html, re.DOTALL
    )
    return _strip_tags(match.group(1)) if match else None


class WeebCentralClient:
    def __init__(
        self,
        config: Config | None = None,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        # config is accepted for the shared plugin constructor contract; there
        # is no [sources.weebcentral] section yet.
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )

    @staticmethod
    def looks_like_id(query: str) -> bool:
        """Whether a CLI argument is a series id rather than a title to search."""
        return bool(_ID_RE.match(query))

    def _request_raw(self, method: str, path: str, **kwargs) -> httpx.Response:
        delay = 1.0
        for attempt in range(MAX_RETRIES + 1):
            resp = self._client.request(method, path, **kwargs)
            if resp.status_code < 400:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == MAX_RETRIES:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
        raise AssertionError("unreachable")

    def _get(self, path: str, **kwargs) -> str:
        return self._request_raw("GET", path, **kwargs).text

    def search(self, title: str) -> list[Series]:
        html = self._request_raw(
            "POST", "/search/simple", params={"location": "main"},
            data={"text": title},
        ).text
        return [Series(id=sid, title=unescape(name))
                for sid, name in _SEARCH_RE.findall(html)]

    def get_series(self, series_id: str) -> Series:
        html = self._get(f"/series/{series_id}")
        match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse series page for {series_id!r}")
        released = _field(html, "Released")
        assoc = re.search(
            r"Associated Name\(s\).*?<ul[^>]*>(.*?)</ul>", html, re.DOTALL
        )
        alt = [_strip_tags(m) for m in re.findall(r"<li[^>]*>(.*?)</li>", assoc.group(1), re.DOTALL)] if assoc else []
        return Series(
            id=series_id,
            title=_strip_tags(match.group(1)),
            alt_titles=[a for a in alt if a],
            year=int(released) if released and released.isdigit() else None,
            status=_field(html, "Status"),
        )

    def chapters(self, manga_id: str, langs: list[str] | None = None) -> list[Chapter]:
        """All hosted chapters; the site is English-only, so if a language
        list is supplied and does not include "en", nothing matches."""
        if langs is not None and "en" not in langs:
            return []
        html = self._get(f"/series/{manga_id}/full-chapter-list")
        result = []
        for chapter_id, label, published in _CHAPTER_RE.findall(html):
            num = _CHAPTER_NUM_RE.search(label)
            result.append(Chapter(
                id=chapter_id,
                chapter=num.group(1) if num else None,
                title=None,
                lang="en",
                pages=0,  # page count is only known from the reader endpoint
                published_at=published or None,
            ))
        return result

    def page_urls(self, chapter_id: str) -> list[str]:
        html = self._get(
            f"/chapters/{chapter_id}/images",
            params={"is_prev": "False", "reading_style": "long_strip"},
        )
        return _IMG_RE.findall(html)

    def download_pages(self, chapter_id: str, dest_dir: Path) -> list[Path]:
        """Download all pages of a chapter into dest_dir as page-NNN.<ext>.

        Existing files are skipped (cache). The extension is derived from the
        actual image bytes, not the URL, because some hosts serve JPEGs from
        `.png` URLs. Returns the sorted page paths.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        urls = self.page_urls(chapter_id)
        paths: list[Path] = []
        for index, url in enumerate(urls, start=1):
            url_ext = Path(url.split("?", 1)[0]).suffix or ".jpg"
            dest = dest_dir / page_dest_from_url(url, index)
            if not dest.exists():
                resp = self._request_raw("GET", url)
                data = resp.content
                ext = canonical_image_ext(data, fallback=url_ext)
                dest = dest_dir / f"page-{index:03d}{ext}"
                dest.write_bytes(data)
            paths.append(dest)
        return paths
