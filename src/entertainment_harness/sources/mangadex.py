"""MangaDex client: search and paginated chapter feed.

Public REST API, no auth needed for reading. Polite behavior: descriptive
User-Agent and retry/backoff on 429/5xx (honoring Retry-After).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from entertainment_harness.config import Config
from entertainment_harness.sources.utils import canonical_image_ext, page_dest_from_url

BASE_URL = "https://api.mangadex.org"
USER_AGENT = "entertainment-harness/0.1 (personal local manga recap tool)"
PAGE_SIZE = 100
MAX_RETRIES = 4

CONTENT_RATINGS = ["safe", "suggestive", "erotica"]

_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class Series:
    id: str
    title: str
    alt_titles: list[str] = field(default_factory=list)
    year: int | None = None
    status: str | None = None


@dataclass
class Chapter:
    id: str
    chapter: str | None  # chapter number as string, may be None/"0"
    title: str | None
    lang: str
    pages: int
    published_at: str | None = None


class MangaDexClient:
    def __init__(
        self,
        config: Config | None = None,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        # config is accepted for the shared plugin constructor contract; there
        # is no [sources.mangadex] section yet.
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

    def _request(self, method: str, path: str, **kwargs) -> dict:
        return self._request_raw(method, path, **kwargs).json()

    def search(self, title: str, limit: int = 20) -> list[Series]:
        data = self._request(
            "GET", "/manga", params={"title": title, "limit": limit}
        )
        return [self._parse_series(m) for m in data.get("data", [])]

    def get_series(self, manga_id: str) -> Series:
        data = self._request("GET", f"/manga/{manga_id}")
        return self._parse_series(data["data"])

    def chapters(self, manga_id: str, langs: list[str] | None = None) -> list[Chapter]:
        chapters: list[Chapter] = []
        offset = 0
        while True:
            params: dict = {
                "order[chapter]": "asc",
                "limit": PAGE_SIZE,
                "offset": offset,
                "contentRating[]": CONTENT_RATINGS,
            }
            if langs:
                params["translatedLanguage[]"] = list(langs)
            data = self._request("GET", f"/manga/{manga_id}/feed", params=params)
            batch = [self._parse_chapter(c) for c in data.get("data", [])]
            chapters.extend(batch)
            offset += len(batch)
            if offset >= data.get("total", offset) or not batch:
                break
        return chapters

    def page_urls(self, chapter_id: str) -> list[str]:
        """Resolve page image URLs via the at-home server endpoint."""
        data = self._request("GET", f"/at-home/server/{chapter_id}")
        base = data["baseUrl"]
        chapter = data["chapter"]
        chapter_hash = chapter["hash"]
        return [
            f"{base}/data/{chapter_hash}/{filename}"
            for filename in chapter.get("data", [])
        ]

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

    @staticmethod
    def _parse_series(m: dict) -> Series:
        attrs = m.get("attributes", {})
        title = attrs.get("title", {})
        title_text = title.get("en") or next(iter(title.values()), "")
        alt = [
            next(iter(t.values()))
            for t in attrs.get("altTitles", [])
            if t
        ]
        return Series(
            id=m["id"],
            title=title_text,
            alt_titles=alt,
            year=attrs.get("year"),
            status=attrs.get("status"),
        )

    @staticmethod
    def _parse_chapter(c: dict) -> Chapter:
        attrs = c.get("attributes", {})
        return Chapter(
            id=c["id"],
            chapter=attrs.get("chapter"),
            title=attrs.get("title"),
            lang=attrs.get("translatedLanguage", ""),
            pages=attrs.get("pages", 0),
            published_at=attrs.get("publishAt"),
        )
