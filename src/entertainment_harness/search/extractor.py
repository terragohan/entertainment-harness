"""Fetch and extract readable text from blogs, Reddit, and YouTube transcripts."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from entertainment_harness.search import SearchResult

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MAX_SIZE_BYTES = 1_000_000
FETCH_TIMEOUT = 20.0


@dataclass
class ExtractedSource:
    title: str
    url: str
    source_type: str
    text: str


def _is_youtube(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc in {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}


def _is_reddit(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc in {"reddit.com", "www.reddit.com", "old.reddit.com"}


def _youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0] or None
    if "v" in (q := _parse_qs(parsed.query)):
        return q["v"][0]
    if parsed.path.startswith("/embed/"):
        return parsed.path.split("/")[2]
    return None


def _parse_qs(query: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for part in query.split("&"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result.setdefault(key, []).append(value)
    return result


def _strip_html_tags(raw: str) -> str:
    """Minimal HTML-to-text using stdlib html.parser."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fetch(url: str) -> str:
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        if len(resp.content) > MAX_SIZE_BYTES:
            raise ValueError(f"Page too large (> {MAX_SIZE_BYTES} bytes)")
        return resp.text


def _extract_youtube(url: str) -> str:
    video_id = _youtube_video_id(url)
    if not video_id:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError("youtube-transcript-api is not installed") from exc
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    except Exception:
        return ""
    return " ".join(entry.text for entry in transcript)


def _extract_reddit(url: str) -> str:
    parsed = urlparse(url)
    json_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}.json"
    try:
        text = _fetch(json_url)
    except Exception:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ""
    pieces: list[str] = []
    # Reddit JSON is [post, comments].
    for listing in data if isinstance(data, list) else [data]:
        for child in listing.get("data", {}).get("children", []):
            post = child.get("data", {})
            if post.get("selftext"):
                pieces.append(post["selftext"])
            if post.get("title"):
                pieces.append(post["title"])
            body = post.get("body")
            if body:
                pieces.append(body)
    return "\n\n".join(pieces)


def _extract_generic(url: str) -> str:
    raw = _fetch(url)
    return _strip_html_tags(raw)


def extract_text(url: str, source_type: str | None = None) -> str:
    if _is_youtube(url):
        return _extract_youtube(url)
    if _is_reddit(url):
        return _extract_reddit(url)
    return _extract_generic(url)


def gather_sources(
    results: list[SearchResult],
    max_sources: int = 6,
) -> list[ExtractedSource]:
    sources: list[ExtractedSource] = []
    for result in results:
        if len(sources) >= max_sources:
            break
        try:
            text = extract_text(result.url, result.source_type)
        except Exception:
            continue
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) < 100:
            continue
        sources.append(
            ExtractedSource(
                title=result.title,
                url=result.url,
                source_type=result.source_type,
                text=cleaned,
            )
        )
    return sources
