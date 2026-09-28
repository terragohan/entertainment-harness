"""Import local books and comics into the library ('eh import').

Sources are local files or user-supplied URLs pointing directly at a file —
no site scraping. Books (EPUB/TXT/MD) are split into text parts under
works/<work>/chapters/ch-NNN/source/ and summarized with the text-role model
only (no vision pass). Comics (CBZ/CBR/image folder) are extracted into the
same source/ page layout, so the recap/visuals/store paths work untouched.

EPUB/CBZ are parsed with the stdlib (zipfile + xml.etree + html.parser); CBR
needs a system 'unar' or 'unrar' binary. No new dependencies.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
import tempfile
import uuid
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

import httpx

from entertainment_harness.library import works
from entertainment_harness.config import Config
from entertainment_harness.db import utcnow

BOOK_EXTS = {".epub", ".txt", ".md"}
COMIC_EXTS = {".cbz", ".cbr"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
TXT_CHUNK_WORDS = 4000  # fallback split when a text has no chapter headings
MIN_PART_WORDS = 300  # smaller parts are merged into a neighbor


class ImportFailure(Exception):
    pass


# ---------------------------------------------------------------------------
# kind detection / input acquisition


def detect_kind(path: Path) -> str:
    """'book' or 'comic' from the file extension (directories are comics)."""
    if path.is_dir():
        return "comic"
    ext = path.suffix.lower()
    if ext in BOOK_EXTS:
        return "book"
    if ext in COMIC_EXTS:
        return "comic"
    raise ImportFailure(
        f"Cannot tell whether {path.name!r} is a book or a comic "
        f"(known: {', '.join(sorted(BOOK_EXTS | COMIC_EXTS))}). "
        "Pass --kind book|comic."
    )


def _filename_from_response(url: str, resp: httpx.Response) -> str:
    cd = resp.headers.get("content-disposition", "")
    match = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.I)
    if match:
        return unquote(match.group(1)).strip()
    name = Path(urlparse(str(resp.url)).path).name
    return unquote(name) if name else "download"


def fetch_url(url: str, dest_dir: Path, log=print) -> Path:
    """Download a direct file link into dest_dir. HTML pages are refused."""
    log(f"Downloading {url}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=120.0
    ) as resp:
        resp.raise_for_status()
        filename = _filename_from_response(url, resp)
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if not Path(filename).suffix and ctype == "text/html":
            raise ImportFailure(
                f"{url} serves a web page, not a file — pass a direct link to "
                "an EPUB/TXT/CBZ/CBR file."
            )
        dest = dest_dir / filename
        written = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    dest.unlink(missing_ok=True)
                    raise ImportFailure(
                        f"Download exceeds the 500 MB cap: {url}"
                    )
                fh.write(chunk)
    log(f"  saved {dest.name} ({written / 1e6:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# book parsing


class _TextExtractor(HTMLParser):
    """Strip XHTML to plain text; block-level tags become newlines."""

    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _html_to_text(raw: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    return parser.text()


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = container.find(".//c:rootfile", ns)
    if rootfile is None or not rootfile.get("full-path"):
        raise ImportFailure("Malformed EPUB: no rootfile in container.xml")
    return rootfile.get("full-path")


def read_epub_author(path: Path) -> str | None:
    """The EPUB's first dc:creator, if any — the byline for video credits."""
    ns = {
        "opf": "http://www.idpf.org/2007/opf",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    try:
        with zipfile.ZipFile(path) as zf:
            opf = ElementTree.fromstring(zf.read(_opf_path(zf)))
    except (zipfile.BadZipFile, ImportFailure, ElementTree.ParseError, KeyError):
        return None
    creator = opf.find(".//opf:metadata/dc:creator", ns)
    if creator is None or not (creator.text or "").strip():
        return None
    return creator.text.strip()


def read_epub(path: Path) -> tuple[list[str], bytes | None, str | None]:
    """Return (spine-ordered part texts, cover bytes, cover extension)."""
    ns = {"opf": "http://www.idpf.org/2007/opf"}
    try:
        with zipfile.ZipFile(path) as zf:
            opf_path = _opf_path(zf)
            opf_dir = PurePosixPath(opf_path).parent
            opf = ElementTree.fromstring(zf.read(opf_path))
            manifest = {
                item.get("id"): item
                for item in opf.findall(".//opf:manifest/opf:item", ns)
            }

            def read_href(item) -> bytes:
                href = unquote(item.get("href"))
                full = href if str(opf_dir) == "." else str(opf_dir / href)
                return zf.read(full)

            parts = []
            for ref in opf.findall(".//opf:spine/opf:itemref", ns):
                item = manifest.get(ref.get("idref"))
                if item is None:
                    continue
                media_type = item.get("media-type", "")
                if "html" not in media_type and "xml" not in media_type:
                    continue
                try:
                    parts.append(_html_to_text(read_href(item)))
                except KeyError:
                    continue

            cover_item = next(
                (
                    item
                    for item in manifest.values()
                    if "cover-image" in (item.get("properties") or "")
                ),
                None,
            )
            if cover_item is None:
                meta = opf.find(".//opf:metadata/opf:meta[@name='cover']", ns)
                if meta is not None:
                    cover_item = manifest.get(meta.get("content"))
            cover = cover_ext = None
            if cover_item is not None:
                try:
                    cover = read_href(cover_item)
                    cover_ext = Path(cover_item.get("href")).suffix.lower()
                except KeyError:
                    pass
    except zipfile.BadZipFile as exc:
        raise ImportFailure(f"{path.name} is not a valid EPUB (zip) file") from exc
    return parts, cover, cover_ext


_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chapter|part|book|prologue|epilogue)\b", re.IGNORECASE
)


def word_chunks(text: str, chunk_words: int) -> list[str]:
    parts, current, count = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        words = len(para.split())
        if current and count + words > chunk_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        parts.append("\n\n".join(current))
    return parts


def _merge_small(parts: list[str], min_words: int = MIN_PART_WORDS) -> list[str]:
    """Merge parts smaller than min_words into the following (or previous)
    part — EPUB spines are littered with title/half-title pages."""
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1].split()) < min_words:
            merged[-1] = merged[-1] + "\n\n" + part
        else:
            merged.append(part)
    if len(merged) > 1 and len(merged[-1].split()) < min_words:
        merged[-2] = merged[-2] + "\n\n" + merged[-1]
        merged.pop()
    return merged


def split_text(text: str, chunk_words: int = TXT_CHUNK_WORDS) -> list[str]:
    """Split a plain-text/markdown book into parts: on chapter headings when
    present, else into ~chunk_words blocks on paragraph boundaries."""
    parts, current = [], []
    for line in text.splitlines():
        if _CHAPTER_HEADING_RE.match(line) and current:
            parts.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        parts.append("\n".join(current))
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = word_chunks(text, chunk_words)
    return _merge_small(parts)


# ---------------------------------------------------------------------------
# comic extraction


def _collect_images(src_dir: Path) -> list[Path]:
    return sorted(
        p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def _extract_cbr(path: Path, dest_dir: Path) -> None:
    tool = shutil.which("unar") or shutil.which("unrar")
    if tool is None:
        raise ImportFailure(
            "CBR (RAR) comics need a system 'unar' or 'unrar' binary — "
            "install one (e.g. 'brew install unar') or convert to CBZ."
        )
    with tempfile.TemporaryDirectory() as tmp:
        cmd = (
            [tool, "-o", tmp, str(path)]
            if Path(tool).name == "unar"
            else [tool, "x", str(path), tmp + "/"]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ImportFailure(f"{Path(tool).name} failed on {path.name}")
        _place_images(_collect_images(Path(tmp)), dest_dir)


def _place_images(images: list[Path], dest_dir: Path) -> int:
    if not images:
        raise ImportFailure("No image pages found (expected jpg/png/webp).")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images, start=1):
        shutil.copyfile(image, dest_dir / f"page-{i:03d}{image.suffix.lower()}")
    return len(images)


# ---------------------------------------------------------------------------
# library insertion


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _insert_series(conn, title: str, kind: str) -> str:
    source_id = _slug(title)
    if conn.execute(
        "SELECT id FROM series WHERE source = 'import' AND source_id = ?",
        (source_id,),
    ).fetchone():
        raise ImportFailure(f"{title!r} is already imported.")
    series_id = uuid.uuid4().hex
    now = utcnow()
    conn.execute(
        "INSERT INTO series (id, title, alt_titles, source, source_id, status,"
        " added_at, kind) VALUES (?, ?, '', 'import', ?, 'imported', ?, ?)",
        (series_id, title, source_id, now, kind),
    )
    works.write_work_metadata(
        works.WorkMetadata(
            id=series_id,
            title=title,
            source="import",
            source_id=source_id,
            added_at=now,
            kind=kind,
            status="imported",
        )
    )
    return series_id


def _insert_chapter(
    conn, series_id: str, num: int, title: str, lang: str, pages: int | None
) -> str:
    chapter_id = f"{series_id}-{num:04d}"
    conn.execute(
        "INSERT INTO chapters (id, series_id, chapter_num, title, lang, pages,"
        " published_at, fetched_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
        (chapter_id, series_id, float(num), title, lang, pages, utcnow()),
    )
    return chapter_id


def import_work(
    conn: sqlite3.Connection,
    config: Config,
    source: str,
    title: str | None = None,
    kind: str | None = None,
    log=print,
) -> sqlite3.Row:
    """Import a book or comic from a local path or direct-file URL.

    Returns the new series row.
    """
    tmp: tempfile.TemporaryDirectory | None = None
    try:
        if source.startswith(("http://", "https://")):
            tmp = tempfile.TemporaryDirectory()
            path = fetch_url(source, Path(tmp.name), log)
        else:
            path = Path(source).expanduser()
            if not path.exists():
                raise ImportFailure(f"No such file or directory: {source}")
        kind = kind or detect_kind(path)
        if kind not in ("book", "comic"):
            raise ImportFailure(f"--kind must be 'book' or 'comic', got {kind!r}")
        title = title or re.sub(r"[-_]+", " ", path.stem).strip().title()
        series_id = _insert_series(conn, title, kind)

        if kind == "book":
            _import_book(conn, series_id, path, config, log)
            chapter_id = None
        else:
            chapter_id = _import_comic(conn, series_id, path, title, config, log)
        conn.commit()
        return conn.execute(
            "SELECT * FROM series WHERE id = ?", (series_id,)
        ).fetchone()
    except Exception:
        conn.rollback()
        raise
    finally:
        if tmp is not None:
            tmp.cleanup()


def _import_book(conn, series_id: str, path: Path, config: Config, log) -> None:
    ext = path.suffix.lower()
    if ext == ".epub":
        parts, cover, cover_ext = read_epub(path)
        author = read_epub_author(path)
        if author:
            meta = works.read_work_metadata(series_id)
            if meta is not None:
                meta.author = author
                works.write_work_metadata(meta)
    else:
        parts, cover, cover_ext = split_text(path.read_text(errors="replace")), None, None
    parts = _merge_small([p for p in (pt.strip() for pt in parts) if p])
    if not parts:
        raise ImportFailure(f"No text found in {path.name}.")

    book_dir = works.work_dir(series_id)
    book_dir.mkdir(parents=True, exist_ok=True)
    if cover:
        (book_dir / f"cover{cover_ext or '.jpg'}").write_bytes(cover)
    lang = config.library.langs[0] if config.library.langs else "en"
    now = utcnow()
    for i, part in enumerate(parts, start=1):
        chapter_id = _insert_chapter(conn, series_id, i, f"Part {i}", lang, None)
        # Metadata first: it creates the ch-NNN dir that source_dir resolves to.
        works.write_chapter_metadata(
            series_id,
            works.ChapterMetadata(
                id=chapter_id,
                chapter_num=float(i),
                title=f"Part {i}",
                lang=lang,
                pages=None,
                published_at=None,
                fetched_at=now,
            ),
        )
        part_dir = works.source_dir(series_id, chapter_id)
        part_dir.mkdir(parents=True, exist_ok=True)
        (part_dir / f"ch-{i:03d}.txt").write_text(part)
    log(f"  book: {len(parts)} part(s)"
        + (", cover extracted" if cover else ""))


def _import_comic(
    conn, series_id: str, path: Path, title: str, config: Config, log
) -> str:
    lang = config.library.langs[0] if config.library.langs else "en"
    chapter_id = _insert_chapter(conn, series_id, 1, title, lang, None)
    now = utcnow()
    # Metadata first: it creates the ch-NNN dir that source_dir resolves to.
    works.write_chapter_metadata(
        series_id,
        works.ChapterMetadata(
            id=chapter_id,
            chapter_num=1.0,
            title=title,
            lang=lang,
            pages=None,
            published_at=None,
            fetched_at=now,
        ),
    )
    dest_dir = works.source_dir(series_id, chapter_id)
    if path.is_dir():
        count = _place_images(_collect_images(path), dest_dir)
    elif path.suffix.lower() == ".cbz":
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                n for n in zf.namelist()
                if Path(n).suffix.lower() in IMAGE_EXTS and not n.startswith("__MACOSX")
            )
            if not names:
                raise ImportFailure(f"No image pages found in {path.name}.")
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, name in enumerate(names, start=1):
                dest = dest_dir / f"page-{i:03d}{Path(name).suffix.lower()}"
                dest.write_bytes(zf.read(name))
            count = len(names)
    else:  # .cbr
        _extract_cbr(path, dest_dir)
        count = len(list(dest_dir.iterdir()))
    conn.execute(
        "UPDATE chapters SET pages = ? WHERE id = ?", (count, chapter_id)
    )
    works.write_chapter_metadata(
        series_id,
        works.ChapterMetadata(
            id=chapter_id,
            chapter_num=1.0,
            title=title,
            lang=lang,
            pages=count,
            published_at=None,
            fetched_at=now,
        ),
    )
    log(f"  comic: {count} pages")
    return chapter_id
