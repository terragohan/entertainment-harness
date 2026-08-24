"""Import-layer tests: EPUB/TXT book parsing, CBZ/folder comic extraction,
URL downloads, kind detection, duplicate refusal. No network (respx-mocked).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
import respx
from PIL import Image

from entertainment_harness import db
from entertainment_harness.config import Config, data_dir
from entertainment_harness.library.importer import (
    ImportFailure,
    _merge_small,
    detect_kind,
    import_work,
    split_text,
    word_chunks,
)


def _png_bytes(color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (20, 30), color).save(buf, format="PNG")
    return buf.getvalue()


CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="cover" href="cover.png" media-type="image/png" properties="cover-image"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>
"""


def _xhtml(heading: str, words: int = 400) -> str:
    return (
        f"<html><body><h1>{heading}</h1><p>{'word ' * words}</p>"
        "<script>ignore me</script></body></html>"
    )


def _make_epub(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", OPF)
        zf.writestr("OEBPS/ch1.xhtml", _xhtml("Chapter 1"))
        zf.writestr("OEBPS/ch2.xhtml", _xhtml("Chapter 2"))
        zf.writestr("OEBPS/cover.png", _png_bytes())
    return path


def _make_cbz(path: Path, pages: int = 3) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(1, pages + 1):
            zf.writestr(f"page-{i:03d}.png", _png_bytes())
        zf.writestr("__MACOSX/junk.png", _png_bytes())  # must be ignored
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("EH_DATA_DIR", str(tmp_path))
    conn = db.connect()
    yield conn, tmp_path
    conn.close()


def test_import_epub(env):
    conn, tmp = env
    epub = _make_epub(tmp / "test-book.epub")
    series = import_work(conn, Config(), str(epub), log=lambda m: None)

    assert series["kind"] == "book"
    assert series["source"] == "import"
    assert series["title"] == "Test Book"
    chapters = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ? ORDER BY chapter_num",
        (series["id"],),
    ).fetchall()
    assert len(chapters) == 2
    assert all(c["lang"] == "en" for c in chapters)

    from entertainment_harness.library import works

    work_dir = works.work_dir(series["id"])
    part1 = (works.source_dir(series["id"], chapters[0]["id"]) / "ch-001.txt").read_text()
    assert "Chapter 1" in part1
    assert "ignore me" not in part1  # script stripped
    assert "<p>" not in part1  # tags stripped
    assert (work_dir / "cover.png").exists()


def test_import_txt_splits_on_headings(env):
    conn, tmp = env
    text = "\n\n".join(
        f"Chapter {i}\n\n{'text ' * 400}" for i in range(1, 4)
    )
    path = tmp / "novel.txt"
    path.write_text(text)
    series = import_work(conn, Config(), str(path), log=lambda m: None)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM chapters WHERE series_id = ?", (series["id"],)
    ).fetchone()["n"]
    assert count == 3


def test_split_text_falls_back_to_word_chunks():
    text = "\n\n".join(f"{'word ' * 50}" for _ in range(10))  # 500 words, no headings
    parts = word_chunks(text, 100)
    assert len(parts) == 5
    # through split_text the small chunks merge back together (they are below
    # MIN_PART_WORDS), but no text is lost either way
    joined = " ".join(split_text(text, chunk_words=100))
    assert joined.count("word") == 500


def test_merge_small_parts():
    parts = ["tiny intro", "big " * 100, "another big " * 100, "tiny outro"]
    merged = _merge_small(parts, min_words=10)
    assert len(merged) == 2
    assert merged[0].startswith("tiny intro")
    assert merged[1].endswith("tiny outro")


def test_import_cbz(env):
    conn, tmp = env
    cbz = _make_cbz(tmp / "comic.cbz", pages=3)
    series = import_work(conn, Config(), str(cbz), log=lambda m: None)

    assert series["kind"] == "comic"
    chapter = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ?", (series["id"],)
    ).fetchone()
    assert chapter["pages"] == 3
    from entertainment_harness.library import works

    page_dir = works.source_dir(series["id"], chapter["id"])
    assert sorted(
        p.name for p in page_dir.iterdir() if p.name.startswith("page-")
    ) == ["page-001.png", "page-002.png", "page-003.png"]


def test_import_cbz_writes_metadata(env):
    conn, tmp = env
    cbz = _make_cbz(tmp / "comic.cbz", pages=3)
    series = import_work(conn, Config(), str(cbz), log=lambda m: None)
    chapter = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ?", (series["id"],)
    ).fetchone()

    from entertainment_harness.library import works

    work_meta = works.read_work_metadata(series["id"])
    assert work_meta is not None
    assert work_meta.id == series["id"]
    assert work_meta.title == "Comic"

    chapter_meta = works.read_chapter_metadata(series["id"], chapter["id"])
    assert chapter_meta is not None
    assert chapter_meta.id == chapter["id"]
    assert chapter_meta.chapter_num == 1.0
    # Human-readable dirs: title slug + ch-NNN.
    assert works.work_dir(series["id"]).name == "comic"
    assert works.chapter_dir(series["id"], chapter["id"]).name == "ch-001"


def test_import_image_folder(env):
    conn, tmp = env
    folder = tmp / "comic-folder"
    folder.mkdir()
    for i in range(1, 3):
        (folder / f"{i}.png").write_bytes(_png_bytes())
    (folder / "notes.txt").write_text("not a page")
    series = import_work(conn, Config(), str(folder), log=lambda m: None)
    chapter = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ?", (series["id"],)
    ).fetchone()
    assert chapter["pages"] == 2


@respx.mock
def test_import_from_url(env):
    conn, tmp = env
    cbz = _make_cbz(tmp / "src.cbz", pages=2)
    respx.get("https://files.example.com/remote.cbz").mock(
        return_value=httpx.Response(200, content=cbz.read_bytes())
    )
    series = import_work(
        conn, Config(), "https://files.example.com/remote.cbz", log=lambda m: None
    )
    assert series["kind"] == "comic"
    chapter = conn.execute(
        "SELECT * FROM chapters WHERE series_id = ?", (series["id"],)
    ).fetchone()
    assert chapter["pages"] == 2


@respx.mock
def test_import_url_html_refused(env):
    conn, _ = env
    respx.get("https://example.com/some-book").mock(
        return_value=httpx.Response(
            200, content=b"<html>...</html>", headers={"content-type": "text/html"}
        )
    )
    with pytest.raises(ImportFailure, match="web page"):
        import_work(conn, Config(), "https://example.com/some-book", log=lambda m: None)


def test_duplicate_import_refused(env):
    conn, tmp = env
    cbz = _make_cbz(tmp / "comic.cbz")
    import_work(conn, Config(), str(cbz), log=lambda m: None)
    with pytest.raises(ImportFailure, match="already imported"):
        import_work(conn, Config(), str(cbz), log=lambda m: None)


def test_detect_kind_unknown_extension(tmp_path):
    mystery = tmp_path / "file.xyz"
    mystery.write_text("?")
    with pytest.raises(ImportFailure, match="--kind"):
        detect_kind(mystery)


def test_import_missing_file(env):
    conn, _ = env
    with pytest.raises(ImportFailure, match="No such file"):
        import_work(conn, Config(), "/nonexistent/book.epub", log=lambda m: None)
