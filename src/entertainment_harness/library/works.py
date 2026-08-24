"""Resource-centric filesystem layout.

All data for a single work (manga, book, comic, tiktok search) lives under
`data/works/<work-dir>/`. Work directories are human-readable (a slug of the
title, e.g. `one-piece`); chapter directories are `ch-NNN`. The real ids
(ULIDs / source ids) live in work.json / chapter.json, so every public
function here still takes ids — a resolution layer maps id -> directory,
with a fallback to legacy id-named directories. The filesystem is
authoritative; SQLite is a rebuildable index.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from entertainment_harness.config import data_dir

WORK_METADATA_FILENAME = "work.json"
CHAPTER_METADATA_FILENAME = "chapter.json"
RECAP_FILENAME = "recap.json"
NARRATION_FILENAME = "narration.json"
TRANSLATION_FILENAME = "translation.json"
VIDEO_METADATA_FILENAME = "video.json"
SOURCE_DIR = "source"
TRANSLATED_DIR = "translated"
VIDEO_RECAP_DIR = "video-recap"
VIDEO_NARRATION_DIR = "video-narration"
TIKTOK_DIR = "tiktok"
ANIME_SCENE_DIR = "anime-scene"


def works_root() -> Path:
    return data_dir() / "works"


# --- id -> directory resolution ---
#
# Directory names are human-readable; ids live in the metadata files. The
# scans below are memoized per data dir and updated by the write_* functions,
# so lookups stay O(1) after the first scan. Directories named after the id
# itself (legacy layout, store pulls) always resolve directly.

_work_cache: tuple[Path, dict[str, str]] | None = None
_chapter_cache: dict[tuple[Path, str], dict[str, str]] = {}


def _work_map() -> dict[str, str]:
    global _work_cache
    root = works_root()
    if _work_cache is None or _work_cache[0] != root:
        mapping: dict[str, str] = {}
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir():
                    meta = _read_json(d / WORK_METADATA_FILENAME)
                    if isinstance(meta, dict) and meta.get("id"):
                        mapping[str(meta["id"])] = d.name
        _work_cache = (root, mapping)
    return _work_cache[1]


def _chapter_map(series_id: str) -> dict[str, str]:
    key = (works_root(), series_id)
    if key not in _chapter_cache:
        mapping: dict[str, str] = {}
        base = work_dir(series_id) / "chapters"
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir():
                    meta = _read_json(d / CHAPTER_METADATA_FILENAME)
                    if isinstance(meta, dict) and meta.get("id"):
                        mapping[str(meta["id"])] = d.name
        _chapter_cache[key] = mapping
    return _chapter_cache[key]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _unique_work_slug(title: str, series_id: str) -> str:
    base = slugify(title)
    taken = set(_work_map().values())
    name = base
    suffix = 2
    while name in taken or (works_root() / name).exists():
        name = f"{base}-{suffix}"
        suffix += 1
    return name


def chapter_dir_name(chapter_num: float | None, chapter_id: str) -> str:
    if chapter_num is None:
        return slugify(chapter_id)
    whole = int(chapter_num)
    frac = chapter_num - whole
    if frac == 0:
        return f"ch-{whole:03d}"
    return f"ch-{whole:03d}{f'{frac:.4f}'.lstrip('0').rstrip('0')}"


def refresh_paths() -> None:
    """Drop memoized id -> dir mappings (after external renames)."""
    global _work_cache
    _work_cache = None
    _chapter_cache.clear()


def work_dir(series_id: str) -> Path:
    root = works_root()
    legacy = root / series_id
    if legacy.is_dir():
        return legacy
    return root / _work_map().get(series_id, series_id)


def chapter_dir(series_id: str, chapter_id: str) -> Path:
    base = work_dir(series_id) / "chapters"
    legacy = base / chapter_id
    if legacy.is_dir():
        return legacy
    return base / _chapter_map(series_id).get(chapter_id, chapter_id)


def source_dir(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / SOURCE_DIR


def translated_dir(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / TRANSLATED_DIR


def video_recap_dir(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / VIDEO_RECAP_DIR


def video_narration_dir(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / VIDEO_NARRATION_DIR


def anime_scene_dir(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / ANIME_SCENE_DIR


def tiktok_dir(series_id: str) -> Path:
    return work_dir(series_id) / TIKTOK_DIR


def find_cover(series_id: str) -> Path | None:
    """The work's extracted cover image (importer writes cover.<ext> into the
    work dir), or None when there is none."""
    wdir = work_dir(series_id)
    if not wdir.is_dir():
        return None
    return next(iter(sorted(wdir.glob("cover.*"))), None)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@dataclass
class WorkProgress:
    last_read_chapter: float | None = None
    updated_at: str | None = None


@dataclass
class WorkContext:
    rolling_summary: str = ""
    through_chapter: float | None = None


@dataclass
class OnlineSummary:
    provider: str = ""
    query: str = ""
    summary: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    steering_prompt: str = ""
    created_at: str | None = None


@dataclass
class WorkMetadata:
    id: str
    title: str
    source: str
    source_id: str
    added_at: str
    kind: str = "manga"
    alt_titles: list[str] = field(default_factory=list)
    status: str | None = None
    progress: WorkProgress = field(default_factory=WorkProgress)
    context: WorkContext = field(default_factory=WorkContext)
    online_summary: OnlineSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.online_summary is None:
            d["online_summary"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkMetadata":
        progress = WorkProgress(**d.get("progress", {}))
        context = WorkContext(**d.get("context", {}))
        online = d.get("online_summary")
        if online:
            online = OnlineSummary(**online)
        return cls(
            id=d["id"],
            title=d["title"],
            source=d["source"],
            source_id=d["source_id"],
            added_at=d["added_at"],
            kind=d.get("kind", "manga"),
            alt_titles=d.get("alt_titles", []),
            status=d.get("status"),
            progress=progress,
            context=context,
            online_summary=online,
        )


@dataclass
class ChapterMetadata:
    id: str
    chapter_num: float | None
    title: str | None
    lang: str | None
    pages: int | None
    published_at: str | None
    fetched_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChapterMetadata":
        return cls(**d)


@dataclass
class RecapMetadata:
    summary: str
    model: str | None
    created_at: str
    detail: str = "standard"  # gist | brief | standard | detailed | full
    instruction: str = ""  # steering direction the artifact was written with
    standalone: bool = False  # generated without the rolling story-so-far
                              # (--video gap fill behind the read frontier)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecapMetadata":
        return cls(**d)


@dataclass
class NarrationMetadata:
    text: str
    model: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NarrationMetadata":
        return cls(**d)


@dataclass
class TranslationMetadata:
    pages: int | None
    model: str | None
    created_at: str
    bubbles: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranslationMetadata":
        return cls(
            pages=d.get("pages"),
            model=d.get("model"),
            created_at=d["created_at"],
            bubbles=d.get("bubbles", []),
        )


@dataclass
class VideoMetadata:
    kind: str  # "recap" | "narration" | "tiktok"
    duration_s: float | None
    model: str | None
    tts_engine: str | None
    created_at: str
    video_gen_provider: str | None = None
    compress: str | None = None
    wiped_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VideoMetadata":
        return cls(**d)


# --- Work metadata I/O ---


def work_metadata_path(series_id: str) -> Path:
    return work_dir(series_id) / WORK_METADATA_FILENAME


def write_work_metadata(meta: WorkMetadata) -> Path:
    wdir = work_dir(meta.id)
    if not wdir.is_dir():
        wdir = works_root() / _unique_work_slug(meta.title, meta.id)
        _work_map()[meta.id] = wdir.name
    path = wdir / WORK_METADATA_FILENAME
    _write_json(path, meta.to_dict())
    return path


def read_work_metadata(series_id: str) -> WorkMetadata | None:
    return read_work_metadata_at(work_dir(series_id))


def read_work_metadata_at(wdir: Path) -> WorkMetadata | None:
    d = _read_json(wdir / WORK_METADATA_FILENAME)
    if d is None:
        return None
    return WorkMetadata.from_dict(d)


def work_exists(series_id: str) -> bool:
    return work_metadata_path(series_id).exists()


def remove_work(series_id: str) -> None:
    d = work_dir(series_id)
    if d.exists():
        shutil.rmtree(d)
    _work_map().pop(series_id, None)
    _chapter_cache.pop((works_root(), series_id), None)


# --- Chapter metadata I/O ---


def chapter_metadata_path(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / CHAPTER_METADATA_FILENAME


def write_chapter_metadata(series_id: str, meta: ChapterMetadata) -> Path:
    cdir = chapter_dir(series_id, meta.id)
    if not cdir.is_dir():
        base = work_dir(series_id) / "chapters"
        name = chapter_dir_name(meta.chapter_num, meta.id)
        taken = set(_chapter_map(series_id).values())
        if name in taken or (base / name).exists():
            suffix = 2
            candidate = f"{name}-{suffix}"
            while candidate in taken or (base / candidate).exists():
                suffix += 1
                candidate = f"{name}-{suffix}"
            name = candidate
        cdir = base / name
        _chapter_map(series_id)[meta.id] = name
    path = cdir / CHAPTER_METADATA_FILENAME
    _write_json(path, meta.to_dict())
    return path


def read_chapter_metadata(series_id: str, chapter_id: str) -> ChapterMetadata | None:
    return read_chapter_metadata_at(chapter_dir(series_id, chapter_id))


def read_chapter_metadata_at(cdir: Path) -> ChapterMetadata | None:
    d = _read_json(cdir / CHAPTER_METADATA_FILENAME)
    if d is None:
        return None
    return ChapterMetadata.from_dict(d)


def list_chapter_dirs(series_id: str) -> list[Path]:
    return list_chapter_dirs_at(work_dir(series_id))


def list_chapter_dirs_at(wdir: Path) -> list[Path]:
    chapters_root = wdir / "chapters"
    if not chapters_root.is_dir():
        return []
    return sorted(d for d in chapters_root.iterdir() if d.is_dir())


# --- Recap I/O ---


def recap_path(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / RECAP_FILENAME


def write_recap(series_id: str, chapter_id: str, meta: RecapMetadata) -> Path:
    path = recap_path(series_id, chapter_id)
    _write_json(path, meta.to_dict())
    return path


def read_recap(series_id: str, chapter_id: str) -> RecapMetadata | None:
    d = _read_json(recap_path(series_id, chapter_id))
    if d is None:
        return None
    return RecapMetadata.from_dict(d)


def has_recap(series_id: str, chapter_id: str) -> bool:
    return recap_path(series_id, chapter_id).exists()


# --- Narration I/O ---


def narration_path(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / NARRATION_FILENAME


def write_narration(series_id: str, chapter_id: str, meta: NarrationMetadata) -> Path:
    path = narration_path(series_id, chapter_id)
    _write_json(path, meta.to_dict())
    return path


def read_narration(series_id: str, chapter_id: str) -> NarrationMetadata | None:
    d = _read_json(narration_path(series_id, chapter_id))
    if d is None:
        return None
    return NarrationMetadata.from_dict(d)


def has_narration(series_id: str, chapter_id: str) -> bool:
    return narration_path(series_id, chapter_id).exists()


# --- Translation I/O ---


def translation_path(series_id: str, chapter_id: str) -> Path:
    return chapter_dir(series_id, chapter_id) / TRANSLATION_FILENAME


def write_translation(
    series_id: str, chapter_id: str, meta: TranslationMetadata
) -> Path:
    path = translation_path(series_id, chapter_id)
    _write_json(path, meta.to_dict())
    return path


def read_translation(series_id: str, chapter_id: str) -> TranslationMetadata | None:
    d = _read_json(translation_path(series_id, chapter_id))
    if d is None:
        return None
    return TranslationMetadata.from_dict(d)


def has_translation(series_id: str, chapter_id: str) -> bool:
    return translation_path(series_id, chapter_id).exists()


# --- Video I/O ---


def video_dir_for_kind(series_id: str, chapter_id: str, kind: str) -> Path:
    if kind == "narration":
        return video_narration_dir(series_id, chapter_id)
    return video_recap_dir(series_id, chapter_id)


def video_metadata_path(series_id: str, chapter_id: str, kind: str) -> Path:
    return video_dir_for_kind(series_id, chapter_id, kind) / VIDEO_METADATA_FILENAME


def tiktok_metadata_path(series_id: str) -> Path:
    return tiktok_dir(series_id) / VIDEO_METADATA_FILENAME


def write_video_metadata(
    series_id: str, chapter_id: str | None, meta: VideoMetadata
) -> Path:
    if chapter_id is None:
        path = tiktok_metadata_path(series_id)
    else:
        path = video_metadata_path(series_id, chapter_id, meta.kind)
    _write_json(path, meta.to_dict())
    return path


def read_video_metadata(
    series_id: str, chapter_id: str | None, kind: str
) -> VideoMetadata | None:
    if chapter_id is None:
        path = tiktok_metadata_path(series_id)
    else:
        path = video_metadata_path(series_id, chapter_id, kind)
    d = _read_json(path)
    if d is None:
        return None
    return VideoMetadata.from_dict(d)


def video_file_path(series_id: str, chapter_id: str | None, kind: str, suffix: str = "out.mp4") -> Path:
    if chapter_id is None:
        return tiktok_dir(series_id) / suffix
    return video_dir_for_kind(series_id, chapter_id, kind) / suffix


def list_video_files(series_id: str, chapter_id: str) -> list[tuple[Path, str, str]]:
    """Return (path, kind, filename) for rendered mp4s of a chapter."""
    result: list[tuple[Path, str, str]] = []
    for kind, d in (
        ("recap", video_recap_dir(series_id, chapter_id)),
        ("narration", video_narration_dir(series_id, chapter_id)),
    ):
        if d.is_dir():
            for p in sorted(d.glob("*.mp4")):
                result.append((p, kind, p.name))
    return result


# --- Helpers for discovery ---


def list_work_dirs() -> list[Path]:
    root = works_root()
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir())
