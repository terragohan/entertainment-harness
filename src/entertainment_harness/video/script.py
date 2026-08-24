"""Stage 1: chapter recap -> narration script (text-role model).

The script is a list of spoken segments, each tagged with the story moment it
covers. Persisted as script.json; later stages (tts, visuals) fill in
duration_s and pages on the same structure, so it is the single source of
truth for the whole video.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from entertainment_harness.models.base import ModelAdapter


class VideoError(Exception):
    pass


class VideoConfigError(VideoError):
    """A misconfiguration (bad model id, missing capability), not a
    per-panel failure: assembly aborts the render instead of degrading —
    retrying every anchor against a broken config just storms the API."""
    pass


@dataclass
class Segment:
    index: int
    text: str  # spoken narration
    moment: str = ""  # short label of the story beat
    pages: list[int] = field(default_factory=list)  # 1-based; visuals stage
    motion: str = ""  # ken burns spec; visuals stage
    duration_s: float = 0.0  # from rendered audio (authoritative); tts stage
    # Silence inserted after this segment's audio (set by apply_pauses). The
    # segment's visuals hold through the pause; its subtitle cue does not.
    pause_after_s: float = 0.0
    # Viewport anchors for scroll-mode hold-and-glide (grounding stage,
    # video/grounding.py): [{"page": 3, "kind": "hold", "box": [x0,y0,x1,y1]},
    # {"page": 4, "kind": "pan"}, ...] in reading order. Empty = ungrounded,
    # assembly uses the linear descent.
    regions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> Segment:
        return Segment(
            index=int(data["index"]),
            text=str(data["text"]),
            moment=str(data.get("moment", "")),
            pages=[int(p) for p in data.get("pages", [])],
            motion=str(data.get("motion", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            pause_after_s=float(data.get("pause_after_s", 0.0)),
            regions=[dict(r) for r in data.get("regions", [])],
        )


SCRIPT_PROMPT = """You are writing the narration script for a recap video of chapter {chapter} of the manga "{title}".

Turn the recap below into spoken narration. Requirements:
- 25 to 40 short narration beats; 450-600 words in total (about 3-4 minutes read aloud). Cover the whole recap; do not skip scenes.
- Each beat is ONE spoken sentence, typically 12-22 words (about 4-8 seconds read aloud). Cut at the story's natural beats; never pad or merge sentences just to hit a count.
- Tell the story in the recap's order.
- Plain spoken English, present tense, no headers, no stage directions.
- Faithful to the recap: no new events, names, or details.{cast}
- Tag each beat with a short "moment" label naming the story beat (e.g. "Shin hunts the boar"); reuse the same label for consecutive beats in the same scene.

Output ONLY a JSON array, no other text:
[{{"text": "...", "moment": "..."}}, ...]

Recap:
---
{recap}
---"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


_SEGMENT_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _salvage_segment_array(raw: str) -> list[dict] | None:
    """Recover segments from a truncated/corrupt JSON array: parse every
    complete {"text": ...} object individually and keep the valid ones.
    Returns None when nothing usable survives."""
    items = []
    for candidate in _SEGMENT_OBJ_RE.findall(raw):
        try:
            item = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            items.append(item)
    return items or None


def parse_script(raw: str, salvage: bool = True) -> list[Segment]:
    """Extract the JSON segment array from raw model output.

    Tolerates markdown fences and preamble. With `salvage` (the default),
    also keeps the complete segments from a truncated or corrupt array;
    raises VideoError when nothing usable survives.
    """
    match = _JSON_ARRAY_RE.search(raw)
    if match is None:
        # No complete JSON array at all — the output may have been truncated
        # before the closing bracket. Salvage whatever segments survive.
        data = _salvage_segment_array(raw) if salvage else None
        if data is None:
            raise VideoError("Script model did not return a JSON array")
    else:
        blob = match.group(0)
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            # The provider sometimes truncates mid-array; salvaged segments
            # are better than nothing but incomplete, so callers wanting the
            # full script can retry first (salvage=False) and fall back.
            data = _salvage_segment_array(blob) if salvage else None
            if data is None:
                raise VideoError(
                    f"Script model returned invalid JSON: {exc}"
                ) from exc
    segments = []
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            raise VideoError(f"Script segment {i} is missing a 'text' field")
        segments.append(
            Segment(index=i, text=str(item["text"]).strip(),
                    moment=str(item.get("moment", "")).strip())
        )
    if not segments:
        raise VideoError("Script model returned zero segments")
    return segments


SCRIPT_ATTEMPTS = 3  # the script endpoint flakes occasionally; don't die on it


def generate_script(
    adapter: ModelAdapter,
    model: str,
    recap: str,
    title: str,
    chapter_num: float,
    max_attempts: int = SCRIPT_ATTEMPTS,
    cast: str = "",
) -> list[Segment]:
    prompt = SCRIPT_PROMPT.format(
        chapter=f"{chapter_num:g}", title=title, recap=recap, cast=cast
    )
    last_error: VideoError | None = None
    for attempt in range(max_attempts):
        # A truncated array is worth retrying for completeness; only the
        # final attempt settles for salvaged segments.
        try:
            raw = adapter.generate(model, prompt)
            return split_long_segments(parse_script(raw, salvage=attempt == max_attempts - 1))
        except VideoError as exc:
            last_error = exc
    raise VideoError(
        f"Script model failed after {max_attempts} attempts: {last_error}"
    )


_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def _word_count(text: str) -> int:
    return len(text.split())


MAX_BEAT_WORDS = 30  # safety net: longer model output splits at sentences


def split_long_segments(
    segments: list[Segment], max_words: int = MAX_BEAT_WORDS
) -> list[Segment]:
    """Re-chunk segments over max_words at sentence boundaries, preserving the
    wording verbatim and copying the moment label to each piece. A single
    over-long sentence stays whole — pacing targets never butcher prose."""
    out: list[Segment] = []
    for seg in segments:
        sentences = [s for s in _SENTENCE_RE.split(seg.text) if s.strip()]
        if _word_count(seg.text) <= max_words or len(sentences) <= 1:
            out.append(seg)
            continue
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and _word_count(candidate) > max_words:
                out.append(Segment(index=-1, text=current, moment=seg.moment))
                current = sentence
            else:
                current = candidate
        if current:
            out.append(Segment(index=-1, text=current, moment=seg.moment))
    return [
        Segment(index=i, text=s.text, moment=s.moment) for i, s in enumerate(out)
    ]


BEAT_PAUSE_S = 0.15  # breath between beats in the same scene
TRANSITION_PAUSE_S = 0.30  # longer beat where the moment label changes


def apply_pauses(
    segments: list[Segment],
    beat_s: float = BEAT_PAUSE_S,
    transition_s: float = TRANSITION_PAUSE_S,
) -> None:
    """Stamp pause_after_s: a short breath between beats, a longer pause where
    the moment label changes (scene transition). Deterministic from the
    script, so cached scripts pick pauses up on load; assembly inserts the
    actual silence."""
    for i, seg in enumerate(segments):
        if i + 1 == len(segments):
            seg.pause_after_s = 0.0
        elif seg.moment == segments[i + 1].moment:
            seg.pause_after_s = beat_s
        else:
            seg.pause_after_s = transition_s


def segments_from_narration(
    text: str, *, min_words: int = 10, target_words: int = 30
) -> list[Segment]:
    """Split a narration into TTS segments without a model call — the
    narration is already spoken-form, so its wording is preserved verbatim.

    Segments are kept short (a sentence or two, ~7s of speech) so page
    assignment and subtitles track the narration closely. Paragraphs
    (blank-line separated) are the natural beats: consecutive short
    paragraphs merge until they reach min_words; paragraphs longer than
    ~2x target_words split at sentence boundaries into chunks near
    target_words.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        raise VideoError("Narration is empty")

    # Merge short paragraphs forward.
    merged: list[str] = []
    for para in paragraphs:
        if merged and _word_count(merged[-1]) < min_words:
            merged[-1] = f"{merged[-1]} {para}"
        else:
            merged.append(para)

    # Split oversized paragraphs at sentence boundaries.
    chunks: list[str] = []
    for para in merged:
        if _word_count(para) <= target_words * 2:
            chunks.append(para)
            continue
        current = ""
        for sentence in _SENTENCE_RE.split(para):
            candidate = f"{current} {sentence}".strip()
            if current and _word_count(candidate) > target_words:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)

    return [Segment(index=i, text=chunk) for i, chunk in enumerate(chunks)]


def save_script(
    path: Path,
    segments: list[Segment],
    title: str,
    chapter_num: float,
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chapter": chapter_num,
        "title": title,
        "model": model,
        "segments": [s.to_dict() for s in segments],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_script(path: Path) -> tuple[list[Segment], str]:
    """Return (segments, model tag) from a saved script.json."""
    payload = json.loads(path.read_text())
    return [Segment.from_dict(s) for s in payload["segments"]], payload.get("model", "")
