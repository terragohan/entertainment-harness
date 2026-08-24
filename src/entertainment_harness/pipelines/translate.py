"""Translation pipeline: chapter pages -> per-bubble translations -> overlay
rendering. Calls models via models/registry only.

The pipeline is driven by scanlation jobs (``[jobs.scanlation.<name>]``). Each
job names the backend and model for extract, translate, render, and optional
judge stages. When no jobs are configured the pipeline falls back to the legacy
model roles (``[models.vision]``, ``[models.text]``, etc.).

Three-stage per page:
1. Extract: identify bubble/caption boxes and the original text (vision model
   or LFM backend).
2. Translate: convert the extracted originals into the target language.
3. Render: OpenCV snaps each box to the dominant light (or dark) connected
   component; Pillow draws the translated text onto a copy of the page.

Spike verdict (2026-08-23, samples/kenja-ch0): detection + translation
accurate on real pt-br pages; raw boxes loose in ~half of cases, snap fixed
them. SFX and scanlation credits are excluded by prompt.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from entertainment_harness.library import works
from entertainment_harness.config import (
    Config,
    ModelRoleConfig,
    ScanlationJobConfig,
    StageConfig,
    data_dir,
)
from entertainment_harness.db import utcnow
from entertainment_harness.hardware import HardwareProfile
from entertainment_harness.pipelines.judge import (
    ATTEMPTS,
    MAX_ATTEMPTS,
    RenderIssue,
    judge_render,
    judge_translation,
)
from entertainment_harness.models.base import ModelAdapter
from entertainment_harness.models.registry import (
    Selection,
    get_judge_model,
    get_translation_model,
    get_vision_model,
    resolve,
)
from entertainment_harness.pipelines.recap import chapter_pages, model_tag
from entertainment_harness.models import lfm

_T = TypeVar("_T")

EXTRACT_PROMPT = """This is page {page} of {total} of chapter {chapter} of the manga "{title}". The text is in {lang}.

Find every speech bubble and caption box containing story text. IGNORE pure sound effects (SFX — including untranslated Japanese kana SFX such as ドン, ギシ!, or はぁ left in the art) and scanlation credits/watermarks.

For EACH bubble/caption output one entry:
- "box": bounding box [x1, y1, x2, y2] as normalized floats 0.0-1.0 (origin top-left), TIGHT around that single bubble or caption — not the whole panel, never spanning multiple text areas
- "original": the exact text as printed

Rules:
- One entry per bubble. Never merge separate bubbles or captions into one entry.
- Only use text actually printed on the page; no outside knowledge of any manga or anime.
- If the page contains no story text, output an empty array.

Output ONLY a JSON array:
[{{"box": [0.12, 0.05, 0.45, 0.20], "original": "..."}}]"""

TRANSLATE_PROMPT = """You are translating manga chapter {chapter} of "{title}".

This is page {page} of {total}. The original text is in {lang}; translate it into {target}.

Translate each numbered line below. Preserve meaning, tone, and character voice. Use names as printed. Do not add or remove information. Translate EVERY line — never leave a line untranslated, empty, or in the original language; if a line is a sound effect that slipped through, render it as a short {target} sound effect (e.g. "sigh...", "thud").{cast}

{originals}

Output ONLY a JSON array with one entry per line, in the same order:
[{{"translation": "..."}}, {{"translation": "..."}}]"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

MIN_BOX_AREA = 0.0005  # normalized; drops noise specks
BUBBLE_PAD_PX = 8

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _source_dir(series_id: str, chapter_id: str) -> Path:
    """Prefer the new works layout, but fall back to the legacy manga cache."""
    new_dir = works.source_dir(series_id, chapter_id)
    old_dir = data_dir() / "manga" / series_id / chapter_id
    if new_dir.is_dir() or not old_dir.is_dir():
        return new_dir
    return old_dir


class TranslateError(Exception):
    pass


@dataclass
class Bubble:
    box: list[float]  # normalized [x1, y1, x2, y2]
    original: str
    translation: str


@dataclass
class ResolvedScanlationPipeline:
    """Fully-resolved adapters for one scanlation job."""

    extract: Selection | None
    lfm_locator: lfm.LFMLocator | None
    translate: Selection
    judge: Selection | None
    render: Selection


@dataclass
class BoxOverride:
    """Per-bubble render adjustments applied on top of the VLM box."""

    pad_left: int = 0
    pad_right: int = 0
    pad_top: int = 0
    pad_bottom: int = 0
    font_size_delta: int = 0


def _escape_unescaped_newlines(text: str) -> str:
    """Escape raw newlines/carriage returns that appear inside JSON strings.

    VLMs often emit multi-line translations inside a JSON string, which is
    invalid. We walk the text and only touch characters that are inside
    unescaped double quotes.
    """
    out: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch in ("\n", "\r"):
            out.append("\\n")
            continue
        out.append(ch)
    return "".join(out)


def _repair_json(text: str) -> str:
    """Best-effort cleanup of model-generated JSON (object or array).

    Fixes:
    - raw newlines inside strings
    - trailing commas before ] or }
    - common leading/trailing markdown fences and prose
    """
    # Strip markdown fences and any surrounding prose on the same lines.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    stripped = text.strip()
    if stripped.startswith("["):
        match = _JSON_ARRAY_RE.search(stripped)
        body = match.group(0) if match else stripped
    elif stripped.startswith("{"):
        body = stripped
    else:
        body = stripped
    body = _escape_unescaped_newlines(body)
    # Remove trailing commas before closing brackets/braces.
    body = re.sub(r",(\s*[\]\}])", r"\1", body)
    return body


def _extract_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects from text using brace-depth scanning.

    Respects quoted strings, so braces inside translations are not confused
    with object boundaries. Used as a last-resort salvage when the full array
    won't parse.
    """
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : i + 1])
                    start = None
    return objects


def _parse_bubble_object(text: str, *, require_translation: bool = True,
                         image_size: tuple[int, int] | None = None) -> Bubble | None:
    """Try to parse a single bubble object after best-effort repair.

    During extraction the vision model may omit "translation"; set
    ``require_translation=False`` to allow that and return an empty string.
    Models sometimes return pixel coordinates instead of normalized floats;
    when ``image_size`` is given and any coordinate exceeds 1.0, the box is
    treated as pixels and normalized.
    """
    for candidate in (text, _repair_json(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            return None
        box = data.get("box")
        if not (isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)):
            return None
        coords = [float(v) for v in box]
        if image_size is not None and max(coords) > 1.0:
            w, h = image_size
            coords = [coords[0] / w, coords[1] / h, coords[2] / w, coords[3] / h]
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in coords)
        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            return None
        original = str(data.get("original") or "").strip()
        if not original:
            return None
        translation = str(data.get("translation") or "").strip()
        if require_translation and not translation:
            return None
        return Bubble(
            box=[x1, y1, x2, y2],
            original=original,
            translation=translation,
        )
    return None


def parse_bubbles(raw: str, *, page_label: str = "", require_translation: bool = True,
                  image_size: tuple[int, int] | None = None) -> list[Bubble]:
    """Extract a validated bubble list from model output.

    Tolerates code fences/preamble, repairs common JSON mistakes (raw
    newlines in strings, trailing commas), normalizes pixel-coordinate boxes
    when ``image_size`` is given, clamps boxes to [0,1] and drops degenerate,
    inverted, or speck-sized entries and empty originals.

    If the full array won't parse, we fall back to salvaging individual
    ``{"box": [...], ...}`` objects so one malformed bubble doesn't kill an
    entire page.
    """
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        raise TranslateError("Model did not return a JSON array")
    candidates = [match.group(0), _repair_json(match.group(0))]
    data = None
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue

    if data is not None and isinstance(data, list):
        bubbles = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            bubble = _parse_bubble_object(
                json.dumps(entry), require_translation=require_translation,
                image_size=image_size,
            )
            if bubble is not None:
                bubbles.append(bubble)
        return bubbles

    # Full-array parse failed. Try to salvage individual objects.
    salvaged: list[Bubble] = []
    for obj_text in _extract_json_objects(raw):
        bubble = _parse_bubble_object(
            obj_text, require_translation=require_translation,
            image_size=image_size,
        )
        if bubble is not None:
            salvaged.append(bubble)
    if salvaged:
        return salvaged

    label = f" ({page_label})" if page_label else ""
    snippet = raw[:800].replace("\n", "\\n")
    raise TranslateError(
        f"Model returned invalid JSON{label}: {last_exc}\n"
        f"Raw snippet: {snippet}"
    ) from last_exc


def _snap_box(gray: np.ndarray, box: list[float]) -> tuple[tuple[int, int, int, int], bool]:
    """Snap a normalized box to the dominant light component inside it.

    Returns ((px1, py1, px2, py2), is_dark). Dark-caption boxes (median
    brightness < 128) are kept as-is and flagged for inverted rendering.
    """
    h, w = gray.shape
    px1, py1 = int(box[0] * w), int(box[1] * h)
    px2, py2 = int(box[2] * w), int(box[3] * h)
    crop = gray[py1:py2, px1:px2]
    if crop.size == 0:
        return (px1, py1, px2, py2), False
    # A large bright component wins first: bubbles are bright even when the
    # surrounding panel art pulls the crop median dark.
    mask = (crop > 200).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] >= 0.05 * crop.size:
            sx, sy = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP]
            sw, sh = stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]
            return (px1 + sx, py1 + sy, px1 + sx + sw, py1 + sy + sh), False
    if float(np.median(crop)) < 128:
        return (px1, py1, px2, py2), True
    return (px1, py1, px2, py2), False


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str,
              width: int, height: int,
              font_size_delta: int = 0) -> tuple[ImageFont.ImageFont, list[str]]:
    """Word-wrap text to width, shrinking the font until it fits height.

    ``font_size_delta`` offsets the starting font size (negative shrinks,
    positive grows), bounded at a minimum of 10 px.
    """
    size = max(10, min(28, height // 4) + font_size_delta)
    while True:
        font = _load_font(size)
        lines: list[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            line = ""
            for word in words:
                trial = f"{line} {word}".strip()
                if draw.textlength(trial, font=font) <= width or not line:
                    line = trial
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        line_height = size + 2
        if len(lines) * line_height <= height or size <= 10:
            return font, lines
        size -= 2


def _brightness(color: tuple[int, ...]) -> float:
    r, g, b = color[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _bubble_mask(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a mask of the dominant bright bubble/caption region.

    For bright bubbles this is the largest bright connected component, which
    is usually the bubble interior. For dark captions the whole crop is kept
    and ``is_dark`` is set so the renderer looks for light text instead of
    dark text.
    """
    bright = (gray > 150).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] >= 0.05 * gray.size:
            return (labels == largest).astype(np.uint8), False
    if float(np.median(gray)) < 128:
        return np.ones_like(gray, dtype=np.uint8), True
    # No clear bubble found; assume the whole crop is the bubble.
    return np.ones_like(gray, dtype=np.uint8), False


def _render_bubble_replacement(
    img: Image.Image,
    bubble: Bubble,
    override: BoxOverride | None,
) -> None:
    """Crop a bubble, erase only the interior text, and draw the translation.

    This preserves the bubble outline and surrounding artwork by first
    snapping to the dominant bright bubble (or dark caption) region, then
    building a text mask from connected components that lie mostly inside that
    region. The original text is filled with the median background color and
    the translation is centered in the cleared area.
    """
    ovr = override or BoxOverride()
    w, h = img.size
    x1 = int(bubble.box[0] * w)
    y1 = int(bubble.box[1] * h)
    x2 = int(bubble.box[2] * w)
    y2 = int(bubble.box[3] * h)
    if x2 <= x1 or y2 <= y1:
        return

    pad_left = BUBBLE_PAD_PX + ovr.pad_left
    pad_right = BUBBLE_PAD_PX + ovr.pad_right
    pad_top = BUBBLE_PAD_PX + ovr.pad_top
    pad_bottom = BUBBLE_PAD_PX + ovr.pad_bottom
    cx1 = max(0, x1 - pad_left)
    cy1 = max(0, y1 - pad_top)
    cx2 = min(w, x2 + pad_right)
    cy2 = min(h, y2 + pad_bottom)

    crop = img.crop((cx1, cy1, cx2, cy2))
    crop_np = np.array(crop)
    ch, cw = crop_np.shape[:2]
    gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)

    # Find the actual bubble/caption region so we only erase text inside it.
    bubble_mask, is_dark = _bubble_mask(gray)
    # Fill text-shaped holes in the bright bubble mask so the interior is a
    # solid region we can mask against.
    if not is_dark:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, close_kernel)

    # Select text pixels: dark text for bright bubbles, light text for dark
    # captions. Use a low threshold so anti-aliased/faded ink is caught.
    if is_dark:
        _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Restrict analysis to the solid bubble interior.
    masked_binary = cv2.bitwise_and(binary, bubble_mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(masked_binary, 8)

    # Keep connected components that are dark pixels inside the bubble. The
    # area floor is low because manga strokes are thin, but isolated noise is
    # dropped.
    text_mask = np.zeros_like(binary)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4 or area > 0.5 * cw * ch:
            continue
        text_mask[labels == i] = 255

    if cv2.countNonZero(text_mask) == 0:
        return

    # Dilate aggressively to fully cover anti-aliased stroke edges and
    # bridge fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    text_mask = cv2.dilate(text_mask, kernel, iterations=3)

    # Fill the text with the median background color of the bubble interior.
    bubble_pixels = crop_np[bubble_mask > 0]
    fill_color = (
        tuple(int(v) for v in np.median(bubble_pixels, axis=0))
        if len(bubble_pixels)
        else (255, 255, 255)
    )
    crop_np[text_mask > 0] = fill_color
    cleaned = Image.fromarray(crop_np)

    # Region for replacement text: bounding box of the erased text area.
    ys, xs = np.where(text_mask > 0)
    tx1 = max(0, int(xs.min()) + 4)
    ty1 = max(0, int(ys.min()) + 4)
    tx2 = min(cw, int(xs.max()) - 4)
    ty2 = min(ch, int(ys.max()) - 4)
    if tx2 <= tx1 or ty2 <= ty1:
        return

    draw = ImageDraw.Draw(cleaned)
    font, lines = _fit_text(
        draw, bubble.translation, tx2 - tx1, ty2 - ty1,
        font_size_delta=ovr.font_size_delta,
    )
    line_height = font.size + 2 if hasattr(font, "size") else 12
    total_h = len(lines) * line_height
    y = ty1 + max(0, (ty2 - ty1 - total_h) // 2)
    text_color = "black" if _brightness(fill_color) > 128 else "white"
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = tx1 + (tx2 - tx1 - line_w) // 2
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_height

    img.paste(cleaned, (cx1, cy1))


def render_page(
    src: Path,
    dest: Path,
    bubbles: list[Bubble],
    overrides: dict[int, BoxOverride] | None = None,
) -> Path:
    """Render translated text onto a copy of the page.

    ``overrides`` maps a bubble index to a ``BoxOverride`` with extra padding
    and/or font-size adjustments. This is used by the render-quality judge
    loop to fix visible defects without re-running extraction/translation.

    The renderer crops each bubble, removes only the interior text, and draws
    the translation centered in the cleared area. This preserves the bubble
    outline and surrounding artwork.
    """
    img = Image.open(src).convert("RGB")
    overrides = overrides or {}
    for i, bubble in enumerate(bubbles):
        _render_bubble_replacement(img, bubble, overrides.get(i))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    return dest


MAX_RENDER_ATTEMPTS = 3
MAX_EXTRA_PAD_PX = 75
FONT_SHRINK_STEP = 2

_OCR_CROP_PROMPT = """Read the exact printed text inside this manga speech bubble or caption box.

Return a JSON object: {"text": "the exact text"}.
If there is no readable story text, return {"text": ""}.
Output ONLY the JSON object."""


def _overrides_from_issues(
    issues: list[RenderIssue],
    current: dict[int, BoxOverride],
) -> dict[int, BoxOverride]:
    """Build a new override map from the render judge's structured issues.

    Adjustments are cumulative but capped per direction by ``MAX_EXTRA_PAD_PX``.
    Unknown problem types are ignored.
    """
    out: dict[int, BoxOverride] = {
        idx: BoxOverride(**vars(ovr)) for idx, ovr in current.items()
    }
    for issue in issues:
        idx = issue.bubble_index
        if idx is None:
            continue
        if idx not in out:
            out[idx] = BoxOverride()
        ovr = out[idx]
        if issue.problem == "original_text_visible":
            for direction in issue.directions:
                if direction == "left":
                    ovr.pad_left = min(MAX_EXTRA_PAD_PX, ovr.pad_left + 25)
                elif direction == "right":
                    ovr.pad_right = min(MAX_EXTRA_PAD_PX, ovr.pad_right + 25)
                elif direction == "top":
                    ovr.pad_top = min(MAX_EXTRA_PAD_PX, ovr.pad_top + 25)
                elif direction == "bottom":
                    ovr.pad_bottom = min(MAX_EXTRA_PAD_PX, ovr.pad_bottom + 25)
        elif issue.problem == "text_overflow":
            ovr.font_size_delta -= FONT_SHRINK_STEP
            ovr.pad_right = min(MAX_EXTRA_PAD_PX, ovr.pad_right + 20)
            ovr.pad_bottom = min(MAX_EXTRA_PAD_PX, ovr.pad_bottom + 20)
        elif issue.problem == "unreadable":
            ovr.font_size_delta += FONT_SHRINK_STEP
            ovr.pad_left = min(MAX_EXTRA_PAD_PX, ovr.pad_left + 20)
            ovr.pad_right = min(MAX_EXTRA_PAD_PX, ovr.pad_right + 20)
            ovr.pad_top = min(MAX_EXTRA_PAD_PX, ovr.pad_top + 20)
            ovr.pad_bottom = min(MAX_EXTRA_PAD_PX, ovr.pad_bottom + 20)
    return out


def _render_with_judge(
    vision_adapter: ModelAdapter,
    vision_model: str,
    page: Path,
    bubbles: list[Bubble],
    out_path: Path,
    prompt_args: dict,
    max_attempts: int = MAX_RENDER_ATTEMPTS,
    log: Callable[[str], None] = lambda m: None,
) -> Path:
    """Render a page, judge the overlay quality, and adjust if needed.

    On a failing verdict the renderer applies deterministic fixes (box
    expansion, font shrinkage) and re-renders. On persistent failure the
    **first** render is restored to avoid feedback poisoning.
    """
    label = f"page {prompt_args['page']} render"
    overrides: dict[int, BoxOverride] = {}

    # First render is saved to out_path; subsequent attempts render to the
    # same path but we restore the first version if the judge never passes.
    render_page(page, out_path, bubbles, overrides=overrides)

    for attempt in range(1, max_attempts + 1):
        verdict = judge_render(
            vision_adapter, vision_model,
            page, out_path, bubbles,
            prompt_args["title"], float(prompt_args["chapter"]),
            int(prompt_args["page"]),
        )
        if verdict.passed:
            log(f"  {label}: render judge passed.")
            return out_path

        issue_summary = "; ".join(
            f"#{issue.bubble_index} {issue.problem}"
            + (f" ({','.join(issue.directions)})" if issue.directions else "")
            for issue in verdict.issues
        )
        log(
            f"  {label}: render judge rejected attempt {attempt}/{max_attempts}:"
            f" {issue_summary}"
        )

        if attempt == max_attempts:
            break

        new_overrides = _overrides_from_issues(verdict.issues, overrides)
        if not new_overrides:
            # Judge reported unactionable issues; stop retrying.
            break
        overrides = new_overrides
        render_page(page, out_path, bubbles, overrides=overrides)

    log(
        f"  warning: {label} failed render judge {max_attempts} times;"
        " keeping first render."
    )
    render_page(page, out_path, bubbles, overrides={})
    return out_path


def _retry_once(attempt: Callable[[], _T]) -> _T:
    """Run attempt(); on bad model output (TranslateError), retry once with no
    feedback. The second failure propagates."""
    try:
        return attempt()
    except TranslateError:
        return attempt()


def _extract_page(
    adapter: ModelAdapter,
    model: str,
    page: Path,
    prompt_args: dict,
) -> list[Bubble]:
    """Run the vision model on one page and return bubbles with original text."""
    prompt = EXTRACT_PROMPT.format(**prompt_args)
    label = f"page {prompt_args['page']} of chapter {prompt_args['chapter']}"
    with Image.open(page) as img:
        image_size = img.size

    def attempt() -> list[Bubble]:
        raw = adapter.generate(model, prompt, images=[page])
        return parse_bubbles(raw, page_label=label, require_translation=False,
                             image_size=image_size)

    return _retry_once(attempt)


def _ocr_bubble_crop(
    adapter: ModelAdapter,
    model: str,
    page: Path,
    box: list[float],
    log: Callable[[str], None] = lambda m: None,
) -> str:
    """Run the vision model on a single bubble crop to fill missing OCR text."""
    import tempfile

    with Image.open(page) as img:
        w, h = img.size
        x1 = max(0, int(box[0] * w) - BUBBLE_PAD_PX)
        y1 = max(0, int(box[1] * h) - BUBBLE_PAD_PX)
        x2 = min(w, int(box[2] * w) + BUBBLE_PAD_PX)
        y2 = min(h, int(box[3] * h) + BUBBLE_PAD_PX)
        crop = img.crop((x1, y1, x2, y2))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            crop_path = Path(fh.name)
        crop.save(crop_path)

    try:
        raw = adapter.generate(model, _OCR_CROP_PROMPT, images=[crop_path])
    except Exception as exc:  # pragma: no cover - runtime model failures
        log(f"  OCR crop failed: {exc}")
        return ""
    finally:
        crop_path.unlink(missing_ok=True)

    # Prefer a JSON {"text": "..."} field.
    try:
        data = json.loads(_repair_json(raw))
        if isinstance(data, dict):
            return str(data.get("text") or "").strip()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return str(
                data[0].get("text") or data[0].get("original") or ""
            ).strip()
    except Exception:
        pass
    # Fallback: first reasonable line of raw text.
    for line in raw.splitlines():
        line = line.strip().strip('"').removeprefix("text:")
        if line:
            return line
    return ""


def _extract_page_lfm(
    locator: lfm.LFMLocator,
    page: Path,
    prompt_args: dict,
    ocr_adapter: ModelAdapter | None = None,
    ocr_model: str = "",
    log: Callable[[str], None] = lambda m: None,
) -> list[Bubble]:
    """Run the LFM2.5-VL locator on one page and return bubbles with originals.

    When LFM returns boxes but no text, each empty bubble is cropped and sent
    to the configured vision model for OCR fallback. Boxes are snapped to the
    dominant light component so loose model boxes fit the actual bubble interior.
    """
    results = locator.locate(page, include_text=True)
    if not results:
        return []

    with Image.open(page) as img:
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        page_w, page_h = img.size

    # Manga reading order: top-to-bottom, right-to-left within a row.
    results.sort(key=lambda r: (r["box"][1], -r["box"][0]))

    bubbles: list[Bubble] = []
    for entry in results:
        # Snap the model box to the actual bubble/caption region.
        snapped, _ = _snap_box(gray, entry["box"])
        x1, y1, x2, y2 = snapped
        entry["box"] = [x1 / page_w, y1 / page_h, x2 / page_w, y2 / page_h]

        original = entry.get("original", "").strip()
        if not original and ocr_adapter is not None:
            original = _ocr_bubble_crop(
                ocr_adapter, ocr_model, page, entry["box"], log=log
            )
        if not original:
            continue
        box = entry["box"]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        bubbles.append(
            Bubble(box=box, original=original, translation="")
        )
    return bubbles


def _parse_translations(raw: str, expected: int, *, page_label: str = "") -> list[str]:
    """Parse the text model's translation array.

    Expects ``expected`` entries. If the parse yields a different count,
    fall back to one translation per original by repeating or truncating.
    """
    try:
        data = json.loads(_repair_json(raw))
    except json.JSONDecodeError as exc:
        label = f" ({page_label})" if page_label else ""
        snippet = raw[:400].replace("\n", "\\n")
        raise TranslateError(
            f"Translation model returned invalid JSON{label}: {exc}\n"
            f"Raw snippet: {snippet}"
        ) from exc
    if not isinstance(data, list):
        label = f" ({page_label})" if page_label else ""
        raise TranslateError(f"Translation model did not return a JSON array{label}")
    translations = []
    for entry in data:
        if isinstance(entry, dict):
            translations.append(str(entry.get("translation") or "").strip())
        elif isinstance(entry, str):
            translations.append(entry.strip())
        else:
            translations.append("")
    # Best-effort alignment if the model returns the wrong count.
    if len(translations) < expected:
        translations.extend([""] * (expected - len(translations)))
    return translations[:expected]


def _translate_bubbles(
    adapter: ModelAdapter,
    model: str,
    bubbles: list[Bubble],
    prompt_args: dict,
    feedback: str = "",
    cast: str = "",
) -> list[Bubble]:
    """Send the extracted originals to the text model and fill translations."""
    if not bubbles:
        return []
    originals = "\n".join(
        f"{i}. {b.original}" for i, b in enumerate(bubbles, start=1)
    )
    prompt = (
        TRANSLATE_PROMPT.format(originals=originals, cast=cast, **prompt_args)
        + feedback
    )
    label = f"page {prompt_args['page']} of chapter {prompt_args['chapter']}"

    def attempt() -> list[str]:
        raw = adapter.generate(model, prompt)
        return _parse_translations(raw, len(bubbles), page_label=label)

    translations = _retry_once(attempt)
    return [
        Bubble(box=b.box, original=b.original, translation=t)
        for b, t in zip(bubbles, translations)
        if t  # drop bubbles the text model left empty
    ]


def _translate_page(
    pipeline: ResolvedScanlationPipeline,
    page: Path,
    prompt_args: dict,
    *,
    judge_enabled: bool = False,
    title: str = "",
    chapter_num: float = 0,
    source_lang: str = "?",
    target_lang: str = "en",
    max_attempts: int = MAX_ATTEMPTS,
    cast: str = "",
    log: Callable[[str], None] = lambda m: None,
) -> list[Bubble]:
    """One page: extract bubbles, translate, then optionally judge.

    Extraction uses the pipeline's LFM locator if present, otherwise the
    resolved extract adapter. A failing translation verdict regenerates with
    feedback, bounded by ``max_attempts``.
    """
    if pipeline.lfm_locator is not None:
        bubbles = _extract_page_lfm(
            pipeline.lfm_locator, page, prompt_args,
            ocr_adapter=pipeline.render.adapter if pipeline.render else None,
            ocr_model=pipeline.render.info.name if pipeline.render else "",
            log=log,
        )
    elif pipeline.extract is not None:
        bubbles = _extract_page(
            pipeline.extract.adapter, pipeline.extract.info.name, page, prompt_args
        )
    else:
        bubbles = []

    def generate(feedback: str) -> list[Bubble]:
        # Feedback from the judge is appended only to the translation prompt;
        # extraction is independent of translation quality and is not retried.
        return _translate_bubbles(
            pipeline.translate.adapter, pipeline.translate.info.name,
            bubbles, prompt_args, feedback=feedback, cast=cast,
        )

    if not judge_enabled or pipeline.judge is None:
        return generate("")
    from entertainment_harness.pipelines.recap import judge_loop

    page_num = int(prompt_args["page"])

    def judge_fn(bubbles: list[Bubble]) -> object:
        return judge_translation(
            pipeline.judge.adapter, pipeline.judge.info.name, bubbles,
            title, chapter_num, page_num, source_lang, target_lang, cast=cast,
        )

    bubbles, _verdict = judge_loop(
        generate, judge_fn, log, f"page {prompt_args['page']} translation",
        max_attempts,
    )
    return bubbles


def _lfm_locator_from_stage(
    stage: StageConfig,
    log: Callable[[str], None] = print,
) -> lfm.LFMLocator | None:
    """Return an LFM locator if the stage is configured to use the LFM backend.

    "lfm" is not a special case here: it's a registered model_backend plugin
    (models/lfm.py LFMAdapter), resolved through the registry like any other
    backend name."""
    model_id = stage.model.strip()
    if not model_id or stage.backend != "lfm":
        return None
    if not lfm.is_available():
        log(
            "Warning: scanlation job uses backend 'lfm' but the dataset extra"
            " is not installed. Run: uv sync --extra dataset --group dev"
        )
        return None
    log(f"Using LFM model {model_id} for bubble extraction/OCR.")
    from entertainment_harness.models.registry import REGISTRY as model_backends

    adapter = model_backends.create("lfm", None, model_id=model_id, log=log)
    return adapter.locator()


def _lfm_locator_from_config(
    config: Config,
    log: Callable[[str], None] = print,
) -> lfm.LFMLocator | None:
    """Backwards-compatible LFM locator from the legacy [models] key."""
    stage = StageConfig(
        backend="lfm" if config.models.lfm_model.strip() else "",
        model=config.models.lfm_model,
    )
    return _lfm_locator_from_stage(stage, log=log)


def _active_scanlation_job(config: Config) -> ScanlationJobConfig:
    """Return the active scanlation job, or a synthesized one from model roles."""
    if config.jobs.scanlation:
        name = config.scanlation.job or next(iter(config.jobs.scanlation))
        return config.jobs.scanlation.get(name) or next(
            iter(config.jobs.scanlation.values())
        )
    return _synthesize_scanlation_job_from_roles(config)


def _synthesize_scanlation_job_from_roles(
    config: Config,
) -> ScanlationJobConfig:
    """Build a scanlation job from the legacy vision/text/translation/judge roles."""
    models = config.models
    if models.lfm_model.strip():
        extract = StageConfig(backend="lfm", model=models.lfm_model)
    else:
        extract = StageConfig(backend=models.vision.backend, model=models.vision.model)

    translation = models.translation
    if not translation.model:
        translation = models.text
    judge = models.judge
    if not judge.model:
        judge = models.text

    return ScanlationJobConfig(
        extract=extract,
        translate=StageConfig(backend=translation.backend, model=translation.model),
        judge=StageConfig(backend=judge.backend, model=judge.model),
        render=StageConfig(backend=models.vision.backend, model=models.vision.model),
    )


def _resolve_scanlation_stage(
    config: Config,
    profile: HardwareProfile,
    stage: StageConfig,
    label: str,
    log: Callable[[str], None],
) -> Selection:
    """Resolve a scanlation stage to an adapter + model info."""
    role = ModelRoleConfig(backend=stage.backend, model=stage.model)
    sel = resolve(role, profile, config, config.models.quant_policy)
    if sel.warning:
        log(f"Warning: {sel.warning}")
    log(f"Using {model_tag(sel.info)} for scanlation {label}.")
    sel.adapter.ensure(sel.info.name)
    return sel


def _resolve_pipeline_from_job(
    config: Config,
    profile: HardwareProfile,
    job: ScanlationJobConfig,
    thinking: str,
    log: Callable[[str], None],
) -> ResolvedScanlationPipeline:
    """Resolve an explicitly configured scanlation job."""
    extract = None
    lfm_locator = None
    if job.extract.backend == "lfm":
        lfm_locator = _lfm_locator_from_stage(job.extract, log=log)
    else:
        extract = _resolve_scanlation_stage(
            config, profile, job.extract, "extract", log
        )

    translate = _resolve_scanlation_stage(
        config, profile, job.translate, "translate", log
    )
    judge = None
    if thinking in ("medium", "high"):
        judge = _resolve_scanlation_stage(
            config, profile, job.judge_stage, "judge", log
        )
    render = _resolve_scanlation_stage(config, profile, job.render, "render", log)

    return ResolvedScanlationPipeline(
        extract=extract,
        lfm_locator=lfm_locator,
        translate=translate,
        judge=judge,
        render=render,
    )


def _resolve_pipeline_from_roles(
    config: Config,
    profile: HardwareProfile,
    thinking: str,
    log: Callable[[str], None],
) -> ResolvedScanlationPipeline:
    """Resolve the legacy role-based scanlation pipeline for backwards compatibility."""
    vision = get_vision_model(config, profile)
    if vision.warning:
        log(f"Warning: {vision.warning}")
    log(f"Using {model_tag(vision.info)} for extraction/grounding.")
    vision.adapter.ensure(vision.info.name)

    translation = get_translation_model(config, profile)
    if translation.warning:
        log(f"Warning: {translation.warning}")
    log(f"Using {model_tag(translation.info)} for translation.")
    translation.adapter.ensure(translation.info.name)

    judge = None
    if thinking in ("medium", "high"):
        judge = get_judge_model(config, profile)
        if judge.warning:
            log(f"Warning: {judge.warning}")
        log(f"Using {model_tag(judge.info)} for judging.")
        judge.adapter.ensure(judge.info.name)

    lfm_locator = _lfm_locator_from_config(config, log=log)
    extract = None if lfm_locator is not None else vision
    return ResolvedScanlationPipeline(
        extract=extract,
        lfm_locator=lfm_locator,
        translate=translation,
        judge=judge,
        render=vision,
    )


def translate_chapters(
    conn: sqlite3.Connection,
    series: sqlite3.Row,
    config: Config,
    profile: HardwareProfile,
    chapter_num: float | None = None,
    all_chapters: bool = False,
    force: bool = False,
    client=None,
    thinking: str = "medium",
    progress=None,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Translate chapters of a series into config.library.langs[0] (overlay pages).

    Chapters already in one of the preferred languages are skipped (nothing
    to do), as are chapters with a translations row unless force=True.
    thinking controls critiquing depth per page: "low" skips all judges,
    "medium" judges the original/translation pairs as text and the rendered
    page visually, "high" widens the attempt budget. progress, if given,
    receives stage() updates for the live UI. Returns the ids of chapters
    translated.
    """
    from entertainment_harness.sources import get_client

    _client = client

    def client_factory():
        nonlocal _client
        if _client is None:
            _client = get_client(series["source"])
        return _client

    langs = config.library.langs
    target = langs[0] if langs else "en"

    job = _active_scanlation_job(config)
    explicit_job = bool(config.jobs.scanlation)
    if explicit_job:
        pipeline = _resolve_pipeline_from_job(config, profile, job, thinking, log)
    else:
        pipeline = _resolve_pipeline_from_roles(config, profile, thinking, log)
    max_attempts = ATTEMPTS.get(thinking, MAX_ATTEMPTS)
    cast = ""
    if config.pipeline.characters:
        # Canonical spellings from the work's registry (character-bible):
        # the translation romanizes recurring characters consistently.
        from entertainment_harness.pipelines.characters import (
            cast_translation_block,
            load_cast_block,
        )
        cast = load_cast_block(conn, series["id"], block=cast_translation_block)

    if chapter_num is not None:
        row = conn.execute(
            "SELECT * FROM chapters WHERE series_id = ? AND chapter_num = ?",
            (series["id"], chapter_num),
        ).fetchone()
        if row is None:
            raise TranslateError(
                f"Chapter {chapter_num:g} is not synced for this series."
            )
        todo = [row]
    else:
        todo = conn.execute(
            "SELECT c.* FROM chapters c LEFT JOIN translations t"
            " ON t.chapter_id = c.id"
            " WHERE c.series_id = ? AND t.chapter_id IS NULL"
            " AND c.chapter_num IS NOT NULL ORDER BY c.chapter_num ASC",
            (series["id"],),
        ).fetchall()
        if not all_chapters:
            todo = todo[:1]

    done: list[str] = []
    for chapter in todo:
        chapter_id = chapter["id"]
        out_dir = works.translated_dir(series["id"], chapter_id)
        legacy_out_dir = data_dir() / "manga" / series["id"] / chapter_id / "translated"
        if chapter["lang"] in langs and not force:
            log(f"Chapter {chapter['chapter_num']:g} is already in a preferred language; skipped.")
            continue
        log(f"Translating chapter {chapter['chapter_num']:g}"
            f" ({chapter['lang']} -> {target})...")
        if progress is not None:
            progress.stage("translate", f"ch {chapter['chapter_num']:g}")
        dest = _source_dir(series["id"], chapter_id)
        pages = chapter_pages(
            config, client_factory, series["id"], chapter_id, dest, log
        )

        page_records = []
        prior: dict = {}
        prior_json = (
            legacy_out_dir / "translation.json"
            if (legacy_out_dir / "translation.json").exists()
            else out_dir / "translation.json"
        )
        if prior_json.exists() and not force:
            try:
                prior = {p["page"]: p
                         for p in json.loads(prior_json.read_text())["pages"]}
            except (json.JSONDecodeError, KeyError):
                prior = {}
        translated = 0
        for i, page in enumerate(pages, start=1):
            out_path = out_dir / f"page-{i:03d}.png"
            if out_path.exists() and not force:
                if page.name in prior:
                    page_records.append(prior[page.name])
                translated += 1
                log(f"  page {i}/{len(pages)}: cached.")
                if progress is not None:
                    progress.stage("translate", f"page {i}/{len(pages)} (cached)")
                continue
            if progress is not None:
                progress.stage("translate", f"page {i}/{len(pages)}")
            prompt_args = {
                "page": i, "total": len(pages),
                "chapter": f"{chapter['chapter_num']:g}",
                "title": series["title"],
                "lang": chapter["lang"] or "?",
                "target": target,
            }
            bubbles = _translate_page(
                pipeline, page, prompt_args,
                judge_enabled=(thinking in ("medium", "high")),
                title=series["title"],
                chapter_num=chapter["chapter_num"],
                source_lang=chapter["lang"] or "?",
                target_lang=target,
                max_attempts=max_attempts,
                cast=cast,
                log=log,
            )
            if not bubbles:
                log(f"  page {i}/{len(pages)}: no story text found.")
            else:
                log(f"  page {i}/{len(pages)}: {len(bubbles)} bubble(s).")
            if thinking == "low" or not bubbles:
                # No render judge for empty pages — nothing to critique.
                render_page(page, out_path, bubbles)
            else:
                _render_with_judge(
                    pipeline.render.adapter, pipeline.render.info.name,
                    page, bubbles, out_path, prompt_args,
                    max_attempts=max_attempts, log=log,
                )
            page_records.append({
                "page": page.name,
                "bubbles": [{"box": b.box, "original": b.original,
                             "translation": b.translation} for b in bubbles],
            })
            translated += 1
            # Keep the legacy translated layout alive during transition.
            legacy_out_path = legacy_out_dir / out_path.name
            if legacy_out_path != out_path:
                legacy_out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out_path, legacy_out_path)

        if pipeline.lfm_locator is not None:
            extract_label = pipeline.lfm_locator.model_id
        elif pipeline.extract is not None:
            extract_label = model_tag(pipeline.extract.info)
        else:
            extract_label = "none"
        model_attribution = (
            f"extract: {extract_label};"
            f" translate: {model_tag(pipeline.translate.info)};"
            f" render_judge: {model_tag(pipeline.render.info)}"
        )
        translation_payload = {
            "model": model_attribution,
            "target_lang": target,
            "pages": page_records,
        }
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "translation.json").write_text(
            json.dumps(translation_payload, indent=2)
        )
        legacy_out_dir.mkdir(parents=True, exist_ok=True)
        (legacy_out_dir / "translation.json").write_text(
            json.dumps(translation_payload, indent=2)
        )
        now = utcnow()
        all_bubbles = [
            bubble
            for page in page_records
            for bubble in page.get("bubbles", [])
        ]
        works.write_translation(
            series["id"],
            chapter_id,
            works.TranslationMetadata(
                pages=translated,
                model=model_attribution,
                created_at=now,
                bubbles=all_bubbles,
            ),
        )
        conn.execute(
            "INSERT INTO translations (chapter_id, pages, model, created_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(chapter_id) DO UPDATE SET pages = excluded.pages,"
            " model = excluded.model, created_at = excluded.created_at",
            (chapter_id, translated, model_attribution, now),
        )
        conn.commit()
        done.append(chapter_id)
        log(f"Chapter {chapter['chapter_num']:g} translated -> {out_dir}")
    if not done:
        log("Nothing to translate.")
    return done
