"""Judge stage tests: verdict parsing and prompt construction. No network."""

from __future__ import annotations

from pathlib import Path

from entertainment_harness.pipelines.judge import (
    VISION_JUDGE_MAX_PX,
    VISION_JUDGE_PAGE_CAP,
    judge_context,
    judge_narration,
    judge_recap,
    judge_render,
    parse_render_verdict,
    parse_verdict,
    sample_pages,
)
from entertainment_harness.models.base import ModelInfo


class RecordingAdapter:
    name = "fake"

    def __init__(self, output: str = '{"pass": true}') -> None:
        self.output = output
        self.prompts: list[str] = []
        self.images: list[list[Path]] = []
        self.image_sizes: list[list[int]] = []

    def supports(self, model: str) -> ModelInfo:
        return ModelInfo(model, "fake", 4.0, "Q8_0", 3 * 10**9)

    def ensure(self, model: str, quant: str | None = None) -> None:
        pass

    def list_available(self) -> list[ModelInfo]:
        return []

    def generate(self, model: str, prompt: str, images: list[Path] | None = None) -> str:
        self.prompts.append(prompt)
        imgs = images or []
        self.images.append(imgs)
        sizes = []
        for p in imgs:
            try:
                from PIL import Image

                with Image.open(p) as im:
                    sizes.append(max(im.size))
            except Exception:
                sizes.append(-1)
        self.image_sizes.append(sizes)
        return self.output


def test_parse_verdict_pass():
    assert parse_verdict('{"pass": true}').passed


def test_parse_verdict_fail_with_issues():
    verdict = parse_verdict(
        '{"pass": false, "issues": ["invented a name", "meta-commentary"]}'
    )
    assert not verdict.passed
    assert verdict.issues == ["invented a name", "meta-commentary"]


def test_parse_verdict_tolerates_fences_and_preamble():
    raw = 'Evaluation:\n```json\n{"pass": false, "issues": ["off-topic"]}\n```'
    verdict = parse_verdict(raw)
    assert not verdict.passed
    assert verdict.issues == ["off-topic"]


def test_parse_verdict_garbage_is_a_pass():
    """A broken judge must not stall the pipeline."""
    assert parse_verdict("I cannot evaluate this.").passed
    assert parse_verdict('{"pass": ').passed


def test_parse_verdict_fail_without_issues_gets_a_note():
    verdict = parse_verdict('{"pass": false}')
    assert not verdict.passed
    assert verdict.issues == ["judge gave no specific issues"]


def test_judge_recap_prompt_carries_all_inputs():
    adapter = RecordingAdapter()
    verdict = judge_recap(
        adapter, "m", "The recap.", ["batch one", "batch two"],
        "Story so far.", "Test Manga", 3.0,
    )
    assert verdict.passed
    prompt = adapter.prompts[0]
    assert "The recap." in prompt
    assert "batch one" in prompt and "batch two" in prompt
    assert "Story so far." in prompt
    assert '"Test Manga"' in prompt
    assert "chapter 3" in prompt
    assert "outside knowledge" in prompt.lower() or "Outside knowledge" in prompt


def test_judge_context_prompt_carries_all_inputs():
    adapter = RecordingAdapter('{"pass": false, "issues": ["dropped Merlin"]}')
    verdict = judge_context(
        adapter, "m", "candidate context", "previous context",
        "chapter recap", "Test Manga",
    )
    assert not verdict.passed
    prompt = adapter.prompts[0]
    assert "candidate context" in prompt
    assert "previous context" in prompt
    assert "chapter recap" in prompt
    assert "crossover" in prompt  # the unrelated-bonus-chapter exception


def test_judge_recap_shows_instruction_with_mandated_omission_rule():
    """A steering instruction is shown to the artifact judge: mandated
    omissions/style are not issues, the quoting rules stay intact."""
    adapter = RecordingAdapter()
    verdict = judge_recap(
        adapter, "m", "The recap.", ["batch one"], None, "Test Manga", 3.0,
        instruction="skip chapter-opening recap pages",
    )
    assert verdict.passed
    prompt = adapter.prompts[0]
    assert "USER DIRECTION" in prompt
    assert "skip chapter-opening recap pages" in prompt
    assert "NOT issues" in prompt  # the mandated-omission rule
    assert "quote the exact phrase" in prompt  # evidence rules survive


def test_judge_narration_shows_instruction_with_mandated_omission_rule():
    adapter = RecordingAdapter()
    judge_narration(
        adapter, "m", "The narration.", ["batch one"], None, "Test Manga", 3.0,
        instruction="write in Gen Z slang",
    )
    prompt = adapter.prompts[0]
    assert "USER DIRECTION" in prompt
    assert "write in Gen Z slang" in prompt
    assert "NOT issues" in prompt
    assert "Completeness" in prompt  # the narration-only axis stays


def test_judge_without_instruction_has_no_direction_block():
    adapter = RecordingAdapter()
    judge_recap(adapter, "m", "The recap.", ["batch one"], None, "Test Manga", 3.0)
    assert "USER DIRECTION" not in adapter.prompts[0]


def test_sample_pages_evenly_spaced_and_capped():
    pages = list(range(100))
    sampled = sample_pages(pages)
    assert len(sampled) == VISION_JUDGE_PAGE_CAP
    assert sampled[0] == 0
    # small lists return all pages
    assert sample_pages([1, 2, 3]) == [1, 2, 3]


def test_judge_recap_resizes_sample_pages(tmp_path):
    from PIL import Image

    pages = []
    for i in range(20):
        p = tmp_path / f"page_{i:02d}.png"
        Image.new("RGB", (1600, 2400), color=(i, i, i)).save(p)
        pages.append(p)

    adapter = RecordingAdapter()
    verdict = judge_recap(
        adapter, "m", "recap text", ["batch"], "context",
        "Test Manga", 3.0, pages=pages,
    )
    assert verdict.passed
    assert len(adapter.images) == 1
    sent = adapter.images[0]
    assert len(sent) <= VISION_JUDGE_PAGE_CAP
    assert all(p.suffix == ".jpg" for p in sent)
    assert all(s <= VISION_JUDGE_MAX_PX for s in adapter.image_sizes[0] if s != -1)


# --- parse_render_verdict -----------------------------------------------------


def test_parse_render_verdict_pass():
    verdict = parse_render_verdict('{"pass": true}')
    assert verdict.passed
    assert verdict.issues == []


def test_parse_render_verdict_fail_with_structured_issues():
    raw = (
        '{"pass": false, "issues": ['
        '{"bubble_index": 1, "problem": "original_text_visible", "directions": ["left"]},'
        '{"bubble_index": 3, "problem": "text_overflow"}'
        ']}'
    )
    verdict = parse_render_verdict(raw)
    assert not verdict.passed
    assert len(verdict.issues) == 2
    assert verdict.issues[0].bubble_index == 1
    assert verdict.issues[0].problem == "original_text_visible"
    assert verdict.issues[0].directions == ["left"]
    assert verdict.issues[1].bubble_index == 3
    assert verdict.issues[1].problem == "text_overflow"


def test_parse_render_verdict_garbage_is_a_pass():
    """A broken render judge must not stall the pipeline."""
    verdict = parse_render_verdict("looks fine to me")
    assert verdict.passed


def test_judge_render_sends_both_pages_and_bubble_list(tmp_path):
    from PIL import Image

    original = tmp_path / "orig.png"
    rendered = tmp_path / "rendered.png"
    Image.new("RGB", (900, 1275), color="white").save(original)
    Image.new("RGB", (900, 1275), color="white").save(rendered)

    class Bubble:
        def __init__(self, translation: str) -> None:
            self.translation = translation

    adapter = RecordingAdapter('{"pass": true}')
    verdict = judge_render(
        adapter, "m", original, rendered,
        [Bubble("Hello"), Bubble("World")],
        "Test Manga", 1.0, 4,
    )
    assert verdict.passed
    assert len(adapter.images) == 1
    assert len(adapter.images[0]) == 2
    prompt = adapter.prompts[0]
    assert "Hello" in prompt
    assert "World" in prompt
    assert "Test Manga" in prompt
    assert "Chapter 1" in prompt
    assert "page 4" in prompt
