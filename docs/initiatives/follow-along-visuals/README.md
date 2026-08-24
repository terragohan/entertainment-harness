# Follow-along visuals

Status: **dropped** (absorbed into [anchored-scroll](../anchored-scroll/) on
2026-09-16: its Phase 1 regions and Phase 2 vision panel ranking became that
initiative's Phases 1–2, extended with a grounding judge and hold-and-glide
pacing — the "looping" complaint was resolved in favor of full position-driven
pacing rather than panel crops alone)

## Goal

Videos where the viewport follows the narration: crop to panels instead of whole pages, pan panel-to-panel in reading order, and draw a highlight on the active panel — all baked into the rendered mp4. No interactivity needed; plays in any video player with seek/speed/subtitles for free.

## Non-goals

- Word-level sync. Narration text is a **retelling**, not the manga's dialogue, so true "highlight what is being read" is impossible in principle. The achievable target is: highlight the panel that illustrates the current sentence.
- Interactive pan/zoom by the viewer. That's the native-experience initiative; only pursue it if Tier 1 leaves the "looping" feeling unresolved (see Decision gate).

## Rationale

Today the video shows whole pages with Ken Burns for 20–30 s per pick ("just kind of loops"). Page metadata already carries bubble coordinates (`PageMetadata.bubbles` in `works.py`), so panel-level crops don't need new detection machinery. Panel granularity with per-panel time slices multiplies visual change without touching TTS or scripts.

## Design notes

- **Panel segmentation**: derive panels from bubble clusters (merge bubbles into bounding boxes, pad). Pages with no bubble data fall back to whole-page pan (current behavior) — graceful degradation per page.
- **Segment → panel mapping**: the vision model already assigns pages per segment; extend it to also rank panels within an assigned page, or distribute a segment's time across its pages' panels proportionally (simpler v1 — no new model call).
- **Time allocation**: segment duration divided across its panels; MIN_PAGE_SECONDS becomes MIN_PANEL_SECONDS (~1.5–2 s), giving 5–15× more visual changes per chapter than today.
- **Highlight**: subtle border or dim-the-rest overlay on the active panel; keep it tasteful and subtitle-visible.
- Output stays a normal mp4 — concat, compress, store, wipe all work unchanged.

## Phases

1. **Panel pipeline (no model calls).** Bubble-cluster → panels; per-page fallback; segment time split across panels; panel crop + highlight in the clip renderer.
   - Gate: render kenja ch 23 alongside the current render; panel changes ≥ 3× more frequent; pages without bubbles render correctly.
2. **Vision panel ranking (optional).** Ask the page-assignment model to pick the panel within its assigned page per segment, replacing proportional distribution where confident.
   - Gate: spot-check 5 chapters — active panel matches the sentence ≥ 80% of the time (eyeball verdict is fine).

## Decision gate

Live with Tier 1 (Phase 1) for a week of actual watching. If the desire to *manually* move the page persists, that's the signal for the native-experience initiative. If auto-follow is enough, close this as done and never build the app.

## Dependencies

None (uses existing bubble metadata; degrades when absent).
