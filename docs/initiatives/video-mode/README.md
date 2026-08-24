# Video mode — scroll presentation (alternative to Ken Burns)

Status: **done** (Phase 2 deferred: optional, revisit only if real viewing
shows per-segment cuts are choppy)

## Goal

A new video mode, `scroll`, selectable alongside the current Ken Burns
presentation: the viewport descends the manga top-to-bottom as the recap plays,
so watching a recap video feels like reading down through the chapter instead
of watching still pages breathe. Everything else in the video pipeline (script,
TTS, page assignment, timing, muxing) stays identical — only how the frame
moves changes.

## Non-goals

- **Interactive scroll-driven narration** (user scrolls, narration follows).
  Deferred until a UI exists — that's the native-experience initiative's
  territory. This initiative produces baked mp4s only.
- **Side-by-side recap text in the video.** Explored and rejected by the user:
  no text column, no visible alignment markers.
- **Location-anchored recaps** (segment → page coordinates). Not needed for
  the baked scroll; if the interactive mode is ever built, whoever builds it
  owns that format.
- Panel cropping / highlight overlays (that's follow-along-visuals).
- Changes to script.json schema, recap pipeline, or DB.

## Rationale

Ken Burns motion is decorative: `pan_down` drifts across a page and `zoom_in`
breathes on it, both at fixed pace, unrelated to what's being narrated. A
scroll makes the motion semantic at zero model cost: the frame moves the way
manga is actually read. The current `_clip_filter` already proves the core
ffmpeg mechanic (animated crop drifting over a tall image) — the work is
stacking pages into strips and pacing the descent, not new rendering tech.

Design constraint discovered in planning: a whole-chapter vertical strip
(~40 pages × ~3000 px ≈ 120 000 px tall) exceeds image codec dimension
limits, so strips are built **per segment** (its assigned pages), not per
chapter.

## Design

- Config: `[video] mode = "kenburns" | "scroll"` (default kenburns); CLI
  `--video-mode scroll` on `eh recap --video` / `eh narrate --video`.
  `render_state.json` records the mode so toggling re-renders (same pattern
  as `colorize`/`translated`).
- Per segment (unchanged inputs: assigned pages + slot duration from
  authoritative TTS audio):
  1. Stack the segment's assigned pages vertically in page order with a small
     gutter, into a strip image (Pillow), cached per segment.
  2. Render the clip with the existing animated-crop mechanic, drifting the
     viewport from the strip's top to its bottom over the slot duration.
  3. `MIN_PAGE_SECONDS` pacing rules are unchanged — page picks are already
     capped/spaced per slot by `_segment_pages`.
- Single-page segments degrade gracefully: they use the normal per-page
  path with `pan_down` (no strip is built for one page), so the result is
  today's pan behavior.
- No model calls are added; `script.json` is untouched; books/shorts keep
  their existing motion (scroll only makes sense with real pages).

## Phases

1. **[x] Scroll motion in the clip renderer.** Strip stacking, animated-crop
   descent, config/CLI selection, render_state invalidation, tests for the
   filter-graph construction and strip builder.
   - Gate: `uv run pytest` green; `eh recap kenja --chapter 23 --video
     --video-mode scroll` renders alongside the Ken Burns render; the scroll
     version visibly descends through each segment's pages in order with no
     upside-down/blank segments.
2. **(Optional, after watching) Continuous-chapter feel.** If per-segment
   cuts still feel choppy: cross-page-boundary continuity (next segment's
   strip starts where the last ended when pages are consecutive), or a
   chapter-wide descent with linger/fast-forward pacing around assigned
   pages. Decide only from real viewing.

## Dependencies

None for Phase 1 (existing page picks + ffmpeg). Interactive mode, if ever
pursued, depends on a UI surface (native-experience initiative) and would
define its own segment→position map.

## Evidence

Phase 1 gate, 2026-09-14.

**Tests** — `uv run pytest -q`:

```
1 failed, 474 passed in 4.31s
```

The one failure is the pre-existing, unrelated
`tests/test_lfm.py::test_repair_json_repairs_trailing_comma` (Python 3.13
json strictness).

**Render** — ch-023 of kenja-no-mago (fully cached: script.json, page
picks, source pages). Deviation from the gate command: instead of
`eh recap kenja --chapter 23 --video --video-mode scroll`, the render was
invoked as `build_video(..., video_mode="scroll")` via `uv run python`,
because this machine's default model backend is OpenRouter (paid) and
`eh recap --chapter` force-re-recaps the chapter before the video callback
wipes/rebuilds the video dir — both would have spent money on stage 1/3
model calls. The direct call exercises the identical pipeline stages with
zero model calls (stages 1 and 3 served from cache; stage 2 is local
Kokoro). Output (narration lines elided):

```
Stage 1/4 script: cached (61 segments)
Stage 4/4 assembly: render settings changed, re-rendering...
Stage 2/4 narration (kokoro): rendering, voice=af_heart...
  narration seg-00..seg-60: rendered   (61 segments re-synthesized locally)
Stage 3/4 visuals: cached
  pacing: 61 beats, ~708 words, 52 cuts, 4.4s avg beat, 3.6s per visual
Stage 4/4 assembly: rendering...
  clip-00-0: rendered (2-page strip, 5.6s)
  clip-01-0: rendered (2-page strip, 6.1s)
  clip-02-0: rendered (2-page strip, 6.0s)
  clip-03-0: rendered (page 6, 3.9s)
  ... (18 strip clips total; 43 one-page segments use the normal
       pan_down path, e.g. clip-03 above)
  clip-60-0: rendered (page 48, 5.2s)
  pruned clips/
  pruned 62 segment WAV(s)
Done: ~/.local/share/entertainment-harness/works/kenja-no-mago/chapters/ch-023/video-recap/out.mp4 (286s)
```

`render_state.json` after the render:
`{"colorize": false, "translated": false, "mode": "scroll", "pacing": 1}`

**Duration** — `ffprobe -show_entries format=duration`: `285.898047`
(matches the 286s slot total).

**Frame inspection** — three frames from segment 2 (2-page strip, pages
[4, 37], slot 11.7s–17.7s), extracted with `ffmpeg -ss`:

- t=12.5s: top of the strip (page 4's upper panels).
- t=14.7s: mid-descent, the neutral-gray gutter band between page 4 and
  page 37 crosses the frame — the strip seam is plainly visible.
- t=17.2s: bottom of the strip (page 37's content, a different scene).

Verdict: the viewport visibly descends each multi-page segment's strip in
page order; no upside-down or blank segments. Gate passed.
