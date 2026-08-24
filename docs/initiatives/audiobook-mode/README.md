# Audiobook mode (`eh listen`)

Status: **proposed**

## Goal

Hear narration chapters without persisting video or audio on disk: synthesize TTS per segment, play as it goes, delete after playing. Peak temp disk usage is one ~7 s WAV.

## Non-goals

- Page visuals (that's follow-along-visuals' domain; a static page-display phase is an optional extension, not a requirement).
- Full video output (`--video` keeps existing behavior).
- Podcast-style RSS/downloads.

## Rationale

The pipeline is already segment-based: narration text → `segments_from_narration` (already split fine, ~7 s chunks) → per-segment TTS. Concat and assembly are the only stages a listen mode skips. TTS throughput on Apple Silicon exceeds real-time by a wide margin with kokoro and is workable with 1–2 segment lookahead for qwen3-clone.

## Design notes

- Playback: start with `afplay` per segment (simple, gapless enough at segment granularity); upgrade to mpv with a FIFO only if seams bother us.
- Resume: `last_listened_chapter` pointer per series, stored in `progress` (separate from `last_read_chapter` — listening must NOT advance read progress, since recap gating depends on it).
- Wiped chapters listen without any wipe interplay: nothing is created.

## Phases

1. **Skeleton: segment loop + afplay.** For each narration segment: synthesize (reuse pipeline stage 2), `afplay`, delete WAV; lookahead 1 segment. `eh listen <series> [--chapter N]`.
   - Gate: listen to kenja ch 23 start-to-finish. Peak temp disk < 10 MB. No files under the chapter's video workdir afterward.
2. **Resume + progress.** `last_listened` pointer; restart mid-chapter resumes within one segment; `eh list` shows listening progress.
   - Gate: stop mid-chapter, restart, resumes at the correct segment (verify by which text is heard).
3. **(Optional) qwen3-clone voice support.** Multi-segment lookahead sized to engine speed.
   - Gate: listen with a cloned voice, no stutters.

## Dependencies

None.
