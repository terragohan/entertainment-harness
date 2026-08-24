# Initiatives

Tracked multi-phase work, one directory per initiative (each holding its `README.md` plus any supporting artifacts — specs, spikes, measurements). The `README.md` carries the goal, non-goals, phases with verifiable gates, and status — the same progressive-confidence format as `../plan.md`.

## Statuses

`proposed` → `active` → `done` | `dropped`. Exactly one initiative is `active` at a time.

## Process rules

- A phase is `done` only when its gate evidence is pasted into the initiative's file (command output, measurements) — never on vibes.
- New sessions: read the `active` initiative's `README.md` before touching code.
- Dropping an initiative is fine; record why in the file rather than deleting the directory.
- Initiatives may spawn side artifacts (time-map formats, config keys); keep the canonical description in the initiative directory, not scattered across `design.md`.

## Index

| Initiative | Status | Depends on |
|---|---|---|
| [desktop-ui](desktop-ui/) — Electrobun macOS app: library, playback, run triggering, settings; `eh serve` localhost bridge | done | — |
| [audiobook-mode](audiobook-mode/) — `eh listen`: streaming TTS narration, no files persisted | proposed | — |
| [streaming](streaming/) — R2 for static video, localhost bridge for dynamic content; no custom protocols | proposed | faststart flag (Phase 1) |
| [follow-along-visuals](follow-along-visuals/) — panel-crop + highlight baked into video | dropped (absorbed into anchored-scroll) | — |
| [anchored-scroll](anchored-scroll/) — position-driven video pacing: hold-and-glide over grounded regions, panel-first narration | done | — |
| [video-mode](video-mode/) — scroll presentation as a video mode alternative to Ken Burns | done | — |
| [unify-recap-narrate](unify-recap-narrate/) — one pipeline with `--detail gist\|brief\|standard\|detailed\|full`; narrations fold into recaps, one video per chapter, steering instructions | done | — |
| [player-integration](player-integration/) — mpv as the playback engine for play/stream/read-along | proposed | streaming (Phase 2), follow-along-visuals, audiobook-mode (Phases 2–3) |
| [native-experience](native-experience/) — native macOS app for interactive pan/highlight | blocked on follow-along-visuals Tier-1 evaluation | follow-along-visuals |
| [manga-to-anime-scene](manga-to-anime-scene/) — guided manga→anime slice via Runway | done | — |
| [character-bible](character-bible/) — per-work cast registry injected into recap prompts | done | — |
| [video-styles](video-styles/) — four new presentation modes for chapter videos | done | — |
| [panel-animation](panel-animation/) — `animate` video mode (AI-generated frames per beat) | done | — |
| [frame-sequence](frame-sequence/) — frame-by-frame panel animation video mode (`sequence`) | done | panel-animation |
| [plugin-extensibility](plugin-extensibility/) — capabilities as data, specific roles, zero-core-edit plugins | done | — |
