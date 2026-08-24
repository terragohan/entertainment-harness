# Native experience (app)

Status: **blocked** — decision gate is the last section of [follow-along-visuals](../follow-along-visuals/README.md). Do not start before that evaluation.

> Note (2026-09-20): the [desktop-ui](../desktop-ui/README.md) initiative (Electrobun app + `eh serve` bridge) supersedes this initiative's SwiftUI viewer approach for the v1 desktop experience and realizes its `eh serve --local` sketch. If desktop-ui ships and covers the interactive-viewing need, mark this `dropped` with that reason.

## Goal (if pursued)

A platform-native app for interactive narration viewing: page rendering with free pan/zoom (trackpad/mouse), automatic highlight following the narration, tap/click-a-panel-to-seek, playback speed and scrubbing that feel like a real media player. First-class citizen on macOS (SwiftUI); the time-map format should be platform-neutral so other frontends are possible later.

## Non-goals

- Cross-platform on day one (macOS first; Linux/Windows only if the backend shape demands it).
- Replacing the CLI for pipeline operations — the app is a *viewer*, generation stays in `eh`.
- Video re-encoding or a video pipeline inside the app — it consumes pages + audio + the time-map, nothing heavier.

## Why native, not a web UI

The interactive surface here is exactly what native toolkits are good at: buttery pan/zoom gestures, media-key playback control, system-wide volume/seek behavior, zero server process to babysit. A localhost web app would rebuild half of that with a trackpad-disappointing imitation. The cost is real (Swift, Xcode, app packaging), which is precisely why this initiative is gated behind proving the interactive need exists.

## Sketch (so the decision is informed)

- **Backend**: keep everything in Python. The app shells out to / embeds a small local bridge (e.g. a Python XPC helper or `eh serve --local` one-shot) that streams TTS audio (audiobook-mode lookahead) or serves cached segment WAVs, plus the time-map JSON: `[{t_start, panel_bbox, page}]`.
- **Frontend**: SwiftUI. Page image in a `ScrollView`/`MagnifyGesture` composition; highlight layer drawn from the time-map; audio via `AVPlayer`; panel hit-testing for tap-to-seek.
- **Shared contract**: the time-map format is designed with follow-along-visuals Phase 1, so the baked-in video and the app stay in lockstep.
- Effort estimate: 3–5 days for a solid macOS MVP including the Python bridge.

## Decision gate

1. Follow-along-visuals Phase 1 ships and is watched for ~1 week.
2. If the wish for *manual* control is still there → promote to `proposed`, start by freezing the time-map format shared with Tier 1.
3. If not → mark `dropped` with the reason recorded here.

## Dependencies

follow-along-visuals (evaluation + time-map format), audiobook-mode (audio source).
