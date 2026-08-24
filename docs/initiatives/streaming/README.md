# Streaming

Status: **proposed**

Absorbs the former `r2-streaming` initiative: one umbrella for how `eh` delivers video/audio to players, with a strict division of labor between static remote storage and dynamic local generation.

## Goal

Any chapter is watchable/listenable on demand: finished videos stream from R2 (wiped locally, still playable), and dynamic content (on-the-fly TTS, read-along muxing) is served by a local HTTP bridge that every player can consume without special support.

## Non-goals

- A custom wire protocol. Evaluated and rejected (see Rationale): HTTP range requests already solve seekable streaming, and a bespoke protocol only works on players we control. The only protocol here is HTTP.
- Streaming during render (render → push → wipe stays the lifecycle).
- HLS/segmented streaming (faststart mp4 over range requests suffices at these bitrates).
- Public bucket / CDN — the bucket stays private; access is presigned URLs. The bridge binds to localhost only.

## Rationale

**Division of labor.** R2 serves static bytes — nothing else. The localhost bridge generates responses on the fly (muxing images + highlight + audio into a fragmented mp4 the player can't distinguish from a file). Static → R2, dynamic → bridge. This also gives wiped chapters a clean story: rendered-and-wiped streams from R2, never-rendered is generated live by the bridge.

**Local HTTP bridge, not a custom protocol.** A `foo://` scheme would work on mpv (Lua protocol handlers) and nowhere else — QuickTime/`open` accept only `http(s)://` and `file://`, and AVPlayer needs a custom resource loader. A localhost proxy serves plain `http://127.0.0.1:PORT/...` to *every* player, including the future native app (AVPlayer just points at the same URL). Jellyfin/Plex/yt-dlp all converged here for the same reason. The thing worth designing is the data layer (time-map), not the transport.

**R2 economics.** Storage ~$0.015/GB/mo (the whole ~60 GB library ≈ $1/mo); egress is free, so repeat watching costs nothing.

## Phases

1. **faststart everywhere.** Add `+faststart` to assembly output and `video/compress.py` encodes; verify moov position.
   - Gate: a fresh mp4 starts playing from a pipe/stream within seconds (`cat file.mp4 | mpv -`).
2. **R2 static streaming.** Presigned GET URLs (boto3 or manual SigV4); `eh play` hands the player the URL instead of pulling back; store push covers compressed copies too.
   - Gate: wipe a compressed chapter locally, `eh play` streams it from R2, forward/backward seek works.
3. **Local HTTP bridge (`eh serve`).** Stdlib/Starlette server on localhost: endpoints for chapter audio (streamed TTS via audiobook lookahead, or cached segment WAVs) and read-along fMP4 (images + ASS highlight + audio, muxed on demand from the follow-along time-map).
   - Note (2026-09-20): [desktop-ui](../desktop-ui/README.md) Phase 1 is building `eh serve` (FastAPI) for library/progress/video-streaming endpoints. The audio/read-along endpoints below should extend that same server rather than create a second one.
   - Gate: read-along for kenja ch 23 plays in mpv *and* QuickTime with panel highlight following narration; zero files persist after the player closes.
   - Note: the bridge only needs to exist once read-along ships; Phases 1–2 don't depend on it.

## Dependencies

`store/r2.py` (exists), `video/compress.py` presets (exists). Phase 3 depends on audiobook-mode (audio streaming) and follow-along-visuals (time-map format).
