# Player integration (mpv)

Status: **proposed**

> Named "rtv player" in the original request — no such product exists; this initiative assumes **mpv**, the player discussed for follow-along visuals. Retarget if a different player was meant.

## Goal

Standardize on mpv as the playback engine for everything `eh` plays: `eh play` launches mpv instead of the OS default, R2 streaming hands mpv a presigned URL, and the read-along mode (audiobook + follow-along time-map) renders through mpv's image-playlist + ASS-highlight path. One player, one config, all surfaces.

## Non-goals

- Bundling mpv (it's a brew install; we detect and error helpfully).
- Replacing `open` entirely on macOS — keep it as fallback when mpv is absent.
- mpv Lua scripting / custom osc themes.

## Rationale

Today `eh play` shells out to `open`, which launches whatever the OS picks — fine for mp4s, useless for the things coming next: streaming URLs (mpv plays them with seek via HTTP ranges), image-playlist read-along, and ASS highlight overlays (mpv bundles libass even though our ffmpeg lacks drawtext). mpv also gives every surface the same transport controls, speed, and subtitle handling for free.

## Design notes

- Config: `[player] executable` (default `mpv`), `[player] args` (extra CLI args, e.g. `--fs`). Absent executable → fall back to `open` with a one-line notice.
- `eh play`/`eh concat` output paths stay unchanged; only the launch step changes.
- Subtitles: videos already embed a mov_text track; mpv shows it by default. No SRT sidecars needed.

## Phases

1. **`eh play` via mpv.** Launch through mpv when available; `open` fallback. `[player]` config keys.
   - Gate: `eh play kenja --narration --chapter 23` opens in mpv with subtitles visible; with mpv uninstalled, falls back to `open` without error.
2. **Streaming target for R2** (depends on streaming Phase 2). `eh play` hands mpv the presigned URL instead of pulling the file back.
   - Gate: a wiped chapter streams from R2 with forward/backward seek working (requires faststart).
3. **Read-along mode in mpv** (depends on follow-along-visuals time-map + audiobook-mode). Build an on-the-fly playlist (page/panel images with durations from the time-map) + ASS highlight track + streamed segment audio; launch mpv; delete temp files on exit.
   - Gate: kenja ch 23 plays start-to-finish with panel highlight following the narration; zero files persist after mpv closes.

## Dependencies

streaming (Phase 2), follow-along-visuals (time-map), audiobook-mode (audio streaming). Phase 1 is independent and can start immediately.
