# Working notes (implementation facts discovered)

## Repo facts (verified by reading code)

- `works.py`: constants VIDEO_RECAP_DIR="video-recap", VIDEO_NARRATION_DIR,
  TIKTOK_DIR; helpers `chapter_dir(series_id, chapter_id)`, `source_dir(...)`,
  `_write_json/_read_json`, `read_chapter_metadata_at(cdir)`,
  `list_chapter_dirs(series_id)`. Chapter dirs named `ch-%03d`-ish via
  `chapter_dir_name()`. Pages in source_dir are `page-001.png` etc.
- `library.parse_chapter_spec("8-10")` exists (used by `eh wipe`) — reuse for
  `--pages`.
- `cli/video.py`: command pattern — `@app.command()`, `_pipeline_config(...)`,
  `db.connect()`, `library.resolve_series(conn, name)` (title search, NOT
  works slug; "kenja" works, "kenja-no-mago" doesn't), `console/err_console`
  from `cli/__init__.py`, catch `(library.LibraryError, VideoError, ModelError,
  PluginError)` → err_console + typer.Exit(1).
- `models.registry.get_vision_model(config, profile)` → Selection(adapter,
  info); adapter.generate(model, prompt, images=[Path,...]); adapter.ensure(name).
- `video/gen/runway.py` current: RUNWAY_BASE "https://api.runwayml.com",
  RUNWAY_VERSION "2024-11-06", POLL_INTERVAL 5, MAX_POLL 600. RunwayProvider
  (config): api_key from config.video_gen.runway.api_key or env RUNWAY_API_KEY;
  model default gen3a_turbo. `_prepare_image` hard-crops to 768x1280 JPEG.
  `_submit(image,text,duration)` POSTs /v1/image_to_video {model, promptImage:
  data_uri, promptText[:512], duration 5|10, ratio "768:1280", watermark False}.
  `_poll(task_id)` GET /v1/tasks/<id>, status SUCCEEDED/FAILED, output[0] url.
  `_download(url,dest)`. `generate_segment(image, segment, duration, workdir)`
  → seg-runway-NN.mp4.
- `tests/test_video_gen.py` mocks via respx at https://api.runwayml.com/... —
  when base changes to api.dev.runwayml.com, update those URLs (modernization;
  behavior preserved). Tests: registry names ["local","runway"], animated flag,
  api key required, env key, _prepare_image → 768x1280 jpeg, generate_segment
  polls+downloads (payload contains gen3a_turbo + 768:1280), poll FAILED raises.
- `config.py`: VideoGenConfig{provider="local", runway=RunwayConfig(api_key="",
  model="gen3a_turbo")}; load_config parses [video_gen.runway]. ADD `[anime]`
  section: keyframe_model="gen4_image", video_model="gen4.5" (+ maybe
  keyframe_ratio). Env RUNWAY_API_KEY stays.
- Tests conventions: pytest, tmp_path, monkeypatch EH_DATA_DIR; fakes inline;
  respx for httpx mocking; NO paid network. PIL for images. FFmpeg allowed in
  tests only where existing tests do (shutil.which guard).
- Prior session's diff is committed? NO — working tree has uncommitted pacing
  changes (video/script.py, assemble.py, pipeline.py, visuals.py, tests,
  design.md) + this initiative. All tests green (427 passed, 1 pre-existing
  test_lfm failure on pristine tree too).
- Kenja ch0: series id 01J76XYBTCD15FW69W889WKHB0, chapter id
  01J76XYXKWZ72RD7K9JDZV9PAA, 11 pages cached in source dir. samples/kenja-ch0/
  has page-01.jpg … page-16.jpg for fixtures.

## Design decisions

- New package `src/entertainment_harness/anime/`: `scene.py` (Shot/SceneSpec
  dataclasses, to/from_dict, validation, normalized crop [l,t,r,b] 0–1 with
  Pillow crop), `planner.py` (PLANNER_PROMPT v1, parse/validate), `pipeline.py`
  (stages, fingerprints, state.json, assembly).
- Dir helper `works.anime_scene_dir(series_id, chapter_id)` =
  chapter_dir/"anime-scene"; constant ANIME_SCENE_DIR.
- Runway refactor: RUNWAY_BASE → https://api.dev.runwayml.com; generic
  `_submit_task(endpoint, payload)`, `_poll`, `_download`, `data_uri(image,
  ratio=(w,h))`; new `text_to_image(prompt, references=[(tag, Path)], ratio)`
  and `image_to_video(image, prompt, duration, ratio, model)`. TikTok
  `generate_segment` keeps gen3a_turbo/768:1280 defaults via image_to_video.
- Keyframe prompt per task spec; shot 0 omits @Anchor/@Previous lines.
- Animation prompt per task spec (motion-only wrapper around shot fields).
- Assembly: ffmpeg per-clip normalize (scale/pad 1280x720, fps 30, yuv420p,
  -an) → concat demuxer → out.mp4; ffprobe duration ≤ 30 assert.
- CLI `anime-scene` in cli/video.py: series, --chapter (float, required),
  --pages "8-10" (parse_chapter_spec), --instruction (required str),
  --stop-after {plan,keyframes}, --regenerate-shot N, --vision-model/--backend
  overrides via _pipeline_config. No DB rows; no store push.
- Shot.source_page = index INTO the selected page range or absolute page
  number? DECISION: absolute page number within chapter (1-based), validated
  against selected range — more human-meaningful when editing scene.json.
- Planner prompt: as specified in the task, + JSON schema block, + shot fields
  list, + validation retry once on invalid (mirror script.py salvage style:
  simple retry loop max 2).

## Task essentials (full text may truncate)

`eh anime-scene` vertical slice, Kenja ch0 pages 8-10 example. SceneSpec
fields: title, chapter, source_pages, instruction, scene_summary, continuity,
shots[]. Shot: index, duration_s, source_page, source_crop (opt [l,t,r,b]),
shot_type, action, composition, camera, continuity, keyframe_prompt,
animation_prompt. Validate: 3–5 shots, all 5 s, ≤30 s total, pages in range,
non-empty action/keyframe/animation, crop bounds. Keyframe refs: shot0
@Source only; shot N: @Source+@Anchor(shot-00)+@Previous(N-1 when distinct).
gen4_image text_to_image 1280:720; gen4.5 image_to_video 1280:720 duration 5.
Mock all paid calls. samples/kenja-ch0 fixtures. ffmpeg assembly separate from
recap muxer. --stop-after plan|keyframes, --regenerate-shot N. Initiative +
gate evidence. Report: architecture, files, schema, prompts, models, cache
mechanics, test results, Runway API assumptions, deferrals, 4 exact commands.
Do NOT broaden scope. Do NOT change recap/tiktok behavior.
