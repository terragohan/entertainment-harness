# Contributing

Thanks for helping out. This document covers dev setup, the gates every
change must pass, and the conventions that keep the project navigable.

## Dev setup

```sh
git clone https://github.com/terragohan/entertainment-harness
cd entertainment-harness
uv sync                  # python deps, including the dev group (pytest, pyinstaller)
cd ui && bun install     # desktop app deps (Bun + TypeScript)
```

Optional extras, only needed for specific features:

```sh
uv sync --extra dataset  # huggingface backend (LFM2.5-VL via transformers)
uv sync --extra sam      # Segment Anything panel detection
uv sync --extra qwen3    # voice cloning via mlx-audio
```

Use `uv run …` for everything Python (`uv run eh`, `uv run pytest`) so you
always hit the project environment.

## Gates — run these before opening a PR

```sh
uv run pytest -q                 # python suite
cd ui && bun run typecheck       # tsc --noEmit
cd ui && bun run smoke           # headless UI test against a real eh serve
```

- The python suite has **one known pre-existing failure**:
  `tests/test_lfm.py::test_repair_json_repairs_trailing_comma` (Python 3.13
  json strictness). Everything else must be green.
- Run only one app/`eh serve` instance per data directory — SQLite is
  single-writer. Point `EH_DATA_DIR` at a scratch dir in dev.

## Conventions

- **Docs move with behavior.** `docs/design.md` is the canonical
  architecture/schema reference — update it in the same change that alters
  documented behavior.
- **Database migrations are additive-only** `ALTER TABLE`s in `db.py`.
- The CLI is [Typer](https://typer.tiangolo.com) under
  `src/entertainment_harness/cli/`; library/works helpers live in
  `library/works.py`.
- Match the surrounding code style; don't add dependencies without a
  discussion in an issue first.
- Tests: add them when the project area already has tests; mirror the
  existing fixtures (`tests/conftest.py`, `tests/test_cli.py` for CLI
  patterns).

## Initiatives (multi-phase features)

Larger features are tracked as *initiatives* under `docs/initiatives/` —
one directory each, with goal, non-goals, phases, and gate evidence. Rules:

- Exactly one initiative is `active` at a time; check
  [`docs/initiatives/README.md`](docs/initiatives/README.md) before
  starting feature work.
- A phase is done only when its gate evidence (command output,
  measurements) is pasted into the initiative file.
- Dropping an initiative is fine — record the reason, keep the directory.

## Reporting bugs

Open an issue with:

- macOS version, Python version (`python3 --version`), ffmpeg version
  (`ffmpeg -version`), and how you installed (repo clone / app)
- the exact command or app action, and the full error output
- relevant log lines from your data dir

**Redact API keys.** Never paste your `config.toml` — it contains your
OpenRouter/Runway keys. See [`SECURITY.md`](SECURITY.md).

## Pull requests

- One thing per PR; reference the issue it closes.
- Gates green (above), `docs/design.md` updated if behavior changed.
- Describe *what* changed and *why*; paste before/after output for
  user-facing changes.

## Conduct

Be respectful and constructive. Assume good intent; disagree on technical
merits. That's the whole policy — maintainers may remove
contributions/commenters that can't manage it.

## License of contributions

By submitting a contribution you agree it is licensed under the project's
license: [GPL-3.0](LICENSE) with the
[Commons Clause](COMMONS-CLAUSE.md) condition, copyright the project
licensor (Terra Gohan). If that's a problem (e.g. your employer's
policy), talk to us first — open an issue:
<https://github.com/terragohan/entertainment-harness/issues>.
