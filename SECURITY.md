# Security Policy

## Reporting a vulnerability

Please report security issues privately to **abraham@elmahrek.com** rather
than opening a public issue. Include a description, reproduction steps, and
the affected version/commit. You'll get an acknowledgement within a few
days.

## Scope notes

Entertainment Harness is a local-first desktop/CLI application:

- Your library, configuration, and generated artifacts live only in your
  data directory (`~/.local/share/entertainment-harness/`). Nothing is
  transmitted anywhere unless you configure a remote model backend, a cloud
  store, or use a feature that fetches from the network (sources, search,
  model downloads).
- `config.toml` in the data dir contains your API keys (OpenRouter, Runway,
  cloud store). **Never paste it, or screenshots of it, into issues or PRs.**
- `eh serve` binds to localhost and is not designed to be exposed to a
  network. Don't put it behind a public port.
- The app executes local tools (ffmpeg, Ollama) and renders content
  fetched from third-party sources; treat untrusted input files (CBZ/EPUB
  imports from the internet) with the usual caution.

## Supported versions

Only the latest commit on `main` (and the most recent tagged release, when
those exist) receives fixes.
