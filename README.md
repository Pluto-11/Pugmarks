# Pugmark 🐅

Turn hunting and natural-history novels into illustrated, evaluated bestiaries.

> Status: v1 in development. See `docs/superpowers/specs/2026-05-01-pugmark-design.md`.

## Quick start

```bash
uv sync
cp .env.example .env  # then fill in keys
uv run pugmark --help
```

## Layout

- `pugmark/` — package
- `prompts/` — versioned prompt templates
- `eval/` — DeepEval harness + ground truth
- `tests/` — pytest suite
- `app.py` — Gradio HF Space entry
- `docs/superpowers/` — design spec + implementation plan
