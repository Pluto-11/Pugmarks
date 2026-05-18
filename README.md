---
title: Pugmark
emoji: 🐅
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: mit
---

# Pugmark 🐅

Turn hunting and natural-history novels into illustrated, evaluated bestiaries.

Pugmark reads a chapter from a book like *The Kenneth Anderson Omnibus*, extracts every animal, plant, and fungus mentioned (literal names AND lyrical references like *"the great striped one"*), validates each against Wikidata, and produces a gallery of cards with real photographs from Wikimedia Commons, Wikipedia summaries, and the page numbers where each species appears.

## Architecture highlights

- **Five-stage async pipeline:** ingest → extract → validate → enrich → render.
- **LLM provider abstraction** via LiteLLM — Gemini 2.0 Flash primary, Groq fallback, Ollama for offline dev.
- **Versioned prompts** in `prompts/`, fetched via a registry that supports Langfuse overlay.
- **File-based cache** with version-keyed invalidation across all five stages, atomic writes, corrupt-file self-eviction.
- **Automated ground-truth labeling** via LLM-as-judge (Gemini 2.5 Pro) with 3-shot majority voting + Wikidata round-trip — no manual annotation needed to scale to new books.
- **CLI regression gate**: `pugmark eval --strict` exits non-zero if F1 drops more than 5% vs the most recent prior run.
- **Langfuse tracing** for every LLM and HTTP call (no-op without keys).

## v1 baseline (Sivanipalli chapter)

Fill these in after running `pugmark autolabel` to produce ground truth, then `pugmark eval` to measure:

| Metric | Value |
|---|---|
| Extraction F1 | _pending eval_ |
| Validation QID accuracy | _pending eval_ |
| Hallucination rate | _pending eval_ |
| Latency | _pending eval_ ms |

## Quick start (local)

```bash
git clone https://github.com/Ansumanbhujabal/Pugmarks pugmark
cd pugmark
uv sync
cp .env.example .env  # fill in GEMINI_API_KEY at minimum
```

List chapters in a PDF:
```bash
uv run pugmark chapters path/to/book.pdf
```

Build a gallery from a chapter:
```bash
uv run pugmark extract path/to/book.pdf --chapter 1 --out gallery.html
```

Auto-label a chapter for the eval harness (LLM-as-judge → Wikidata roundtrip):
```bash
uv run pugmark autolabel path/to/book.pdf \
  --chapter 1 --out eval/ground_truth/sivanipalli.json
```

Run the eval harness with regression gate:
```bash
uv run pugmark eval path/to/book.pdf \
  --chapter 1 \
  --ground-truth eval/ground_truth/sivanipalli.json \
  --strict
```

## Deployment (HuggingFace Spaces)

This repo doubles as a HuggingFace Space. The YAML frontmatter at the top of this README, combined with `requirements.txt` and `app.py`, is everything the HF Spaces stock-Gradio builder needs.

To deploy:
```bash
git remote add hf https://huggingface.co/spaces/<your-username>/pugmark
git push hf main
```

Then in Space Settings → Variables and secrets, add:
- `GEMINI_API_KEY` (required)
- `GROQ_API_KEY` (optional, for fallback)
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (optional, for tracing)

The cache automatically routes to `/data/.cache/pugmark` on HF Spaces (detected via the `SPACE_ID` env var).

## Roadmap

- v1: real photos via Wikimedia Commons, auto-labeled Sivanipalli ground truth, HF Space deploy
- v2: AI-generated illustrations via Colab (SDXL Turbo / Flux Schnell)
- v3: reading-companion view + self-hosted Langfuse
- v4: full omnibus + cross-chapter analytics

See `docs/superpowers/specs/2026-05-01-pugmark-design.md` for the complete design and `docs/superpowers/plans/2026-05-01-pugmark-v1.md` for the implementation plan.

## Credits

Built on PyMuPDF, LiteLLM, Wikidata, Wikipedia, Wikimedia Commons, Langfuse, Gradio, and DeepEval. All gallery images are CC-licensed; attribution is rendered visibly on every card.
