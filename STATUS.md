# Pugmark — Build Status

> Living handoff document. Update at the end of every work session.

**Last updated:** 2026-05-18
**Branch:** `main` (22 commits ahead of origin — push when ready)
**Remote:** `https://github.com/Ansumanbhujabal/Pugmarks.git`

---

## TL;DR — Where to resume

**Tasks T3–T15 (all automated implementation tasks) are done.** The full v1 pipeline ships green: 61/61 tests pass, `ruff check .` clean, smoke-tested via `uv run pugmark chapters tests/fixtures/sample_chapter.pdf`.

The next thing to do is **T16: hand-label the Sivanipalli chapter ground truth** — this is manual work that cannot be delegated to a subagent. It blocks T17 (regression gate) and T18 (HF Spaces deploy).

```bash
# Resume cold:
git clone https://github.com/Ansumanbhujabal/Pugmarks.git pugmark
cd pugmark
uv sync
cp .env.example .env  # paste GEMINI_API_KEY + LANGFUSE_* at minimum
uv run pytest -q       # → 61 passed
uv run ruff check .    # → clean
uv run pugmark chapters tests/fixtures/sample_chapter.pdf  # synthetic fixture works
```

---

## Done ✅

### Task 0 — Brainstorming + Spec + Plan
- 7 sequenced design decisions captured (output format, image source, project name, extraction approach, scope, run mode, LLM provider)
- Design spec at `docs/superpowers/specs/2026-05-01-pugmark-design.md` (491 lines)
- Implementation plan at `docs/superpowers/plans/2026-05-01-pugmark-v1.md` (3809 lines, 18 TDD tasks)

### Task 1 — Project Scaffolding (`bd330df`, `ed2c523`)

### Task 2 — Pydantic Schemas (`aad05a4`, `f56bb09`)
10 models + `Chapter.offset_to_page()`. Code review found 2 Important items, fixed in follow-up: `fuzzy_score` bounded `[0.0, 1.0]`, parametrized boundary test for `offset_to_page`.

### Task 3 — File-based cache (`761d6f9`, `08f247d`)
Atomic write-then-rename in `set()`, corrupt-file recovery in `get()`, HF Spaces detection via `SPACE_ID` env var (not `HF_HOME`). Plan code had 3 latent bugs that would have hit T8-T10 once `asyncio.gather` ran concurrent cache writes.

### Task 4 — Observability (`adae180`, `41e9119`)
`init_observability()` adds `"langfuse"` to LiteLLM's success/failure callbacks if `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set; otherwise no-op. Follow-up commit added a callback-registration test that catches typo/dedup regressions.

### Task 5 — LLM client (`08cb799`, `4933789`)
Async LiteLLM client with `Gemini → Groq → Ollama` fallback chain. `LLMConfig.from_env()` reads `PUGMARK_PROVIDERS` + `PUGMARK_PRIMARY_MODEL`. Follow-up hardened `from_env` against typos (silently ignores unknown providers instead of `KeyError`) and tightened the `metadata` kwarg assertion that pins the Langfuse trace contract.

### Task 6 — Prompt registry + first template (`de55820`)
`PromptRegistry` loads `prompts/{name}.{version}.j2`; default `get(name)` picks highest version. Few-shot extraction prompt with two Anderson-flavored examples (tiger / great striped one; peepul / sambhur).

### Task 7 — Ingest stage (`0a767df`, `7c22b11`)
PyMuPDF outline-based chapter detection + text normalization (dehyphenation, smart quotes, whitespace collapse). Follow-up fixed a TOC-level bug that would have treated subsection bookmarks as chapters once a real Kenneth Anderson PDF lands, and tightened the page-boundary test.

### Task 8 — Extract stage (`2a9856f`, `e6a329d`)
Chapter → list[Candidate] via LiteLLM structured output. Cache key includes prompt_version + provider list. Follow-up converted the test's direct class-attribute mutation to `monkeypatch.setattr` — the plan's pattern leaked the mock into `test_llm.py` and broke 3 tests.

### Task 9 — Validate stage (`907a227`, `0c62792`)
Wikidata SPARQL with exact → alias → fuzzy (rapidfuzz threshold 85) → unresolved. Async fan-out with `Semaphore(10)`, per-name caching, many-to-one collapse on QID. Follow-up added the missing alias-path and fuzzy-fallback tests (previously the most complex code path had zero coverage).

### Task 10 — Enrich stage (`1714791`)
ConfirmedTaxon → TaxonCard via Wikidata sitelinks (Wikipedia REST) + Commons API (image with extmetadata license + attribution). Cache-hit refreshes sightings from the current chapter.

### Task 11 — Render stage (`e6b77b0`)
HTML via Jinja2 (`pugmark/templates/gallery.html.j2`, inline CSS, visible attribution per card) + lazy-imported Gradio Blocks (`render_gradio`). Same `Gallery` model feeds both.

### Task 12 — CLI (`5ece153`)
Click-based CLI: `pugmark chapters <pdf>` and `pugmark extract <pdf> --chapter N --out F.html`. Loads `.env`, initializes observability. Smoke-tested against the synthetic fixture PDF.

### Task 13 — Gradio app (`4bcde93`)
`app.py` at repo root — HF Spaces entry point. Upload PDF → pick chapter → see gallery + summary. Reuses the same pipeline functions as the CLI.

### Task 14 — Eval metrics (`ac25f9a`)
`eval/metrics.py`: `compute_extraction_metrics` (precision/recall/F1/hallucination) + `compute_validation_metrics` (qid_accuracy, confusion_matrix, unresolved_rate). Hallucination = surface_form absent from chapter text.

### Task 15 — Eval runner (`56f402d`)
`eval/runner.py::run_eval`: loads ground truth, runs ingest → extract → validate → enrich, computes metrics, writes timestamped `EvalRun` JSON to `runs/`. DeepEval integration deferred to v1.5 once we have multiple runs to compare.

---

## Pending — Manual + final-mile

| # | Task | Files | Notes |
|---|---|---|---|
| **T16** | **Hand-label Sivanipalli ground truth** | `eval/ground_truth/sivanipalli.json` | **Manual: ~30–60 min. You read the chapter and label every taxon. Ansuman-only task. Blocks T17, T18.** |
| T17 | Wire eval into CLI + regression gate | modify `pugmark/cli.py` | Adds `pugmark eval --chapter NAME [--strict]` ; `--strict` returns exit 1 on F1 drop > 5%. Needs T16. |
| T18 | HF Spaces deployment + README polish | `requirements.txt`, README frontmatter, baseline metrics | Push to HF Space, fill in v1 baseline metrics in README. Needs all prior. |

---

## State of the world

- **Tests**: 61 passing, 0 failing
- **Lint**: clean
- **Code paths exercised by tests**:
  - All pipeline stages with mocked LLM / network
  - Cache: roundtrip, version-bump, corrupt file, clear (stage + all), HF Spaces detection (HF_HOME + SPACE_ID)
  - LLM client: provider success, fallback, all-fail, `from_env` (default, primary swap, unknown provider), metadata kwarg
  - Ingest: chapter listing, normalization (dehyphenation), TOC level filtering, page-boundary mapping
  - Extract: returns candidates, second call hits cache
  - Validate: exact match, alias match, unresolved, many-to-one collapse, fuzzy fallback
  - Enrich: card production with Wikipedia + Commons
  - Render: HTML output contains taxon/attribution/unresolved
  - CLI: chapters subcommand, help includes extract
  - Gradio: build_app returns gr.Blocks
  - Eval metrics: perfect / missed / hallucination
  - Eval runner: writes EvalRun JSON, computes correct metrics
- **Not exercised in tests** (intentional / known): real LLM calls, real Wikidata queries, real Wikipedia/Commons fetches, real Langfuse network. End-to-end smoke test against a real PDF is a manual run.

## Known follow-up items (none block T16)

- `pugmark/validate.py::_sparql_query` f-string interpolation breaks on names containing `"`. Low-likelihood in Anderson corpus but worth escaping before public deployment.
- `pugmark/validate.py` cache key omits `kingdom_hint` — by design today, but if v2 uses kingdom in disambiguation the hash must include it.
- `eval/runner.py` `_cards` is discarded (`enrich_taxa` is called for side-effect of populating cache). Worth flagging when CLI integration lands in T17 — the runner could optionally save the gallery JSON.

---

## Decisions locked in (do not re-litigate)

These came out of the 7-question brainstorming on 2026-05-01. Do not relitigate during execution; if something feels wrong, escalate as a design revision rather than ad-hoc deviation.

- **Output:** B then A — gallery first, reading-companion later
- **Images:** A then C — real photos via Wikimedia/iNaturalist first, AI-generated illustrations on Colab later (v2)
- **Name:** Pugmark
- **Extraction:** staged C — LLM baseline → Wikidata validator → eval harness with regression gate
- **Scope:** all taxonomic entities (animalia, plantae, fungi)
- **Run mode:** D — CLI + Gradio HF Space, same package backs both
- **LLM:** LiteLLM multi-provider — Gemini 2.0 Flash primary, Groq fallback, Ollama for offline dev
- **Async:** async-throughout (httpx, LiteLLM acompletion, asyncio.gather, Semaphore(10))
- **PDF:** PyMuPDF (`fitz`) — Marker reserved for scanned-PDF future case
- **Cache:** file-based, `~/.cache/pugmark/`, HF Spaces `/data` aware
- **Observability:** Langfuse cloud free tier (self-host deferred to v3)
- **Prompts:** in-repo Jinja2 + Langfuse runtime overlay with file fallback

---

## Repo URL note

The remote is `https://github.com/Ansumanbhujabal/Pugmarks.git` (with an `s`) but the local project is named `pugmark`. Decide before T18 (HF deployment) whether to rename the GitHub repo to `Pugmark` for consistency, or leave it as-is.
