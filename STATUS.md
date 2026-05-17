# Pugmark — Build Status

> Living handoff document. Update at the end of every work session.

**Last updated:** 2026-05-17
**Branch:** `main` (2 commits ahead of origin — push when ready)
**Remote:** `https://github.com/Ansumanbhujabal/Pugmarks.git`

---

## TL;DR — Where to resume

**Task 2 (Pydantic schemas) is done.** The next thing to do is **Task 3 (File-based cache)** — `docs/superpowers/plans/2026-05-01-pugmark-v1.md` line 566.

Resume via:
- `superpowers:subagent-driven-development` and dispatch an implementer subagent for Task 3, OR
- Execute it manually following the TDD steps in the plan.

Plan, spec, and conversation context are all committed.

```bash
git clone https://github.com/Ansumanbhujabal/Pugmarks.git pugmark
cd pugmark
uv sync
cp .env.example .env  # paste in GEMINI_API_KEY at minimum
uv run pytest -q       # should pass cleanly (no tests yet)
uv run ruff check .    # should pass cleanly
```

---

## Done ✅

### Task 0 — Brainstorming + Spec + Plan
- 7 sequenced design decisions captured (output format, image source, project name, extraction approach, scope, run mode, LLM provider)
- Design spec at `docs/superpowers/specs/2026-05-01-pugmark-design.md` (491 lines)
- Implementation plan at `docs/superpowers/plans/2026-05-01-pugmark-v1.md` (3809 lines, 18 TDD tasks)

### Task 1 — Project Scaffolding
- `pyproject.toml` (uv-managed, 10 runtime deps + 5 dev deps)
- `.gitignore`, `.env.example`, `README.md`
- `pugmark/`, `tests/` packages with `__init__.py`
- `tests/conftest.py` with `fixtures_dir` + `_disable_langfuse` fixtures
- Empty directories: `prompts/`, `eval/ground_truth/`, `eval/runs/.gitkeep`, `data/samples/`, `tests/fixtures/`
- `uv.lock` committed
- Verified: `uv sync`, all imports OK, `pytest -q` clean, `ruff check .` clean
- Spec compliance review ✅ + code quality review ✅ (one minor unused-import issue fixed)

### Task 2 — Pydantic Schemas (commits `aad05a4`, `f56bb09`)
- `pugmark/schemas.py`: 10 models (`Chapter`, `Candidate`, `ConfirmedTaxon`, `ImageRef`, `Sighting`, `TaxonCard`, `ExtractionMetrics`, `ValidationMetrics`, `EvalRun`, `Gallery`) + `Chapter.offset_to_page()` helper
- `tests/test_schemas.py`: 13 tests pass — base 6 + 6 parametrized `offset_to_page` boundaries + 1 `fuzzy_score` bounds test
- Spec compliance review ✅ (`# noqa: F401` on `Sighting`/`TaxonCard` test imports accepted as minimal-friction resolution)
- Code quality review APPROVED_WITH_NITS → 2 Important findings fixed in follow-up commit `f56bb09`:
  - `ConfirmedTaxon.fuzzy_score` now bounded `[0.0, 1.0]` (matches `best_score/100` convention in Task 9 validate stage)
  - `Chapter.offset_to_page` now has parametrized coverage at boundaries
- `ruff check .` clean

---

## Pending — In recommended execution order

Each task is fully specified in `docs/superpowers/plans/2026-05-01-pugmark-v1.md`.
Each step in the plan includes complete code, exact commands, and expected outputs.

| # | Task | Files | Dep on |
|---|---|---|---|
| **T3** | File-based cache | `pugmark/cache.py`, `tests/test_cache.py` | T2 |
| T4 | Observability (Langfuse + LiteLLM callback) | `pugmark/observability.py`, test | none |
| T5 | LLM client (LiteLLM + provider fallback) | `pugmark/llm.py`, test | T2 |
| T6 | Prompt registry + extract_taxa.v1.j2 | `pugmark/prompt_registry.py`, `prompts/extract_taxa.v1.j2`, test | none |
| T7 | Ingest stage (PDF → Chapter via PyMuPDF) | `pugmark/ingest.py`, test, fixture PDF | T2 |
| T8 | Extract stage (Chapter → Candidates) | `pugmark/extract.py`, test | T2,T3,T5,T6 |
| T9 | Validate stage (Wikidata SPARQL) | `pugmark/validate.py`, test, mock JSONs | T2,T3 |
| T10 | Enrich stage (Wikipedia + Commons) | `pugmark/enrich.py`, test, mock JSONs | T2,T3 |
| T11 | Render stage (HTML + Gradio) | `pugmark/render.py`, `pugmark/templates/gallery.html.j2`, test | T2 |
| T12 | CLI (Click) | `pugmark/cli.py`, test | T1–T11 |
| T13 | Gradio app | `app.py`, test | T1–T11 |
| T14 | Eval metrics | `eval/metrics.py`, test | T2 |
| T15 | Eval runner | `eval/runner.py`, test | T7–T11, T14 |
| **T16** | **Hand-label Sivanipalli ground truth (manual, ~30-60 min)** | `eval/ground_truth/sivanipalli.json` | T7 |
| T17 | Wire eval into CLI + regression gate | modify `pugmark/cli.py` | T12, T15, T16 |
| T18 | HF Spaces deployment + README polish | `requirements.txt`, README frontmatter | all |

**Independent tasks (can be done in any order if executed serially):** T4, T6.
**Critical-path tasks:** T2 → (T3, T5) → T8 → T12.
**Manual task that needs you, not an agent:** T16 (you read the chapter and label every taxon).

---

## Definition of v1 done

- [ ] `pugmark chapters book.pdf` lists chapters
- [ ] `pugmark extract book.pdf --chapter N --out gallery.html` produces a gallery
- [ ] `pugmark eval --chapter sivanipalli` returns metrics + writes a run JSON
- [ ] `pugmark eval --strict` returns exit 1 on F1 drop > 5%
- [ ] Gradio app on HF Space accepts uploads, produces galleries
- [ ] Langfuse dashboard shows traces from ≥ 5 end-to-end runs
- [ ] README has v1 baseline metrics filled in
- [ ] All tests pass, all lint passes

---

## Decisions locked in (do not re-litigate)

These came out of the 7-question brainstorming on 2026-05-01. Do not relitigate during execution; if something feels wrong, escalate it as a design revision rather than ad-hoc deviation.

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

The remote is `https://github.com/Ansumanbhujabal/Pugmarks.git` (with an `s`) but the local project is named `pugmark` (no `s` — matches `pyproject.toml`, the design spec, the plan, and the CLI entrypoint `pugmark = "pugmark.cli:cli"`). When you resume, decide:
- **Option A:** Rename the GitHub repo to `Pugmark` (Settings → Rename) — keeps things consistent.
- **Option B:** Leave it as-is — `Pugmarks` becomes the public name, `pugmark` stays the Python package.

Either is fine. Most users will discover the project through GitHub, so the repo name is the public-facing one.
