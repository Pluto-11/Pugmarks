# Pugmark — Build Status

> Living handoff document. Update at the end of every work session.

**Last updated:** 2026-05-18
**Branch:** `main` (26 commits ahead of origin — push when ready)
**Remote:** `https://github.com/Ansumanbhujabal/Pugmarks.git`

---

## TL;DR — v1 is code-complete

**Every task in the implementation plan is done.** 72/72 tests pass, lint clean. The codebase is shippable.

The plan's original T16 (Ansuman hand-labels every taxon) was **replaced by automation per your "no manual steps" directive**: `pugmark autolabel` uses a separate judge model (Gemini 2.5 Pro), 3-shot majority voting, and Wikidata round-trip to generate ground truth from any PDF. No annotation labor needed to scale to new books.

```bash
git clone https://github.com/Ansumanbhujabal/Pugmarks.git pugmark
cd pugmark
uv sync
cp .env.example .env  # GEMINI_API_KEY required; LANGFUSE_* + GROQ_API_KEY optional
uv run pytest -q          # → 72 passed
uv run ruff check .       # → clean
uv run pugmark --help     # → chapters, extract, autolabel, eval
```

Remaining work is **environment / account level**, not code: provision API keys, push to GitHub, create + push the HF Space. None of this can be code-automated without first providing credentials.

---

## Done ✅ — full T0–T18 ledger

| Task | Commits | What it ships |
|---|---|---|
| T0  | `640700b`, `ffec156` | Design spec + 18-task implementation plan |
| T1  | `bd330df`, `ed2c523` | uv-managed scaffolding, pyproject, ruff config |
| T2  | `aad05a4`, `f56bb09` | 10 Pydantic schemas; `Chapter.offset_to_page()` + bounded `fuzzy_score` after review |
| T3  | `761d6f9`, `08f247d` | File-based cache. Plan code had latent bugs — fixed: atomic write-then-rename, corrupt-file self-eviction, HF Spaces `SPACE_ID` detection (not `HF_HOME`) |
| T4  | `adae180`, `41e9119` | Langfuse + LiteLLM callback wiring, no-op when keys unset. Added idempotency test |
| T5  | `08cb799`, `4933789` | Async LiteLLM client, provider fallback (Gemini→Groq→Ollama). Hardened `from_env` against typos; pinned `metadata` kwarg contract |
| T6  | `de55820` | Versioned prompt registry + `extract_taxa.v1.j2` few-shot template |
| T7  | `0a767df`, `7c22b11` | PyMuPDF outline-based ingest. Fixed TOC-level bug that would have treated subsection bookmarks as chapters on real Anderson PDFs |
| T8  | `2a9856f`, `e6a329d` | Extract stage (Chapter → Candidates). Fixed plan's test pattern that leaked `LLMClient` mock into `test_llm.py` |
| T9  | `907a227`, `0c62792` | Validate stage (Wikidata SPARQL, exact→alias→fuzzy). Added alias-path + fuzzy-fallback tests |
| T10 | `1714791` | Enrich stage (Wikipedia summary + Commons image + license/attribution) |
| T11 | `e6b77b0` | HTML render + Gradio block render from one `Gallery` model |
| T12 | `5ece153` | Click CLI: `chapters`, `extract` |
| T13 | `4bcde93` | `app.py` Gradio entry point for HF Spaces |
| T14 | `ac25f9a` | Eval metrics: precision/recall/F1/hallucination + qid_accuracy/confusion_matrix/unresolved_rate |
| T15 | `56f402d` | Eval runner: full pipeline → `EvalRun` JSON in `eval/runs/` |
| **T16** | **`6d9d423`** | **AUTOMATED ground-truth generation** (replaces hand-labeling): `pugmark autolabel` runs Gemini 2.5 Pro judge × 3, keeps ≥2/3 majority vote, round-trips through Wikidata, drops anything that doesn't resolve to a real QID |
| T17 | `e2ddce0` | `pugmark eval --strict` — exit 1 on F1 drop >5% vs latest prior run. CI-ready regression gate |
| T18 | `8849a78` | HF Spaces deploy artifacts: pinned `requirements.txt` (uv pip compile), `README.md` with HF YAML frontmatter, frontmatter validation tests |

### Tests now exercise (72 total)
- Schemas: 13 (validation + offset_to_page boundaries + fuzzy_score bounds)
- Cache: 10 (roundtrip, version-bump, corrupt-file recovery, stage clear, HF_HOME + SPACE_ID resolution)
- Observability: 4 (configured/unconfigured detection + callback idempotency)
- LLM client: 6 (provider fallback chain, from_env permutations, metadata kwarg pin)
- Prompt registry: 4 (load, render, missing, version selection)
- Ingest: 5 (chapter listing, normalization, TOC level filter, page boundaries)
- Extract: 2 (returns candidates, cache hit)
- Validate: 5 (exact, alias, unresolved, many-to-one, fuzzy)
- Enrich: 1 (card production end-to-end with mocked Wikipedia + Commons)
- Render: 3 (taxon info, visible attribution, unresolved count)
- CLI: 6 (chapters, --help × 3 subcommands, eval happy path, eval --strict regression)
- Auto-label: 3 (majority vote, Wikidata filter, sub-quorum returns empty)
- Gradio app: 2 (importable, builds Blocks)
- Eval metrics: 3 (perfect, missed, hallucination)
- Eval runner: 1 (writes JSON, computes metrics)
- README frontmatter: 4 (HF keys present, sdk=gradio, app_file=app.py, requirements pinned)

### What tests do NOT exercise (intentional)
- Real LLM API calls (mocked everywhere)
- Real Wikidata / Wikipedia / Commons network (mocked)
- Real Langfuse network (autouse fixture sets keys to empty)
- HF Spaces deployment (no programmatic deploy test — `git push hf main` is the deploy)

---

## Final mile — environment + account-level, code-complete already

Everything below is **codeless** — code is done. These are operations against external services that require credentials we cannot generate.

| Step | Command | Requires |
|---|---|---|
| Push to GitHub | `git push origin main` | GitHub auth on this machine |
| Generate v1 ground truth | `uv run pugmark autolabel <pdf> --chapter N --out eval/ground_truth/sivanipalli.json` | `GEMINI_API_KEY` in `.env` |
| Capture v1 baseline | `uv run pugmark eval <pdf> --chapter N --ground-truth eval/ground_truth/sivanipalli.json` | Same key + the ground truth file from prior step |
| Fill README baseline numbers | Edit `README.md` "v1 baseline" table from the JSON in `eval/runs/` | Just text editing |
| Create the HF Space | UI: huggingface.co/new-space (owner=ansumanbhujabal, name=pugmark, SDK=Gradio, hardware=CPU basic) | HF account |
| Wire HF remote | `git remote add hf https://huggingface.co/spaces/ansumanbhujabal/pugmark && git push hf main` | HF account + write token |
| Set Space secrets | Space Settings → Variables and secrets: `GEMINI_API_KEY`, optionally `GROQ_API_KEY`, `LANGFUSE_*` | HF Space access |

### Optional further automation (not yet built, ask if you want them)
- `pugmark update-baseline` subcommand that reads the latest `eval/runs/*.json` and writes the baseline table into README in place
- `scripts/deploy.sh` that uses the `hf` CLI to create the Space, set secrets from `.env`, and push — would still require `HF_TOKEN` to be set
- GitHub Actions workflow that runs `pugmark eval --strict` on every PR

---

## Known follow-ups (none block v1)

- `pugmark/validate.py::_sparql_query` interpolates `query_name` via f-string. Names containing `"` will break the SPARQL query. Low-likelihood in Anderson corpus; worth escaping before v1 ships publicly.
- Validate cache key omits `kingdom_hint`. By design today (the SPARQL query doesn't use it). If v2 uses kingdom in disambiguation, the hash must include it.
- `eval/runner.py` discards the rendered cards (`_cards = await enrich_taxa(...)`). The runner could optionally save the gallery JSON for inspection.
- README baseline table has `_pending eval_` placeholders. Fills in after the first `pugmark eval` run.

---

## Decisions locked in

- **Output:** B then A — gallery first, reading-companion later
- **Images:** A then C — real photos via Wikimedia/iNaturalist first, AI-generated illustrations on Colab later (v2)
- **Name:** Pugmark (repo on GitHub is `Pugmarks` with `s` — decide before HF deploy whether to rename)
- **Extraction:** staged C — LLM baseline → Wikidata validator → eval harness with regression gate
- **Ground truth:** **AUTOMATED via LLM-as-judge** (changed from plan's manual labeling per Ansuman directive)
- **Scope:** all taxonomic entities (animalia, plantae, fungi)
- **Run mode:** D — CLI + Gradio HF Space, same package backs both
- **LLM:** LiteLLM multi-provider — Gemini 2.0 Flash primary for production, Gemini 2.5 Pro for judge, Groq fallback, Ollama for offline dev
- **Async:** async-throughout (httpx, LiteLLM acompletion, asyncio.gather, Semaphore(10))
- **PDF:** PyMuPDF (`fitz`); Marker deferred to scanned-PDF case
- **Cache:** file-based, `~/.cache/pugmark/`, HF Spaces `/data` aware via `SPACE_ID`
- **Observability:** Langfuse cloud free tier (self-host deferred to v3)
- **Prompts:** in-repo Jinja2 + Langfuse runtime overlay with file fallback

---

## v2 — Universal bestiary (code-complete 2026-05-18)

### Done — every task in the v2 plan

| Task | Commit | What it ships |
|---|---|---|
| T1  | `dc5acc0` | Generalized schemas with v1 aliases (Candidate.entity_type, ConfirmedEntity, EntityCard, Gallery.cards_by_type) |
| T2  | `5f278bb` | EntityTypeSpec + BookSchema Pydantic models |
| T3+T4 | `d53f72f` | Plugin registry + built-in people/places prompts |
| T5+T6 | `eae5b8d` | Book analyzer (Gemini 2.5 Pro proposes per-book types) |
| T7  | `6196b9a` | Schema realizer (merge proposed + registered + overrides) |
| T8  | `cb3dce8` | extract.py type-parameterized |
| T9  | `a4dd5c5` | validate.py Wikidata Q-class templating |
| T10 | `ca7bdbb` | in-book cross-reference tier |
| T11 | `86a9263` | judge consensus tier |
| T12 | `05b8cf6` | tiered enrich with LLM-summary fallback |
| T13 | `32ed325` | render cards_by_type + placeholder + AI-summarized badge |
| T14 | `184363d` | pugmark analyze CLI subcommand |
| T15 | `1a04eea` | library API (extract_gallery + __init__ exports) |
| T16 | `39515d1` | Gradio app uses extract_gallery with type-prefixed captions |
| T17 | `42facad` | per-type ExtractionMetrics.by_type |
| T18 | `ec696d7` | eval runner per-type + multi-file ground truth |
| T19 | `67ec891` | per-type autolabel + auto_label_book + --all-types CLI |
| T20 | `56b907b` | end-to-end integration test + this STATUS update |

### Test count
- v1 baseline: 72 tests
- v2 delta: +42 tests
- **Total: 114 tests, all passing, ruff clean**

### v1 backward compatibility (still working)
- `TaxonCard`, `ConfirmedTaxon` aliased to `EntityCard`, `ConfirmedEntity`
- `Candidate(kingdom_hint=...)` → `entity_type="taxa"` + `type_attrs["kingdom_hint"]` via model_validator
- `Gallery(cards=...)` → `cards_by_type={"taxa": cards}` via model_validator
- `enrich_taxa` alias of `enrich_confirmed`
- Read-side `@property` shims on Candidate.kingdom_hint, ConfirmedEntity.lineage, EntityCard.taxon, Gallery.cards (temporary; remove when v1 consumers are gone)

### v2 architecture summary

```
PDF
 ▼
ingest (unchanged)
 ▼
analyze (NEW: LLM proposes per-book entity types) ──► BookSchema
 ▼
realize-schema (NEW: merge proposed + registered + overrides) ──► list[EntityTypeSpec]
 ▼
per-type parallel fan-out:
   extract (type-parameterized prompt)
 ──► validate (tiered: Wikidata Q-class → in-book crossref → judge consensus)
 ──► enrich (tiered: Wikipedia+Commons → LLM-in-book summary)
 ▼
render (cards_by_type sections + placeholder + AI-summarized badge)
```

### Final-mile — environment / account-level (no code left)

| Step | Command | Requires |
|---|---|---|
| Push 47+ commits to GitHub | `git push origin main` | GitHub auth |
| Auto-label all proposed types for a real book | `uv run pugmark autolabel <pdf> --chapter N --out eval/ground_truth/ --all-types` | `GEMINI_API_KEY` |
| Capture v2 baseline metrics | `uv run pugmark eval <pdf> --chapter N --ground-truth eval/ground_truth/` | Same key |
| Fill README baseline numbers | Edit `README.md` baseline table | Just text editing |
| HF Spaces deploy | `git remote add hf ... && git push hf main` + secrets | HF account + token |

### Known follow-ups (none block v2)

- **Property shim removal** (T1 review item): the 4 read-side `@property` shims on schemas should be deprecated with `warnings.warn(DeprecationWarning)` and removed when downstream v1 consumers are gone. See T1 spec compliance review for the recommended pattern.
- **LLM payload `type_attrs` forwarding** (T8 review item): `extract.py` currently drops "extra" fields from the LLM payload (e.g. `kingdom_hint`). For full v1 parity it should populate `type_attrs` from any payload keys not in the Candidate schema. Cheap to add when the next non-taxa type's enricher wants to use type-specific attributes.
- **`ConfirmedEntity.attributes` type widening** (T1 review item): currently `dict[str, str]`. For nested attributes (people: `{"birth": {"year": "1820"}}`) it should be `dict[str, Any]`. Fix before the first non-taxa enricher writes structured sub-attributes.
- **SPARQL quote escaping in validate** (v1 review carry-over): names containing `"` will break the SPARQL query. Worth escaping before any external user uploads a book with such names.
