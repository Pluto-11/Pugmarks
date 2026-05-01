# Pugmark — Design Spec

**Date:** 2026-05-01
**Author:** Ansuman SS Bhujabala
**Status:** Approved (brainstorming complete; ready for implementation plan)
**Workspace:** `/opt/CodeRepo/pugmark/`

---

## 1. North Star

**Pugmark turns hunting and natural-history novels into illustrated, evaluated bestiaries.**

Given a PDF of a book like *The Kenneth Anderson Omnibus* or *The Jim Corbett Omnibus*, Pugmark extracts every taxonomic mention from a chapter (animals, plants, fungi), validates each against Wikidata, and produces a gallery of cards — one per species — with a real photograph, a Wikipedia summary, taxonomic lineage, and the page numbers where the species is mentioned in the chapter.

The project doubles as a portfolio piece demonstrating production-grade AI-engineering practices: provider-agnostic LLM abstraction, async I/O concurrency, file-based caching with versioned invalidation, distributed tracing via Langfuse, versioned prompts in git, and an automated eval harness with regression gates.

### Design tenets

1. **One pipeline, three faces.** A clean Python package powers a CLI, a Gradio app on HuggingFace Spaces, and Colab notebooks. The same pure functions back all three.
2. **Sequential at the chapter level, async within stages.** Async lives inside `enrich` and `validate` for I/O fanout; the chapter-level pipeline stays a readable top-to-bottom flow.
3. **Cache everything, version everything.** Every stage caches its output keyed by content hash plus a version string. Re-runs are free; logic changes invalidate cleanly.
4. **Evals are first-class.** Hand-labeled ground truth, automated metrics, regression gates, and Langfuse traces aren't an afterthought — they're the most interview-relevant artifact in the repo.
5. **YAGNI ruthlessly.** Features that don't serve v1 are listed explicitly under "Out of Scope" and deferred to numbered roadmap milestones.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Pugmark                              │
│                                                              │
│   PDF / text  ──►  ingest  ──►  extract  ──►  validate      │
│                      │            │             │            │
│                      ▼            ▼             ▼            │
│                   chapters    candidates    confirmed taxa   │
│                                                              │
│                                          ──►  enrich  ──►   │
│                                                  │           │
│                                                  ▼           │
│                                           taxon cards        │
│                                          (image+wiki+meta)   │
│                                                  │           │
│                                                  ▼           │
│                                              render          │
│                                          (HTML / Gradio)     │
└─────────────────────────────────────────────────────────────┘

Three faces, one core:
  • CLI         pugmark extract <pdf> --chapter N --out gallery.html
  • Library     from pugmark import build_gallery
  • Gradio      app on HF Spaces (calls library)
```

Five stages, each a module, each independently testable. Inter-stage communication is via Pydantic models (Section 3). Cross-cutting concerns (LLM client, cache, observability) are injected, not imported globally — they appear in test fixtures and in real wiring at the same boundaries.

---

## 3. Data Model

All inter-stage data is Pydantic. Models are also the cache format (each stage's output written as JSON, keyed by content hash).

```python
# pugmark/schemas.py

class Chapter(BaseModel):
    book: str                            # e.g., "Nine Man-Eaters and One Rogue"
    number: int                          # ordinal within the omnibus
    title: str
    source_pdf: Path
    page_start: int
    page_end: int
    raw_text: str
    normalized_text: str                 # dehyphenated, smart-quotes fixed, whitespace collapsed
    page_offsets: list[int]              # char offset where each page starts in normalized_text
    ingest_version: str                  # for cache invalidation

class Candidate(BaseModel):              # one per LLM-extracted mention
    surface_form: str                    # exact phrase: "great striped one"
    proposed_name: str                   # LLM normalized: "tiger"
    kingdom_hint: Literal["animalia","plantae","fungi","unknown"]
    context_sentence: str                # the sentence containing the mention
    context_window: str                  # ~3-sentence window for reading-companion future
    char_offset: int                     # position in chapter.normalized_text
    page: int                            # derived via page_offsets
    llm_confidence: float                # self-reported, 0-1
    extractor_version: str               # bumped when prompt or model changes

class ConfirmedTaxon(BaseModel):         # one per validated species (many-to-one over Candidates)
    canonical_name: str                  # e.g., "Panthera tigris tigris"
    vernacular: str                      # e.g., "Bengal tiger"
    wikidata_qid: str                    # e.g., "Q26836"
    rank: str                            # species/genus/family/...
    lineage: dict[str, str]              # only filled levels: kingdom, phylum, class, ...
    validation_method: Literal["sparql_exact","sparql_fuzzy","alias","manual"]
    fuzzy_score: float | None            # only for fuzzy matches
    source_candidates: list[Candidate]   # every Candidate that resolved to this taxon

class ImageRef(BaseModel):
    url: HttpUrl
    license: str                         # e.g., "CC BY-SA 4.0"
    attribution: str                     # required for display
    source: Literal["wikimedia","inaturalist","wikipedia","ai_generated"]
    width: int | None
    height: int | None

class Sighting(BaseModel):               # bridge from Gallery to future Reading Companion
    page: int
    paragraph: str                       # 3-sentence window around the mention

class TaxonCard(BaseModel):
    taxon: ConfirmedTaxon
    wikipedia_url: HttpUrl
    wikipedia_summary: str               # first paragraph
    primary_image: ImageRef
    alt_images: list[ImageRef]           # populated in v2 with AI-gen illustrations
    sightings: list[Sighting]
    enrich_version: str

class EvalRun(BaseModel):
    chapter_id: str
    extraction: ExtractionMetrics
    validation: ValidationMetrics
    cost_usd: float
    latency_ms: int
    pugmark_version: str
    llm_provider: str                    # "gemini-2.0-flash"
    prompt_version: str                  # for prompt A/B testing
    timestamp: datetime

class Gallery(BaseModel):                # the chapter-level artifact
    chapter: Chapter
    cards: list[TaxonCard]
    unresolved: list[Candidate]          # honestly surfaced, not discarded
    generated_at: datetime
    pugmark_version: str
    eval_metrics: EvalRun | None         # populated when eval was run
```

### Architectural choices in the model

- **`source_candidates: list[Candidate]`** is the many-to-one bridge. Every mention of a species rolls up to a single `ConfirmedTaxon`, but the gallery can still show all page references and surrounding context. This is the entire data foundation for the v3 reading companion.
- **`unresolved` lives on `Gallery`, not discarded.** Surfacing the candidates the validator could not confirm is honest, and the unresolved-rate is a headline eval metric.
- **Version strings on every cached output** — `ingest_version`, `extractor_version`, `enrich_version`, `pugmark_version`. When prompts or models change, the cache invalidates correctly without manual cleanup.
- **License + attribution are non-negotiable fields on `ImageRef`** — required for any public deployment of the gallery.

---

## 4. Pipeline Stages

### 4.1 `ingest.py` — PDF → Chapter

- **Library:** `pymupdf` (`import fitz`). 5-10x faster than pypdf, cleaner extraction, coordinate-aware.
- **Chapter detection:** PDF outline (`doc.get_toc()`) is the primary source; the Anderson omnibus has proper bookmarks. Fallback: regex match on chapter-title pages (`r'^CHAPTER\s+\w+'`) using `page.get_text("blocks")`.
- **Normalization:**
  - Dehyphenate end-of-line splits (`r'-\n([a-z])' → r'\1'`)
  - Replace smart quotes with ASCII equivalents
  - Collapse runs of whitespace
  - Preserve case (Anderson named animals like "Stripes" must keep their capitalization)
- **Page tracking:** `page_offsets[i]` = char position in `normalized_text` where page `i` begins. Char offset → page is then a single `bisect_right` lookup.
- **CLI surface:** `pugmark chapters <pdf>` lists detected chapters; `pugmark ingest <pdf> --chapter N` extracts one.
- **Cache key:** `hash(pdf_bytes + chapter_number + ingest_version)`.

### 4.2 `extract.py` — Chapter → list[Candidate]

- **Engine:** LiteLLM `acompletion` to Gemini 2.0 Flash. Structured output via Pydantic schema (`response_format`).
- **Prompt strategy:** few-shot with three Anderson-flavored examples covering literal name, lyrical reference, and ambiguous case. Single-pass — Gemini's 1M context handles full chapters trivially.
- **Prompt source:** `prompts/extract_taxa.v1.j2`, fetched via `prompt_registry.get("extract_taxa")` which checks Langfuse first, falls back to in-repo file.
- **Page derivation:** for each candidate's `char_offset`, look up the containing page via `chapter.page_offsets`.
- **Cache key:** `hash(chapter.normalized_text + extractor_version + prompt_version + model_id)`. Cache hit → zero LLM cost.
- **Failure mode:** if LLM returns malformed JSON, retry once with stricter prompt; if still malformed, raise — this is a real bug, not a runtime exception to swallow.

### 4.3 `validate.py` — Candidate → ConfirmedTaxon (or unresolved)

- **Primary lookup:** Wikidata SPARQL against `wdt:P31/wdt:P279* wd:Q16521` (any taxon).
- **Match strategy:**
  1. Exact match on `rdfs:label` (English)
  2. Alias match via `skos:altLabel` (catches "Bengal tiger" ↔ "Panthera tigris tigris")
  3. Fuzzy match via `rapidfuzz.fuzz.ratio` with threshold 0.85
  4. Otherwise → unresolved (added to `Gallery.unresolved`)
- **Many-to-one collapse:** candidates resolving to the same `wikidata_qid` merge into one `ConfirmedTaxon` with a list of `source_candidates`.
- **Concurrency:** SPARQL queries are I/O-bound; batch with `asyncio.gather`, semaphore-limited to 10 concurrent requests against Wikidata.
- **Caching:** per `wikidata_qid` lookup, 24h TTL via `requests-cache`-style approach using our own `cache.py`.

### 4.4 `enrich.py` — ConfirmedTaxon → TaxonCard

- **Wikipedia summary:** REST API endpoint `GET /api/rest_v1/page/summary/{title}` derived from Wikidata's `sitelinks`. Returns summary text and thumbnail in one call.
- **Image strategy (v1, real photos only):** prefer Wikimedia Commons "featured" or "valued image" tags; fallback chain: Commons → iNaturalist → Wikipedia thumbnail.
- **License capture:** every image must record license string and attribution; if neither is parseable, the image is skipped and the next fallback is tried.
- **Sightings:** for each `source_candidate`, slice a 3-sentence window from `chapter.normalized_text` around `char_offset`. This is the bridge to v3 reading companion mode.
- **Concurrency:** per-card enrichment fans out to 3 endpoints concurrently (`wiki_summary`, `commons_image`, `inat_fallback`), gathered with `asyncio.gather`.
- **Caching:** per `wikidata_qid`, 7-day TTL. Wikipedia content for taxon pages rarely changes day-to-day.

### 4.5 `render.py` — list[TaxonCard] → output

- **Two render functions, same input:**
  - `render_html(gallery: Gallery) → str` — single self-contained HTML file via Jinja2 template, inline CSS, no external assets except image URLs.
  - `render_gradio(gallery: Gallery) → gr.Blocks` — `gr.Gallery` of cards with click-through detail panel; uses the same Jinja2 partial for card HTML internally.
- **Eval badge:** if `gallery.eval_metrics` is populated, the render shows a small badge — "P=0.94, R=0.89 on chapter X." Honest, transparent, demonstrable.
- **Attribution surface:** every image's license and attribution renders directly on the card. Non-negotiable for public deployment.

---

## 5. Cross-Cutting Concerns

### 5.1 `llm.py` — LiteLLM Client + Provider Fallback

- Thin wrapper exposing `LLMClient.extract_taxa(chapter_text: str) → list[Candidate]`.
- Provider order from config: Gemini 2.0 Flash → Groq Llama 3.3 70B → Ollama (local Phi-3.5/Qwen2.5).
- Per-provider retry with exponential backoff (LiteLLM native).
- Structured output enforced via Pydantic `response_format`.
- Every call traced via Langfuse callback (registered once at startup).
- Token counts, cost, latency, and fallback events recorded in trace metadata.

### 5.2 `cache.py` — File-Based Cache

- Storage location:
  - Local: `~/.cache/pugmark/{stage}/{hash}.json`
  - HF Spaces: `/data/.cache/pugmark/{stage}/{hash}.json` (detected via `HF_HOME` env)
- Key composition: `hash(stage_input + version_string)`.
- TTL configurable per stage (24h for SPARQL, 7d for Wikipedia, indefinite for ingest/extract since version bumps invalidate explicitly).
- No SQLite — premature for a single-user tool.

### 5.3 `observability.py` — Langfuse Wiring

- Initialize Langfuse client from env vars (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- Register Langfuse as a LiteLLM callback at startup — every LLM call automatically traced.
- Manual spans for non-LLM operations: SPARQL queries, Wikipedia fetches, Commons fetches, eval runs.
- Self-hosting deferred to v3 (cloud free tier is sufficient for v1).

### 5.4 `prompt_registry.py` — Versioned Prompts

- Layered: in-repo Jinja2 files in `prompts/` are the canonical source of truth.
- Runtime fetch order: Langfuse Prompt Registry → in-repo file fallback.
- Allows production A/B testing without redeploying — bump active version in Langfuse, next pipeline run picks it up.
- Every `EvalRun` records the active `prompt_version` so prompt-driven regressions are detectable from a single metrics diff.

---

## 6. Evaluation

### 6.1 What we evaluate

| Layer | Question | Failure mode |
|---|---|---|
| Extraction | Did the LLM find every taxonomic mention? | Misses + hallucinations |
| Validation | Did each candidate resolve to the right Wikidata QID? | Wrong species |
| End-to-end | Is the final gallery card useful? | Wrong image, wrong wiki, missing license |

### 6.2 Ground truth

Hand-label one chapter once. Recommended: *The Black Panther of Sivanipalli* — short, dense, varied taxa.

Stored in `eval/ground_truth/sivanipalli.json` as a versioned dataset:
```json
[
  {
    "surface_form": "panther",
    "page": 12,
    "expected_wikidata_qid": "Q34706",
    "expected_kingdom": "animalia"
  },
  ...
]
```

One-time effort: ~30-45 minutes. Every subsequent eval run is automatic.

### 6.3 Metrics

```python
class ExtractionMetrics(BaseModel):
    precision: float          # correctly_extracted / total_extracted
    recall: float             # correctly_extracted / true_total
    f1: float
    hallucination_rate: float # surface_forms with no support in chapter text

class ValidationMetrics(BaseModel):
    qid_accuracy: float       # correctly_resolved / should_resolve
    confusion_matrix: dict    # broken down by kingdom
    unresolved_rate: float    # candidates that should have resolved but didn't
```

### 6.4 Tooling

- **DeepEval** as the framework — `pytest.mark.evals` integration, LLM-as-judge support, JSON export.
- **Custom `eval/runner.py`** wraps DeepEval for our specific shape — runs the full pipeline against ground truth, dumps `EvalRun` JSON to `eval/runs/{timestamp}.json`, prints a markdown summary.
- **Judge model ≠ extraction model** to avoid circularity (Gemini Pro judges Gemini Flash output, or Claude Sonnet judges Gemini).

### 6.5 Where evals show up

1. **In the gallery** — `Gallery.eval_metrics` renders as a badge on the gallery page if populated.
2. **Markdown report** — `eval/runs/latest.md` auto-renders a comparison table across runs with trend lines (provider swaps, prompt changes, model versions).
3. **CLI command** — `pugmark eval --chapter sivanipalli` runs eval, updates `latest.md`, exits non-zero if F1 dropped >5% vs baseline (regression gate).
4. **v2 dashboard** — Streamlit/Gradio tab on the HF Space showing all runs over time. Deferred to v2.

---

## 7. Repo Layout

```
pugmark/                              # /opt/CodeRepo/pugmark/
├── pyproject.toml                    # uv-managed
├── uv.lock                           # committed
├── .env.example                      # template, no secrets
├── .gitignore
├── README.md
├── pugmark/                          # the package
│   ├── __init__.py
│   ├── schemas.py
│   ├── ingest.py                     # PDF → Chapter (PyMuPDF)
│   ├── extract.py                    # Chapter → Candidates (LLM)
│   ├── validate.py                   # Candidates → ConfirmedTaxa (Wikidata)
│   ├── enrich.py                     # Confirmed → TaxonCard (wiki+image)
│   ├── render.py                     # TaxonCard → HTML/Gradio
│   ├── llm.py                        # LiteLLM client + provider fallback
│   ├── cache.py                      # disk cache, HF /data aware
│   ├── observability.py              # Langfuse init + LiteLLM callback
│   ├── prompt_registry.py            # Langfuse-fetch w/ in-repo fallback
│   └── cli.py                        # Click-based CLI
├── prompts/
│   ├── extract_taxa.v1.j2
│   └── validate_alias.v1.j2
├── eval/
│   ├── runner.py
│   ├── metrics.py
│   ├── ground_truth/
│   │   └── sivanipalli.json
│   └── runs/                         # auto-generated EvalRun JSONs + latest.md
├── tests/
│   ├── test_ingest.py
│   ├── test_extract.py
│   ├── test_validate.py
│   ├── test_enrich.py
│   ├── test_render.py
│   ├── test_llm.py
│   ├── test_cache.py
│   └── test_e2e.py                   # mocked LLM/HTTP
├── app.py                            # Gradio HF Space entry
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-05-01-pugmark-design.md
└── data/
    └── samples/                      # sample chapter PDFs/text for tests
```

---

## 8. Dependencies

```toml
[project]
name = "pugmark"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pymupdf>=1.24",
    "litellm>=1.40",
    "httpx>=0.27",
    "rapidfuzz>=3.6",
    "jinja2>=3.1",
    "click>=8.1",
    "gradio>=4.40",
    "langfuse>=2.40",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
    "deepeval>=0.21",
    "ruff>=0.4",
    "ty>=0.0.1a1",
]
```

9 runtime dependencies, 6 dev dependencies. Lean.

---

## 9. Roadmap

### v1 — "Single chapter, real photos, evals" (target: 2-3 weekends)

- Ingest one Anderson chapter end-to-end via PyMuPDF
- LLM extraction via Gemini Flash (LiteLLM, with Groq + Ollama fallback paths wired)
- Wikidata validator with exact + alias + fuzzy match
- Wikipedia + Commons enrichment (real photos only)
- HTML gallery output + Gradio app
- DeepEval harness with hand-labeled Sivanipalli ground truth
- Langfuse tracing on every LLM + HTTP call
- Versioned prompts in `prompts/`
- Deployed to HuggingFace Spaces with public link
- README with one-screenshot demo

#### Definition of v1 done

1. `pugmark extract <pdf> --chapter sivanipalli --out gallery.html` produces a complete, attribution-correct gallery
2. `pugmark eval --chapter sivanipalli` returns F1 ≥ 0.85 (the v1 baseline)
3. Gradio app on HF Spaces accepts a chapter, runs the pipeline, displays the gallery
4. Langfuse dashboard shows traces for at least 5 end-to-end runs
5. README contains a screenshot demo and quickstart instructions

### v2 — "AI-generated illustrations" (target: 1 weekend)

- Colab notebook: `pugmark precompute_illustrations` runs SDXL Turbo / Flux Schnell on confirmed taxa
- `TaxonCard.alt_images` populated alongside real photos
- Gallery toggle: "Photo / Illustration / Both"
- Prompt template per kingdom (animal vs plant vs fungus → different style guidance)

### v3 — "Reading companion + self-hosted Langfuse" (target: 1-2 weekends)

- New render mode: `render_reading_companion()` — chapter text inline with sidebar pop-outs at every sighting
- Reuses `Sighting` data already on `TaxonCard`
- Toggle on the Gradio app: "Gallery / Reader"
- Self-host Langfuse via Docker Compose on a $5 VPS

### v4 — "Full omnibus + cross-chapter analytics"

- Batch process all 7 books in the omnibus
- Cross-book queries: "Every species mentioned in both Anderson and Corbett"
- Frequency heatmap, sighting timelines
- Multi-page interactive site

---

## 10. Out of Scope (v1)

Explicitly deferred to keep v1 shippable in 2-3 weekends:

- ❌ Async batch processing of multiple chapters in parallel — sequential per chapter; concurrency is *within* a chapter
- ❌ User accounts or saved galleries — stateless app
- ❌ Custom domain — `https://huggingface.co/spaces/ansumanbhujabal/pugmark` is fine
- ❌ Mobile-optimized UI — Gradio defaults are decent
- ❌ Cross-gallery search — single-chapter-at-a-time
- ❌ Self-hosted Langfuse — cloud free tier
- ❌ Local image gen — defer to Colab in v2
- ❌ AI-generated illustrations — defer to v2
- ❌ Reading companion view — defer to v3

---

## 11. Risks and Open Questions

### Risks

| Risk | Mitigation |
|---|---|
| Wikidata SPARQL flakiness or rate limits | Per-query caching + concurrency cap + retry with backoff |
| Gemini free tier quota exhaustion during demos | LiteLLM fallback to Groq pre-wired |
| Image license ambiguity from Wikimedia | Strict license parser; skip-and-fallback on unparseable cases; never display unlicensed images |
| LLM hallucinates plausible-but-fake species | Wikidata validator catches these; unresolved-rate metric tracks the gap |
| HF Spaces free tier compute caps | Single-chapter latency budget ≤ 60s; cache layer ensures repeat runs are sub-second |

### Open Questions (resolve during planning, not blocking spec approval)

1. Concrete chapter selection for v1 — *The Black Panther of Sivanipalli* recommended; confirm on first ingest run.
2. Initial prompt template content — drafted during plan, reviewed before first eval run.
3. F1 baseline of 0.85 — set after first hand-labeled ground truth run; may revise up or down based on actual numbers.
4. `ty` (Astral type checker) is alpha — fall back to `mypy` if it produces too many false positives during dev.

---

## 12. Decision Log

This spec is the result of seven sequenced decisions during brainstorming on 2026-05-01:

| # | Question | Choice | Rationale |
|---|---|---|---|
| 1 | Output format | B then A: gallery first, reading companion as natural extension | Clean evolution path; same data model |
| 2 | Image source | A then C: real photos first, AI gen later via Colab | Ship real-photo pipeline fast; AI gen is portfolio enhancement, not v1 blocker |
| 3 | Project name | Pugmark | Evocative; signals tracking-through-text metaphor; great GitHub-profile name |
| 4 | Extraction approach | Staged C: LLM baseline → Wikidata validator → evals | Classic architect-mindset pipeline: fuzzy understanding + deterministic validation + measurement |
| 5 | Scope | C: all taxonomic entities (animals, plants, fungi) | Single Wikidata `taxon` class covers all; validator filters non-matchable generics |
| 6 | Run mode | D: CLI core + Gradio HF Space | Same separation-of-concerns pattern used in NeuroLens and ultradoc |
| 7 | LLM provider | D: LiteLLM multi-provider (Gemini 2.0 Flash primary, Groq fallback, Ollama offline) | 1M context for full chapters; abstraction enables provider A/B without code changes |

Plus three follow-on architectural decisions:

| Topic | Choice |
|---|---|
| Concurrency | Async-throughout: `httpx.AsyncClient`, LiteLLM `acompletion`, `asyncio.gather`, `Semaphore(10)` |
| PDF library | PyMuPDF (`fitz`) — Marker reserved for scanned-PDF future case |
| Observability | Langfuse cloud free tier (v1) → self-host v3; LiteLLM callback + manual spans; prompts in repo + Langfuse registry with file fallback |

---
