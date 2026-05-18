# Pugmark v2 — Self-Adapting Universal Bestiary

**Date:** 2026-05-18
**Author:** Ansuman SS Bhujabala
**Status:** Brainstorming complete; awaiting user review of this spec.
**Workspace:** `/opt/CodeRepo/pugmark/`
**Supersedes:** Pugmark v1 design (`docs/superpowers/specs/2026-05-01-pugmark-design.md`) — v1 is shipped as the taxa-specific specialization; v2 generalizes it.

---

## 1. North Star

**Pugmark v2 turns any book PDF into an illustrated, evaluated bestiary — where "bestiary" is auto-defined per book.**

A field guide produces a taxa gallery (today's behavior). A cookbook produces a recipes-and-ingredients gallery. A mystery novel produces characters-and-locations. A sci-fi novel produces characters, planets, technologies, and factions. The user does not pick the entity types — Pugmark reads the book and proposes them, then runs the existing extract/validate/enrich/render pipeline per type.

This makes Pugmark a **self-adapting reading companion**: drop in any PDF, get a structured gallery tailored to that book's content. It also makes Pugmark a **library** — users can import it, register custom entity types, and reuse the pipeline for their own corpora.

### Design tenets (carried from v1, extended)

1. **Same pipeline, three faces.** CLI, Gradio, library. The library API is now first-class and equal to the CLI.
2. **Self-adapting schema.** The set of entity types is derived from the book, not hardcoded. The pipeline is type-agnostic; only prompts and validators are type-specific.
3. **Cache everything, version everything.** Cache key now includes `entity_type` + `analyzer_version` so a re-analyzed book re-extracts cleanly.
4. **Tiered validation, no silent hallucinations.** Wikidata round-trip when available. In-book cross-reference + judge-LLM consensus when not. Never trust a single LLM call.
5. **Backward-compat without forks.** v1 schema names (`TaxonCard`, `ConfirmedTaxon`) keep working as aliases. Existing tests pass.
6. **YAGNI ruthlessly.** AI-generated images, REST API, multi-book analytics — out of scope, deferred.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Pugmark v2                                  │
│                                                                      │
│  PDF ──► ingest ──► analyze ──► realize-schema                       │
│                       │              │                               │
│                       ▼              ▼                               │
│                 BookSchema    list[EntityTypeSpec]                   │
│                                      │                               │
│                                      ▼                               │
│              ┌────────────── per entity type ──────────────┐         │
│              │                                              │         │
│              │  extract ──► validate (tiered) ──► enrich   │         │
│              │     │             │                  │       │         │
│              │     ▼             ▼                  ▼       │         │
│              │ Candidate   ConfirmedEntity     EntityCard  │         │
│              └──────────────────────────────────────────────┘         │
│                                      │                               │
│                                      ▼                               │
│                          render(cards_by_type)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Three faces, one core:

```
CLI:      pugmark/cli.py    — Click subcommands (chapters, analyze, extract, autolabel, eval)
Gradio:   app.py            — HF Spaces UI; same pipeline functions
Library:  pugmark/__init__.py exports analyze_book, extract_gallery, register_entity_type
```

---

## 3. Data Model

### New / renamed schemas in `pugmark/schemas.py`

```python
# Renamed (v1 aliases provided for backward compat)
class ConfirmedEntity(BaseModel):           # was: ConfirmedTaxon
    canonical_name: str
    vernacular: str
    entity_type: str                         # NEW — "taxa", "people", "wines"…
    wikidata_qid: str | None                 # NEW — optional (was required str)
    rank: str                                # for taxa; generic descriptor otherwise
    attributes: dict[str, str]               # was: lineage — generic key/value
    validation_method: Literal[
        "sparql_exact", "sparql_fuzzy", "alias", "manual",
        "in_book_crossref", "judge_consensus", "hybrid"   # NEW — last three
    ]
    fuzzy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    crossref_count: int | None = None        # NEW — for in_book_crossref
    judge_votes: int | None = None           # NEW — for judge_consensus
    source_candidates: list[Candidate]

class EntityCard(BaseModel):                 # was: TaxonCard
    entity: ConfirmedEntity
    wikipedia_url: HttpUrl | None            # NEW — None when no QID
    wikipedia_summary: str                   # may be LLM-generated when no QID
    summary_source: Literal["wikipedia", "llm_in_book", "none"]  # NEW
    primary_image: ImageRef | None           # NEW — None when no Commons image
    alt_images: list[ImageRef] = Field(default_factory=list)
    sightings: list[Sighting]
    enrich_version: str

# Modified
class Candidate(BaseModel):
    surface_form: str
    proposed_name: str
    entity_type: str                         # NEW — was: kingdom_hint Literal
    type_attrs: dict[str, str] = Field(default_factory=dict)
                                             # NEW — type-specific extras
                                             # (e.g. kingdom_hint for taxa)
    context_sentence: str
    context_window: str
    char_offset: int
    page: int
    llm_confidence: float = Field(..., ge=0.0, le=1.0)
    extractor_version: str

class Gallery(BaseModel):
    chapter: Chapter
    cards_by_type: dict[str, list[EntityCard]]   # NEW shape (was: cards)
    unresolved: list[Candidate]
    generated_at: datetime
    pugmark_version: str
    book_schema: BookSchema                  # NEW — what we extracted and why
    eval_metrics: EvalRun | None = None
```

### New schemas

```python
class EntityTypeSpec(BaseModel):
    """Description of one entity type the pipeline should extract.

    Either Pugmark's analyzer proposes this from book content, or the user
    registers it explicitly via register_entity_type().
    """
    name: str                                # "taxa", "characters", "wines"
    description: str                         # one-line, LLM-generated or user-provided
    wikidata_qclass: str | None              # e.g. "Q16521" or None
    extraction_prompt_template: str          # Jinja2 source (rendered at runtime)
    judge_prompt_template: str               # Jinja2 source for auto-label
    min_book_occurrences: int = 2            # cross-ref threshold for non-Wikidata
    min_judge_votes: int = 2                 # ≥votes-of-3 for non-Wikidata
    examples: list[str] = Field(default_factory=list)
                                             # examples to include in prompt
    spec_version: str = "v1"                 # bump invalidates cache cleanly

class BookSchema(BaseModel):
    """The set of entity types Pugmark will extract from one specific book."""
    book_id: str                             # pdf hash or stem
    proposed_types: list[EntityTypeSpec]
    analyzer_version: str
    analyzed_at: datetime
```

### Backward-compat aliases

In `pugmark/schemas.py`:
```python
TaxonCard = EntityCard
ConfirmedTaxon = ConfirmedEntity
```

v1 tests continue to pass; v1 ground-truth files (`eval/ground_truth/sivanipalli.json`) continue to work — the loader sets `entity_type="taxa"` when the file lacks the new field.

---

## 4. New Pipeline Components

### 4.1 `pugmark/analyzer.py` — book → entity type proposals

```python
async def analyze_book(
    pdf: Path,
    *,
    cache: Cache,
    llm_config: LLMConfig | None = None,
    max_sample_chapters: int = 3,
) -> BookSchema:
    """Read TOC + sample chapter text → propose entity types.

    The analyzer LLM is given:
      - the book's TOC (chapter titles)
      - first 2000 chars of the first chapter
      - random samples (500 chars each) from up to max_sample_chapters more chapters

    and is prompted to propose 2-6 entity types valuable to extract, each with:
      - name (snake_case, lowercase)
      - description (one line)
      - 2-3 in-book example surface forms
      - suggested Wikidata Q-class (or null)

    Cached by (pdf hash, analyzer_version) so re-analysis of the same PDF is free.
    """
```

The analyzer LLM uses a separate prompt template (`prompts/book_analyzer.v1.j2`) and is encouraged to use the **strongest available model** (Gemini 2.5 Pro by default). Output is JSON, parsed via Pydantic.

**Refusal handling:** if the analyzer proposes a non-Wikidata type that's too granular ("specific anecdotes about grandmother" — useless), the realizer filters them out (no Wikidata Q-class AND name doesn't match a known type class pattern → drop with warning).

### 4.2 `pugmark/entity_registry.py` — pluggable EntityType registry

```python
_REGISTERED: dict[str, EntityTypeSpec] = {}

def register_entity_type(spec: EntityTypeSpec) -> None:
    """Register a user-defined entity type.

    Registered types are auto-included if the analyzer also proposes them
    (matching on name, case-insensitive). They can also be force-included
    via pugmark.extract_gallery(..., types=[name]).
    """
    _REGISTERED[spec.name.lower()] = spec

def get_registered() -> dict[str, EntityTypeSpec]:
    return dict(_REGISTERED)
```

Pugmark ships built-in defaults for `taxa`, `people`, `places` registered at import time. The analyzer can still propose these; if it does, the registered spec wins (registered specs are tuned; analyzer-proposed ones use a generic template).

### 4.3 `pugmark/schema_realizer.py` — proposals → realized specs

```python
def realize_schema(
    book_schema: BookSchema,
    *,
    overrides: dict[str, EntityTypeSpec] | None = None,
) -> list[EntityTypeSpec]:
    """Merge analyzer proposals with registered + user overrides.

    Precedence (highest to lowest):
      1. user overrides (passed in `overrides`)
      2. registered types (entity_registry._REGISTERED)
      3. analyzer-proposed types (raw, with generic templates)

    Returns the final list of EntityTypeSpec to run extraction with.
    """
```

---

## 5. Modified Components

### 5.1 `pugmark/extract.py` — type-parameterized

```python
async def extract_candidates(
    chapter: Chapter,
    *,
    entity_type: EntityTypeSpec,             # NEW — was: hardcoded extract_taxa prompt
    llm_config: LLMConfig,
    cache: Cache,
) -> list[Candidate]:
    """Same logic as v1 but the prompt template comes from entity_type.

    Cache key now: hash(text + extractor_version + prompt_version + provider + type_name + spec_version)
    """
```

### 5.2 `pugmark/validate.py` — tiered validation

```python
async def validate_candidates(
    candidates: list[Candidate],
    *,
    entity_type: EntityTypeSpec,
    chapter: Chapter,                        # NEW — needed for in-book crossref
    cache: Cache,
) -> tuple[list[ConfirmedEntity], list[Candidate]]:
    """Tiered: Wikidata first, then in-book + judge consensus.

    Tier 1 — Wikidata path (when entity_type.wikidata_qclass is set):
      - Reuse v1 SPARQL exact → alias → fuzzy logic
      - Q-class is templated into the SPARQL (was hardcoded Q16521)
      - validation_method ∈ {sparql_exact, sparql_fuzzy, alias}

    Tier 2 — in-book + judge path (when wikidata_qclass is None):
      - Cross-reference: count occurrences of surface_form (case-insensitive,
        word-boundary) in chapter.normalized_text. Must be ≥ entity_type.min_book_occurrences.
      - Judge consensus: query the judge LLM N times for "is this a real <type>
        in this book?" — must get ≥ entity_type.min_judge_votes yes.
      - validation_method = "in_book_crossref" if only cross-ref passes,
        "judge_consensus" if both pass, "hybrid" if both with QID
      - wikidata_qid stays None; canonical_name = proposed_name

    Returns (confirmed, unresolved) same shape as v1.
    """
```

The v1 `validate_candidates` signature is preserved via a thin adapter:
```python
async def validate_candidates_legacy(
    candidates: list[Candidate], *, cache: Cache
) -> tuple[list[ConfirmedEntity], list[Candidate]]:
    """v1-compat wrapper. Assumes entity_type='taxa' for every candidate."""
```

### 5.3 `pugmark/enrich.py` — tiered enrichment

```python
async def enrich_confirmed(
    entities: list[ConfirmedEntity],
    *,
    chapter: Chapter,
    llm_config: LLMConfig,
    cache: Cache,
) -> list[EntityCard]:
    """Tiered enrichment per entity.

    Tier 1 (entity.wikidata_qid is not None):
      - Existing Wikipedia + Commons flow
      - summary_source = "wikipedia"
      - primary_image is the Commons image with license + attribution

    Tier 2 (entity.wikidata_qid is None):
      - LLM-generated summary from concatenated context_windows of
        entity.source_candidates (capped at ~2000 chars input).
      - summary_source = "llm_in_book"
      - primary_image = None
      - wikipedia_url = None

    Cards without external image render with a typographic placeholder
    in the gallery (initials + entity_type icon) — handled in render.py.
    """
```

### 5.4 `pugmark/render.py` — grouped by type

```python
def render_html(gallery: Gallery) -> str:
    """Same template signature, but template is updated to iterate
    cards_by_type and render section headers.
    """
```

Template change (`pugmark/templates/gallery.html.j2`):
- Outer loop: `for type_name, cards in gallery.cards_by_type.items()`
- Section heading per type
- Cards within a section use the same card template
- Cards with `primary_image is None` render a typographic placeholder div instead of `<img>`
- `summary_source == "llm_in_book"` cards show a small "AI-summarized from book" badge to distinguish from Wikipedia-sourced summaries (epistemic honesty)

### 5.5 `pugmark/cli.py` — new subcommands

```bash
pugmark analyze <pdf>                              # show proposed entity types
pugmark extract <pdf> --chapter N --out F.html     # auto-analyze + extract (DEFAULT)
pugmark extract <pdf> --chapter N --types taxa,characters --out F.html  # override types
pugmark autolabel <pdf> --chapter N --types ...    # type-aware ground truth gen
pugmark eval <pdf> --chapter N --types ...         # type-aware metrics
```

### 5.6 Library API surface (`pugmark/__init__.py`)

```python
from .analyzer import analyze_book
from .entity_registry import register_entity_type, get_registered
from .entity_type import EntityTypeSpec
from .schemas import (
    BookSchema, Candidate, Chapter, ConfirmedEntity, EntityCard, Gallery,
    # v1 aliases
    ConfirmedTaxon, TaxonCard,
)

async def extract_gallery(
    pdf: Path | str,
    chapter_number: int,
    *,
    types: list[str] | None = None,          # subset filter, e.g. ["taxa","places"]
    cache: Cache | None = None,
    llm_config: LLMConfig | None = None,
) -> Gallery:
    """Convenience: analyze + realize + extract + validate + enrich + assemble.

    If `types` is given, the proposed schema is filtered to those types
    (force-added if registered but not auto-proposed).
    """
```

---

## 6. Customization Surface

A user using Pugmark as a library can customize at any layer:

| Layer | How to customize |
|---|---|
| **Add an entity type** | `pugmark.register_entity_type(EntityTypeSpec(...))` |
| **Override a prompt** | Place a Jinja2 file at `prompts/<type_name>.<version>.j2`; the prompt registry picks it up automatically |
| **Override a validator** | Implement `async def validate(candidates, entity_type, chapter, cache) -> (confirmed, unresolved)` and pass it to `extract_gallery(..., validator=my_validator)` |
| **Override an enricher** | Same pattern with `enricher=my_enricher` |
| **Override the analyzer LLM** | `extract_gallery(..., analyzer_llm_config=LLMConfig(providers=["claude-3-5-sonnet"]))` |
| **Skip analysis entirely** | `extract_gallery(..., types=[...explicit list...])` — bypasses the analyzer |
| **Custom cache backend** | Pass `cache=MyCache(...)` — duck-typed on `Cache.get/set` |

Hook points are explicit kwargs on `extract_gallery`, not magic config files — easier to discover, easier to type-check.

---

## 7. Eval Harness Updates

The v1 eval harness (`eval/metrics.py`, `eval/runner.py`) is type-aware:

```python
# ExtractionMetrics + ValidationMetrics gain per-type breakdowns
class ExtractionMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    hallucination_rate: float
    by_type: dict[str, "ExtractionMetrics"] | None = None  # NEW — recursive optional
```

`eval/auto_label.py` becomes type-aware: each entity type has its own auto-labeled ground truth file (`eval/ground_truth/<book_id>__<type>.json`). The eval runner reports per-type and aggregate metrics.

**Regression gate** in `pugmark eval --strict` triggers on **either** aggregate F1 dropping >5% **or** any individual type's F1 dropping >10% (per-type swings are noisier).

---

## 8. Migration from v1

| Concern | v1 → v2 path |
|---|---|
| Existing tests | All pass — `TaxonCard`/`ConfirmedTaxon` are aliases; `validate_candidates_legacy` covers the v1 signature |
| `Candidate.kingdom_hint` callers | Pydantic `model_validator(mode="before")` on `Candidate`: if the input dict contains `kingdom_hint` (v1 keyword), set `entity_type="taxa"` and stash the value at `type_attrs["kingdom_hint"]`. v1 test fixtures and ground-truth files keep working unchanged. |
| Existing ground truth | `eval/ground_truth/sivanipalli.json` is loaded with implicit `entity_type="taxa"` |
| Existing prompts | `prompts/extract_taxa.v1.j2` becomes the default extraction template for the registered `taxa` EntityTypeSpec |
| Existing cache entries | v1 cache keys did not include `entity_type` → invalidated naturally (cache key changes). One-time re-fetch on first v2 run |
| `pugmark extract` default | Now auto-analyzes. To pin to v1 behavior: `pugmark extract <pdf> --types taxa` |
| `pugmark eval` ground truth | Backward-compat loader infers `entity_type="taxa"` from v1 files |

---

## 9. Testing Strategy

Test pyramid mirrors v1:

- **Schemas** (~20 tests): all new types, backward-compat aliases, type_attrs flexibility
- **Analyzer** (~5 tests): mocked LLM returns fixed schema; filter out junk proposals; cache hit on second call
- **Entity registry** (~5 tests): register/get, override behavior, built-in defaults
- **Schema realizer** (~5 tests): merge order, user override beats registered, registered beats proposed
- **Validate tiered** (~10 tests): Wikidata path (existing 5), in-book crossref (new), judge consensus (new), hybrid (new), insufficient crossref → unresolved (new)
- **Enrich tiered** (~5 tests): Wikipedia path (existing), LLM-summary path (new), placeholder image flag in card (new)
- **Render** (~6 tests): cards_by_type sections, placeholder rendering, "AI-summarized" badge
- **Library API** (~5 tests): `extract_gallery` end-to-end with mocked stages; `register_entity_type` round-trip
- **Eval per-type** (~5 tests): per-type metrics correctness; regression gate triggers on per-type drop

Existing 72 v1 tests stay green via aliases + legacy adapter. v2 target: ~130 tests total.

---

## 10. Out of Scope (v2)

Deferred to v3+:
- AI-generated illustrations for entities without Commons photos (placeholder only in v2)
- REST API / long-running service mode
- Multi-book / cross-book analytics
- Interactive chat-with-book (RAG) — separate product
- Auto-suggesting entity types based on similarity to other books' schemas
- Fine-tuning a smaller model on auto-labeled data
- Self-hosted Langfuse migration

---

## 11. Risks and Open Questions

### Risks

| Risk | Mitigation |
|---|---|
| Analyzer proposes useless / too-granular types | Realizer filters proposals lacking either a Wikidata Q-class or matching a registered/cross-referenceable pattern. Log dropped proposals. |
| In-book cross-reference produces false positives on common nouns | Use word-boundary regex + case-sensitive match when surface_form starts with capital. Min occurrence threshold defaults to 2 (override per type). |
| Judge LLM ensemble blesses hallucinations | Same judge ≥2/3 voting pattern as auto-label proven in v1. Add explicit "is this a real X in this book?" question (yes/no), not free-form. |
| LLM-generated summaries hallucinate "facts" not in the book | Pin the prompt to use only the supplied context_windows; mark cards with `summary_source="llm_in_book"` and a visible "AI-summarized from book" badge so users understand the epistemic difference |
| Cost: more LLM calls (analyzer + per-type judge ensemble) | Aggressive caching at analyzer (per pdf), at judge (per type+text), at extract (existing). Re-runs on the same book ≈ free after first run. |
| Breaking changes to v1 callers (library users) | Backward-compat aliases for `TaxonCard`/`ConfirmedTaxon`; legacy `validate_candidates_legacy`; conservative defaults |

### Open Questions (decide during planning, not blocking spec approval)

1. **Built-in types ship with v2**: `taxa`, `people`, `places` — agreed at brainstorming. Anything else worth shipping pre-registered (e.g., `wines`, `chess_openings`)? Probably not — register externally to demo the plugin API.
2. **Per-type concurrency**: Run all types in `asyncio.gather` at the top of the pipeline, or sequentially? Defaults to gather; can be tuned with `--max-type-concurrency`.
3. **Render: ordering of type sections**: Alphabetical, or analyzer-priority, or a configurable `display_order` field on EntityTypeSpec? Default analyzer-priority (the order the analyzer returned).
4. **Ground truth file naming**: `sivanipalli.json` (v1) → `sivanipalli__taxa.json` (v2)? Or one file with a type field? Single file with type field is cleaner — migrate v1 files on first read.

---

## 12. Roadmap

### v2 milestone — "Universal bestiary" (target: ~5 working days)

Implementation plan to be written next; rough breakdown:
- **Day 1**: schemas + backward-compat aliases + entity_registry + EntityTypeSpec
- **Day 2**: analyzer (`prompts/book_analyzer.v1.j2` + tests) + schema_realizer
- **Day 3**: tiered validate + tiered enrich
- **Day 4**: render updates + CLI extensions + library API + Gradio updates
- **Day 5**: eval harness updates (per-type metrics + multi-file ground truth) + integration tests + STATUS handoff

### v3 — "AI illustrations + relationship graphs" (target: 1-2 weekends)
- SDXL Turbo / Flux Schnell illustrations for entities without Commons photos
- Relationship graphs (per book, per type) — e.g., character co-occurrence network

### v4 — "Multi-book + cross-corpus" (target: 1 weekend)
- Cross-chapter analytics for one book
- Cross-book queries: "where does 'tiger' appear across all Kenneth Anderson books"

---

## 13. Decision Log

Captured during the 2026-05-18 brainstorm (this spec):

| # | Decision | Why |
|---|---|---|
| 1 | Replace static EntityType registry with dynamic per-book schema | User goal: "auto-adjusting, works for all books." Static list can't cover every book type. |
| 2 | Approach B: generalize schemas in place, alias v1 names | Honest semantics; alias keeps v1 callers + tests working. Approach A (adapter) leaves dead fields; C (side-by-side) creates two codebases. |
| 3 | Tiered validation: Wikidata → in-book crossref + judge consensus | Most general-domain entity types aren't in Wikidata. Need a fallback that's still rigorous. |
| 4 | LLM-summary fallback for enrichment when no QID | Better than blank cards. Marked with a badge so users understand the epistemic difference. |
| 5 | Plugin API: `register_entity_type()` + per-stage hooks via kwargs | "Highly customizable lib" — needs first-class extensibility |
| 6 | No AI-generated images in v2 | Scope cut; placeholder cards in v2; SDXL/Flux in v3 |
| 7 | No REST API in v2 | Existing CLI + Gradio + library scope is enough for portfolio + personal use |
| 8 | Backward-compat aliases for v1 schemas | v1 tests still green; v1 ground-truth files still load |
