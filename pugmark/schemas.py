"""All Pydantic data models for Pugmark.

These are pure data shapes — no business logic. They flow between pipeline stages
and serve as the on-disk cache format (JSON-serialized).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Chapter(BaseModel):
    """One chapter extracted from a source PDF."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    book: str
    number: int
    title: str
    source_pdf: Path
    page_start: int
    page_end: int
    raw_text: str
    normalized_text: str
    page_offsets: list[int] = Field(
        ..., description="Char offset in normalized_text where each page starts"
    )
    ingest_version: str

    def offset_to_page(self, char_offset: int) -> int:
        """Map a char offset in normalized_text to a page number (1-indexed)."""
        from bisect import bisect_right

        idx = bisect_right(self.page_offsets, char_offset)
        return self.page_start + max(idx - 1, 0)


class Candidate(BaseModel):
    """One LLM-extracted taxonomic mention from a chapter."""

    surface_form: str
    proposed_name: str
    kingdom_hint: Literal["animalia", "plantae", "fungi", "unknown"]
    context_sentence: str
    context_window: str
    char_offset: int
    page: int
    llm_confidence: float = Field(..., ge=0.0, le=1.0)
    extractor_version: str


class ConfirmedTaxon(BaseModel):
    """A Candidate (or several) resolved to a real Wikidata taxon."""

    canonical_name: str
    vernacular: str
    wikidata_qid: str
    rank: str
    lineage: dict[str, str] = Field(default_factory=dict)
    validation_method: Literal["sparql_exact", "sparql_fuzzy", "alias", "manual"]
    fuzzy_score: float | None = None
    source_candidates: list[Candidate]


class ImageRef(BaseModel):
    """A licensed image reference. License + attribution are required."""

    url: HttpUrl
    license: str
    attribution: str
    source: Literal["wikimedia", "inaturalist", "wikipedia", "ai_generated"]
    width: int | None = None
    height: int | None = None


class Sighting(BaseModel):
    """One mention of a confirmed taxon in the chapter, with surrounding context."""

    page: int
    paragraph: str


class TaxonCard(BaseModel):
    """The display unit of a Pugmark gallery."""

    taxon: ConfirmedTaxon
    wikipedia_url: HttpUrl
    wikipedia_summary: str
    primary_image: ImageRef
    alt_images: list[ImageRef] = Field(default_factory=list)
    sightings: list[Sighting] = Field(default_factory=list)
    enrich_version: str


class ExtractionMetrics(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)


class ValidationMetrics(BaseModel):
    qid_accuracy: float = Field(..., ge=0.0, le=1.0)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    unresolved_rate: float = Field(..., ge=0.0, le=1.0)


class EvalRun(BaseModel):
    chapter_id: str
    extraction: ExtractionMetrics
    validation: ValidationMetrics
    cost_usd: float
    latency_ms: int
    pugmark_version: str
    llm_provider: str
    prompt_version: str
    timestamp: datetime


class Gallery(BaseModel):
    """The chapter-level artifact rendered to HTML/Gradio."""

    chapter: Chapter
    cards: list[TaxonCard]
    unresolved: list[Candidate] = Field(default_factory=list)
    generated_at: datetime
    pugmark_version: str
    eval_metrics: EvalRun | None = None
