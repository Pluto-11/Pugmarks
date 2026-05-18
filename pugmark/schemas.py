"""All Pydantic data models for Pugmark.

v2 generalizes the v1 taxa-specific models. v1 names are preserved as aliases.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


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
    """One LLM-extracted entity mention from a chapter.

    v2: `entity_type` replaces v1's `kingdom_hint` Literal. v1 callers that pass
    `kingdom_hint=...` are silently converted: `entity_type="taxa"` and
    `type_attrs["kingdom_hint"] = <value>`.
    """

    surface_form: str
    proposed_name: str
    entity_type: str = "taxa"
    type_attrs: dict[str, str] = Field(default_factory=dict)
    context_sentence: str
    context_window: str
    char_offset: int
    page: int
    llm_confidence: float = Field(..., ge=0.0, le=1.0)
    extractor_version: str

    @model_validator(mode="before")
    @classmethod
    def _v1_kingdom_hint_compat(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "kingdom_hint" in data and "entity_type" not in data:
            kh = data.pop("kingdom_hint")
            # Preserve v1 Literal validation: only the original set is allowed.
            if kh not in {"animalia", "plantae", "fungi", "unknown"}:
                raise ValueError(
                    f"kingdom_hint must be one of animalia/plantae/fungi/unknown, "
                    f"got {kh!r}"
                )
            data["entity_type"] = "taxa"
            data.setdefault("type_attrs", {})
            data["type_attrs"]["kingdom_hint"] = kh
        return data

    @property
    def kingdom_hint(self) -> str | None:
        """v1 read-side compat: surfaces type_attrs['kingdom_hint'] when present."""
        return self.type_attrs.get("kingdom_hint")


class ConfirmedEntity(BaseModel):
    """A Candidate (or several) resolved into a real entity."""

    canonical_name: str
    vernacular: str
    entity_type: str = "taxa"
    wikidata_qid: str | None = None
    rank: str
    attributes: dict[str, str] = Field(default_factory=dict)
    validation_method: Literal[
        "sparql_exact",
        "sparql_fuzzy",
        "alias",
        "manual",
        "in_book_crossref",
        "judge_consensus",
        "hybrid",
    ]
    fuzzy_score: float | None = Field(default=None, ge=0.0, le=1.0)
    crossref_count: int | None = None
    judge_votes: int | None = None
    source_candidates: list[Candidate]

    @model_validator(mode="before")
    @classmethod
    def _v1_lineage_compat(cls, data: object) -> object:
        """v1 used `lineage` for taxonomic ancestry; v2 uses generic `attributes`."""
        if isinstance(data, dict) and "lineage" in data and "attributes" not in data:
            data["attributes"] = data.pop("lineage")
        return data

    @property
    def lineage(self) -> dict[str, str]:
        """v1 read-side compat: aliases `attributes`."""
        return self.attributes


class ImageRef(BaseModel):
    """A licensed image reference. License + attribution are required."""

    url: HttpUrl
    license: str
    attribution: str
    source: Literal["wikimedia", "inaturalist", "wikipedia", "ai_generated"]
    width: int | None = None
    height: int | None = None


class Sighting(BaseModel):
    """One mention of a confirmed entity in the chapter."""

    page: int
    paragraph: str


class EntityCard(BaseModel):
    """The display unit of a Pugmark gallery."""

    entity: ConfirmedEntity
    wikipedia_url: HttpUrl | None = None
    wikipedia_summary: str
    summary_source: Literal["wikipedia", "llm_in_book", "none"] = "wikipedia"
    primary_image: ImageRef | None = None
    alt_images: list[ImageRef] = Field(default_factory=list)
    sightings: list[Sighting] = Field(default_factory=list)
    enrich_version: str

    @model_validator(mode="before")
    @classmethod
    def _v1_taxon_compat(cls, data: object) -> object:
        """v1 used `taxon=` for the confirmed entity field."""
        if isinstance(data, dict) and "taxon" in data and "entity" not in data:
            data["entity"] = data.pop("taxon")
        return data

    @property
    def taxon(self) -> ConfirmedEntity:
        """v1 read-side compat: aliases `entity`."""
        return self.entity


class ExtractionMetrics(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    by_type: dict[str, ExtractionMetrics] | None = None


ExtractionMetrics.model_rebuild()


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
    cards_by_type: dict[str, list[EntityCard]] = Field(default_factory=dict)
    unresolved: list[Candidate] = Field(default_factory=list)
    generated_at: datetime
    pugmark_version: str
    book_schema: object | None = None
    eval_metrics: EvalRun | None = None

    @model_validator(mode="before")
    @classmethod
    def _v1_cards_compat(cls, data: object) -> object:
        """v1 used a flat `cards` list. v2 keys by type; v1 cards land under 'taxa'."""
        if isinstance(data, dict) and "cards" in data and "cards_by_type" not in data:
            cards = data.pop("cards")
            data["cards_by_type"] = {"taxa": cards} if cards else {}
        return data

    @property
    def cards(self) -> list[EntityCard]:
        """v1 read-side compat: flat list across all type buckets."""
        flat: list[EntityCard] = []
        for bucket in self.cards_by_type.values():
            flat.extend(bucket)
        return flat


# v1 backward-compat aliases
ConfirmedTaxon = ConfirmedEntity
TaxonCard = EntityCard
