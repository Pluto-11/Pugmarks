"""Pugmark — self-adapting universal bestiary."""
from pugmark.analyzer import analyze_book
from pugmark.api import extract_gallery
from pugmark.entity_registry import (
    get_registered,
    register_entity_type,
)
from pugmark.entity_type import BookSchema, EntityTypeSpec
from pugmark.schemas import (
    Candidate,
    Chapter,
    ConfirmedEntity,
    ConfirmedTaxon,
    EntityCard,
    EvalRun,
    ExtractionMetrics,
    Gallery,
    ImageRef,
    Sighting,
    TaxonCard,
    ValidationMetrics,
)

__all__ = [
    "BookSchema",
    "Candidate",
    "Chapter",
    "ConfirmedEntity",
    "ConfirmedTaxon",
    "EntityCard",
    "EntityTypeSpec",
    "EvalRun",
    "ExtractionMetrics",
    "Gallery",
    "ImageRef",
    "Sighting",
    "TaxonCard",
    "ValidationMetrics",
    "analyze_book",
    "extract_gallery",
    "get_registered",
    "register_entity_type",
]
