"""Library API surface tests.

Verifies that import pugmark + extract_gallery() works end-to-end with
mocked LLM + network. This is the entrypoint a downstream library user calls.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import pugmark
from pugmark.entity_type import BookSchema, EntityTypeSpec
from pugmark.schemas import (
    Candidate,
    ConfirmedEntity,
    EntityCard,
    Gallery,
    ImageRef,
)

FIXTURE_PDF = Path("tests/fixtures/sample_chapter.pdf")


def test_top_level_exports() -> None:
    assert hasattr(pugmark, "analyze_book")
    assert hasattr(pugmark, "extract_gallery")
    assert hasattr(pugmark, "register_entity_type")
    assert hasattr(pugmark, "EntityTypeSpec")
    assert hasattr(pugmark, "Gallery")
    # v1 aliases
    assert hasattr(pugmark, "ConfirmedTaxon")
    assert hasattr(pugmark, "TaxonCard")


@pytest.mark.asyncio
async def test_extract_gallery_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pipeline with mocked LLM + validator + enricher → returns Gallery."""
    # Mock analyze_book to return a taxa-only schema
    spec = EntityTypeSpec(
        name="taxa",
        description="x",
        wikidata_qclass="Q16521",
        extraction_prompt_template="extract {{ chapter_text }}",
        judge_prompt_template="x",
    )
    fake_schema = BookSchema(
        book_id="sample_chapter",
        proposed_types=[spec],
        analyzer_version="v1",
        analyzed_at=datetime.now(),
    )

    async def fake_analyze(*args, **kwargs):
        return fake_schema

    monkeypatch.setattr("pugmark.api.analyze_book", fake_analyze)

    # Mock extract_candidates
    fake_cand = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        entity_type="taxa",
        type_attrs={"kingdom_hint": "animalia"},
        context_sentence="A tiger appeared.",
        context_window="A tiger appeared.",
        char_offset=0,
        page=1,
        llm_confidence=0.95,
        extractor_version="v2",
    )

    async def fake_extract(chapter, *, entity_type, llm_config, cache):
        return [fake_cand]

    monkeypatch.setattr("pugmark.api.extract_candidates", fake_extract)

    # Mock validate
    confirmed_entity = ConfirmedEntity(
        canonical_name="Panthera tigris",
        vernacular="Tiger",
        entity_type="taxa",
        wikidata_qid="Q15324",
        rank="species",
        attributes={},
        validation_method="sparql_exact",
        source_candidates=[fake_cand],
    )

    async def fake_validate(cands, *, entity_type, chapter, cache):
        return [confirmed_entity], []

    monkeypatch.setattr("pugmark.api.validate_candidates", fake_validate)

    # Mock enrich
    fake_card = EntityCard(
        entity=confirmed_entity,
        wikipedia_url="https://en.wikipedia.org/wiki/Tiger",
        wikipedia_summary="A big cat.",
        summary_source="wikipedia",
        primary_image=ImageRef(
            url="https://example.org/tiger.jpg",
            license="CC0",
            attribution="x",
            source="wikimedia",
        ),
        alt_images=[],
        sightings=[],
        enrich_version="v2",
    )

    async def fake_enrich(entities, *, chapter, cache, llm_config=None):
        return [fake_card]

    monkeypatch.setattr("pugmark.api.enrich_confirmed", fake_enrich)

    gallery = await pugmark.extract_gallery(FIXTURE_PDF, chapter_number=1)
    assert isinstance(gallery, Gallery)
    assert "taxa" in gallery.cards_by_type
    assert len(gallery.cards_by_type["taxa"]) == 1
    assert gallery.cards_by_type["taxa"][0].entity.canonical_name == "Panthera tigris"
    assert gallery.book_schema is not None
