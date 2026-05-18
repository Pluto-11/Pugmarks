"""Pydantic schemas have correct shape and validation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pugmark.schemas import (
    Candidate,
    Chapter,
    ConfirmedTaxon,
    EvalRun,
    ExtractionMetrics,
    Gallery,
    ImageRef,
    Sighting,  # noqa: F401  # imported to verify schema exists
    ValidationMetrics,
)


def test_chapter_requires_page_offsets() -> None:
    chapter = Chapter(
        book="Nine Man-Eaters and One Rogue",
        number=1,
        title="The Black Panther of Sivanipalli",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=20,
        raw_text="raw",
        normalized_text="norm",
        page_offsets=[0, 100, 200],
        ingest_version="v1",
    )
    assert chapter.page_offsets == [0, 100, 200]


def test_candidate_kingdom_hint_is_enum() -> None:
    with pytest.raises(ValidationError):
        Candidate(
            surface_form="tiger",
            proposed_name="tiger",
            kingdom_hint="vegetable",  # invalid
            context_sentence="A tiger appeared.",
            context_window="A tiger appeared.",
            char_offset=0,
            page=1,
            llm_confidence=0.9,
            extractor_version="v1",
        )


def test_confirmed_taxon_collapses_candidates() -> None:
    c1 = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        kingdom_hint="animalia",
        context_sentence="A tiger appeared.",
        context_window="A tiger appeared.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v1",
    )
    c2 = c1.model_copy(update={"page": 5, "char_offset": 500})
    taxon = ConfirmedTaxon(
        canonical_name="Panthera tigris",
        vernacular="Tiger",
        wikidata_qid="Q15324",
        rank="species",
        lineage={"kingdom": "Animalia"},
        validation_method="sparql_exact",
        fuzzy_score=None,
        source_candidates=[c1, c2],
    )
    assert len(taxon.source_candidates) == 2


def test_image_ref_requires_attribution() -> None:
    img = ImageRef(
        url="https://example.org/x.jpg",
        license="CC BY-SA 4.0",
        attribution="Photographer Y",
        source="wikimedia",
    )
    assert img.source == "wikimedia"


def test_eval_run_records_provider() -> None:
    run = EvalRun(
        chapter_id="sivanipalli",
        extraction=ExtractionMetrics(
            precision=0.9, recall=0.85, f1=0.875, hallucination_rate=0.05
        ),
        validation=ValidationMetrics(
            qid_accuracy=0.92, confusion_matrix={}, unresolved_rate=0.08
        ),
        cost_usd=0.0,
        latency_ms=15000,
        pugmark_version="0.1.0",
        llm_provider="gemini-2.0-flash",
        prompt_version="v1",
        timestamp=datetime.now(),
    )
    assert run.llm_provider == "gemini-2.0-flash"


def test_gallery_can_have_no_eval_metrics() -> None:
    chapter = Chapter(
        book="X",
        number=1,
        title="T",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=2,
        raw_text="r",
        normalized_text="n",
        page_offsets=[0],
        ingest_version="v1",
    )
    gallery = Gallery(
        chapter=chapter,
        cards=[],
        unresolved=[],
        generated_at=datetime.now(),
        pugmark_version="0.1.0",
        eval_metrics=None,
    )
    assert gallery.eval_metrics is None


@pytest.mark.parametrize(
    ("char_offset", "expected_page"),
    [
        (0, 1),     # at first offset → page 1
        (50, 1),    # mid first page
        (100, 2),   # at second offset boundary → page 2
        (199, 2),   # just before third boundary
        (200, 3),   # at third offset boundary → page 3
        (999, 3),   # past last offset → clamps to last page
    ],
)
def test_chapter_offset_to_page(char_offset: int, expected_page: int) -> None:
    chapter = Chapter(
        book="X",
        number=1,
        title="T",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=3,
        raw_text="",
        normalized_text="",
        page_offsets=[0, 100, 200],
        ingest_version="v1",
    )
    assert chapter.offset_to_page(char_offset) == expected_page


def test_confirmed_taxon_fuzzy_score_bounds() -> None:
    c = Candidate(
        surface_form="x",
        proposed_name="x",
        kingdom_hint="animalia",
        context_sentence="x",
        context_window="x",
        char_offset=0,
        page=1,
        llm_confidence=0.5,
        extractor_version="v1",
    )
    with pytest.raises(ValidationError):
        ConfirmedTaxon(
            canonical_name="X",
            vernacular="x",
            wikidata_qid="Q1",
            rank="species",
            lineage={},
            validation_method="sparql_fuzzy",
            fuzzy_score=1.5,  # out of [0.0, 1.0]
            source_candidates=[c],
        )


def test_candidate_v1_kingdom_hint_still_works() -> None:
    """v1 callers passing kingdom_hint should produce entity_type='taxa'."""
    c = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        kingdom_hint="animalia",
        context_sentence="A tiger appeared.",
        context_window="A tiger appeared.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v1",
    )
    assert c.entity_type == "taxa"
    assert c.type_attrs["kingdom_hint"] == "animalia"


def test_candidate_v2_explicit_entity_type() -> None:
    c = Candidate(
        surface_form="Hercule Poirot",
        proposed_name="hercule poirot",
        entity_type="people",
        context_sentence="Poirot entered.",
        context_window="Poirot entered.",
        char_offset=0,
        page=1,
        llm_confidence=0.95,
        extractor_version="v2",
    )
    assert c.entity_type == "people"
    assert c.type_attrs == {}


def test_confirmed_entity_alias_to_confirmed_taxon() -> None:
    from pugmark.schemas import ConfirmedEntity, ConfirmedTaxon

    assert ConfirmedEntity is ConfirmedTaxon


def test_entity_card_alias_to_taxon_card() -> None:
    from pugmark.schemas import EntityCard, TaxonCard

    assert EntityCard is TaxonCard


def test_confirmed_entity_without_qid() -> None:
    from pugmark.schemas import ConfirmedEntity

    c = Candidate(
        surface_form="The Inquirer",
        proposed_name="the inquirer",
        entity_type="organizations",
        context_sentence="The Inquirer reported.",
        context_window="The Inquirer reported.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v2",
    )
    e = ConfirmedEntity(
        canonical_name="The Inquirer",
        vernacular="The Inquirer",
        entity_type="organizations",
        wikidata_qid=None,
        rank="organization",
        attributes={},
        validation_method="in_book_crossref",
        crossref_count=4,
        source_candidates=[c],
    )
    assert e.wikidata_qid is None
    assert e.validation_method == "in_book_crossref"
    assert e.crossref_count == 4


def test_entity_card_no_image_no_wikipedia() -> None:
    from pugmark.schemas import ConfirmedEntity, EntityCard

    c = Candidate(
        surface_form="The Cabal",
        proposed_name="the cabal",
        entity_type="factions",
        context_sentence="The Cabal struck.",
        context_window="The Cabal struck.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v2",
    )
    e = ConfirmedEntity(
        canonical_name="The Cabal",
        vernacular="The Cabal",
        entity_type="factions",
        wikidata_qid=None,
        rank="faction",
        attributes={},
        validation_method="judge_consensus",
        judge_votes=3,
        source_candidates=[c],
    )
    card = EntityCard(
        entity=e,
        wikipedia_url=None,
        wikipedia_summary="A shadowy organization mentioned in chapters 3 and 7.",
        summary_source="llm_in_book",
        primary_image=None,
        alt_images=[],
        sightings=[],
        enrich_version="v2",
    )
    assert card.primary_image is None
    assert card.wikipedia_url is None
    assert card.summary_source == "llm_in_book"


def test_gallery_cards_by_type() -> None:
    chapter = Chapter(
        book="X",
        number=1,
        title="T",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=2,
        raw_text="r",
        normalized_text="n",
        page_offsets=[0],
        ingest_version="v1",
    )
    g = Gallery(
        chapter=chapter,
        cards_by_type={},
        unresolved=[],
        generated_at=datetime.now(),
        pugmark_version="0.2.0",
        book_schema=None,
        eval_metrics=None,
    )
    assert g.cards_by_type == {}
