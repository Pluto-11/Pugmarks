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
    TaxonCard,  # noqa: F401  # imported to verify schema exists
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
