from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pugmark.render import render_html
from pugmark.schemas import (
    Candidate,
    Chapter,
    ConfirmedTaxon,
    Gallery,
    ImageRef,
    Sighting,
    TaxonCard,
)


@pytest.fixture
def gallery() -> Gallery:
    chapter = Chapter(
        book="Test",
        number=1,
        title="T",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=1,
        raw_text="A tiger appeared.",
        normalized_text="A tiger appeared.",
        page_offsets=[0],
        ingest_version="v1",
    )
    cand = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        kingdom_hint="animalia",
        context_sentence="A tiger appeared.",
        context_window="A tiger appeared.",
        char_offset=2,
        page=1,
        llm_confidence=0.95,
        extractor_version="v1",
    )
    taxon = ConfirmedTaxon(
        canonical_name="Panthera tigris",
        vernacular="Tiger",
        wikidata_qid="Q15324",
        rank="species",
        lineage={},
        validation_method="sparql_exact",
        fuzzy_score=None,
        source_candidates=[cand],
    )
    card = TaxonCard(
        taxon=taxon,
        wikipedia_url="https://en.wikipedia.org/wiki/Tiger",
        wikipedia_summary="The tiger is the largest cat species.",
        primary_image=ImageRef(
            url="https://example.org/tiger.jpg",
            license="CC BY-SA 4.0",
            attribution="Photographer X",
            source="wikimedia",
        ),
        alt_images=[],
        sightings=[Sighting(page=1, paragraph="A tiger appeared.")],
        enrich_version="v1",
    )
    return Gallery(
        chapter=chapter,
        cards=[card],
        unresolved=[],
        generated_at=datetime.now(),
        pugmark_version="0.1.0",
        eval_metrics=None,
    )


def test_render_html_contains_taxon_info(gallery: Gallery) -> None:
    html = render_html(gallery)
    assert "Tiger" in html
    assert "Panthera tigris" in html
    assert "https://example.org/tiger.jpg" in html
    assert "CC BY-SA 4.0" in html
    assert "Photographer X" in html
    assert "https://en.wikipedia.org/wiki/Tiger" in html


def test_render_html_attribution_is_visible(gallery: Gallery) -> None:
    html = render_html(gallery)
    # Must render a visible attribution block, not just an HTML attribute
    assert "Photographer X" in html
    assert html.count("CC BY-SA 4.0") >= 1


def test_render_html_shows_unresolved_count(gallery: Gallery) -> None:
    cand = gallery.cards[0].taxon.source_candidates[0]
    g = gallery.model_copy(update={"unresolved": [cand]})
    html = render_html(g)
    assert "1 unresolved" in html or "Unresolved (1)" in html
