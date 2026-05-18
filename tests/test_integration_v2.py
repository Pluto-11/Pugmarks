"""End-to-end integration test for Pugmark v2 with everything mocked.

Verifies: analyze → realize → per-type extract/validate/enrich → render HTML.
Two entity types are exercised: taxa (Wikidata path) and factions (tier-2 path).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pugmark.cache import Cache

FIXTURE_PDF = Path("tests/fixtures/sample_chapter.pdf")


@pytest.mark.asyncio
async def test_v2_two_types_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both a Wikidata-validated type and a tier-2 type land in the gallery."""
    from datetime import datetime as _dt

    import pugmark
    from pugmark.entity_type import BookSchema, EntityTypeSpec

    taxa_spec = EntityTypeSpec(
        name="taxa",
        description="x",
        wikidata_qclass="Q16521",
        extraction_prompt_template="extract taxa from {{ chapter_text }}",
        judge_prompt_template="x",
    )
    factions_spec = EntityTypeSpec(
        name="factions",
        description="political factions",
        wikidata_qclass=None,
        extraction_prompt_template="extract factions from {{ chapter_text }}",
        judge_prompt_template="Is {{ candidate_name }} a faction?",
    )
    fake_schema = BookSchema(
        book_id="sample_chapter",
        proposed_types=[taxa_spec, factions_spec],
        analyzer_version="v1",
        analyzed_at=_dt.now(),
    )

    async def fake_analyze(*args, **kwargs):
        return fake_schema

    monkeypatch.setattr("pugmark.api.analyze_book", fake_analyze)

    from pugmark.schemas import Candidate

    async def fake_extract(chapter, *, entity_type, llm_config, cache):
        if entity_type.name == "taxa":
            return [
                Candidate(
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
            ]
        return [
            Candidate(
                surface_form="Cabal",
                proposed_name="Cabal",
                entity_type="factions",
                context_sentence="The Cabal struck.",
                context_window="The Cabal struck.",
                char_offset=0,
                page=1,
                llm_confidence=0.9,
                extractor_version="v2",
            )
        ]

    monkeypatch.setattr("pugmark.api.extract_candidates", fake_extract)

    from pugmark.schemas import ConfirmedEntity

    async def fake_validate(cands, *, entity_type, chapter, cache):
        if not cands:
            return [], []
        if entity_type.name == "taxa":
            return [
                ConfirmedEntity(
                    canonical_name="Panthera tigris",
                    vernacular="Tiger",
                    entity_type="taxa",
                    wikidata_qid="Q15324",
                    rank="species",
                    attributes={},
                    validation_method="sparql_exact",
                    source_candidates=cands,
                )
            ], []
        return [
            ConfirmedEntity(
                canonical_name="Cabal",
                vernacular="Cabal",
                entity_type="factions",
                wikidata_qid=None,
                rank="faction",
                attributes={},
                validation_method="judge_consensus",
                crossref_count=3,
                judge_votes=3,
                source_candidates=cands,
            )
        ], []

    monkeypatch.setattr("pugmark.api.validate_candidates", fake_validate)

    from pugmark.schemas import EntityCard, ImageRef

    async def fake_enrich(entities, *, chapter, cache, llm_config=None):
        cards = []
        for e in entities:
            if e.entity_type == "taxa":
                cards.append(
                    EntityCard(
                        entity=e,
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
                )
            else:
                cards.append(
                    EntityCard(
                        entity=e,
                        wikipedia_url=None,
                        wikipedia_summary="A shadowy organization.",
                        summary_source="llm_in_book",
                        primary_image=None,
                        alt_images=[],
                        sightings=[],
                        enrich_version="v2",
                    )
                )
        return cards

    monkeypatch.setattr("pugmark.api.enrich_confirmed", fake_enrich)

    cache = Cache(root=tmp_path / "cache")
    gallery = await pugmark.extract_gallery(FIXTURE_PDF, 1, cache=cache)

    assert "taxa" in gallery.cards_by_type
    assert "factions" in gallery.cards_by_type
    assert gallery.cards_by_type["taxa"][0].summary_source == "wikipedia"
    assert gallery.cards_by_type["factions"][0].summary_source == "llm_in_book"
    assert gallery.cards_by_type["factions"][0].primary_image is None

    # Render to HTML and confirm both sections appear with appropriate decorations
    from pugmark.render import render_html

    html = render_html(gallery)
    assert "taxa" in html.lower() and "factions" in html.lower()
    assert "AI-summarized" in html
    assert "no-image-placeholder" in html
