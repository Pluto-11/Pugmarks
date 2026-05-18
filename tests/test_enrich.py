from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from pugmark.cache import Cache
from pugmark.enrich import enrich_confirmed
from pugmark.schemas import Candidate, Chapter, ConfirmedTaxon


@pytest.fixture
def chapter() -> Chapter:
    text = "A tiger appeared in the clearing."
    return Chapter(
        book="X",
        number=1,
        title="T",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=1,
        raw_text=text,
        normalized_text=text,
        page_offsets=[0],
        ingest_version="v1",
    )


@pytest.fixture
def confirmed() -> list[ConfirmedTaxon]:
    cand = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        kingdom_hint="animalia",
        context_sentence="A tiger appeared in the clearing.",
        context_window="A tiger appeared in the clearing.",
        char_offset=2,
        page=1,
        llm_confidence=0.95,
        extractor_version="v1",
    )
    return [
        ConfirmedTaxon(
            canonical_name="Panthera tigris",
            vernacular="Tiger",
            wikidata_qid="Q15324",
            rank="species",
            lineage={},
            validation_method="sparql_exact",
            fuzzy_score=None,
            source_candidates=[cand],
        )
    ]


@pytest.mark.asyncio
async def test_enrich_produces_taxon_card(
    chapter: Chapter, confirmed: list[ConfirmedTaxon], tmp_path: Path
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    wp = json.loads((fixtures / "wikipedia_responses" / "tiger.json").read_text())
    commons = json.loads((fixtures / "commons_responses" / "tiger_image.json").read_text())

    async def fake_wp(qid: str) -> dict:
        return wp

    async def fake_commons(qid: str) -> dict:
        return commons

    cache = Cache(root=tmp_path / "cache")

    with (
        patch("pugmark.enrich._fetch_wikipedia", new=AsyncMock(side_effect=fake_wp)),
        patch("pugmark.enrich._fetch_commons_image", new=AsyncMock(side_effect=fake_commons)),
    ):
        cards = await enrich_confirmed(confirmed, chapter=chapter, cache=cache)

    assert len(cards) == 1
    card = cards[0]
    assert "tiger" in card.wikipedia_summary.lower()
    assert card.primary_image.license == "CC BY-SA 4.0"
    assert card.primary_image.attribution.startswith("Hollingsworth")
    assert len(card.sightings) == 1
    assert card.sightings[0].page == 1


@pytest.mark.asyncio
async def test_enrich_uses_llm_summary_when_no_qid(
    chapter: Chapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirmed entities without a Wikidata QID get an LLM-summarized card."""
    from pugmark.schemas import Candidate, ConfirmedEntity

    cand = Candidate(
        surface_form="The Cabal",
        proposed_name="The Cabal",
        entity_type="factions",
        context_sentence="The Cabal struck.",
        context_window="The Cabal struck silently at dawn.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v2",
    )
    entity = ConfirmedEntity(
        canonical_name="The Cabal",
        vernacular="The Cabal",
        entity_type="factions",
        wikidata_qid=None,
        rank="faction",
        attributes={},
        validation_method="judge_consensus",
        crossref_count=3,
        judge_votes=3,
        source_candidates=[cand],
    )

    class _Summary(BaseModel):
        text: str

    async def fake_summary(*args, **kwargs):
        return _Summary(
            text="The Cabal is a shadowy organization referenced repeatedly. (faction)"
        ), "gemini/gemini-2.0-flash"

    monkeypatch.setattr(
        "pugmark.enrich.LLMClient.complete_structured",
        AsyncMock(side_effect=fake_summary),
    )

    cache = Cache(root=tmp_path / "cache")
    cards = await enrich_confirmed([entity], chapter=chapter, cache=cache)

    assert len(cards) == 1
    card = cards[0]
    assert card.primary_image is None
    assert card.wikipedia_url is None
    assert card.summary_source == "llm_in_book"
    assert "shadowy" in card.wikipedia_summary.lower()
