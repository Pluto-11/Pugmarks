from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from eval.runner import run_eval


@pytest.mark.asyncio
async def test_run_eval_writes_run_json(tmp_path: Path) -> None:
    pdf_fixture = Path("tests/fixtures/sample_chapter.pdf")
    truth = [
        {
            "surface_form": "tiger",
            "expected_wikidata_qid": "Q15324",
            "expected_kingdom": "animalia",
        },
    ]
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Mock heavy stages
    from pugmark.schemas import (
        Candidate,
        ConfirmedTaxon,
        ImageRef,
        TaxonCard,
    )

    cand = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        kingdom_hint="animalia",
        context_sentence="A tiger appeared.",
        context_window="A tiger appeared.",
        char_offset=0,
        page=1,
        llm_confidence=0.95,
        extractor_version="v1",
    )

    async def fake_extract(chapter, **kw):
        return [cand]

    async def fake_validate(cands, **kw):
        confirmed = [
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
        return confirmed, []

    async def fake_enrich(taxa, **kw):
        return [
            TaxonCard(
                taxon=taxa[0],
                wikipedia_url="https://en.wikipedia.org/wiki/Tiger",
                wikipedia_summary="...",
                primary_image=ImageRef(
                    url="https://x.org/x.jpg",
                    license="CC0",
                    attribution="x",
                    source="wikimedia",
                ),
                alt_images=[],
                sightings=[],
                enrich_version="v1",
            )
        ]

    with (
        patch("eval.runner.extract_candidates", new=AsyncMock(side_effect=fake_extract)),
        patch("eval.runner.validate_candidates", new=AsyncMock(side_effect=fake_validate)),
        patch("eval.runner.enrich_taxa", new=AsyncMock(side_effect=fake_enrich)),
    ):
        run = await run_eval(
            pdf=pdf_fixture,
            chapter_number=1,
            ground_truth_path=truth_path,
            runs_dir=runs_dir,
        )

    assert run.extraction.precision == 1.0
    assert run.extraction.recall == 1.0
    assert run.validation.qid_accuracy == 1.0
    written = list(runs_dir.glob("*.json"))
    assert len(written) == 1
