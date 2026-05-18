from __future__ import annotations

import json
from datetime import datetime
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
    from pugmark.entity_type import BookSchema, EntityTypeSpec
    from pugmark.schemas import (
        Candidate,
        ConfirmedTaxon,
        ImageRef,
        TaxonCard,
    )

    cand = Candidate(
        surface_form="tiger",
        proposed_name="tiger",
        entity_type="taxa",
        context_sentence="A tiger appeared.",
        context_window="A tiger appeared.",
        char_offset=0,
        page=1,
        llm_confidence=0.95,
        extractor_version="v2",
    )

    async def fake_extract(chapter, **kw):
        return [cand]

    async def fake_validate(cands, *, entity_type, chapter, cache):
        confirmed = [
            ConfirmedTaxon(
                canonical_name="Panthera tigris",
                vernacular="Tiger",
                entity_type=entity_type.name,
                wikidata_qid="Q15324",
                rank="species",
                attributes={},
                validation_method="sparql_exact",
                fuzzy_score=None,
                source_candidates=[cand],
            )
        ]
        return confirmed, []

    async def fake_enrich(taxa, **kw):
        if not taxa:
            return []
        return [
            TaxonCard(
                entity=taxa[0],
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

    fake_schema = BookSchema(
        book_id="sample_chapter",
        proposed_types=[
            EntityTypeSpec(
                name="taxa",
                description="x",
                wikidata_qclass="Q16521",
                extraction_prompt_template="x",
                judge_prompt_template="x",
            ),
        ],
        analyzer_version="v1",
        analyzed_at=datetime.now(),
    )

    async def fake_analyze(*args, **kwargs):
        return fake_schema

    with (
        patch("eval.runner.analyze_book", new=AsyncMock(side_effect=fake_analyze)),
        patch("eval.runner.extract_candidates", new=AsyncMock(side_effect=fake_extract)),
        patch("eval.runner.validate_candidates", new=AsyncMock(side_effect=fake_validate)),
        patch("eval.runner.enrich_confirmed", new=AsyncMock(side_effect=fake_enrich)),
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


@pytest.mark.asyncio
async def test_run_eval_multi_file_ground_truth(tmp_path: Path) -> None:
    """Ground truth can be a list of files (one per type) or a single file."""
    from pathlib import Path as _P

    from pugmark.entity_type import BookSchema, EntityTypeSpec
    from pugmark.schemas import Candidate, ConfirmedEntity

    pdf_fixture = _P("tests/fixtures/sample_chapter.pdf")
    truth_taxa = [
        {"surface_form": "tiger", "expected_wikidata_qid": "Q15324", "entity_type": "taxa"},
    ]
    truth_people = [
        {"surface_form": "Anderson", "expected_wikidata_qid": "Q1", "entity_type": "people"},
    ]
    (tmp_path / "sivanipalli__taxa.json").write_text(json.dumps(truth_taxa))
    (tmp_path / "sivanipalli__people.json").write_text(json.dumps(truth_people))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    cand_tiger = Candidate(
        surface_form="tiger", proposed_name="tiger", entity_type="taxa",
        context_sentence="x", context_window="x", char_offset=0, page=1,
        llm_confidence=0.9, extractor_version="v2",
    )
    cand_anderson = Candidate(
        surface_form="Anderson", proposed_name="Anderson", entity_type="people",
        context_sentence="x", context_window="x", char_offset=0, page=1,
        llm_confidence=0.9, extractor_version="v2",
    )

    async def fake_extract_per_type(chapter, *, entity_type, llm_config, cache):
        if entity_type.name == "taxa":
            return [cand_tiger]
        return [cand_anderson]

    async def fake_validate(cands, *, entity_type, chapter, cache):
        if not cands:
            return [], []
        return [
            ConfirmedEntity(
                canonical_name=cands[0].proposed_name,
                vernacular=cands[0].proposed_name,
                entity_type=entity_type.name,
                wikidata_qid="Q1",
                rank="x",
                attributes={},
                validation_method="sparql_exact",
                source_candidates=cands,
            )
        ], []

    async def fake_enrich(entities, *, chapter, cache, llm_config=None):
        return []

    fake_schema = BookSchema(
        book_id="sample_chapter",
        proposed_types=[
            EntityTypeSpec(name="taxa", description="x", wikidata_qclass="Q16521",
                           extraction_prompt_template="x", judge_prompt_template="x"),
            EntityTypeSpec(name="people", description="x", wikidata_qclass="Q5",
                           extraction_prompt_template="x", judge_prompt_template="x"),
        ],
        analyzer_version="v1",
        analyzed_at=datetime.now(),
    )

    async def fake_analyze(*args, **kwargs):
        return fake_schema

    with (
        patch("eval.runner.analyze_book", new=AsyncMock(side_effect=fake_analyze)),
        patch("eval.runner.extract_candidates", new=AsyncMock(side_effect=fake_extract_per_type)),
        patch("eval.runner.validate_candidates", new=AsyncMock(side_effect=fake_validate)),
        patch("eval.runner.enrich_confirmed", new=AsyncMock(side_effect=fake_enrich)),
    ):
        from eval.runner import run_eval

        run = await run_eval(
            pdf=pdf_fixture,
            chapter_number=1,
            ground_truth_path=tmp_path,  # directory mode
            runs_dir=runs_dir,
        )

    assert run.extraction.precision == 1.0
    assert run.extraction.recall == 1.0
    # Per-type breakdown populated
    assert run.extraction.by_type is not None
    assert "taxa" in run.extraction.by_type
    assert "people" in run.extraction.by_type
