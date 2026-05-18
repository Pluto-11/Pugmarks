from __future__ import annotations

import pytest

from eval.metrics import compute_extraction_metrics
from pugmark.schemas import Candidate


def _candidate(name: str, page: int = 1) -> Candidate:
    return Candidate(
        surface_form=name,
        proposed_name=name.lower(),
        kingdom_hint="animalia",
        context_sentence=f"A {name} appeared.",
        context_window=f"A {name} appeared.",
        char_offset=0,
        page=page,
        llm_confidence=0.9,
        extractor_version="v1",
    )


def test_perfect_extraction() -> None:
    truth = [
        {"surface_form": "tiger", "expected_wikidata_qid": "Q15324"},
        {"surface_form": "panther", "expected_wikidata_qid": "Q34706"},
    ]
    extracted = [_candidate("tiger"), _candidate("panther")]
    m = compute_extraction_metrics(extracted, truth, chapter_text="A tiger and a panther.")
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.hallucination_rate == 0.0


def test_missed_extraction() -> None:
    truth = [
        {"surface_form": "tiger", "expected_wikidata_qid": "Q15324"},
        {"surface_form": "panther", "expected_wikidata_qid": "Q34706"},
    ]
    extracted = [_candidate("tiger")]
    m = compute_extraction_metrics(extracted, truth, chapter_text="A tiger and a panther.")
    assert m.recall == 0.5
    assert m.precision == 1.0


def test_hallucination() -> None:
    truth = [{"surface_form": "tiger", "expected_wikidata_qid": "Q15324"}]
    extracted = [_candidate("tiger"), _candidate("dragon")]
    m = compute_extraction_metrics(extracted, truth, chapter_text="A tiger appeared.")
    # "dragon" not in chapter_text → hallucination
    assert m.hallucination_rate == 0.5
    assert m.precision == 0.5


def test_extraction_metrics_per_type_breakdown() -> None:
    """compute_extraction_metrics returns per-type breakdown when entity_type field is set."""
    truth = [
        {"surface_form": "tiger", "expected_wikidata_qid": "Q15324", "entity_type": "taxa"},
        {"surface_form": "Anderson", "expected_wikidata_qid": "Q1", "entity_type": "people"},
        {"surface_form": "Sivanipalli", "expected_wikidata_qid": "Q2", "entity_type": "places"},
    ]
    extracted = [
        _candidate("tiger"),
        _candidate("Anderson").model_copy(update={"entity_type": "people"}),
        # places missed
    ]
    m = compute_extraction_metrics(
        extracted, truth, chapter_text="A tiger appeared. Anderson watched in Sivanipalli."
    )
    # Aggregate
    assert m.recall == pytest.approx(2 / 3, abs=1e-6)
    # Per-type breakdown
    assert m.by_type is not None
    assert m.by_type["taxa"].recall == 1.0
    assert m.by_type["people"].recall == 1.0
    assert m.by_type["places"].recall == 0.0
