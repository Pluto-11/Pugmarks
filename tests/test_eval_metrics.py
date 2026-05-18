from __future__ import annotations

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
