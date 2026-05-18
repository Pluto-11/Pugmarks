"""Tests for the LLM-as-judge auto-labeling pipeline.

Mocks both the judge LLM and the Wikidata roundtrip so tests are network-free.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pugmark.cache import Cache
from pugmark.schemas import ConfirmedTaxon

FIXTURE_PDF = Path("tests/fixtures/sample_chapter.pdf")


def _judge_resp(candidates: list[dict]):
    """Build a fake (_JudgeResponse, provider) tuple."""
    from eval.auto_label import _JudgeCandidate, _JudgeResponse

    return _JudgeResponse(
        candidates=[_JudgeCandidate(**c) for c in candidates]
    ), "gemini/gemini-2.5-pro"


@pytest.mark.asyncio
async def test_majority_vote_keeps_two_of_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entries appearing in >=2 of 3 judge calls survive."""
    from eval import auto_label

    tiger_entry = {
        "surface_form": "tiger",
        "proposed_name": "tiger",
        "kingdom_hint": "animalia",
        "context_sentence": "A tiger appeared.",
        "llm_confidence": 0.95,
    }
    peepul_entry = {
        "surface_form": "peepul tree",
        "proposed_name": "peepul",
        "kingdom_hint": "plantae",
        "context_sentence": "Under a peepul tree.",
        "llm_confidence": 0.9,
    }
    sambhur_entry = {
        "surface_form": "sambhur",
        "proposed_name": "sambar deer",
        "kingdom_hint": "animalia",
        "context_sentence": "A sambhur grazed.",
        "llm_confidence": 0.88,
    }
    one_off_entry = {
        "surface_form": "dragon",
        "proposed_name": "dragon",
        "kingdom_hint": "animalia",
        "context_sentence": "x",
        "llm_confidence": 0.86,
    }

    # 3 calls: tiger in all 3, peepul in 2, sambhur in 2, dragon in 1 (should be dropped).
    call_responses = [
        _judge_resp([tiger_entry, peepul_entry, sambhur_entry]),
        _judge_resp([tiger_entry, peepul_entry, one_off_entry]),
        _judge_resp([tiger_entry, sambhur_entry]),
    ]
    judge_mock = AsyncMock(side_effect=call_responses)
    monkeypatch.setattr(
        "eval.auto_label.LLMClient.complete_structured", judge_mock
    )

    # All three survivors resolve to fake QIDs via mocked validator.
    async def fake_validate(cands, **kw):
        confirmed = [
            ConfirmedTaxon(
                canonical_name=c.proposed_name.title(),
                vernacular=c.proposed_name,
                wikidata_qid=f"Q{1000 + i}",
                rank="species",
                lineage={},
                validation_method="sparql_exact",
                fuzzy_score=None,
                source_candidates=[c],
            )
            for i, c in enumerate(cands)
        ]
        return confirmed, []

    monkeypatch.setattr("eval.auto_label.validate_candidates", fake_validate)

    cache = Cache(root=tmp_path / "cache")
    truth = await auto_label.auto_label_chapter(
        FIXTURE_PDF, 1, cache=cache, prompt_dir=Path("prompts")
    )

    # dragon (1 vote) excluded; tiger (3) + peepul (2) + sambhur (2) included.
    surface_forms = sorted(t["surface_form"] for t in truth)
    assert surface_forms == ["peepul tree", "sambhur", "tiger"]
    assert all("expected_wikidata_qid" in t for t in truth)
    assert all("expected_kingdom" in t for t in truth)
    assert all("page" in t for t in truth)
    assert judge_mock.await_count == 3, "expected 3 independent judge calls"


@pytest.mark.asyncio
async def test_wikidata_roundtrip_filters_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Survivors that fail Wikidata lookup are excluded from ground truth."""
    from eval import auto_label

    entry = {
        "surface_form": "schmoogle",
        "proposed_name": "schmoogle",
        "kingdom_hint": "animalia",
        "context_sentence": "A schmoogle appeared.",
        "llm_confidence": 0.95,
    }
    call_responses = [
        _judge_resp([entry]),
        _judge_resp([entry]),
        _judge_resp([entry]),
    ]
    monkeypatch.setattr(
        "eval.auto_label.LLMClient.complete_structured",
        AsyncMock(side_effect=call_responses),
    )

    # Wikidata can't resolve "schmoogle" → empty confirmed, full unresolved.
    async def fake_validate(cands, **kw):
        return [], list(cands)

    monkeypatch.setattr("eval.auto_label.validate_candidates", fake_validate)

    cache = Cache(root=tmp_path / "cache")
    truth = await auto_label.auto_label_chapter(
        FIXTURE_PDF, 1, cache=cache, prompt_dir=Path("prompts")
    )
    assert truth == [], "unresolved survivors must be dropped"


@pytest.mark.asyncio
async def test_below_quorum_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If every entry shows up only once across 3 calls, ground truth is empty."""
    from eval import auto_label

    call_responses = [
        _judge_resp([{
            "surface_form": "alpha",
            "proposed_name": "alpha",
            "kingdom_hint": "animalia",
            "context_sentence": "x",
            "llm_confidence": 0.9,
        }]),
        _judge_resp([{
            "surface_form": "beta",
            "proposed_name": "beta",
            "kingdom_hint": "animalia",
            "context_sentence": "x",
            "llm_confidence": 0.9,
        }]),
        _judge_resp([{
            "surface_form": "gamma",
            "proposed_name": "gamma",
            "kingdom_hint": "animalia",
            "context_sentence": "x",
            "llm_confidence": 0.9,
        }]),
    ]
    monkeypatch.setattr(
        "eval.auto_label.LLMClient.complete_structured",
        AsyncMock(side_effect=call_responses),
    )

    async def fake_validate(cands, **kw):
        # Should never be called; assert it isn't.
        raise AssertionError("validate_candidates should be skipped on empty survivors")

    monkeypatch.setattr("eval.auto_label.validate_candidates", fake_validate)

    cache = Cache(root=tmp_path / "cache")
    truth = await auto_label.auto_label_chapter(
        FIXTURE_PDF, 1, cache=cache, prompt_dir=Path("prompts")
    )
    assert truth == []
