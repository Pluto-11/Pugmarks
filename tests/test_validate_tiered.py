from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pugmark.cache import Cache
from pugmark.entity_type import EntityTypeSpec
from pugmark.schemas import Candidate, Chapter
from pugmark.validate import validate_candidates


@pytest.fixture
def _mock_judge_all_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to all-yes judge consensus so crossref tests keep passing."""

    async def fake_consensus(*args, **kwargs):  # type: ignore[no-untyped-def]
        return kwargs.get("n_calls", 3)

    monkeypatch.setattr("pugmark.validate._judge_consensus", fake_consensus)


def _recipes_spec(min_occ: int = 2) -> EntityTypeSpec:
    return EntityTypeSpec(
        name="recipes",
        description="recipes in a cookbook",
        wikidata_qclass=None,
        extraction_prompt_template="x",
        judge_prompt_template="x",
        min_book_occurrences=min_occ,
    )


def _candidate(name: str) -> Candidate:
    return Candidate(
        surface_form=name,
        proposed_name=name.lower(),
        entity_type="recipes",
        context_sentence=f"The {name} recipe.",
        context_window=f"The {name} recipe.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v2",
    )


def _chapter_with_text(text: str) -> Chapter:
    return Chapter(
        book="cookbook",
        number=1,
        title="Pasta",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=1,
        raw_text=text,
        normalized_text=text,
        page_offsets=[0],
        ingest_version="v1",
    )


@pytest.mark.asyncio
async def test_in_book_crossref_promotes_to_confirmed(
    tmp_path: Path, _mock_judge_all_yes: None
) -> None:
    """An entity with no Wikidata Q-class but >=2 in-book occurrences passes."""
    cache = Cache(root=tmp_path / "cache")
    chapter = _chapter_with_text(
        "The Carbonara recipe begins at noon. We made Carbonara again at night."
    )
    confirmed, unresolved = await validate_candidates(
        [_candidate("Carbonara")],
        entity_type=_recipes_spec(min_occ=2),
        chapter=chapter,
        cache=cache,
    )
    assert len(confirmed) == 1
    assert confirmed[0].validation_method == "judge_consensus"
    assert confirmed[0].judge_votes == 3
    assert confirmed[0].crossref_count == 2
    assert confirmed[0].wikidata_qid is None
    assert confirmed[0].entity_type == "recipes"
    assert len(unresolved) == 0


@pytest.mark.asyncio
async def test_in_book_crossref_below_threshold_unresolved(
    tmp_path: Path, _mock_judge_all_yes: None
) -> None:
    """Only 1 occurrence with threshold 2 -> unresolved."""
    cache = Cache(root=tmp_path / "cache")
    chapter = _chapter_with_text("The Aglio e Olio recipe. Once is not enough.")
    confirmed, unresolved = await validate_candidates(
        [_candidate("Aglio e Olio")],
        entity_type=_recipes_spec(min_occ=2),
        chapter=chapter,
        cache=cache,
    )
    assert len(confirmed) == 0
    assert len(unresolved) == 1


@pytest.mark.asyncio
async def test_in_book_crossref_case_insensitive_word_boundary(
    tmp_path: Path, _mock_judge_all_yes: None
) -> None:
    """'Risotto' should NOT match 'risottoso' (false positive guard)."""
    cache = Cache(root=tmp_path / "cache")
    chapter = _chapter_with_text(
        "The Risotto recipe. We tried Risotto again. (Bonus: risottoso, an Italian word.)"
    )
    confirmed, _ = await validate_candidates(
        [_candidate("Risotto")],
        entity_type=_recipes_spec(min_occ=2),
        chapter=chapter,
        cache=cache,
    )
    assert len(confirmed) == 1
    assert confirmed[0].crossref_count == 2  # not 3


@pytest.mark.asyncio
async def test_judge_consensus_promotes_when_majority_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When in-book + >=2/3 judge agreement: validation_method='judge_consensus'."""
    cache = Cache(root=tmp_path / "cache")
    chapter = _chapter_with_text(
        "The Cabal struck at dawn. Later, the Cabal regrouped. "
        "Some say the Cabal will return."
    )

    # Judge returns yes-no-yes (2/3)
    from pugmark.validate import _JudgeYesNo

    judge_responses = [
        (_JudgeYesNo(yes=True), "gemini/gemini-2.5-pro"),
        (_JudgeYesNo(yes=False), "gemini/gemini-2.5-pro"),
        (_JudgeYesNo(yes=True), "gemini/gemini-2.5-pro"),
    ]
    monkeypatch.setattr(
        "pugmark.validate.LLMClient.complete_structured",
        AsyncMock(side_effect=judge_responses),
    )

    factions_spec = EntityTypeSpec(
        name="factions",
        description="political factions in the novel",
        wikidata_qclass=None,
        extraction_prompt_template="x",
        judge_prompt_template="Is {{ candidate_name }} a faction in this book?",
        min_book_occurrences=2,
        min_judge_votes=2,
    )
    cand = _candidate("Cabal").model_copy(update={"entity_type": "factions"})
    confirmed, _ = await validate_candidates(
        [cand],
        entity_type=factions_spec,
        chapter=chapter,
        cache=cache,
    )
    assert len(confirmed) == 1
    assert confirmed[0].validation_method == "judge_consensus"
    assert confirmed[0].judge_votes == 2
    assert confirmed[0].crossref_count == 3


@pytest.mark.asyncio
async def test_judge_below_threshold_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-book passes but judge says no twice → unresolved."""
    cache = Cache(root=tmp_path / "cache")
    chapter = _chapter_with_text("The Cabal. The Cabal. The Cabal.")

    from pugmark.validate import _JudgeYesNo

    judge_responses = [
        (_JudgeYesNo(yes=False), "x"),
        (_JudgeYesNo(yes=False), "x"),
        (_JudgeYesNo(yes=True), "x"),
    ]
    monkeypatch.setattr(
        "pugmark.validate.LLMClient.complete_structured",
        AsyncMock(side_effect=judge_responses),
    )

    factions_spec = EntityTypeSpec(
        name="factions",
        description="x",
        wikidata_qclass=None,
        extraction_prompt_template="x",
        judge_prompt_template="x",
        min_book_occurrences=2,
        min_judge_votes=2,
    )
    cand = _candidate("Cabal").model_copy(update={"entity_type": "factions"})
    confirmed, unresolved = await validate_candidates(
        [cand],
        entity_type=factions_spec,
        chapter=chapter,
        cache=cache,
    )
    assert len(confirmed) == 0
    assert len(unresolved) == 1
