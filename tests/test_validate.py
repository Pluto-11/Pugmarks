from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pugmark.cache import Cache
from pugmark.entity_type import EntityTypeSpec
from pugmark.schemas import Candidate, Chapter
from pugmark.validate import validate_candidates


def _candidate(name: str, kingdom: str = "animalia") -> Candidate:
    return Candidate(
        surface_form=name,
        proposed_name=name,
        kingdom_hint=kingdom,  # type: ignore[arg-type]
        context_sentence=f"A {name} appeared.",
        context_window=f"A {name} appeared.",
        char_offset=0,
        page=1,
        llm_confidence=0.9,
        extractor_version="v1",
    )


def _load_fixture(name: str) -> dict:
    p = Path(__file__).parent / "fixtures" / "wikidata_responses" / name
    return json.loads(p.read_text())


def _taxa_spec() -> EntityTypeSpec:
    return EntityTypeSpec(
        name="taxa",
        description="taxa",
        wikidata_qclass="Q16521",
        extraction_prompt_template="x",
        judge_prompt_template="x",
    )


def _dummy_chapter() -> Chapter:
    return Chapter(
        book="X",
        number=1,
        title="T",
        source_pdf=Path("/tmp/x.pdf"),
        page_start=1,
        page_end=1,
        raw_text="A tiger appeared. A peepul tree stood nearby.",
        normalized_text="A tiger appeared. A peepul tree stood nearby.",
        page_offsets=[0],
        ingest_version="v1",
    )


@pytest.mark.asyncio
async def test_exact_match_resolves(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    fake_response = _load_fixture("tiger_exact.json")

    async def fake_query(name: str, qclass: str) -> dict:
        return fake_response

    with patch("pugmark.validate._sparql_query", new=AsyncMock(side_effect=fake_query)):
        confirmed, unresolved = await validate_candidates(
            [_candidate("tiger")],
            entity_type=_taxa_spec(),
            chapter=_dummy_chapter(),
            cache=cache,
        )
    assert len(confirmed) == 1
    assert confirmed[0].wikidata_qid == "Q15324"
    assert confirmed[0].validation_method == "sparql_exact"
    assert len(unresolved) == 0


@pytest.mark.asyncio
async def test_unmatched_goes_to_unresolved(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    with patch(
        "pugmark.validate._sparql_query", new=AsyncMock(return_value=_load_fixture("empty.json"))
    ):
        confirmed, unresolved = await validate_candidates(
            [_candidate("schmoogle")],
            entity_type=_taxa_spec(),
            chapter=_dummy_chapter(),
            cache=cache,
        )
    assert len(confirmed) == 0
    assert len(unresolved) == 1


@pytest.mark.asyncio
async def test_many_to_one_collapse(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    c1 = _candidate("tiger")
    c2 = _candidate("tiger").model_copy(update={"page": 5, "char_offset": 500})

    with patch(
        "pugmark.validate._sparql_query",
        new=AsyncMock(return_value=_load_fixture("tiger_exact.json")),
    ):
        confirmed, _ = await validate_candidates(
            [c1, c2],
            entity_type=_taxa_spec(),
            chapter=_dummy_chapter(),
            cache=cache,
        )

    assert len(confirmed) == 1
    assert len(confirmed[0].source_candidates) == 2


@pytest.mark.asyncio
async def test_alias_match_uses_alias_method(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    with patch(
        "pugmark.validate._sparql_query",
        new=AsyncMock(return_value=_load_fixture("peepul_alias.json")),
    ):
        confirmed, unresolved = await validate_candidates(
            [_candidate("peepul", kingdom="plantae")],
            entity_type=_taxa_spec(),
            chapter=_dummy_chapter(),
            cache=cache,
        )
    assert len(confirmed) == 1
    assert confirmed[0].validation_method == "alias"
    assert confirmed[0].wikidata_qid == "Q193404"
    assert len(unresolved) == 0


@pytest.mark.asyncio
async def test_fuzzy_fallback_attaches_misspelled_to_resolved(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    tiger_fixture = _load_fixture("tiger_exact.json")
    empty_fixture = _load_fixture("empty.json")

    async def fake_query(name: str, qclass: str) -> dict:
        # "tiger" resolves; "tigr" misses → fuzzy fallback should attach it to tiger.
        if name == "tiger":
            return tiger_fixture
        return empty_fixture

    with patch("pugmark.validate._sparql_query", new=AsyncMock(side_effect=fake_query)):
        confirmed, unresolved = await validate_candidates(
            [_candidate("tiger"), _candidate("tigr")],
            entity_type=_taxa_spec(),
            chapter=_dummy_chapter(),
            cache=cache,
        )

    assert len(unresolved) == 0
    assert len(confirmed) == 1
    taxon = confirmed[0]
    assert taxon.wikidata_qid == "Q15324"
    assert taxon.validation_method == "sparql_fuzzy"
    assert taxon.fuzzy_score is not None and taxon.fuzzy_score >= 0.85
    assert len(taxon.source_candidates) == 2
