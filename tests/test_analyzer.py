from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pugmark.analyzer import (
    ANALYZER_VERSION,
    _AnalyzerResponse,
    _AnalyzerType,
    analyze_book,
)
from pugmark.cache import Cache
from pugmark.entity_type import BookSchema, BookType

FIXTURE_PDF = Path("tests/fixtures/sample_chapter.pdf")


def _resp(types: list[dict]) -> tuple[_AnalyzerResponse, str]:
    return (
        _AnalyzerResponse(proposed_types=[_AnalyzerType(**t) for t in types]),
        "gemini/gemini-2.5-pro",
    )


def _book_type_resp() -> tuple[BookType, str]:
    return (
        BookType(
            genre="test",
            period="test",
            setting="test",
            themes=["x"],
            target_reader="x",
            summary="x",
        ),
        "gemini/gemini-2.5-pro",
    )


def _two_pass_mock(proposals: list[dict]) -> AsyncMock:
    """Mock that returns BookType for pass 1, _AnalyzerResponse for pass 2."""
    return AsyncMock(side_effect=[_book_type_resp(), _resp(proposals)])


@pytest.mark.asyncio
async def test_analyze_returns_book_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals = [
        {
            "name": "taxa",
            "description": "Animals + plants in the chapter",
            "examples": ["tiger", "peepul tree"],
            "wikidata_qclass": "Q16521",
        },
        {
            "name": "people",
            "description": "Named human characters",
            "examples": ["Anderson"],
            "wikidata_qclass": "Q5",
        },
    ]
    mock = _two_pass_mock(proposals)
    monkeypatch.setattr("pugmark.analyzer.LLMClient.complete_structured", mock)

    cache = Cache(root=tmp_path / "cache")
    schema = await analyze_book(FIXTURE_PDF, cache=cache)

    assert isinstance(schema, BookSchema)
    assert schema.analyzer_version == ANALYZER_VERSION
    assert len(schema.proposed_types) == 2
    assert {t.name for t in schema.proposed_types} == {"taxa", "people"}
    # Two passes: book-type classifier + entity-type proposer
    assert mock.await_count == 2
    assert schema.book_type is not None
    assert schema.book_type.genre == "test"


@pytest.mark.asyncio
async def test_analyze_second_call_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals = [
        {
            "name": "taxa",
            "description": "x",
            "examples": ["x"],
            "wikidata_qclass": "Q16521",
        }
    ]
    # Pass 1 (book-type) + pass 2 (entity-types) = 2 awaits on first call.
    # Second call should hit cache and add zero awaits.
    mock = AsyncMock(side_effect=[_book_type_resp(), _resp(proposals)])
    monkeypatch.setattr("pugmark.analyzer.LLMClient.complete_structured", mock)

    cache = Cache(root=tmp_path / "cache")
    await analyze_book(FIXTURE_PDF, cache=cache)
    await analyze_book(FIXTURE_PDF, cache=cache)
    assert mock.await_count == 2, "second call should hit cache (no new awaits)"


@pytest.mark.asyncio
async def test_analyze_filters_too_granular_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proposal with no Q-class AND a very long, sentence-like name is dropped."""
    proposals = [
        {
            "name": "taxa",
            "description": "x",
            "examples": ["x"],
            "wikidata_qclass": "Q16521",
        },
        {
            # ~junk: sentence-like, no qclass
            "name": "specific anecdotes about grandmother's garden",
            "description": "x",
            "examples": [],
            "wikidata_qclass": None,
        },
    ]
    monkeypatch.setattr(
        "pugmark.analyzer.LLMClient.complete_structured",
        _two_pass_mock(proposals),
    )

    cache = Cache(root=tmp_path / "cache")
    schema = await analyze_book(FIXTURE_PDF, cache=cache)
    names = [t.name for t in schema.proposed_types]
    assert "taxa" in names
    assert all(" " not in n and len(n) <= 40 for n in names)
