from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from pugmark.entity_type import EntityTypeSpec
from pugmark.extract import EXTRACT_VERSION, extract_candidates
from pugmark.llm import LLMConfig
from pugmark.schemas import Chapter


class _ExtractResponse(BaseModel):
    candidates: list[dict]


@pytest.fixture
def chapter() -> Chapter:
    text = "A tiger appeared near the peepul tree. The sambhur stood frozen."
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


def _taxa_spec() -> EntityTypeSpec:
    return EntityTypeSpec(
        name="taxa",
        description="taxa",
        wikidata_qclass="Q16521",
        extraction_prompt_template="Extract from {{ chapter_text }}",
        judge_prompt_template="x",
    )


@pytest.mark.asyncio
async def test_extract_returns_candidates(
    chapter: Chapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_llm_payload = {
        "candidates": [
            {
                "surface_form": "tiger",
                "proposed_name": "tiger",
                "kingdom_hint": "animalia",
                "context_sentence": "A tiger appeared near the peepul tree.",
                "char_offset": 2,
                "llm_confidence": 0.95,
            },
            {
                "surface_form": "peepul tree",
                "proposed_name": "peepul",
                "kingdom_hint": "plantae",
                "context_sentence": "A tiger appeared near the peepul tree.",
                "char_offset": 26,
                "llm_confidence": 0.9,
            },
        ]
    }

    async def fake_complete_structured(*args: object, **kwargs: object):
        return _ExtractResponse(**fake_llm_payload), "gemini/gemini-2.0-flash"

    from pugmark import extract as extract_mod

    monkeypatch.setattr(
        extract_mod.LLMClient,
        "complete_structured",
        AsyncMock(side_effect=fake_complete_structured),
    )

    from pugmark.cache import Cache
    cache = Cache(root=tmp_path / "cache")

    candidates = await extract_candidates(
        chapter, entity_type=_taxa_spec(), llm_config=LLMConfig(), cache=cache
    )
    assert len(candidates) == 2
    assert candidates[0].surface_form == "tiger"
    assert candidates[0].extractor_version == EXTRACT_VERSION
    assert candidates[1].entity_type == "taxa"


@pytest.mark.asyncio
async def test_extract_uses_cache_on_second_call(
    chapter: Chapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pugmark import extract as extract_mod
    from pugmark.cache import Cache

    fake_llm_payload = {
        "candidates": [
            {
                "surface_form": "tiger",
                "proposed_name": "tiger",
                "kingdom_hint": "animalia",
                "context_sentence": "A tiger appeared.",
                "char_offset": 2,
                "llm_confidence": 0.95,
            }
        ]
    }
    mock = AsyncMock(
        return_value=(_ExtractResponse(**fake_llm_payload), "gemini/gemini-2.0-flash")
    )
    monkeypatch.setattr(extract_mod.LLMClient, "complete_structured", mock)

    cache = Cache(root=tmp_path / "cache")
    await extract_candidates(
        chapter, entity_type=_taxa_spec(), llm_config=LLMConfig(), cache=cache
    )
    await extract_candidates(
        chapter, entity_type=_taxa_spec(), llm_config=LLMConfig(), cache=cache
    )
    assert mock.await_count == 1, "second call should hit cache"


@pytest.mark.asyncio
async def test_extract_uses_entity_type_specific_prompt(
    chapter: Chapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extraction prompt comes from EntityTypeSpec, not a hardcoded file."""
    fake_payload = {
        "candidates": [
            {
                "surface_form": "Sherlock",
                "proposed_name": "Sherlock Holmes",
                "entity_type": "people",
                "context_sentence": "Sherlock entered.",
                "char_offset": 0,
                "llm_confidence": 0.95,
            }
        ]
    }

    class _Resp(BaseModel):
        candidates: list[dict]

    async def fake_call(*args, **kwargs):
        return _Resp(**fake_payload), "gemini/gemini-2.0-flash"

    from pugmark import extract as extract_mod

    monkeypatch.setattr(
        extract_mod.LLMClient,
        "complete_structured",
        AsyncMock(side_effect=fake_call),
    )
    from pugmark.cache import Cache
    cache = Cache(root=tmp_path / "cache")

    people_spec = EntityTypeSpec(
        name="people",
        description="people",
        wikidata_qclass="Q5",
        extraction_prompt_template="Find people in: {{ chapter_text }}",
        judge_prompt_template="x",
    )
    candidates = await extract_candidates(
        chapter, entity_type=people_spec, llm_config=LLMConfig(), cache=cache
    )
    assert len(candidates) == 1
    assert candidates[0].entity_type == "people"
    assert candidates[0].proposed_name == "Sherlock Holmes"
