from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

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


@pytest.mark.asyncio
async def test_extract_returns_candidates(chapter: Chapter, tmp_path: Path) -> None:
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

    extract_mod.LLMClient.complete_structured = AsyncMock(side_effect=fake_complete_structured)

    from pugmark.cache import Cache
    cache = Cache(root=tmp_path / "cache")

    candidates = await extract_candidates(
        chapter, llm_config=LLMConfig(), prompt_dir=Path("prompts"), cache=cache
    )
    assert len(candidates) == 2
    assert candidates[0].surface_form == "tiger"
    assert candidates[0].extractor_version == EXTRACT_VERSION
    assert candidates[1].kingdom_hint == "plantae"


@pytest.mark.asyncio
async def test_extract_uses_cache_on_second_call(chapter: Chapter, tmp_path: Path) -> None:
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
    mock = AsyncMock(return_value=(_ExtractResponse(**fake_llm_payload), "gemini/gemini-2.0-flash"))
    extract_mod.LLMClient.complete_structured = mock

    cache = Cache(root=tmp_path / "cache")
    await extract_candidates(
        chapter, llm_config=LLMConfig(), prompt_dir=Path("prompts"), cache=cache
    )
    await extract_candidates(
        chapter, llm_config=LLMConfig(), prompt_dir=Path("prompts"), cache=cache
    )
    assert mock.await_count == 1, "second call should hit cache"
