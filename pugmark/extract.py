"""Chapter → list[Candidate] via LiteLLM + structured output.

Single-pass: full chapter text in context (Gemini 2.0 Flash has 1M tokens).
Caches by hash(normalized_text + extractor_version + prompt_version + provider).
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from pugmark.cache import Cache
from pugmark.llm import LLMClient, LLMConfig
from pugmark.prompt_registry import PromptRegistry
from pugmark.schemas import Candidate, Chapter

logger = logging.getLogger(__name__)

EXTRACT_VERSION = "v1"
PROMPT_NAME = "extract_taxa"
SYSTEM_PROMPT = (
    "You output strictly valid JSON. Do not add commentary. "
    "Follow the user's schema exactly."
)


class _ExtractResponse(BaseModel):
    """LLM-output shape; we transform to Candidate after."""

    candidates: list[dict] = Field(default_factory=list)


async def extract_candidates(
    chapter: Chapter,
    *,
    llm_config: LLMConfig,
    prompt_dir: Path,
    cache: Cache,
) -> list[Candidate]:
    registry = PromptRegistry(in_repo_dir=prompt_dir)
    prompt = registry.get(PROMPT_NAME)
    user_prompt = prompt.render(chapter_text=chapter.normalized_text)

    cache_key = Cache.compute_hash(
        chapter.normalized_text,
        EXTRACT_VERSION,
        prompt.version,
        ",".join(llm_config.providers),
    )

    cached = cache.get("extract", cache_key, _CandidateBundle)
    if cached is not None:
        logger.info("extract cache hit")
        return cached.candidates

    client = LLMClient(llm_config)
    resp, provider_used = await client.complete_structured(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        schema=_ExtractResponse,
        prompt_version=prompt.version,
    )

    candidates: list[Candidate] = []
    for raw in resp.candidates:
        char_offset = int(raw.get("char_offset", 0))
        page = chapter.offset_to_page(char_offset)
        # Pull a 3-sentence window around char_offset for the future reading-companion mode
        window = _three_sentence_window(chapter.normalized_text, char_offset)
        candidates.append(
            Candidate(
                surface_form=str(raw["surface_form"]),
                proposed_name=str(raw["proposed_name"]),
                kingdom_hint=raw.get("kingdom_hint", "unknown"),
                context_sentence=str(raw.get("context_sentence", "")),
                context_window=window,
                char_offset=char_offset,
                page=page,
                llm_confidence=float(raw.get("llm_confidence", 0.5)),
                extractor_version=EXTRACT_VERSION,
            )
        )

    bundle = _CandidateBundle(candidates=candidates)
    cache.set("extract", cache_key, bundle)
    logger.info(f"extracted {len(candidates)} candidates via {provider_used}")
    return candidates


class _CandidateBundle(BaseModel):
    """Wrapper so we can cache list[Candidate] via the Cache.set/get API."""

    candidates: list[Candidate]


def _three_sentence_window(text: str, char_offset: int) -> str:
    """Return ~3 sentences centered on char_offset for context."""
    import re

    sent_ends = [m.end() for m in re.finditer(r"[.!?](\s+|$)", text)]
    if not sent_ends:
        return text[max(0, char_offset - 200) : char_offset + 200]
    # Find the sentence containing char_offset
    sent_end_idx = next((i for i, e in enumerate(sent_ends) if e > char_offset), len(sent_ends) - 1)
    start_idx = max(0, sent_end_idx - 1)
    end_idx = min(len(sent_ends) - 1, sent_end_idx + 1)
    start = 0 if start_idx == 0 else sent_ends[start_idx - 1]
    end = sent_ends[end_idx]
    return text[start:end].strip()
