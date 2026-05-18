"""Chapter → list[Candidate] via type-parameterized LLM call.

Single-pass: full chapter text in context. Caches by hash(text + extractor_version
+ prompt_version + provider + entity_type + spec_version).
"""
from __future__ import annotations

import logging

from jinja2 import Template
from pydantic import BaseModel, Field

from pugmark.cache import Cache
from pugmark.entity_type import EntityTypeSpec
from pugmark.llm import LLMClient, LLMConfig
from pugmark.schemas import Candidate, Chapter

logger = logging.getLogger(__name__)

EXTRACT_VERSION = "v2"
SYSTEM_PROMPT = (
    "You output strictly valid JSON. Do not add commentary. "
    "Follow the user's schema exactly."
)


class _ExtractResponse(BaseModel):
    candidates: list[dict] = Field(default_factory=list)


class _CandidateBundle(BaseModel):
    candidates: list[Candidate]


async def extract_candidates(
    chapter: Chapter,
    *,
    entity_type: EntityTypeSpec,
    llm_config: LLMConfig,
    cache: Cache,
) -> list[Candidate]:
    user_prompt = Template(entity_type.extraction_prompt_template).render(
        chapter_text=chapter.normalized_text
    )

    cache_key = Cache.compute_hash(
        chapter.normalized_text,
        EXTRACT_VERSION,
        entity_type.name,
        entity_type.spec_version,
        ",".join(llm_config.providers),
    )

    cached = cache.get(f"extract.{entity_type.name}", cache_key, _CandidateBundle)
    if cached is not None:
        logger.info(f"extract cache hit for {entity_type.name}")
        return cached.candidates

    client = LLMClient(llm_config)
    resp, provider_used = await client.complete_structured(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        schema=_ExtractResponse,
        prompt_version=entity_type.spec_version,
    )

    candidates: list[Candidate] = []
    for raw in resp.candidates:
        char_offset = int(raw.get("char_offset", 0))
        page = chapter.offset_to_page(char_offset)
        window = _three_sentence_window(chapter.normalized_text, char_offset)
        candidates.append(
            Candidate(
                surface_form=str(raw["surface_form"]),
                proposed_name=str(raw["proposed_name"]),
                entity_type=entity_type.name,
                context_sentence=str(raw.get("context_sentence", "")),
                context_window=window,
                char_offset=char_offset,
                page=page,
                llm_confidence=float(raw.get("llm_confidence", 0.5)),
                extractor_version=EXTRACT_VERSION,
            )
        )

    bundle = _CandidateBundle(candidates=candidates)
    cache.set(f"extract.{entity_type.name}", cache_key, bundle)
    logger.info(f"extracted {len(candidates)} {entity_type.name} via {provider_used}")
    return candidates


def _three_sentence_window(text: str, char_offset: int) -> str:
    """Return ~3 sentences centered on char_offset for context."""
    import re

    sent_ends = [m.end() for m in re.finditer(r"[.!?](\s+|$)", text)]
    if not sent_ends:
        return text[max(0, char_offset - 200) : char_offset + 200]
    sent_end_idx = next(
        (i for i, e in enumerate(sent_ends) if e > char_offset), len(sent_ends) - 1
    )
    start_idx = max(0, sent_end_idx - 1)
    end_idx = min(len(sent_ends) - 1, sent_end_idx + 1)
    start = 0 if start_idx == 0 else sent_ends[start_idx - 1]
    end = sent_ends[end_idx]
    return text[start:end].strip()
