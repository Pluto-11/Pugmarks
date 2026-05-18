"""Chapter → list[Candidate] via type-parameterized LLM call.

Strategy: chunk the chapter into sentence-aligned overlapping windows, fan out
to the LLM in parallel (bounded by PUGMARK_EXTRACT_CONCURRENCY), merge + dedupe
across chunks. Caches per-chunk so partial re-runs are free.

Chunk size and overlap are driven by env:
  PUGMARK_CHUNK_TOKENS         (default 6000)
  PUGMARK_CHUNK_OVERLAP        (default 600)
  PUGMARK_EXTRACT_CONCURRENCY  (default 3)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

from jinja2 import Template
from pydantic import BaseModel, Field

from pugmark.cache import Cache
from pugmark.chunking import ChunkRef, chunk_text, merge_candidates
from pugmark.entity_type import EntityTypeSpec
from pugmark.llm import LLMClient, LLMConfig
from pugmark.schemas import Candidate, Chapter

logger = logging.getLogger(__name__)

EXTRACT_VERSION = "v3"  # bumped: chunked fan-out invalidates v2 cache hits
SYSTEM_PROMPT = (
    "You output strictly valid JSON. Do not add commentary. "
    "Follow the user's schema exactly."
)


class _ExtractResponse(BaseModel):
    candidates: list[dict] = Field(default_factory=list)


class _CandidateBundle(BaseModel):
    candidates: list[Candidate]


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{key}={raw!r} is not an int; using default {default}")
        return default


_LEADING_INT_RE = re.compile(r"-?\d+")


def _coerce_int(value: object, default: int = 0) -> int:
    """Best-effort int coercion for LLM-returned offsets.

    LLMs sometimes return ranges ("217-222"), strings ("217"), floats, or null.
    We accept what we can and quietly default to 0 on garbage so one bad row
    doesn't kill an entire chunk's extraction.
    """
    if value is None:
        return default
    if isinstance(value, bool):  # bool is a subtype of int — refuse it
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value)
    m = _LEADING_INT_RE.search(s)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return default
    return default


def _coerce_float(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


async def _extract_one_chunk(
    chunk: ChunkRef,
    *,
    entity_type: EntityTypeSpec,
    chapter: Chapter,
    llm_config: LLMConfig,
    cache: Cache,
    sem: asyncio.Semaphore,
) -> list[Candidate]:
    """LLM-extract candidates from a single chunk. Caches per-chunk."""
    user_prompt = Template(entity_type.extraction_prompt_template).render(
        chapter_text=chunk.text
    )
    cache_key = Cache.compute_hash(
        chunk.text,
        EXTRACT_VERSION,
        entity_type.name,
        entity_type.spec_version,
        ",".join(llm_config.providers),
    )
    cached = cache.get(f"extract.{entity_type.name}", cache_key, _CandidateBundle)
    if cached is not None:
        logger.info(
            f"extract chunk {chunk.index} cache hit for {entity_type.name} "
            f"({len(cached.candidates)} candidates)"
        )
        return cached.candidates

    async with sem:
        client = LLMClient(llm_config)
        resp, provider_used = await client.complete_structured(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_ExtractResponse,
            prompt_version=f"{entity_type.spec_version}|chunk{chunk.index}",
        )

    candidates: list[Candidate] = []
    for raw in resp.candidates:
        # LLMs sometimes return ranges ("217-222"), nulls, or floats — coerce safely.
        chunk_local_offset = _coerce_int(raw.get("char_offset"), default=0)
        if "surface_form" not in raw or "proposed_name" not in raw:
            # Skip malformed rows rather than crash the whole chunk
            logger.warning(
                f"chunk {chunk.index}/{entity_type.name}: skipping malformed row {raw!r}"
            )
            continue
        # Use the chapter-level page even at this stage so source_candidates
        # carry meaningful pages even before the merger runs (defensive).
        chapter_offset = chunk_local_offset + chunk.char_offset
        page = chapter.offset_to_page(chapter_offset)
        window = _three_sentence_window(chunk.text, chunk_local_offset)
        candidates.append(
            Candidate(
                surface_form=str(raw["surface_form"]),
                proposed_name=str(raw["proposed_name"]),
                entity_type=entity_type.name,
                context_sentence=str(raw.get("context_sentence", "")),
                context_window=window,
                char_offset=chunk_local_offset,  # merger rebases to chapter
                page=page,
                llm_confidence=_coerce_float(raw.get("llm_confidence"), default=0.5),
                extractor_version=EXTRACT_VERSION,
            )
        )

    bundle = _CandidateBundle(candidates=candidates)
    cache.set(f"extract.{entity_type.name}", cache_key, bundle)
    logger.info(
        f"extract chunk {chunk.index}/{entity_type.name}: "
        f"{len(candidates)} via {provider_used}"
    )
    return candidates


async def extract_candidates(
    chapter: Chapter,
    *,
    entity_type: EntityTypeSpec,
    llm_config: LLMConfig,
    cache: Cache,
) -> list[Candidate]:
    """Chunk chapter → fan out to LLM in parallel → merge candidates.

    See module docstring for env knobs. Falls back to a single chunk for short
    chapters (chunker returns ≤1 chunk under target).
    """
    target_tokens = _env_int("PUGMARK_CHUNK_TOKENS", 6000)
    overlap_chars = _env_int("PUGMARK_CHUNK_OVERLAP", 600)
    concurrency = _env_int("PUGMARK_EXTRACT_CONCURRENCY", 3)

    chunks = chunk_text(
        chapter.normalized_text,
        target_tokens=target_tokens,
        overlap_chars=overlap_chars,
    )
    if not chunks:
        return []
    logger.info(
        f"extract {entity_type.name}: {len(chunks)} chunk(s) "
        f"(target_tokens={target_tokens}, overlap={overlap_chars}, "
        f"concurrency={concurrency})"
    )

    sem = asyncio.Semaphore(concurrency)
    per_chunk = await asyncio.gather(
        *[
            _extract_one_chunk(
                ch,
                entity_type=entity_type,
                chapter=chapter,
                llm_config=llm_config,
                cache=cache,
                sem=sem,
            )
            for ch in chunks
        ]
    )
    merged = merge_candidates(per_chunk, chunk_offsets=[c.char_offset for c in chunks])

    # Recompute page from chapter-rebased char_offset (merger updated offsets).
    finalized: list[Candidate] = []
    for cand in merged:
        finalized.append(
            cand.model_copy(update={"page": chapter.offset_to_page(cand.char_offset)})
        )

    logger.info(
        f"extract {entity_type.name}: {len(finalized)} unique candidates "
        f"after merge across {len(chunks)} chunks"
    )
    return finalized


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
