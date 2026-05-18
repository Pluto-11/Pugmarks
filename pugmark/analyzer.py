"""Book analyzer — proposes per-book entity types via LLM-as-classifier.

Reads the PDF's TOC and short samples from a few chapters, asks the analyzer
LLM to propose 2–6 entity types valuable to extract from this specific book.
Cached by (pdf hash, analyzer_version) so a second call on the same PDF is free.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pugmark.cache import Cache
from pugmark.entity_type import BookSchema, EntityTypeSpec
from pugmark.ingest import list_chapters, load_chapter
from pugmark.llm import LLMClient, LLMConfig
from pugmark.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)

ANALYZER_VERSION = "v1"
ANALYZER_PROMPT_NAME = "book_analyzer"
DEFAULT_ANALYZER_MODEL = "gemini/gemini-2.5-pro"
SAMPLE_PER_CHAPTER_CHARS = 500
FIRST_CHAPTER_HEAD_CHARS = 2000
MAX_SAMPLE_CHAPTERS = 3
ANALYZER_SYSTEM_PROMPT = (
    "You output strictly valid JSON. Do not add commentary. "
    "Follow the user's schema exactly."
)


class _AnalyzerType(BaseModel):
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    wikidata_qclass: str | None = None


class _AnalyzerResponse(BaseModel):
    proposed_types: list[_AnalyzerType] = Field(default_factory=list)


def _build_samples(pdf: Path, chapters: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (toc_text, samples_text) for the analyzer prompt."""
    toc_lines = [
        f"{c['number']}. {c['title']} (pp.{c['page_start']}-{c['page_end']})"
        for c in chapters
    ]
    toc_text = "\n".join(toc_lines)

    samples: list[str] = []
    targets = chapters[: min(len(chapters), MAX_SAMPLE_CHAPTERS)]
    for i, ch_info in enumerate(targets):
        ch = load_chapter(pdf, ch_info["number"])
        text = ch.normalized_text
        if i == 0:
            sample = text[:FIRST_CHAPTER_HEAD_CHARS]
        else:
            mid = len(text) // 2
            sample = text[mid : mid + SAMPLE_PER_CHAPTER_CHARS]
        samples.append(
            f"--- Chapter {ch_info['number']}: {ch_info['title']} ---\n{sample}"
        )
    return toc_text, "\n\n".join(samples)


def _is_valid_proposal(t: _AnalyzerType) -> bool:
    """Filter out too-granular or sentence-shaped proposals."""
    name = t.name.strip()
    if not name:
        return False
    if " " in name and t.wikidata_qclass is None:
        return False
    return len(name) <= 40


def _to_spec(t: _AnalyzerType) -> EntityTypeSpec:
    """Wrap an analyzer proposal as an EntityTypeSpec with generic prompt templates.

    Built-in types (taxa, people, places) get their proper templates via the
    registry override in schema_realizer. Analyzer-only types fall back to a
    generic template inline in the spec.
    """
    generic_extract = (
        "Extract every "
        + t.name
        + " mentioned in the chapter below. "
        + "For each mention, output {surface_form, proposed_name, entity_type='"
        + t.name
        + "', context_sentence, char_offset, llm_confidence}. "
        + "Examples in this book include: "
        + ", ".join(t.examples)
        + '. Output {"candidates": [...]} only.\n\n'
        + "---\n{{ chapter_text }}\n---"
    )
    generic_judge = (
        "You are a careful judge. List ONLY the "
        + t.name
        + " you are highly confident appear in the chapter. "
        + 'Output {"candidates": [{surface_form, proposed_name, entity_type=\''
        + t.name
        + "', context_sentence, llm_confidence}]}. Confidence ≥ 0.85.\n\n"
        + "---\n{{ chapter_text }}\n---"
    )
    return EntityTypeSpec(
        name=t.name,
        description=t.description,
        wikidata_qclass=t.wikidata_qclass,
        extraction_prompt_template=generic_extract,
        judge_prompt_template=generic_judge,
        examples=t.examples,
    )


async def analyze_book(
    pdf: Path,
    *,
    cache: Cache,
    analyzer_model: str = DEFAULT_ANALYZER_MODEL,
    prompt_dir: Path = Path("prompts"),
) -> BookSchema:
    """Propose entity types for one book.

    Returns a BookSchema cached by (pdf_hash, analyzer_version).
    """
    chapters = list_chapters(pdf)
    if not chapters:
        raise ValueError(f"no chapters detected in {pdf}; PDF has no outline")

    book_id = pdf.stem
    cache_key = Cache.compute_hash(book_id, ANALYZER_VERSION)
    hit = cache.get("analyze", cache_key, BookSchema)
    if hit is not None:
        logger.info(f"analyzer cache hit for {book_id}")
        return hit

    toc_text, samples_text = _build_samples(pdf, chapters)

    registry = PromptRegistry(in_repo_dir=prompt_dir)
    prompt = registry.get(ANALYZER_PROMPT_NAME)
    user_prompt = prompt.render(toc=toc_text, samples=samples_text)

    llm_config = LLMConfig(providers=[analyzer_model], max_retries=1, timeout_s=180.0)
    client = LLMClient(llm_config)
    resp, _provider = await client.complete_structured(
        system=ANALYZER_SYSTEM_PROMPT,
        user=user_prompt,
        schema=_AnalyzerResponse,
        prompt_version=prompt.version,
    )

    proposed_specs = [
        _to_spec(t) for t in resp.proposed_types if _is_valid_proposal(t)
    ]
    schema = BookSchema(
        book_id=book_id,
        proposed_types=proposed_specs,
        analyzer_version=ANALYZER_VERSION,
        analyzed_at=datetime.now(),
    )
    cache.set("analyze", cache_key, schema)
    logger.info(
        f"analyzer proposed {len(proposed_specs)} types for {book_id}: "
        + ", ".join(s.name for s in proposed_specs)
    )
    return schema
