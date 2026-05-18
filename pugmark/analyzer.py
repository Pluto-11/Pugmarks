"""Book analyzer — two-pass: classify book type, then propose entity types.

Pass 1 (book-type classifier): samples preface/intro text if found via the
ChapterInfo.kind tag, else first content chapter's head. LLM returns a BookType
{genre, period, setting, themes, target_reader, summary}.

Pass 2 (entity-type proposer): receives the book type plus TOC and content
samples, proposes 2-6 EntityTypeSpec instances biased on the book type.

Cached by (pdf hash, analyzer_version) — bump ANALYZER_VERSION to invalidate.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pugmark.cache import Cache
from pugmark.entity_type import BookSchema, BookType, EntityTypeSpec
from pugmark.ingest import list_chapters, load_chapter
from pugmark.llm import LLMClient, LLMConfig
from pugmark.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)

ANALYZER_VERSION = "v2"  # bumped: now produces BookSchema.book_type
ANALYZER_PROMPT_NAME = "book_analyzer"
BOOK_TYPE_PROMPT_NAME = "book_type_classifier"
SAMPLE_PER_CHAPTER_CHARS = 500
FIRST_CHAPTER_HEAD_CHARS = 2000
MAX_SAMPLE_CHAPTERS = 3
BOOK_TYPE_SAMPLE_CHARS = 3000
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
    """Return (toc_text, samples_text) for the entity-type proposer.

    TOC includes kind labels so the LLM understands which entries are content
    vs front/back-matter. Samples are taken only from content chapters.
    """
    toc_lines = [
        f"{c['number']}. [{c.get('kind', 'content')}] {c['title']} "
        f"(pp.{c['page_start']}-{c['page_end']})"
        for c in chapters
    ]
    toc_text = "\n".join(toc_lines)

    content_chapters = [c for c in chapters if c.get("kind", "content") == "content"]
    if not content_chapters:
        content_chapters = chapters  # degrade gracefully

    samples: list[str] = []
    targets = content_chapters[: min(len(content_chapters), MAX_SAMPLE_CHAPTERS)]
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


def _build_book_type_sample(pdf: Path, chapters: list[dict[str, Any]]) -> str:
    """Build the text sample the book-type classifier sees.

    Prefers preface/intro chapters (kind=front, title matches preface-like
    patterns); falls back to first content chapter's head.
    """
    preface_keywords = ("preface", "foreword", "introduction", "prologue")
    preferred = [
        c for c in chapters
        if c.get("kind") == "front"
        and any(k in c["title"].lower() for k in preface_keywords)
    ]
    if preferred:
        target = preferred[0]
        ch = load_chapter(pdf, target["number"])
        snippet = ch.normalized_text[:BOOK_TYPE_SAMPLE_CHARS]
        return (
            f"--- {target['title']} (preface-like, "
            f"pp.{target['page_start']}-{target['page_end']}) ---\n{snippet}"
        )
    # Fallback: first content chapter head
    content = [c for c in chapters if c.get("kind", "content") == "content"]
    if not content:
        content = chapters
    target = content[0]
    ch = load_chapter(pdf, target["number"])
    snippet = ch.normalized_text[:BOOK_TYPE_SAMPLE_CHARS]
    return (
        f"--- First content chapter: {target['title']} "
        f"(pp.{target['page_start']}-{target['page_end']}) ---\n{snippet}"
    )


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


async def _classify_book_type(
    pdf: Path,
    chapters: list[dict[str, Any]],
    *,
    client: LLMClient,
    registry: PromptRegistry,
) -> BookType | None:
    """LLM-classify the book's genre/period/setting from preface or first chapter.

    Returns None if the LLM call fails — calling code should treat book_type
    as optional context, not a hard requirement.
    """
    sample = _build_book_type_sample(pdf, chapters)
    try:
        prompt = registry.get(BOOK_TYPE_PROMPT_NAME)
        user_prompt = prompt.render(sample=sample)
        prompt_version = prompt.version
    except FileNotFoundError:
        # Fallback inline prompt if the .j2 file isn't there yet.
        user_prompt = (
            "Classify this book from the sample below. Return JSON exactly "
            "matching the schema: "
            '{"genre": <string>, "period": <string>, "setting": <string>, '
            '"themes": [<string>, ...], "target_reader": <string>, '
            '"summary": <string>}. Be concrete and specific (e.g., not just '
            '"fiction" — say "hunting memoir" or "Bildungsroman"). Themes '
            'should be 3-7 high-level recurring topics.\n\n'
            + sample
        )
        prompt_version = "inline-v1"

    try:
        resp, _ = await client.complete_structured(
            system=ANALYZER_SYSTEM_PROMPT,
            user=user_prompt,
            schema=BookType,
            prompt_version=prompt_version,
        )
        logger.info(
            f"book type: {resp.genre} | {resp.period} | {resp.setting} "
            f"({len(resp.themes)} themes)"
        )
        return resp
    except Exception as e:
        logger.warning(f"book-type classification failed: {e!r}; continuing without it")
        return None


async def analyze_book(
    pdf: Path,
    *,
    cache: Cache,
    llm_config: LLMConfig | None = None,
    prompt_dir: Path = Path("prompts"),
) -> BookSchema:
    """Two-pass analyze: classify book type, then propose entity types.

    Returns a BookSchema cached by (pdf_hash, analyzer_version).
    `llm_config` defaults to LLMConfig.from_env("analyzer") so PUGMARK_ANALYZER_MODEL
    / PUGMARK_ANALYZER_PROVIDERS in .env drive model selection with fallback.
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

    if llm_config is None:
        llm_config = LLMConfig.from_env(role="analyzer")
        llm_config.timeout_s = 180.0
    client = LLMClient(llm_config)
    registry = PromptRegistry(in_repo_dir=prompt_dir)

    # Pass 1: book-type classification (preface/intro → BookType)
    book_type = await _classify_book_type(pdf, chapters, client=client, registry=registry)

    # Pass 2: entity-type proposal — receives book_type as biasing context
    toc_text, samples_text = _build_samples(pdf, chapters)
    prompt = registry.get(ANALYZER_PROMPT_NAME)
    book_type_blurb = (
        f"Book type: {book_type.genre} ({book_type.period}, {book_type.setting}). "
        f"Themes: {', '.join(book_type.themes)}. Summary: {book_type.summary}\n\n"
        if book_type is not None
        else "Book type: unknown.\n\n"
    )
    user_prompt = book_type_blurb + prompt.render(toc=toc_text, samples=samples_text)
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
        book_type=book_type,
    )
    cache.set("analyze", cache_key, schema)
    logger.info(
        f"analyzer proposed {len(proposed_specs)} types for {book_id}: "
        + ", ".join(s.name for s in proposed_specs)
    )
    return schema
