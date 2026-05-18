"""Library convenience entrypoint — extract_gallery(pdf, chapter_number).

Wires analyze → realize-schema → per-type extract/validate/enrich and assembles
a Gallery keyed by entity type.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from pugmark.analyzer import analyze_book
from pugmark.cache import Cache
from pugmark.enrich import enrich_confirmed
from pugmark.entity_type import EntityTypeSpec
from pugmark.extract import extract_candidates
from pugmark.ingest import load_chapter
from pugmark.llm import LLMConfig
from pugmark.schema_realizer import realize_schema
from pugmark.schemas import Gallery
from pugmark.validate import validate_candidates

PUGMARK_VERSION = "0.2.0"


async def _extract_one_type(
    chapter,
    entity_type: EntityTypeSpec,
    *,
    llm_config: LLMConfig,
    cache: Cache,
):
    candidates = await extract_candidates(
        chapter, entity_type=entity_type, llm_config=llm_config, cache=cache
    )
    confirmed, unresolved = await validate_candidates(
        candidates, entity_type=entity_type, chapter=chapter, cache=cache
    )
    cards = await enrich_confirmed(
        confirmed, chapter=chapter, cache=cache, llm_config=llm_config
    )
    return cards, unresolved


async def extract_gallery(
    pdf: Path | str,
    chapter_number: int,
    *,
    types: list[str] | None = None,
    overrides: dict[str, EntityTypeSpec] | None = None,
    cache: Cache | None = None,
    llm_config: LLMConfig | None = None,
) -> Gallery:
    """Run analyze → realize → per-type extract/validate/enrich. Return a Gallery.

    If `types` is given, only those types are extracted (force-includes
    registered types not auto-proposed). If `overrides` is given, those specs
    override registered/proposed for matching type names.
    """
    pdf_path = Path(pdf)
    cache = cache or Cache.from_env()
    llm_config = llm_config or LLMConfig.from_env()

    book_schema = await analyze_book(pdf_path, cache=cache)
    realized = realize_schema(book_schema, overrides=overrides, force_types=types)

    chapter = load_chapter(pdf_path, chapter_number)

    results = await asyncio.gather(
        *[
            _extract_one_type(chapter, t, llm_config=llm_config, cache=cache)
            for t in realized
        ]
    )

    cards_by_type: dict[str, list] = {}
    all_unresolved: list = []
    for spec, (cards, unresolved) in zip(realized, results, strict=True):
        cards_by_type[spec.name] = cards
        all_unresolved.extend(unresolved)

    return Gallery(
        chapter=chapter,
        cards_by_type=cards_by_type,
        unresolved=all_unresolved,
        generated_at=datetime.now(),
        pugmark_version=PUGMARK_VERSION,
        book_schema=book_schema,
        eval_metrics=None,
    )
