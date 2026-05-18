"""Eval runner — runs the full pipeline against ground truth, writes EvalRun JSON.

v2: ground_truth_path can be either a single JSON file (v1 compat) or a
directory containing one file per entity type, named '{book_id}__{type}.json'.
Per-type metrics are aggregated into ExtractionMetrics.by_type.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
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
from pugmark.schemas import Chapter, EvalRun
from pugmark.validate import validate_candidates

from .metrics import compute_extraction_metrics, compute_validation_metrics

logger = logging.getLogger(__name__)
PUGMARK_VERSION = "0.2.0"


def _load_ground_truth(path: Path) -> list[dict]:
    """Accept either a single JSON file or a directory of '*__<type>.json' files."""
    if path.is_dir():
        merged: list[dict] = []
        for f in sorted(path.glob("*__*.json")):
            stem = f.stem
            type_name = stem.rsplit("__", 1)[-1]
            entries = json.loads(f.read_text())
            for e in entries:
                e.setdefault("entity_type", type_name)
                merged.append(e)
        return merged
    entries = json.loads(path.read_text())
    for e in entries:
        e.setdefault("entity_type", "taxa")
    return entries


async def _extract_one_type(
    chapter: Chapter,
    entity_type: EntityTypeSpec,
    *,
    llm_config: LLMConfig,
    cache: Cache,
):
    """Per-type pipeline: extract -> validate -> enrich. Mirrors pugmark.api.

    Inlined here (rather than imported from pugmark.api) so that test patches
    targeting ``eval.runner.extract_candidates`` / ``validate_candidates`` /
    ``enrich_confirmed`` take effect. Returns the raw stage outputs so the
    runner can aggregate metrics independent of enrichment success.
    """
    candidates = await extract_candidates(
        chapter, entity_type=entity_type, llm_config=llm_config, cache=cache
    )
    confirmed, unresolved = await validate_candidates(
        candidates, entity_type=entity_type, chapter=chapter, cache=cache
    )
    cards = await enrich_confirmed(
        confirmed, chapter=chapter, cache=cache, llm_config=llm_config
    )
    return candidates, confirmed, unresolved, cards


async def run_eval(
    *,
    pdf: Path,
    chapter_number: int,
    ground_truth_path: Path,
    runs_dir: Path,
    types: list[str] | None = None,
) -> EvalRun:
    truth = _load_ground_truth(ground_truth_path)
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()

    book_schema = await analyze_book(pdf, cache=cache)
    realized = realize_schema(book_schema, force_types=types)

    chapter = load_chapter(pdf, chapter_number)
    t0 = time.perf_counter()

    results = await asyncio.gather(
        *[
            _extract_one_type(chapter, t, llm_config=llm_config, cache=cache)
            for t in realized
        ]
    )

    all_candidates: list = []
    all_confirmed: list = []
    all_unresolved: list = []
    for _spec, (candidates, confirmed, unresolved, _cards) in zip(
        realized, results, strict=True
    ):
        all_candidates.extend(candidates)
        all_confirmed.extend(confirmed)
        all_unresolved.extend(unresolved)

    latency_ms = int((time.perf_counter() - t0) * 1000)

    extraction_m = compute_extraction_metrics(
        all_candidates, truth, chapter_text=chapter.normalized_text
    )
    validation_m = compute_validation_metrics(all_confirmed, all_unresolved, truth)

    run = EvalRun(
        chapter_id=ground_truth_path.name,
        extraction=extraction_m,
        validation=validation_m,
        cost_usd=0.0,
        latency_ms=latency_ms,
        pugmark_version=PUGMARK_VERSION,
        llm_provider=llm_config.providers[0],
        prompt_version="v2",
        timestamp=datetime.now(),
    )

    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{run.timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(run.model_dump_json(indent=2))
    logger.info(f"wrote eval run to {out_path}")
    return run
