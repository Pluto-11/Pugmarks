"""Eval runner — runs the full pipeline against ground truth, writes EvalRun JSON.

DeepEval is intentionally not imported here yet; this is a minimal in-house runner.
DeepEval comes online in a v1.5 task once we have multiple runs to compare.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from pugmark.cache import Cache
from pugmark.enrich import enrich_taxa
from pugmark.extract import extract_candidates
from pugmark.ingest import load_chapter
from pugmark.llm import LLMConfig
from pugmark.schemas import EvalRun
from pugmark.validate import validate_candidates

from .metrics import compute_extraction_metrics, compute_validation_metrics

logger = logging.getLogger(__name__)
PUGMARK_VERSION = "0.1.0"


async def run_eval(
    *,
    pdf: Path,
    chapter_number: int,
    ground_truth_path: Path,
    runs_dir: Path,
    prompts_dir: Path = Path("prompts"),
) -> EvalRun:
    truth = json.loads(ground_truth_path.read_text())
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()

    chapter = load_chapter(pdf, chapter_number)
    t0 = time.perf_counter()

    candidates = await extract_candidates(
        chapter, llm_config=llm_config, prompt_dir=prompts_dir, cache=cache
    )
    confirmed, unresolved = await validate_candidates(candidates, cache=cache)
    _cards = await enrich_taxa(confirmed, chapter=chapter, cache=cache)

    latency_ms = int((time.perf_counter() - t0) * 1000)

    extraction_m = compute_extraction_metrics(
        candidates, truth, chapter_text=chapter.normalized_text
    )
    validation_m = compute_validation_metrics(confirmed, unresolved, truth)

    run = EvalRun(
        chapter_id=ground_truth_path.stem,
        extraction=extraction_m,
        validation=validation_m,
        cost_usd=0.0,
        latency_ms=latency_ms,
        pugmark_version=PUGMARK_VERSION,
        llm_provider=llm_config.providers[0],
        prompt_version="v1",
        timestamp=datetime.now(),
    )

    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{run.timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(run.model_dump_json(indent=2))
    logger.info(f"wrote eval run to {out_path}")
    return run
