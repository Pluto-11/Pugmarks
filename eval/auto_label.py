"""Automated ground-truth generation via LLM-as-judge.

Pipeline:
  1. Chapter text → judge LLM N times (independent calls)
  2. Majority vote: keep only entries seen in >= min_votes calls
  3. Wikidata round-trip via pugmark.validate.validate_candidates — drop anything
     that doesn't resolve to a real Wikidata taxon QID
  4. Emit list[dict] in the schema eval/metrics.py expects.

The judge model is intentionally different from the production extraction model
(Gemini 2.5 Pro vs. Gemini 2.0 Flash) so the evaluation isn't measuring a model's
agreement with its own output.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from pugmark.cache import Cache
from pugmark.ingest import load_chapter
from pugmark.llm import LLMClient, LLMConfig
from pugmark.prompt_registry import PromptRegistry
from pugmark.schemas import Candidate
from pugmark.validate import validate_candidates

logger = logging.getLogger(__name__)

AUTOLABEL_VERSION = "v1"
JUDGE_PROMPT_NAME = "judge_taxa"
DEFAULT_JUDGE_MODEL = "gemini/gemini-2.5-pro"
DEFAULT_N_CALLS = 3
DEFAULT_MIN_VOTES = 2
JUDGE_SYSTEM_PROMPT = (
    "You output strictly valid JSON. Do not add commentary. "
    "Follow the user's schema exactly. Err on the side of false negatives."
)


class _JudgeCandidate(BaseModel):
    surface_form: str
    proposed_name: str
    kingdom_hint: str
    context_sentence: str = ""
    llm_confidence: float = 0.0


class _JudgeResponse(BaseModel):
    candidates: list[_JudgeCandidate] = Field(default_factory=list)


def _vote_key(c: _JudgeCandidate) -> tuple[str, str, str]:
    """Vote key collapses each call's output to a comparable triple."""
    return (
        c.surface_form.lower().strip(),
        c.proposed_name.lower().strip(),
        c.kingdom_hint.lower().strip(),
    )


async def _one_judge_call(
    client: LLMClient,
    system: str,
    user: str,
    prompt_version: str,
) -> list[_JudgeCandidate]:
    resp, _provider = await client.complete_structured(
        system=system,
        user=user,
        schema=_JudgeResponse,
        prompt_version=prompt_version,
    )
    return resp.candidates


async def auto_label_chapter(
    pdf: Path,
    chapter_number: int,
    *,
    cache: Cache,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    n_calls: int = DEFAULT_N_CALLS,
    min_votes: int = DEFAULT_MIN_VOTES,
    prompt_dir: Path = Path("prompts"),
) -> list[dict]:
    """Generate ground-truth labels for one chapter.

    Returns a list of {surface_form, expected_wikidata_qid, expected_kingdom, page}
    suitable for writing to eval/ground_truth/<name>.json.
    """
    chapter = load_chapter(pdf, chapter_number)

    registry = PromptRegistry(in_repo_dir=prompt_dir)
    judge_prompt = registry.get(JUDGE_PROMPT_NAME)
    user_prompt = judge_prompt.render(chapter_text=chapter.normalized_text)

    judge_config = LLMConfig(providers=[judge_model], max_retries=1, timeout_s=120.0)
    client = LLMClient(judge_config)

    all_calls = await asyncio.gather(
        *[
            _one_judge_call(client, JUDGE_SYSTEM_PROMPT, user_prompt, judge_prompt.version)
            for _ in range(n_calls)
        ]
    )

    # Aggregate votes and keep an example candidate per surviving triple so we
    # can carry kingdom_hint + a sample context through to validation.
    counter: Counter[tuple[str, str, str]] = Counter()
    exemplars: dict[tuple[str, str, str], _JudgeCandidate] = {}
    for call_cands in all_calls:
        seen_this_call: set[tuple[str, str, str]] = set()
        for jc in call_cands:
            key = _vote_key(jc)
            if key in seen_this_call:
                continue
            seen_this_call.add(key)
            counter[key] += 1
            exemplars.setdefault(key, jc)

    survivors = [key for key, votes in counter.items() if votes >= min_votes]
    logger.info(
        f"autolabel: {n_calls} judge calls, "
        f"{sum(counter.values())} raw mentions, "
        f"{len(survivors)} survived {min_votes}/{n_calls} vote"
    )

    if not survivors:
        return []

    # Wrap each survivor in a Candidate so we can reuse validate_candidates.
    candidates: list[Candidate] = []
    text_lower = chapter.normalized_text.lower()
    for key in survivors:
        surface, proposed, kingdom = key
        exemplar = exemplars[key]
        # Best-effort page lookup: find the first occurrence of the surface form.
        idx = text_lower.find(surface)
        char_offset = idx if idx >= 0 else 0
        page = chapter.offset_to_page(char_offset) if idx >= 0 else chapter.page_start
        kingdom_norm = kingdom if kingdom in {"animalia", "plantae", "fungi"} else "unknown"
        candidates.append(
            Candidate(
                surface_form=surface,
                proposed_name=proposed,
                kingdom_hint=kingdom_norm,  # type: ignore[arg-type]
                context_sentence=exemplar.context_sentence or "",
                context_window=exemplar.context_sentence or "",
                char_offset=char_offset,
                page=page,
                llm_confidence=exemplar.llm_confidence or 0.85,
                extractor_version=f"judge_{AUTOLABEL_VERSION}",
            )
        )

    confirmed, _unresolved = await validate_candidates(candidates, cache=cache)

    ground_truth: list[dict] = []
    for taxon in confirmed:
        for src in taxon.source_candidates:
            ground_truth.append(
                {
                    "surface_form": src.surface_form,
                    "expected_wikidata_qid": taxon.wikidata_qid,
                    "expected_kingdom": src.kingdom_hint,
                    "page": src.page,
                }
            )

    logger.info(
        f"autolabel: {len(ground_truth)} ground-truth entries after Wikidata roundtrip"
    )
    return ground_truth
