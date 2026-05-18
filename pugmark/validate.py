"""Candidate → ConfirmedTaxon via Wikidata SPARQL.

Strategy: exact label → alias → fuzzy match (rapidfuzz) → unresolved.
Concurrency: asyncio.gather + Semaphore(10) to avoid Wikidata rate limits.
Caches by surface_form for 24h TTL (handled implicitly via hash key + manual sweep).
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import Any

import httpx
from jinja2 import Template
from pydantic import BaseModel
from rapidfuzz import fuzz

from pugmark.cache import Cache
from pugmark.entity_type import EntityTypeSpec
from pugmark.llm import LLMClient, LLMConfig
from pugmark.schemas import Candidate, Chapter, ConfirmedEntity

logger = logging.getLogger(__name__)

VALIDATE_VERSION = "v2"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "Pugmark/0.1 (https://github.com/Ansumanbhujabal/Pugmarks)"
FUZZY_THRESHOLD = 85  # 0-100 rapidfuzz scale
CONCURRENCY = 10


class _JudgeYesNo(BaseModel):
    yes: bool


_JUDGE_SYSTEM = (
    'You are a careful editor. Respond only with strict JSON {"yes": true} '
    'or {"yes": false}. No commentary.'
)


class _CachedResolution(BaseModel):
    qid: str | None
    canonical: str | None
    vernacular: str | None
    rank: str | None
    method: str | None
    fuzzy_score: float | None


async def _sparql_query(query_name: str, qclass: str) -> dict[str, Any]:
    """Issue a SPARQL query against Wikidata, return parsed JSON.

    `query_name` is interpolated into a name-search template. `qclass` is the
    Wikidata Q-identifier for the entity class (Q16521 taxa, Q5 humans, etc.)
    """
    sparql = f"""
    SELECT ?item ?itemLabel ?canonical ?rank ?alias WHERE {{
      VALUES ?searchTerm {{ "{query_name}"@en }}
      ?item rdfs:label ?searchTerm.
      ?item wdt:P31/wdt:P279* wd:{qclass}.
      OPTIONAL {{ ?item wdt:P225 ?canonical. }}
      OPTIONAL {{ ?item wdt:P105/rdfs:label ?rank. FILTER(LANG(?rank)="en") }}
      OPTIONAL {{ ?item skos:altLabel ?alias. FILTER(LANG(?alias)="en") }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5
    """
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(WIKIDATA_ENDPOINT, params={"query": sparql}, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


async def _resolve_one(name: str, qclass: str) -> _CachedResolution:
    data = await _sparql_query(name, qclass)
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return _CachedResolution(
            qid=None, canonical=None, vernacular=None, rank=None, method=None, fuzzy_score=None
        )

    # 1. Exact match — first binding without alias contribution
    first = bindings[0]
    method = "alias" if "alias" in first else "sparql_exact"
    return _CachedResolution(
        qid=_qid_from_uri(first["item"]["value"]),
        canonical=first.get("canonical", {}).get("value"),
        vernacular=first["itemLabel"]["value"],
        rank=first.get("rank", {}).get("value", "species"),
        method=method,
        fuzzy_score=None,
    )


async def validate_candidates(
    candidates: list[Candidate],
    *,
    entity_type: EntityTypeSpec,
    chapter: Chapter,
    cache: Cache,
) -> tuple[list[ConfirmedEntity], list[Candidate]]:
    """Resolve candidates; return (confirmed, unresolved).

    Tier 1 (entity_type.wikidata_qclass is set): SPARQL exact -> alias -> fuzzy.
    Tier 2 (qclass is None): in-book cross-reference. Counts case-insensitive
    word-boundary occurrences of each candidate's surface_form in
    chapter.normalized_text; promotes if count >= min_book_occurrences.
    """
    if entity_type.wikidata_qclass is None:
        return await _validate_in_book(
            candidates, entity_type=entity_type, chapter=chapter
        )

    # === Tier 1: Wikidata path (existing v1+T9 code) ===
    sem = asyncio.Semaphore(CONCURRENCY)
    qclass = entity_type.wikidata_qclass

    async def resolve_with_cache(name: str) -> _CachedResolution:
        async with sem:
            key = Cache.compute_hash(name.lower(), VALIDATE_VERSION, entity_type.name)
            hit = cache.get("validate", key, _CachedResolution)
            if hit is not None:
                return hit
            res = await _resolve_one(name, qclass)
            cache.set("validate", key, res)
            return res

    # Resolve unique names (avoid duplicate API calls)
    unique_names = list({c.proposed_name.lower() for c in candidates})
    resolutions = await asyncio.gather(*[resolve_with_cache(n) for n in unique_names])
    name_to_resolution: dict[str, _CachedResolution] = dict(
        zip(unique_names, resolutions, strict=True)
    )

    # Group source candidates by resolved QID; unmatched go to unresolved
    qid_to_candidates: dict[str, list[Candidate]] = defaultdict(list)
    qid_to_resolution: dict[str, _CachedResolution] = {}
    unresolved: list[Candidate] = []

    for cand in candidates:
        resolution = name_to_resolution[cand.proposed_name.lower()]
        if resolution.qid is None:
            # Try fuzzy fallback against the unique resolved names
            best_score = 0
            best_qid: str | None = None
            best_resolution: _CachedResolution | None = None
            for name, res in name_to_resolution.items():
                if res.qid is None:
                    continue
                score = fuzz.ratio(cand.proposed_name.lower(), name)
                if score > best_score:
                    best_score = score
                    best_qid = res.qid
                    best_resolution = res
            if best_qid and best_score >= FUZZY_THRESHOLD and best_resolution is not None:
                qid_to_candidates[best_qid].append(cand)
                qid_to_resolution[best_qid] = best_resolution.model_copy(
                    update={"method": "sparql_fuzzy", "fuzzy_score": best_score / 100}
                )
            else:
                unresolved.append(cand)
        else:
            qid_to_candidates[resolution.qid].append(cand)
            qid_to_resolution[resolution.qid] = resolution

    confirmed: list[ConfirmedEntity] = []
    for qid, cands in qid_to_candidates.items():
        res = qid_to_resolution[qid]
        confirmed.append(
            ConfirmedEntity(
                canonical_name=res.canonical or res.vernacular or "",
                vernacular=res.vernacular or "",
                entity_type=entity_type.name,
                wikidata_qid=qid,
                rank=res.rank or "species",
                attributes={},
                validation_method=res.method or "sparql_exact",  # type: ignore[arg-type]
                fuzzy_score=res.fuzzy_score,
                source_candidates=cands,
            )
        )

    logger.info(f"validate: {len(confirmed)} confirmed, {len(unresolved)} unresolved")
    return confirmed, unresolved


def _count_word_boundary_occurrences(text: str, surface_form: str) -> int:
    pattern = r"\b" + re.escape(surface_form) + r"\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


async def _judge_one(
    client: LLMClient, prompt_template: str, candidate_name: str, chapter_text: str
) -> bool:
    user_prompt = Template(prompt_template).render(
        candidate_name=candidate_name,
        chapter_text=chapter_text,
    )
    resp, _ = await client.complete_structured(
        system=_JUDGE_SYSTEM,
        user=user_prompt,
        schema=_JudgeYesNo,
        prompt_version="judge-v1",
    )
    return resp.yes


async def _judge_consensus(
    client: LLMClient,
    entity_type: EntityTypeSpec,
    candidate_name: str,
    chapter_text: str,
    n_calls: int = 3,
) -> int:
    """Return the number of 'yes' votes across n_calls judge calls."""
    results = await asyncio.gather(
        *[
            _judge_one(
                client, entity_type.judge_prompt_template, candidate_name, chapter_text
            )
            for _ in range(n_calls)
        ]
    )
    return sum(1 for r in results if r)


async def _validate_in_book(
    candidates: list[Candidate],
    *,
    entity_type: EntityTypeSpec,
    chapter: Chapter,
    judge_model: str = "gemini/gemini-2.5-pro",
) -> tuple[list[ConfirmedEntity], list[Candidate]]:
    """Tier 2: in-book crossref + judge consensus."""
    text = chapter.normalized_text
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    counts: dict[str, int] = {}

    for cand in candidates:
        key = cand.proposed_name.lower()
        grouped[key].append(cand)
        if key not in counts:
            counts[key] = _count_word_boundary_occurrences(text, cand.surface_form)

    judge_config = LLMConfig(providers=[judge_model], max_retries=1, timeout_s=60.0)
    judge_client = LLMClient(judge_config)

    confirmed: list[ConfirmedEntity] = []
    unresolved: list[Candidate] = []

    for key, cands in grouped.items():
        count = counts[key]
        if count < entity_type.min_book_occurrences:
            unresolved.extend(cands)
            continue

        votes = await _judge_consensus(
            judge_client,
            entity_type,
            candidate_name=cands[0].proposed_name,
            chapter_text=text,
        )

        if votes >= entity_type.min_judge_votes:
            confirmed.append(
                ConfirmedEntity(
                    canonical_name=cands[0].proposed_name,
                    vernacular=cands[0].proposed_name,
                    entity_type=entity_type.name,
                    wikidata_qid=None,
                    rank=entity_type.name.rstrip("s"),
                    attributes={},
                    validation_method="judge_consensus",
                    crossref_count=count,
                    judge_votes=votes,
                    source_candidates=cands,
                )
            )
        else:
            unresolved.extend(cands)

    logger.info(
        f"validate (in_book): {len(confirmed)} confirmed, {len(unresolved)} unresolved"
    )
    return confirmed, unresolved
