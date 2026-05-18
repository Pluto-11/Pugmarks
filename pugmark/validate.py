"""Candidate → ConfirmedTaxon via Wikidata SPARQL.

Strategy: exact label → alias → fuzzy match (rapidfuzz) → unresolved.
Concurrency: asyncio.gather + Semaphore(10) to avoid Wikidata rate limits.
Caches by surface_form for 24h TTL (handled implicitly via hash key + manual sweep).
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

import httpx
from pydantic import BaseModel
from rapidfuzz import fuzz

from pugmark.cache import Cache
from pugmark.schemas import Candidate, ConfirmedTaxon

logger = logging.getLogger(__name__)

VALIDATE_VERSION = "v1"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "Pugmark/0.1 (https://github.com/Ansumanbhujabal/Pugmarks)"
FUZZY_THRESHOLD = 85  # 0-100 rapidfuzz scale
CONCURRENCY = 10


class _CachedResolution(BaseModel):
    qid: str | None
    canonical: str | None
    vernacular: str | None
    rank: str | None
    method: str | None
    fuzzy_score: float | None


async def _sparql_query(query_name: str) -> dict[str, Any]:
    """Issue a SPARQL query against Wikidata, return parsed JSON.

    `query_name` is interpolated into a name-search template. This is a thin
    layer separated from validate_candidates for unit-test mocking.
    """
    sparql = f"""
    SELECT ?item ?itemLabel ?canonical ?rank ?alias WHERE {{
      VALUES ?searchTerm {{ "{query_name}"@en }}
      ?item rdfs:label ?searchTerm.
      ?item wdt:P31/wdt:P279* wd:Q16521.
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


async def _resolve_one(name: str) -> _CachedResolution:
    data = await _sparql_query(name)
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
    candidates: list[Candidate], *, cache: Cache
) -> tuple[list[ConfirmedTaxon], list[Candidate]]:
    """Resolve candidates to ConfirmedTaxa; return (confirmed, unresolved).

    Many-to-one: candidates resolving to the same QID merge into one ConfirmedTaxon.
    """
    sem = asyncio.Semaphore(CONCURRENCY)

    async def resolve_with_cache(name: str) -> _CachedResolution:
        async with sem:
            key = Cache.compute_hash(name.lower(), VALIDATE_VERSION)
            hit = cache.get("validate", key, _CachedResolution)
            if hit is not None:
                return hit
            res = await _resolve_one(name)
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

    confirmed: list[ConfirmedTaxon] = []
    for qid, cands in qid_to_candidates.items():
        res = qid_to_resolution[qid]
        confirmed.append(
            ConfirmedTaxon(
                canonical_name=res.canonical or res.vernacular or "",
                vernacular=res.vernacular or "",
                wikidata_qid=qid,
                rank=res.rank or "species",
                lineage={},  # filled in enrich; minimal here
                validation_method=res.method or "sparql_exact",  # type: ignore[arg-type]
                fuzzy_score=res.fuzzy_score,
                source_candidates=cands,
            )
        )

    logger.info(f"validate: {len(confirmed)} confirmed, {len(unresolved)} unresolved")
    return confirmed, unresolved
