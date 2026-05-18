"""ConfirmedTaxon → TaxonCard.

Per taxon: fetch Wikipedia summary + Wikimedia Commons image (with license),
build Sightings from source_candidates, return TaxonCard. Async fan-out per taxon.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from pugmark.cache import Cache
from pugmark.schemas import (
    Chapter,
    ConfirmedTaxon,
    ImageRef,
    Sighting,
    TaxonCard,
)

logger = logging.getLogger(__name__)

ENRICH_VERSION = "v1"
USER_AGENT = "Pugmark/0.1 (https://github.com/Ansumanbhujabal/Pugmarks)"
CONCURRENCY = 10
WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


async def _fetch_wikidata_entity(qid: str) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(WIKIDATA_ENTITY.format(qid=qid), headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _fetch_wikipedia(qid: str) -> dict[str, Any]:
    """Fetch Wikipedia summary by following Wikidata's enwiki sitelink."""
    entity = await _fetch_wikidata_entity(qid)
    sitelinks = entity["entities"][qid]["sitelinks"]
    if "enwiki" not in sitelinks:
        raise ValueError(f"no enwiki sitelink for {qid}")
    title = sitelinks["enwiki"]["title"].replace(" ", "_")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(WIKIPEDIA_REST.format(title=title), headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _fetch_commons_image(qid: str) -> dict[str, Any]:
    """Fetch the primary Commons image for a Wikidata QID via P18."""
    entity = await _fetch_wikidata_entity(qid)
    claims = entity["entities"][qid]["claims"]
    image_claims = claims.get("P18", [])
    if not image_claims:
        raise ValueError(f"no P18 image for {qid}")
    filename = image_claims[0]["mainsnak"]["datavalue"]["value"]
    title = f"File:{filename}"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "titles": title,
    }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(COMMONS_API, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_image(commons_response: dict[str, Any]) -> ImageRef | None:
    pages = commons_response.get("query", {}).get("pages", {})
    for _, page in pages.items():
        infos = page.get("imageinfo")
        if not infos:
            continue
        info = infos[0]
        meta = info.get("extmetadata", {})
        license_str = meta.get("LicenseShortName", {}).get("value")
        attribution = meta.get("Artist", {}).get("value", "Unknown")
        if not license_str:
            continue
        # Strip simple HTML in attribution
        import re

        attribution = re.sub(r"<[^>]+>", "", str(attribution)).strip()
        return ImageRef(
            url=info["url"],
            license=license_str,
            attribution=attribution,
            source="wikimedia",
            width=info.get("width"),
            height=info.get("height"),
        )
    return None


def _build_sightings(taxon: ConfirmedTaxon, chapter: Chapter) -> list[Sighting]:
    return [
        Sighting(page=cand.page, paragraph=cand.context_window)
        for cand in taxon.source_candidates
    ]


async def _enrich_one(
    taxon: ConfirmedTaxon, chapter: Chapter, cache: Cache, sem: asyncio.Semaphore
) -> TaxonCard | None:
    cache_key = Cache.compute_hash(taxon.wikidata_qid, ENRICH_VERSION)
    hit = cache.get("enrich", cache_key, TaxonCard)
    if hit is not None:
        # Refresh sightings from current chapter
        return hit.model_copy(update={"sightings": _build_sightings(taxon, chapter)})

    async with sem:
        try:
            wp_resp, commons_resp = await asyncio.gather(
                _fetch_wikipedia(taxon.wikidata_qid),
                _fetch_commons_image(taxon.wikidata_qid),
            )
        except Exception as e:
            logger.warning(f"enrich failed for {taxon.wikidata_qid}: {e!r}")
            return None

    image = _parse_image(commons_resp)
    if image is None:
        logger.warning(f"no licensed image for {taxon.wikidata_qid}")
        return None

    card = TaxonCard(
        taxon=taxon,
        wikipedia_url=wp_resp["content_urls"]["desktop"]["page"],
        wikipedia_summary=wp_resp.get("extract", ""),
        primary_image=image,
        alt_images=[],
        sightings=_build_sightings(taxon, chapter),
        enrich_version=ENRICH_VERSION,
    )
    cache.set("enrich", cache_key, card)
    return card


async def enrich_taxa(
    taxa: list[ConfirmedTaxon], *, chapter: Chapter, cache: Cache
) -> list[TaxonCard]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*[_enrich_one(t, chapter, cache, sem) for t in taxa])
    cards = [c for c in results if c is not None]
    logger.info(f"enrich: {len(cards)}/{len(taxa)} taxa got cards")
    return cards
