"""ConfirmedEntity → EntityCard with tiered enrichment.

Tier 1 (entity.wikidata_qid is not None):
  - Existing Wikipedia + Commons flow (v1 logic)
  - summary_source = "wikipedia"

Tier 2 (entity.wikidata_qid is None):
  - LLM-summarize from concatenated context_windows of source_candidates
  - summary_source = "llm_in_book"
  - primary_image = None, wikipedia_url = None

Image fallback (applied after Tier 1/2):
  - If primary_image is None AND PUGMARK_AI_IMAGES=1, synthesize an image via
    Azure gpt-image-1.5 (image_gen.generate_image). The result is cached on
    disk and embedded as a file:// ImageRef tagged source="ai_generated".
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Template
from pydantic import BaseModel

from pugmark.cache import Cache
from pugmark.image_gen import build_prompt as build_image_prompt
from pugmark.image_gen import generate_image
from pugmark.llm import LLMClient, LLMConfig
from pugmark.schemas import (
    Chapter,
    ConfirmedEntity,
    EntityCard,
    ImageRef,
    Sighting,
)

logger = logging.getLogger(__name__)

ENRICH_VERSION = "v2"
USER_AGENT = "Pugmark/0.2 (https://github.com/Ansumanbhujabal/Pugmarks)"
CONCURRENCY = 10
WIKIPEDIA_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SUMMARY_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarize_in_book.v1.j2"
MAX_EXCERPT_CHARS = 2000


class _SummaryResp(BaseModel):
    text: str


async def _fetch_wikidata_entity(qid: str) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(WIKIDATA_ENTITY.format(qid=qid), headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _fetch_wikipedia(qid: str) -> dict[str, Any]:
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
    entity = await _fetch_wikidata_entity(qid)
    claims = entity["entities"][qid]["claims"]
    image_claims = claims.get("P18", [])
    if not image_claims:
        raise ValueError(f"no P18 image for {qid}")
    filename = image_claims[0]["mainsnak"]["datavalue"]["value"]
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "titles": f"File:{filename}",
    }
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(COMMONS_API, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _parse_image(commons_response: dict[str, Any]) -> ImageRef | None:
    import re

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


def _build_sightings(entity: ConfirmedEntity) -> list[Sighting]:
    return [
        Sighting(page=cand.page, paragraph=cand.context_window)
        for cand in entity.source_candidates
    ]


async def _summarize_in_book(
    entity: ConfirmedEntity, llm_config: LLMConfig | None = None
) -> str:
    excerpts = "\n\n".join(
        cand.context_window for cand in entity.source_candidates
    )[:MAX_EXCERPT_CHARS]
    template = Template(SUMMARY_PROMPT_PATH.read_text())
    user_prompt = template.render(
        entity_name=entity.canonical_name,
        entity_type=entity.entity_type,
        excerpts=excerpts,
    )
    cfg = llm_config or LLMConfig.from_env()
    client = LLMClient(cfg)
    resp, _ = await client.complete_structured(
        system="You output strictly valid JSON {\"text\": \"<summary>\"}. No commentary.",
        user=user_prompt,
        schema=_SummaryResp,
        prompt_version="summarize-v1",
    )
    return resp.text


async def _enrich_with_qid(
    entity: ConfirmedEntity, sem: asyncio.Semaphore
) -> EntityCard | None:
    async with sem:
        try:
            wp_resp, commons_resp = await asyncio.gather(
                _fetch_wikipedia(entity.wikidata_qid),
                _fetch_commons_image(entity.wikidata_qid),
            )
        except Exception as e:
            logger.warning(f"enrich failed for {entity.wikidata_qid}: {e!r}")
            return None

    image = _parse_image(commons_resp)
    return EntityCard(
        entity=entity,
        wikipedia_url=wp_resp["content_urls"]["desktop"]["page"],
        wikipedia_summary=wp_resp.get("extract", ""),
        summary_source="wikipedia",
        primary_image=image,
        alt_images=[],
        sightings=_build_sightings(entity),
        enrich_version=ENRICH_VERSION,
    )


async def _enrich_without_qid(
    entity: ConfirmedEntity, sem: asyncio.Semaphore, llm_config: LLMConfig | None
) -> EntityCard:
    async with sem:
        summary = await _summarize_in_book(entity, llm_config=llm_config)
    return EntityCard(
        entity=entity,
        wikipedia_url=None,
        wikipedia_summary=summary,
        summary_source="llm_in_book",
        primary_image=None,
        alt_images=[],
        sightings=_build_sightings(entity),
        enrich_version=ENRICH_VERSION,
    )


def _ai_images_enabled() -> bool:
    return os.environ.get("PUGMARK_AI_IMAGES", "0") in ("1", "true", "yes", "on")


async def _maybe_synthesize_image(
    entity: ConfirmedEntity, sem: asyncio.Semaphore
) -> ImageRef | None:
    """Generate an AI image for `entity` and return it as an ImageRef, or None.

    Bounded by the same semaphore as Wikipedia/Commons calls so we don't blow
    past Azure deployment limits.
    """
    if not _ai_images_enabled():
        return None
    context = ""
    if entity.source_candidates:
        context = entity.source_candidates[0].context_sentence or ""
    prompt = build_image_prompt(
        canonical_name=entity.canonical_name or entity.vernacular,
        entity_type=entity.entity_type,
        context_sentence=context,
    )
    async with sem:
        path = await generate_image(prompt)
    if path is None:
        return None
    return ImageRef(
        url=f"file://{Path(path).absolute()}",
        license="AI-generated (Azure gpt-image-1.5)",
        attribution="Synthesized by Pugmark — not a photograph",
        source="ai_generated",
    )


async def _enrich_one(
    entity: ConfirmedEntity,
    chapter: Chapter,
    cache: Cache,
    sem: asyncio.Semaphore,
    llm_config: LLMConfig | None,
) -> EntityCard | None:
    # Cache key includes AI-images-on/off so toggling re-runs the image step.
    cache_key = Cache.compute_hash(
        entity.wikidata_qid or f"in-book:{entity.canonical_name.lower()}",
        ENRICH_VERSION,
        entity.entity_type,
        "ai" if _ai_images_enabled() else "noai",
    )
    hit = cache.get("enrich", cache_key, EntityCard)
    if hit is not None:
        return hit.model_copy(update={"sightings": _build_sightings(entity)})

    if entity.wikidata_qid is not None:
        card = await _enrich_with_qid(entity, sem)
    else:
        card = await _enrich_without_qid(entity, sem, llm_config)

    if card is not None and card.primary_image is None:
        ai_image = await _maybe_synthesize_image(entity, sem)
        if ai_image is not None:
            card = card.model_copy(update={"primary_image": ai_image})

    if card is not None:
        cache.set("enrich", cache_key, card)
    return card


async def enrich_confirmed(
    entities: list[ConfirmedEntity],
    *,
    chapter: Chapter,
    cache: Cache,
    llm_config: LLMConfig | None = None,
) -> list[EntityCard]:
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(
        *[_enrich_one(e, chapter, cache, sem, llm_config) for e in entities]
    )
    cards = [c for c in results if c is not None]
    logger.info(f"enrich: {len(cards)}/{len(entities)} entities got cards")
    return cards


# v1 backward-compat alias
enrich_taxa = enrich_confirmed
