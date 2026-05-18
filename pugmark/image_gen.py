"""AI image generation for entities without real photos (Azure gpt-image-1.5).

When the enrich stage can't pull a Wikipedia/Commons image — either because the
entity isn't on Wikidata, or because Wikidata has the entity but no Commons
photo — we synthesize an illustration. Style is "natural-history plate" so the
result feels like a vintage botanical/zoological illustration regardless of
what the entity is.

Caching: generated images live at ~/.cache/pugmark/images/{hash}.png, keyed by
(prompt + model + style_version). A second call for the same entity is free.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path

import litellm

logger = logging.getLogger(__name__)

IMAGE_GEN_VERSION = "v1"
DEFAULT_STYLE_GUIDANCE = (
    "vintage natural-history illustration in the style of a 19th-century "
    "zoological / botanical plate: precise linework, restrained watercolour "
    "wash, neutral cream background, clearly identifiable subject in profile "
    "or characteristic pose, scholarly and dignified, no human characters, "
    "no text or labels in the image"
)


def _images_cache_dir() -> Path:
    root = Path(os.environ.get("PUGMARK_CACHE_DIR", str(Path.home() / ".cache" / "pugmark")))
    out = root / "images"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cache_key(prompt: str, model: str) -> str:
    h = hashlib.sha256(
        f"{prompt}|{model}|{IMAGE_GEN_VERSION}".encode()
    ).hexdigest()[:32]
    return h


def build_prompt(
    canonical_name: str,
    *,
    entity_type: str,
    context_sentence: str = "",
    style_guidance: str = DEFAULT_STYLE_GUIDANCE,
) -> str:
    """Compose the image-gen prompt from an entity's metadata.

    Keeps the subject specific (canonical_name + entity_type) and the style
    consistent (style_guidance). Context_sentence is included as flavour but
    truncated so the model isn't tempted to render text or scenes.
    """
    ctx = context_sentence.strip().replace("\n", " ")
    if len(ctx) > 240:
        ctx = ctx[:237] + "..."
    parts = [
        f"Subject: {canonical_name}",
        f"Type: {entity_type}",
    ]
    if ctx:
        parts.append(f"Brief context: {ctx}")
    parts.append(f"Style: {style_guidance}")
    return ". ".join(parts)


async def generate_image(
    prompt: str,
    *,
    size: str = "1024x1024",
    quality: str = "medium",
) -> Path | None:
    """Generate (or return cached) image for `prompt`.

    Returns the local file path, or None on failure (missing creds, API error).
    """
    api_key = os.environ.get("AZURE_IMAGE_API_KEY")
    api_base = os.environ.get("AZURE_IMAGE_ENDPOINT")
    api_version = os.environ.get("AZURE_IMAGE_API_VERSION")
    model_env = os.environ.get("AZURE_IMAGE_MODEL", "gpt-image-1.5")
    if not (api_key and api_base and api_version):
        logger.info(
            "image-gen: AZURE_IMAGE_* env not set; returning None (placeholder will be used)"
        )
        return None

    model = f"azure/{model_env}"
    key = _cache_key(prompt, model)
    out_path = _images_cache_dir() / f"{key}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.debug(f"image-gen cache hit: {out_path.name}")
        return out_path

    try:
        resp = await litellm.aimage_generation(
            model=model,
            prompt=prompt,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            n=1,
            size=size,
            quality=quality,
        )
    except Exception as e:
        logger.warning(f"image-gen failed for prompt '{prompt[:60]}...': {e!r}")
        return None

    if not resp.data:
        return None
    d = resp.data[0]
    b64 = getattr(d, "b64_json", None)
    url = getattr(d, "url", None)
    if b64:
        try:
            out_path.write_bytes(base64.b64decode(b64))
        except Exception as e:
            logger.warning(f"image-gen failed decoding b64: {e!r}")
            return None
        logger.info(f"image-gen: wrote {out_path.name} ({out_path.stat().st_size} bytes)")
        return out_path
    if url:
        # Some Azure deployments return URLs; download.
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                out_path.write_bytes(r.content)
        except Exception as e:
            logger.warning(f"image-gen failed downloading {url}: {e!r}")
            return None
        logger.info(f"image-gen: downloaded {out_path.name} ({out_path.stat().st_size} bytes)")
        return out_path
    return None
