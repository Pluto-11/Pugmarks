"""Tests for the AI image-generation module.

We mock the actual LiteLLM call — we never want CI to hit Azure. The tests
cover prompt construction, cache hit behavior, env-var preconditions, and
error tolerance.
"""
from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pugmark.image_gen import build_prompt, generate_image


def test_build_prompt_includes_subject_type_and_style() -> None:
    p = build_prompt("Sloth Bear", entity_type="taxa", context_sentence="A bear cub in the jungle.")
    assert "Sloth Bear" in p
    assert "Type: taxa" in p
    assert "Brief context" in p
    assert "natural-history" in p  # style guidance is appended


def test_build_prompt_truncates_long_context() -> None:
    long_ctx = "x" * 500
    p = build_prompt("X", entity_type="taxa", context_sentence=long_ctx)
    # 240 + "..." inside the prompt
    assert "x" * 240 not in p  # truncation occurred
    assert "..." in p


def test_build_prompt_omits_context_section_when_empty() -> None:
    p = build_prompt("X", entity_type="taxa", context_sentence="")
    assert "Brief context" not in p


@pytest.mark.asyncio
async def test_generate_image_returns_none_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_IMAGE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_IMAGE_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_IMAGE_API_VERSION", raising=False)
    result = await generate_image("any prompt")
    assert result is None


@pytest.mark.asyncio
async def test_generate_image_caches_to_disk_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_IMAGE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_IMAGE_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_IMAGE_API_VERSION", "2024-02-01")
    monkeypatch.setenv("AZURE_IMAGE_MODEL", "gpt-image-1.5")
    monkeypatch.setenv("PUGMARK_CACHE_DIR", str(tmp_path))

    # Synthesize a minimal valid PNG (1x1 red pixel) and b64-encode
    fake_png = bytes([
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  # PNG magic
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,  # IHDR
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xde, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
        0x44, 0xae, 0x42, 0x60, 0x82,
    ])
    fake_data_item = type("ImgItem", (), {
        "b64_json": base64.b64encode(fake_png).decode(),
        "url": None,
    })()
    fake_response = type("Resp", (), {"data": [fake_data_item]})()

    with patch(
        "pugmark.image_gen.litellm.aimage_generation",
        new=AsyncMock(return_value=fake_response),
    ) as m:
        first = await generate_image("vivid prompt")
        # Second call with same prompt → cache hit, no second API call
        second = await generate_image("vivid prompt")

    assert first is not None
    assert first == second
    assert first.exists()
    assert first.read_bytes() == fake_png
    assert m.await_count == 1, "second call should hit cache"


@pytest.mark.asyncio
async def test_generate_image_swallows_api_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AZURE_IMAGE_API_KEY", "k")
    monkeypatch.setenv("AZURE_IMAGE_ENDPOINT", "https://e")
    monkeypatch.setenv("AZURE_IMAGE_API_VERSION", "v")
    monkeypatch.setenv("PUGMARK_CACHE_DIR", str(tmp_path))
    with patch(
        "pugmark.image_gen.litellm.aimage_generation",
        new=AsyncMock(side_effect=RuntimeError("azure down")),
    ):
        result = await generate_image("anything")
    assert result is None
