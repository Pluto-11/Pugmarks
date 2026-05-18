from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from pugmark.llm import LLMClient, LLMConfig


class DummyOut(BaseModel):
    items: list[str]


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(
        providers=["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"],
        max_retries=1,
        timeout_s=30.0,
    )


@pytest.mark.asyncio
async def test_first_provider_succeeds(config: LLMConfig) -> None:
    client = LLMClient(config)
    msg = type("Msg", (), {"content": '{"items": ["a", "b"]}'})()
    choice = type("Choice", (), {"message": msg})()
    fake_response = type("Resp", (), {"choices": [choice]})()
    with patch("pugmark.llm.acompletion", new=AsyncMock(return_value=fake_response)) as m:
        out, used = await client.complete_structured(
            system="sys", user="usr", schema=DummyOut, prompt_version="v1"
        )
    assert out == DummyOut(items=["a", "b"])
    assert used == "gemini/gemini-2.0-flash"
    assert m.await_count == 1


@pytest.mark.asyncio
async def test_falls_back_on_first_provider_error(config: LLMConfig) -> None:
    client = LLMClient(config)
    msg = type("Msg", (), {"content": '{"items": ["x"]}'})()
    choice = type("Choice", (), {"message": msg})()
    fake_response = type("Resp", (), {"choices": [choice]})()

    side_effects = [Exception("boom"), fake_response]
    with patch("pugmark.llm.acompletion", new=AsyncMock(side_effect=side_effects)):
        out, used = await client.complete_structured(
            system="sys", user="usr", schema=DummyOut, prompt_version="v1"
        )
    assert out == DummyOut(items=["x"])
    assert used == "groq/llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_raises_when_all_providers_fail(config: LLMConfig) -> None:
    client = LLMClient(config)
    with patch(
        "pugmark.llm.acompletion", new=AsyncMock(side_effect=Exception("no providers up"))
    ), pytest.raises(RuntimeError, match="all LLM providers failed"):
        await client.complete_structured(
            system="sys", user="usr", schema=DummyOut, prompt_version="v1"
        )
