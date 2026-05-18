from __future__ import annotations

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from pydantic import BaseModel

from pugmark.llm import (
    LLMClient,
    LLMConfig,
    _backoff_seconds,
    _is_rate_limit,
    _parse_retry_after,
    _strip_code_fence,
)


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
    call_kwargs = m.call_args.kwargs
    assert call_kwargs["metadata"] == {
        "prompt_version": "v1",
        "pugmark_provider": "gemini/gemini-2.0-flash",
    }


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


def test_from_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUGMARK_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("PUGMARK_PROVIDERS", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.providers == [
        "gemini/gemini-2.0-flash",
        "groq/llama-3.3-70b-versatile",
    ]


def test_from_env_primary_swap_puts_groq_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUGMARK_PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")
    monkeypatch.setenv("PUGMARK_PROVIDERS", "gemini,groq")
    cfg = LLMConfig.from_env()
    assert cfg.providers == [
        "groq/llama-3.3-70b-versatile",
        "gemini/gemini-2.0-flash",
    ]


def test_from_env_ignores_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUGMARK_PRIMARY_MODEL", "gemini/gemini-2.0-flash")
    monkeypatch.setenv("PUGMARK_PROVIDERS", "gemini,grok,ollama")
    cfg = LLMConfig.from_env()
    assert cfg.providers == [
        "gemini/gemini-2.0-flash",
        "ollama/qwen2.5:7b",
    ]


# ============================================================================
# Retry-with-backoff on rate-limit
# ============================================================================


def test_parse_retry_after_groq_format() -> None:
    err = Exception(
        'GroqException - {"error":{"message":"Rate limit ... Please try again in 1.234s."}}'
    )
    assert _parse_retry_after(err) == pytest.approx(1.234)


def test_parse_retry_after_gemini_structured() -> None:
    err = Exception(
        'GeminiException - {"error":{...},"retryDelay":"49s"}'
    )
    assert _parse_retry_after(err) == pytest.approx(49.0)


def test_parse_retry_after_no_hint_returns_none() -> None:
    assert _parse_retry_after(Exception("some other error")) is None


def test_backoff_seconds_uses_hint_when_present() -> None:
    err = Exception("Please try again in 5.0s")
    # Hint + 0.1s jitter
    assert _backoff_seconds(err, attempt=0, base=1.0) == pytest.approx(5.1)


def test_backoff_seconds_falls_back_to_exponential() -> None:
    err = Exception("no hint")
    assert _backoff_seconds(err, attempt=0, base=1.0) == 1.0
    assert _backoff_seconds(err, attempt=1, base=1.0) == 2.0
    assert _backoff_seconds(err, attempt=2, base=1.0) == 4.0


def test_backoff_seconds_capped_at_30s() -> None:
    # Either hint > 30 or exponential blowup should both cap
    assert _backoff_seconds(Exception("try again in 600s"), attempt=0, base=1.0) == 30.0
    assert _backoff_seconds(Exception("nope"), attempt=10, base=1.0) == 30.0


def test_is_rate_limit_detects_via_class_or_message() -> None:
    assert _is_rate_limit(Exception("rate_limit_exceeded")) is True
    assert _is_rate_limit(Exception("HTTP 429 Too Many Requests")) is True
    assert _is_rate_limit(Exception("tokens per day exceeded")) is True
    assert _is_rate_limit(Exception("network unreachable")) is False


@pytest.mark.asyncio
async def test_retries_on_rate_limit_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 on attempt 1 should sleep + retry; success on attempt 2 returns."""
    cfg = LLMConfig(
        providers=["gemini/gemini-2.0-flash"], max_retries=2, backoff_base_s=0.01
    )
    client = LLMClient(cfg)
    msg = type("Msg", (), {"content": '{"items": ["recovered"]}'})()
    choice = type("Choice", (), {"message": msg})()
    fake_response = type("Resp", (), {"choices": [choice]})()
    side_effects = [
        litellm.RateLimitError(
            "rate limit; try again in 0.01s",
            model="gemini/gemini-2.0-flash",
            llm_provider="gemini",
        ),
        fake_response,
    ]
    with patch("pugmark.llm.acompletion", new=AsyncMock(side_effect=side_effects)) as m:
        out, used = await client.complete_structured(
            system="s", user="u", schema=DummyOut, prompt_version="v"
        )
    assert out == DummyOut(items=["recovered"])
    assert used == "gemini/gemini-2.0-flash"
    assert m.await_count == 2


@pytest.mark.asyncio
async def test_falls_through_provider_only_after_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All retries on provider A hit 429 → fall through to provider B."""
    cfg = LLMConfig(
        providers=["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"],
        max_retries=1,
        backoff_base_s=0.01,
    )
    client = LLMClient(cfg)
    msg = type("Msg", (), {"content": '{"items": ["fallback"]}'})()
    choice = type("Choice", (), {"message": msg})()
    fake_response = type("Resp", (), {"choices": [choice]})()
    side_effects = [
        litellm.RateLimitError("429 #1", model="m", llm_provider="gemini"),
        litellm.RateLimitError("429 #2", model="m", llm_provider="gemini"),
        fake_response,
    ]
    with patch("pugmark.llm.acompletion", new=AsyncMock(side_effect=side_effects)) as m:
        out, used = await client.complete_structured(
            system="s", user="u", schema=DummyOut, prompt_version="v"
        )
    assert out == DummyOut(items=["fallback"])
    assert used == "groq/llama-3.3-70b-versatile"
    # 2 attempts on gemini (initial + 1 retry) + 1 success on groq
    assert m.await_count == 3


@pytest.mark.asyncio
async def test_non_rate_limit_error_does_not_retry() -> None:
    """A non-429 error on provider A should fall through immediately, no retry."""
    cfg = LLMConfig(
        providers=["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"],
        max_retries=5,
        backoff_base_s=0.01,
    )
    client = LLMClient(cfg)
    msg = type("Msg", (), {"content": '{"items": ["ok"]}'})()
    choice = type("Choice", (), {"message": msg})()
    fake_response = type("Resp", (), {"choices": [choice]})()
    side_effects = [
        ValueError("malformed payload"),  # NOT a rate limit
        fake_response,
    ]
    with patch("pugmark.llm.acompletion", new=AsyncMock(side_effect=side_effects)) as m:
        out, used = await client.complete_structured(
            system="s", user="u", schema=DummyOut, prompt_version="v"
        )
    assert out == DummyOut(items=["ok"])
    assert used == "groq/llama-3.3-70b-versatile"
    assert m.await_count == 2  # no retry on the ValueError


# ============================================================================
# Markdown code-fence stripping (GPT-4o sometimes wraps JSON output)
# ============================================================================


def test_strip_code_fence_with_json_label() -> None:
    raw = '```json\n{"ok": true}\n```'
    assert _strip_code_fence(raw) == '{"ok": true}'


def test_strip_code_fence_without_label() -> None:
    raw = '```\n{"ok": true}\n```'
    assert _strip_code_fence(raw) == '{"ok": true}'


def test_strip_code_fence_passes_through_plain_json() -> None:
    raw = '{"ok": true}'
    assert _strip_code_fence(raw) == '{"ok": true}'


def test_strip_code_fence_empty_input() -> None:
    assert _strip_code_fence("") == ""


def test_strip_code_fence_handles_surrounding_whitespace() -> None:
    raw = '  \n```json\n{"x": 1}\n```\n  '
    assert _strip_code_fence(raw) == '{"x": 1}'
