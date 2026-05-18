"""Async LiteLLM client with provider fallback + retry-with-backoff.

Single public method: complete_structured(system, user, schema, prompt_version)
returns (parsed_pydantic, provider_used).

Resilience model:
  - Per provider: retry up to `max_retries` times on RateLimitError (429),
    honoring server `retry-after` hints when parseable; otherwise exponential
    backoff (base 1.0 → 2 → 4 → 8 seconds, capped at 30s).
  - On non-rate-limit exceptions: fall through immediately to next provider.
  - When every provider is exhausted: raise RuntimeError with last error.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TypeVar

import litellm
from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Server-suggested retry hint regex: "Please try again in 1.234s" / "in 49s" etc.
_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9.]+)\s*s", re.IGNORECASE)
_BACKOFF_CAP_SECONDS = 30.0
# Markdown JSON code fences some models (esp. GPT-4o) wrap structured output in.
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


@dataclass
class LLMConfig:
    providers: list[str] = field(
        default_factory=lambda: ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"]
    )
    max_retries: int = 3
    timeout_s: float = 60.0
    backoff_base_s: float = 1.0

    @classmethod
    def from_env(cls, role: str = "primary") -> LLMConfig:
        """Build LLMConfig from env vars.

        role="primary"  reads PUGMARK_PRIMARY_MODEL  / PUGMARK_PROVIDERS
        role="judge"    reads PUGMARK_JUDGE_MODEL    / PUGMARK_JUDGE_PROVIDERS
        role="analyzer" reads PUGMARK_ANALYZER_MODEL / PUGMARK_ANALYZER_PROVIDERS
        """
        role_to_envs = {
            "primary": (
                "PUGMARK_PRIMARY_MODEL", "PUGMARK_PROVIDERS", "gemini/gemini-2.0-flash",
            ),
            "judge": (
                "PUGMARK_JUDGE_MODEL", "PUGMARK_JUDGE_PROVIDERS", "gemini/gemini-2.5-pro",
            ),
            "analyzer": (
                "PUGMARK_ANALYZER_MODEL", "PUGMARK_ANALYZER_PROVIDERS", "gemini/gemini-2.5-pro",
            ),
        }
        if role not in role_to_envs:
            raise ValueError(
                f"unknown LLMConfig role {role!r}; expected one of {list(role_to_envs)}"
            )
        model_env, providers_env, default_primary = role_to_envs[role]
        primary = os.environ.get(model_env, default_primary)
        provider_csv = os.environ.get(providers_env, "gemini,groq").split(",")
        provider_to_model = {
            "azure": "azure/gpt-4.1-mini",
            "azure_strong": "azure/gpt-4o",
            "gemini": "gemini/gemini-2.0-flash",
            "groq": "groq/llama-3.3-70b-versatile",
            "ollama": "ollama/qwen2.5:7b",
        }
        ordered = [primary]
        for p in provider_csv:
            model = provider_to_model.get(p.strip())
            if model and model != primary:
                ordered.append(model)
        return cls(providers=ordered)


def _parse_retry_after(err: Exception) -> float | None:
    """Try to extract a server-suggested wait in seconds from a 429 message.

    Handles common shapes:
      Groq:  "Please try again in 1.234999999s."
      Gemini: retryDelay "49s" in error.details
    Returns None if no hint found.
    """
    msg = str(err)
    m = _RETRY_AFTER_RE.search(msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # Gemini structured retry hint
    m = re.search(r'"retryDelay":\s*"([0-9.]+)s"', msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _backoff_seconds(err: Exception, attempt: int, base: float) -> float:
    """Compute sleep before next retry. Honors server hint when present."""
    hint = _parse_retry_after(err)
    if hint is not None:
        return min(hint + 0.1, _BACKOFF_CAP_SECONDS)
    return min(base * (2**attempt), _BACKOFF_CAP_SECONDS)


def _strip_code_fence(content: str) -> str:
    """Strip markdown ```json ... ``` wrappers some models emit around JSON."""
    if not content:
        return content
    s = content.strip()
    m = _JSON_FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def _is_rate_limit(err: Exception) -> bool:
    if isinstance(err, litellm.RateLimitError):
        return True
    # Some upstream variants raise generic exceptions whose stringified form
    # still says "rate_limit_exceeded" or has HTTP 429.
    s = str(err).lower()
    return "rate_limit" in s or "ratelimit" in s or " 429" in s or "tokens per" in s


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def _call_provider_with_backoff(
        self,
        provider: str,
        system: str,
        user: str,
        schema: type[T],
        prompt_version: str,
    ) -> T:
        """Try one provider up to max_retries on RateLimitError. Other errors raise immediately."""
        last_429: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await acompletion(
                    model=provider,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    timeout=self.config.timeout_s,
                    metadata={"prompt_version": prompt_version, "pugmark_provider": provider},
                )
                content = _strip_code_fence(resp.choices[0].message.content)
                return schema.model_validate_json(content)
            except Exception as e:
                if not _is_rate_limit(e) or attempt >= self.config.max_retries:
                    raise
                last_429 = e
                wait = _backoff_seconds(e, attempt, self.config.backoff_base_s)
                logger.info(
                    f"{provider} hit rate limit (attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}); sleeping {wait:.1f}s then retrying"
                )
                await asyncio.sleep(wait)
        # Should not reach here, but be explicit.
        raise last_429 if last_429 else RuntimeError("unreachable")

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        prompt_version: str,
    ) -> tuple[T, str]:
        """Call LLM with structured output, falling back across providers.

        Returns (parsed_model_instance, provider_used).
        """
        last_err: Exception | None = None
        for provider in self.config.providers:
            try:
                parsed = await self._call_provider_with_backoff(
                    provider, system, user, schema, prompt_version
                )
                logger.info(f"LLM call succeeded on {provider}")
                return parsed, provider
            except Exception as e:
                last_err = e
                logger.warning(f"LLM provider {provider} failed: {e!r}; trying next")
        raise RuntimeError(f"all LLM providers failed; last error: {last_err!r}")
