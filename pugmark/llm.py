"""Async LiteLLM client with provider fallback.

Single public method: complete_structured(system, user, schema, prompt_version)
returns (parsed_pydantic, provider_used). Tries providers in order, falls
back on exception. Raises RuntimeError when every provider fails.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TypeVar

from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMConfig:
    providers: list[str] = field(
        default_factory=lambda: ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"]
    )
    max_retries: int = 1
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        primary = os.environ.get("PUGMARK_PRIMARY_MODEL", "gemini/gemini-2.0-flash")
        provider_csv = os.environ.get("PUGMARK_PROVIDERS", "gemini,groq").split(",")
        provider_to_model = {
            "gemini": "gemini/gemini-2.0-flash",
            "groq": "groq/llama-3.3-70b-versatile",
            "ollama": "ollama/qwen2.5:7b",
        }
        ordered = [primary] + [
            provider_to_model[p] for p in provider_csv if provider_to_model.get(p) != primary
        ]
        return cls(providers=ordered)


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

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
                content = resp.choices[0].message.content
                parsed = schema.model_validate_json(content)
                logger.info(f"LLM call succeeded on {provider}")
                return parsed, provider
            except Exception as e:
                last_err = e
                logger.warning(f"LLM provider {provider} failed: {e!r}; trying next")
        raise RuntimeError(f"all LLM providers failed; last error: {last_err!r}")
