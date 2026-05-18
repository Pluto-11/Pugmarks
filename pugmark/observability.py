"""Langfuse initialization + LiteLLM callback wiring.

Single entry: init_observability() — call once at app/CLI startup. If Langfuse
keys are missing or empty, this is a no-op and the pipeline runs untraced.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def init_observability() -> None:
    """Wire LiteLLM → Langfuse callback if keys are set, else no-op."""
    if not is_langfuse_configured():
        logger.debug("Langfuse keys not set; running without tracing.")
        return

    import litellm

    # LiteLLM picks up LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    # from env automatically when "langfuse" is added as a callback.
    litellm.success_callback = list(set((litellm.success_callback or []) + ["langfuse"]))
    litellm.failure_callback = list(set((litellm.failure_callback or []) + ["langfuse"]))
    logger.info("Langfuse observability enabled for LiteLLM.")
