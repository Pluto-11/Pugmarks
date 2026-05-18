from __future__ import annotations

import pytest

from pugmark.observability import init_observability, is_langfuse_configured


def test_not_configured_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    assert is_langfuse_configured() is False


def test_configured_with_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert is_langfuse_configured() is True


def test_init_no_op_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    # Should not raise
    init_observability()


def test_init_registers_langfuse_callback_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])

    init_observability()
    assert "langfuse" in litellm.success_callback
    assert "langfuse" in litellm.failure_callback

    init_observability()
    assert litellm.success_callback.count("langfuse") == 1
    assert litellm.failure_callback.count("langfuse") == 1
