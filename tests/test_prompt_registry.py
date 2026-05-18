from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pugmark.prompt_registry as pr_module
from pugmark.prompt_registry import PromptRegistry


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "extract_taxa.v1.j2").write_text("Extract from: {{ chapter_text }}")
    (p / "extract_taxa.v2.j2").write_text("v2: {{ chapter_text }}")
    return p


@pytest.fixture(autouse=True)
def _clear_langfuse_lru() -> None:
    """LRU cache on _fetch_langfuse persists across tests; clear before each."""
    pr_module._fetch_langfuse.cache_clear()


# ---- local mode -----------------------------------------------------------


def test_loads_in_repo_prompt(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "local")
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    p = reg.get("extract_taxa", version="v1")
    assert p.template_text == "Extract from: {{ chapter_text }}"
    assert p.version == "v1"
    assert p.source == "local"


def test_renders_with_jinja(prompts_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    p = reg.get("extract_taxa", version="v1")
    rendered = p.render(chapter_text="A tiger appeared.")
    assert rendered == "Extract from: A tiger appeared."


def test_missing_prompt_raises(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "local")
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    with pytest.raises(FileNotFoundError):
        reg.get("nonexistent", version="v1")


def test_default_version_picks_highest(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "local")
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    p = reg.get("extract_taxa")
    assert p.version == "v2"


def test_discover_local_lists_all_versions(prompts_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    found = reg.discover_local()
    assert len(found) == 2
    assert {(n, v) for n, v, _ in found} == {("extract_taxa", "v1"), ("extract_taxa", "v2")}


# ---- langfuse_first mode -------------------------------------------------


def test_langfuse_first_returns_langfuse_when_present(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "langfuse_first")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    with patch.object(pr_module, "_fetch_langfuse", return_value="From Langfuse: {{ x }}"):
        reg = PromptRegistry(in_repo_dir=prompts_dir)
        p = reg.get("extract_taxa", version="v1")
    assert p.template_text == "From Langfuse: {{ x }}"
    assert p.source == "langfuse"
    assert p.version == "v1"


def test_langfuse_first_falls_back_to_local_when_missing(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "langfuse_first")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    with patch.object(pr_module, "_fetch_langfuse", return_value=None):
        reg = PromptRegistry(in_repo_dir=prompts_dir)
        p = reg.get("extract_taxa", version="v1")
    assert p.template_text == "Extract from: {{ chapter_text }}"
    assert p.source == "local"


# ---- langfuse_only mode --------------------------------------------------


def test_langfuse_only_raises_when_missing(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "langfuse_only")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    with patch.object(pr_module, "_fetch_langfuse", return_value=None):
        reg = PromptRegistry(in_repo_dir=prompts_dir)
        with pytest.raises(FileNotFoundError, match="langfuse_only"):
            reg.get("extract_taxa", version="v1")


def test_langfuse_only_returns_langfuse_when_present(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "langfuse_only")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    with patch.object(pr_module, "_fetch_langfuse", return_value="only-from-langfuse"):
        reg = PromptRegistry(in_repo_dir=prompts_dir)
        p = reg.get("extract_taxa", version="v1")
    assert p.source == "langfuse"
    assert p.template_text == "only-from-langfuse"


# ---- default-source resolution -------------------------------------------


def test_default_source_local_when_no_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUGMARK_PROMPTS_SOURCE", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert pr_module._resolve_source() == "local"


def test_default_source_langfuse_first_when_creds_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUGMARK_PROMPTS_SOURCE", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert pr_module._resolve_source() == "langfuse_first"


def test_invalid_source_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUGMARK_PROMPTS_SOURCE", "made-up")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert pr_module._resolve_source() == "local"


# ---- Langfuse SDK error tolerance ----------------------------------------


def test_langfuse_fetch_swallows_sdk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the Langfuse SDK raises, the fetcher must return None (not propagate)."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    fake_client = MagicMock()
    fake_client.get_prompt.side_effect = RuntimeError("langfuse down")
    with patch("langfuse.Langfuse", return_value=fake_client):
        result = pr_module._fetch_langfuse("any_name", "v1")
    assert result is None
