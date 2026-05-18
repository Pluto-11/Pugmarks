from __future__ import annotations

from pathlib import Path

import pytest

from pugmark.prompt_registry import PromptRegistry


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "extract_taxa.v1.j2").write_text("Extract from: {{ chapter_text }}")
    (p / "extract_taxa.v2.j2").write_text("v2: {{ chapter_text }}")
    return p


def test_loads_in_repo_prompt(prompts_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    p = reg.get("extract_taxa", version="v1")
    assert p.template_text == "Extract from: {{ chapter_text }}"
    assert p.version == "v1"


def test_renders_with_jinja(prompts_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    p = reg.get("extract_taxa", version="v1")
    rendered = p.render(chapter_text="A tiger appeared.")
    assert rendered == "Extract from: A tiger appeared."


def test_missing_prompt_raises(prompts_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    with pytest.raises(FileNotFoundError):
        reg.get("nonexistent", version="v1")


def test_default_version_picks_highest(prompts_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    p = reg.get("extract_taxa")  # no version → highest
    assert p.version == "v2"
