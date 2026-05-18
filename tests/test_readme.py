"""Verify the HF Spaces YAML frontmatter in README.md is well-formed.

If this test fails, the HF Spaces deploy will silently skip the Space config
and use defaults — which means wrong SDK, wrong app_file, broken build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"


@pytest.fixture
def frontmatter() -> dict[str, str]:
    text = README.read_text()
    assert text.startswith("---\n"), "README must start with YAML frontmatter for HF Spaces"
    end = text.index("\n---\n", 4)
    block = text[4:end]
    out: dict[str, str] = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def test_frontmatter_has_required_hf_keys(frontmatter: dict[str, str]) -> None:
    required = {"title", "emoji", "sdk", "app_file", "license"}
    missing = required - frontmatter.keys()
    assert not missing, f"README frontmatter missing HF keys: {missing}"


def test_frontmatter_sdk_is_gradio(frontmatter: dict[str, str]) -> None:
    assert frontmatter["sdk"] == "gradio"


def test_frontmatter_app_file_points_at_repo_app(frontmatter: dict[str, str]) -> None:
    assert frontmatter["app_file"] == "app.py"
    assert (README.parent / "app.py").exists(), "app.py declared in frontmatter must exist"


def test_requirements_txt_pinned_and_present() -> None:
    """HF Spaces stock-Gradio builder consumes requirements.txt — must exist + be pinned."""
    req = README.parent / "requirements.txt"
    assert req.exists(), "requirements.txt is required for HF Spaces stock builder"
    contents = req.read_text()
    # uv pip compile pins with ==, never with >=
    pinned_lines = [
        ln for ln in contents.splitlines()
        if ln and not ln.startswith("#") and "==" in ln
    ]
    assert pinned_lines, "requirements.txt should contain pinned (==) versions"
