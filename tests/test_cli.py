from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from pugmark.cli import cli

FIXTURE = Path(__file__).parent / "fixtures" / "sample_chapter.pdf"


def test_chapters_command_lists_chapters() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["chapters", str(FIXTURE)])
    assert result.exit_code == 0
    assert "Black Panther of Sivanipalli" in result.output


def test_help_includes_extract() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.output
    assert "chapters" in result.output


def test_help_includes_autolabel() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "autolabel" in result.output
