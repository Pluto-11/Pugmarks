from __future__ import annotations

from pathlib import Path

import pytest
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


def test_help_includes_eval() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.output


def test_eval_command_writes_run_and_prints_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: eval command runs the pipeline (mocked) and prints metrics."""
    from datetime import datetime

    from pugmark.schemas import EvalRun, ExtractionMetrics, ValidationMetrics

    truth_path = tmp_path / "truth.json"
    truth_path.write_text("[]")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    fake_run = EvalRun(
        chapter_id="truth",
        extraction=ExtractionMetrics(
            precision=0.9, recall=0.8, f1=0.847, hallucination_rate=0.05
        ),
        validation=ValidationMetrics(
            qid_accuracy=0.95, confusion_matrix={}, unresolved_rate=0.05
        ),
        cost_usd=0.0,
        latency_ms=1234,
        pugmark_version="0.1.0",
        llm_provider="gemini/gemini-2.0-flash",
        prompt_version="v1",
        timestamp=datetime.now(),
    )

    async def fake_run_eval(**kwargs):
        return fake_run

    monkeypatch.setattr("pugmark.cli.run_eval", fake_run_eval)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "eval",
            str(FIXTURE),
            "--chapter",
            "1",
            "--ground-truth",
            str(truth_path),
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "F1=0.847" in result.output
    assert "Hallucination: 0.050" in result.output


def test_eval_strict_fails_on_f1_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--strict exits 1 when F1 drops >5% vs the latest prior run."""
    import json as _json
    from datetime import datetime

    from pugmark.schemas import EvalRun, ExtractionMetrics, ValidationMetrics

    truth_path = tmp_path / "truth.json"
    truth_path.write_text("[]")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Seed a prior run with F1=0.9
    prior_run = {
        "chapter_id": "truth",
        "extraction": {
            "precision": 0.9, "recall": 0.9, "f1": 0.9, "hallucination_rate": 0.0
        },
        "validation": {
            "qid_accuracy": 1.0, "confusion_matrix": {}, "unresolved_rate": 0.0
        },
        "cost_usd": 0.0,
        "latency_ms": 1000,
        "pugmark_version": "0.1.0",
        "llm_provider": "gemini/gemini-2.0-flash",
        "prompt_version": "v1",
        "timestamp": "2026-01-01T00:00:00",
    }
    (runs_dir / "20260101T000000.json").write_text(_json.dumps(prior_run))

    # New run with F1=0.5 (>5% drop)
    fake_run = EvalRun(
        chapter_id="truth",
        extraction=ExtractionMetrics(
            precision=0.5, recall=0.5, f1=0.5, hallucination_rate=0.0
        ),
        validation=ValidationMetrics(
            qid_accuracy=1.0, confusion_matrix={}, unresolved_rate=0.0
        ),
        cost_usd=0.0,
        latency_ms=1000,
        pugmark_version="0.1.0",
        llm_provider="gemini/gemini-2.0-flash",
        prompt_version="v1",
        timestamp=datetime.now(),
    )

    async def fake_run_eval(**kwargs):
        return fake_run

    monkeypatch.setattr("pugmark.cli.run_eval", fake_run_eval)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "eval",
            str(FIXTURE),
            "--chapter",
            "1",
            "--ground-truth",
            str(truth_path),
            "--runs-dir",
            str(runs_dir),
            "--strict",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "REGRESSION" in result.output


def test_help_includes_analyze() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output


def test_analyze_command_prints_proposed_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pugmark analyze <pdf>` prints the analyzer's proposed types."""
    from datetime import datetime

    from pugmark.entity_type import BookSchema, EntityTypeSpec

    fake_schema = BookSchema(
        book_id="sample_chapter",
        proposed_types=[
            EntityTypeSpec(
                name="taxa",
                description="animals + plants",
                wikidata_qclass="Q16521",
                extraction_prompt_template="x",
                judge_prompt_template="x",
            ),
            EntityTypeSpec(
                name="people",
                description="characters",
                wikidata_qclass="Q5",
                extraction_prompt_template="x",
                judge_prompt_template="x",
            ),
        ],
        analyzer_version="v1",
        analyzed_at=datetime.now(),
    )

    async def fake_analyze(*args, **kwargs):
        return fake_schema

    monkeypatch.setattr("pugmark.cli.analyze_book", fake_analyze)

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert "taxa" in result.output
    assert "people" in result.output
    assert "Q5" in result.output
