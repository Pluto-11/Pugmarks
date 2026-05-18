"""Pugmark CLI — Click-based.

Subcommands:
  chapters <pdf>                       List chapter titles + page ranges
  extract <pdf> --chapter N --out F    Run pipeline, write HTML gallery to F
  eval --chapter NAME                  Run eval against ground truth
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from eval.auto_label import auto_label_chapter
from eval.runner import run_eval
from pugmark.cache import Cache
from pugmark.enrich import enrich_taxa
from pugmark.extract import extract_candidates
from pugmark.ingest import list_chapters, load_chapter
from pugmark.llm import LLMConfig
from pugmark.observability import init_observability
from pugmark.render import render_html
from pugmark.schemas import Gallery
from pugmark.validate import validate_candidates

PUGMARK_VERSION = "0.1.0"


@click.group()
@click.option("--verbose", is_flag=True, help="Show DEBUG logs.")
def cli(verbose: bool) -> None:
    """Pugmark — illustrated bestiaries from hunting novels."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    init_observability()


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
def chapters(pdf: Path) -> None:
    """List chapters detected in PDF."""
    found = list_chapters(pdf)
    if not found:
        click.echo("No chapters detected (PDF has no outline).")
        return
    for ch in found:
        click.echo(
            f"{ch['number']:>3}.  pp.{ch['page_start']:>4}-{ch['page_end']:<4}  {ch['title']}"
        )


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("--chapter", type=int, required=True, help="Chapter number from `pugmark chapters`")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output HTML path")
def extract(pdf: Path, chapter: int, out: Path) -> None:
    """Run the full pipeline and write an HTML gallery."""
    asyncio.run(_run_pipeline(pdf, chapter, out))


async def _run_pipeline(pdf: Path, chapter_num: int, out: Path) -> None:
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()

    click.echo(f"[1/5] Loading chapter {chapter_num} from {pdf.name}...")
    ch = load_chapter(pdf, chapter_num)

    click.echo(f"[2/5] Extracting taxa via LLM ({llm_config.providers[0]})...")
    candidates = await extract_candidates(
        ch, llm_config=llm_config, prompt_dir=Path("prompts"), cache=cache
    )
    click.echo(f"      → {len(candidates)} candidates")

    click.echo("[3/5] Validating against Wikidata...")
    confirmed, unresolved = await validate_candidates(candidates, cache=cache)
    click.echo(f"      → {len(confirmed)} confirmed · {len(unresolved)} unresolved")

    click.echo("[4/5] Enriching with Wikipedia + Commons...")
    cards = await enrich_taxa(confirmed, chapter=ch, cache=cache)
    click.echo(f"      → {len(cards)} cards built")

    gallery = Gallery(
        chapter=ch,
        cards=cards,
        unresolved=unresolved,
        generated_at=datetime.now(),
        pugmark_version=PUGMARK_VERSION,
        eval_metrics=None,
    )

    click.echo(f"[5/5] Rendering to {out}...")
    out.write_text(render_html(gallery))
    click.echo(f"✓ Done. Open {out} in a browser.")


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("--chapter", type=int, required=True, help="Chapter number")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output JSON path")
@click.option(
    "--judge-model",
    default="gemini/gemini-2.5-pro",
    show_default=True,
    help="LiteLLM model identifier for the judge (must differ from production model)",
)
@click.option("--n-calls", default=3, show_default=True, help="Judge call count")
@click.option("--min-votes", default=2, show_default=True, help="Min votes to survive")
def autolabel(
    pdf: Path,
    chapter: int,
    out: Path,
    judge_model: str,
    n_calls: int,
    min_votes: int,
) -> None:
    """Auto-generate ground truth labels via LLM-as-judge + Wikidata roundtrip."""
    asyncio.run(_run_autolabel(pdf, chapter, out, judge_model, n_calls, min_votes))


async def _run_autolabel(
    pdf: Path,
    chapter_num: int,
    out: Path,
    judge_model: str,
    n_calls: int,
    min_votes: int,
) -> None:
    cache = Cache.from_env()
    click.echo(f"[autolabel] judge={judge_model} n_calls={n_calls} min_votes={min_votes}")
    truth = await auto_label_chapter(
        pdf,
        chapter_num,
        cache=cache,
        judge_model=judge_model,
        n_calls=n_calls,
        min_votes=min_votes,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(truth, indent=2))
    click.echo(f"✓ Wrote {len(truth)} ground-truth entries to {out}")


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("--chapter", type=int, required=True)
@click.option(
    "--ground-truth",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to ground-truth JSON (e.g., eval/ground_truth/sivanipalli.json)",
)
@click.option("--runs-dir", type=click.Path(path_type=Path), default=Path("eval/runs"))
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero if F1 dropped >5% vs latest prior run.",
)
def eval_cmd(
    pdf: Path, chapter: int, ground_truth: Path, runs_dir: Path, strict: bool
) -> None:
    """Run eval against ground truth."""
    run = asyncio.run(
        run_eval(
            pdf=pdf,
            chapter_number=chapter,
            ground_truth_path=ground_truth,
            runs_dir=runs_dir,
        )
    )

    click.echo(
        f"Extraction: P={run.extraction.precision:.3f} "
        f"R={run.extraction.recall:.3f} F1={run.extraction.f1:.3f}"
    )
    click.echo(f"Hallucination: {run.extraction.hallucination_rate:.3f}")
    click.echo(
        f"Validation:  QID-acc={run.validation.qid_accuracy:.3f} "
        f"unresolved-rate={run.validation.unresolved_rate:.3f}"
    )
    click.echo(f"Latency:     {run.latency_ms} ms")

    if strict:
        this_run_stem = run.timestamp.strftime("%Y%m%dT%H%M%S")
        prior = sorted(
            p for p in runs_dir.glob("*.json") if p.stem != this_run_stem
        )
        if prior:
            prev = json.loads(prior[-1].read_text())
            prev_f1 = prev["extraction"]["f1"]
            if run.extraction.f1 < prev_f1 - 0.05:
                click.echo(
                    f"❌ REGRESSION: F1 dropped from {prev_f1:.3f} → {run.extraction.f1:.3f}",
                    err=True,
                )
                sys.exit(1)
            click.echo(f"✓ No regression (prev F1: {prev_f1:.3f})")


# Click does not allow a subcommand named "eval" because of Python's builtin.
# Register under the name "eval":
cli.add_command(eval_cmd, name="eval")


if __name__ == "__main__":
    cli()
