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
from pathlib import Path

import click
from dotenv import load_dotenv

from eval.auto_label import auto_label_chapter
from eval.runner import run_eval
from pugmark.analyzer import analyze_book
from pugmark.api import extract_gallery
from pugmark.cache import Cache
from pugmark.ingest import list_chapters
from pugmark.llm import LLMConfig
from pugmark.observability import init_observability
from pugmark.pdf_export import render_pdf
from pugmark.prompts_cli import prompts as prompts_group
from pugmark.render import render_html

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
@click.option(
    "--content-only",
    is_flag=True,
    help="Hide front- and back-matter (cover, contents, preface, index, author).",
)
def chapters(pdf: Path, content_only: bool) -> None:
    """List chapters detected in PDF, with kind classification."""
    found = list_chapters(pdf)
    if not found:
        click.echo("No chapters detected (PDF has no outline).")
        return
    kind_glyph = {"content": " ", "front": "·", "back": "·"}
    shown = [c for c in found if c["kind"] == "content"] if content_only else found
    for ch in shown:
        glyph = kind_glyph.get(ch["kind"], "?")
        click.echo(
            f"{ch['number']:>3}. {glyph} {ch['kind']:<7} "
            f"pp.{ch['page_start']:>4}-{ch['page_end']:<4}  {ch['title']}"
        )
    if not content_only:
        n_content = sum(1 for c in found if c["kind"] == "content")
        n_front = sum(1 for c in found if c["kind"] == "front")
        n_back = sum(1 for c in found if c["kind"] == "back")
        click.echo(
            f"\n  ({n_content} content / {n_front} front-matter / {n_back} back-matter)"
        )


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
def analyze(pdf: Path) -> None:
    """Show the book type + entity types Pugmark would extract from this PDF."""
    cache = Cache.from_env()
    schema = asyncio.run(analyze_book(pdf, cache=cache))
    bt = schema.book_type
    if bt is not None:
        click.echo("Book type:")
        click.echo(f"  genre        : {bt.genre}")
        click.echo(f"  period       : {bt.period}")
        click.echo(f"  setting      : {bt.setting}")
        click.echo(f"  themes       : {', '.join(bt.themes)}")
        click.echo(f"  target reader: {bt.target_reader}")
        click.echo(f"  summary      : {bt.summary}")
        click.echo("")
    else:
        click.echo("Book type: (classification unavailable)\n")
    click.echo(f"Analyzer proposed {len(schema.proposed_types)} type(s) for {pdf.name}:")
    for spec in schema.proposed_types:
        qclass = spec.wikidata_qclass or "--"
        click.echo(f"  {spec.name:>14}  (Wikidata: {qclass})  -- {spec.description}")


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("--chapter", type=int, required=True, help="Chapter number from `pugmark chapters`")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output HTML path")
def extract(pdf: Path, chapter: int, out: Path) -> None:
    """Run the full pipeline and write an HTML gallery."""
    asyncio.run(_run_pipeline(pdf, chapter, out))


@cli.command("export-pdf")
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("--chapter", type=int, required=True, help="Chapter number from `pugmark chapters`")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output .pdf path")
@click.option(
    "--ai-images/--no-ai-images",
    default=True,
    show_default=True,
    help="Generate AI illustrations for entities without Commons photos "
    "(needs AZURE_IMAGE_API_KEY/_ENDPOINT/_API_VERSION in .env).",
)
def export_pdf(pdf: Path, chapter: int, out: Path, ai_images: bool) -> None:
    """Run the full pipeline and write a print-ready PDF (cover + cards + AI images)."""
    import os as _os
    if ai_images:
        _os.environ["PUGMARK_AI_IMAGES"] = "1"
    else:
        _os.environ["PUGMARK_AI_IMAGES"] = "0"
    asyncio.run(_run_pdf_pipeline(pdf, chapter, out))


async def _run_pdf_pipeline(pdf: Path, chapter_num: int, out: Path) -> None:
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()
    click.echo(
        f"[pdf] PDF={pdf.name} chapter={chapter_num} "
        f"primary={llm_config.providers[0]}"
    )
    click.echo(
        "[pdf] Analyze → realize → per-type extract/validate/enrich "
        "(with AI-image fallback)..."
    )
    gallery = await extract_gallery(pdf, chapter_num, cache=cache, llm_config=llm_config)
    real_sources = ("wikimedia", "wikipedia", "inaturalist")
    for type_name, cards in gallery.cards_by_type.items():
        n_real = sum(
            1 for c in cards
            if c.primary_image and c.primary_image.source in real_sources
        )
        n_ai = sum(
            1 for c in cards
            if c.primary_image and c.primary_image.source == "ai_generated"
        )
        n_none = sum(1 for c in cards if c.primary_image is None)
        click.echo(
            f"  {type_name}: {len(cards)} cards ({n_real} real photo · "
            f"{n_ai} AI-illustrated · {n_none} no image)"
        )
    click.echo(f"  unresolved: {len(gallery.unresolved)}")
    click.echo(f"[pdf] Rendering to {out}...")
    render_pdf(gallery, out)
    click.echo(f"✓ Done. Open {out}.")


async def _run_pipeline(pdf: Path, chapter_num: int, out: Path) -> None:
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()

    click.echo(f"[pipeline] PDF={pdf.name} chapter={chapter_num} primary={llm_config.providers[0]}")
    click.echo("[pipeline] Analyzing book → realizing schema → per-type extract/validate/enrich...")

    gallery = await extract_gallery(
        pdf, chapter_num, cache=cache, llm_config=llm_config
    )

    total_cards = sum(len(cs) for cs in gallery.cards_by_type.values())
    for type_name, cards in gallery.cards_by_type.items():
        click.echo(f"  {type_name}: {len(cards)} cards")
    click.echo(f"  unresolved: {len(gallery.unresolved)}")
    click.echo(f"[render] writing {total_cards} cards to {out}")
    out.write_text(render_html(gallery))
    click.echo(f"✓ Done. Open {out} in a browser.")


@cli.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("--chapter", type=int, required=True, help="Chapter number")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output JSON path")
@click.option(
    "--judge-model",
    default=None,
    show_default=True,
    help="LiteLLM model identifier for the judge. If unset, uses PUGMARK_JUDGE_MODEL / "
    "PUGMARK_JUDGE_PROVIDERS from .env with full fallback chain.",
)
@click.option("--n-calls", default=3, show_default=True, help="Judge call count")
@click.option("--min-votes", default=2, show_default=True, help="Min votes to survive")
@click.option(
    "--all-types",
    is_flag=True,
    help="Run auto-labeling for every analyzer-proposed type, one file per type.",
)
def autolabel(
    pdf: Path,
    chapter: int,
    out: Path,
    judge_model: str | None,
    n_calls: int,
    min_votes: int,
    all_types: bool,
) -> None:
    """Auto-generate ground truth labels via LLM-as-judge + Wikidata roundtrip."""
    asyncio.run(
        _run_autolabel(pdf, chapter, out, judge_model, n_calls, min_votes, all_types)
    )


async def _run_autolabel(
    pdf: Path,
    chapter_num: int,
    out: Path,
    judge_model: str | None,
    n_calls: int,
    min_votes: int,
    all_types: bool,
) -> None:
    cache = Cache.from_env()
    if all_types:
        from eval.auto_label import auto_label_book

        click.echo(f"[autolabel] all-types mode, out_dir={out}")
        paths = await auto_label_book(
            pdf, chapter_num, cache=cache, out_dir=out, judge_model=judge_model
        )
        click.echo(f"✓ Wrote {len(paths)} per-type ground truth file(s):")
        for type_name, path in paths.items():
            click.echo(f"    {type_name}: {path}")
        return

    judge_label = judge_model or "PUGMARK_JUDGE_MODEL/.env"
    click.echo(f"[autolabel] judge={judge_label} n_calls={n_calls} min_votes={min_votes}")
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
cli.add_command(prompts_group, name="prompts")


if __name__ == "__main__":
    cli()
