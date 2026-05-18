"""Pugmark CLI — Click-based.

Subcommands:
  chapters <pdf>                       List chapter titles + page ranges
  extract <pdf> --chapter N --out F    Run pipeline, write HTML gallery to F
  eval --chapter NAME                  Run eval against ground truth
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

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


if __name__ == "__main__":
    cli()
