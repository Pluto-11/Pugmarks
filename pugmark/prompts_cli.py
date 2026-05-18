"""`pugmark prompts {seed,diff,pull,list}` — manage prompts in Langfuse.

Source of truth design:
  - In-repo .j2 files are the source of truth for what *should* be in Langfuse.
  - `seed` pushes local files → Langfuse (overwriting if drift exists).
  - `pull` pulls Langfuse → local files (use when Langfuse was edited via UI).
  - `diff` reports drift without changing either side.
  - `list` shows what's in Langfuse for inspection.

Naming convention:
  - Prompt name in Langfuse  = file stem ("extract_taxa")
  - Label in Langfuse        = file version ("v1")
  - Tag                      = role inferred from prefix:
      extract_*    → "extract"
      judge_*      → "judge"
      summarize_*  → "summarize"
      book_*       → "analyzer"
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from pugmark.prompt_registry import PromptRegistry, _fetch_langfuse, _is_langfuse_configured

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS_DIR = Path("prompts")


def _infer_role_tag(name: str) -> str:
    if name.startswith("extract_"):
        return "extract"
    if name.startswith("judge_"):
        return "judge"
    if name.startswith("summarize_"):
        return "summarize"
    if name.startswith("book_"):
        return "analyzer"
    return "unknown"


def _require_langfuse_or_abort() -> None:
    if not _is_langfuse_configured():
        click.echo(
            "Langfuse credentials missing — set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env",
            err=True,
        )
        sys.exit(2)


@click.group()
def prompts() -> None:
    """Manage Pugmark prompts in Langfuse."""


@prompts.command("list")
@click.option(
    "--prompts-dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    default=DEFAULT_PROMPTS_DIR,
    show_default=True,
)
def list_cmd(prompts_dir: Path) -> None:
    """List local + Langfuse prompts side by side."""
    _require_langfuse_or_abort()
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    local = reg.discover_local()
    click.echo(f"{'name':<22} {'ver':<5} {'local':<7} {'langfuse'}")
    click.echo("-" * 60)
    for name, ver, path in local:
        # Check Langfuse presence (don't print the body — just hit/miss)
        # Bypass the LRU since the user might be debugging
        _fetch_langfuse.cache_clear()
        body = _fetch_langfuse(name, ver)
        click.echo(
            f"{name:<22} {ver:<5} {'✓ ' + str(path.stat().st_size) + 'B':<7} "
            f"{'✓ ' + str(len(body)) + 'B' if body else 'missing'}"
        )


@prompts.command("seed")
@click.option(
    "--prompts-dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    default=DEFAULT_PROMPTS_DIR,
    show_default=True,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be pushed without actually pushing.",
)
def seed_cmd(prompts_dir: Path, dry_run: bool) -> None:
    """Push every local prompt to Langfuse (name + version)."""
    _require_langfuse_or_abort()
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    found = reg.discover_local()
    if not found:
        click.echo(f"No prompts found in {prompts_dir}", err=True)
        sys.exit(1)

    if dry_run:
        click.echo(f"[dry-run] Would seed {len(found)} prompt(s):")
        for name, ver, path in found:
            click.echo(f"  {name}@{ver}  ({path.stat().st_size}B, tag={_infer_role_tag(name)})")
        return

    from langfuse import Langfuse  # imported lazily so tests don't need it

    client = Langfuse()
    pushed = 0
    for name, ver, path in found:
        body = path.read_text()
        tag = _infer_role_tag(name)
        try:
            client.create_prompt(
                name=name,
                prompt=body,
                labels=[ver, "production"] if ver == "v1" else [ver],
                tags=["pugmark", tag],
                type="text",
            )
            click.echo(f"  ✓ {name}@{ver}  ({len(body)}B, tag={tag})")
            pushed += 1
        except Exception as e:
            click.echo(f"  ✗ {name}@{ver}  failed: {e!r}", err=True)
    click.echo(f"Seeded {pushed}/{len(found)} prompts to Langfuse.")


@prompts.command("diff")
@click.option(
    "--prompts-dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    default=DEFAULT_PROMPTS_DIR,
    show_default=True,
)
def diff_cmd(prompts_dir: Path) -> None:
    """Report drift between local files and Langfuse content."""
    _require_langfuse_or_abort()
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    found = reg.discover_local()
    drift = 0
    for name, ver, path in found:
        local_body = path.read_text()
        _fetch_langfuse.cache_clear()
        lf_body = _fetch_langfuse(name, ver)
        if lf_body is None:
            click.echo(f"  {name}@{ver}: missing from Langfuse")
            drift += 1
        elif lf_body != local_body:
            click.echo(
                f"  {name}@{ver}: DRIFT (local={len(local_body)}B, langfuse={len(lf_body)}B)"
            )
            drift += 1
        else:
            click.echo(f"  {name}@{ver}: in sync")
    click.echo(f"\nTotal drift: {drift}/{len(found)} prompts.")
    if drift:
        sys.exit(1)


@prompts.command("pull")
@click.option(
    "--prompts-dir",
    type=click.Path(exists=True, path_type=Path, file_okay=False),
    default=DEFAULT_PROMPTS_DIR,
    show_default=True,
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be written without actually writing.",
)
def pull_cmd(prompts_dir: Path, dry_run: bool) -> None:
    """Fetch all known prompts from Langfuse → overwrite local files."""
    _require_langfuse_or_abort()
    reg = PromptRegistry(in_repo_dir=prompts_dir)
    found = reg.discover_local()
    updated = 0
    for name, ver, path in found:
        _fetch_langfuse.cache_clear()
        lf_body = _fetch_langfuse(name, ver)
        if lf_body is None:
            click.echo(f"  {name}@{ver}: missing from Langfuse — skipping")
            continue
        if path.read_text() == lf_body:
            click.echo(f"  {name}@{ver}: already in sync")
            continue
        if dry_run:
            click.echo(f"  [dry-run] would overwrite {path} ({len(lf_body)}B)")
        else:
            path.write_text(lf_body)
            click.echo(f"  ✓ {name}@{ver}: pulled {len(lf_body)}B → {path}")
        updated += 1
    click.echo(f"\nPulled {updated} prompt(s).")
