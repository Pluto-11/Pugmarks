"""Plugin registry for EntityTypeSpec.

Pugmark ships built-in specs for taxa, people, places at import time. Users
can call `register_entity_type()` to add their own; calling it with a name
that already exists overrides the registered spec.

Prompt templates flow through pugmark.prompt_registry — that means in production
they'll be fetched from Langfuse (with local fallback), in tests from local
files. See pugmark/prompt_registry.py for the resolution rules.
"""
from __future__ import annotations

from pathlib import Path

from pugmark.entity_type import EntityTypeSpec
from pugmark.prompt_registry import PromptRegistry

_REGISTERED: dict[str, EntityTypeSpec] = {}
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def register_entity_type(spec: EntityTypeSpec) -> None:
    """Register a user-defined entity type. Overrides any prior registration."""
    _REGISTERED[spec.name] = spec


def get_registered() -> dict[str, EntityTypeSpec]:
    """Return a copy of the current registry."""
    return dict(_REGISTERED)


def unregister_all() -> None:
    """Clear the registry. Tests only."""
    _REGISTERED.clear()


def _load_via_registry(name: str) -> str:
    """Resolve prompt body via PromptRegistry (Langfuse-first when configured)."""
    return PromptRegistry(in_repo_dir=_PROMPTS_DIR).get(name).template_text


def _install_defaults() -> None:
    """Register the built-in taxa/people/places types.

    Prompts come from PromptRegistry (Langfuse → local fallback). Missing
    prompts in BOTH Langfuse and disk raise FileNotFoundError so a degraded
    install can't silently ship without prompts.
    """
    register_entity_type(
        EntityTypeSpec(
            name="taxa",
            description="Animals, plants, and fungi (Wikidata Q16521 taxa)",
            wikidata_qclass="Q16521",
            extraction_prompt_template=_load_via_registry("extract_taxa"),
            judge_prompt_template=_load_via_registry("judge_taxa"),
        )
    )
    register_entity_type(
        EntityTypeSpec(
            name="people",
            description="Real or fictional persons (Wikidata Q5 + Q15632617)",
            wikidata_qclass="Q5",
            extraction_prompt_template=_load_via_registry("extract_people"),
            judge_prompt_template=_load_via_registry("judge_people"),
        )
    )
    register_entity_type(
        EntityTypeSpec(
            name="places",
            description="Real or fictional locations (Wikidata Q486972 + Q17334923)",
            wikidata_qclass="Q486972",
            extraction_prompt_template=_load_via_registry("extract_places"),
            judge_prompt_template=_load_via_registry("judge_places"),
        )
    )


# Register defaults on import.
_install_defaults()
