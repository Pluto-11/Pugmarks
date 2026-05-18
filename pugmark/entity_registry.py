"""Plugin registry for EntityTypeSpec.

Pugmark ships built-in specs for taxa, people, places at import time. Users
can call `register_entity_type()` to add their own; calling it with a name
that already exists overrides the registered spec.
"""
from __future__ import annotations

from pathlib import Path

from pugmark.entity_type import EntityTypeSpec

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


def _read_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _install_defaults() -> None:
    """Register the built-in taxa/people/places types.

    Reads prompt templates from disk; missing files raise FileNotFoundError so
    a degraded install can't silently ship without prompts.
    """
    register_entity_type(
        EntityTypeSpec(
            name="taxa",
            description="Animals, plants, and fungi (Wikidata Q16521 taxa)",
            wikidata_qclass="Q16521",
            extraction_prompt_template=_read_prompt("extract_taxa.v1.j2"),
            judge_prompt_template=_read_prompt("judge_taxa.v1.j2"),
        )
    )
    register_entity_type(
        EntityTypeSpec(
            name="people",
            description="Real or fictional persons (Wikidata Q5 + Q15632617)",
            wikidata_qclass="Q5",
            extraction_prompt_template=_read_prompt("extract_people.v1.j2"),
            judge_prompt_template=_read_prompt("judge_people.v1.j2"),
        )
    )
    register_entity_type(
        EntityTypeSpec(
            name="places",
            description="Real or fictional locations (Wikidata Q486972 + Q17334923)",
            wikidata_qclass="Q486972",
            extraction_prompt_template=_read_prompt("extract_places.v1.j2"),
            judge_prompt_template=_read_prompt("judge_places.v1.j2"),
        )
    )


# Register defaults on import. People/places prompts must exist in `prompts/`
# at the time this module is imported (they do, courtesy of T4).
_install_defaults()
