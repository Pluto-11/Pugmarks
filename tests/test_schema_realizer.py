from __future__ import annotations

from datetime import datetime

import pytest

from pugmark.entity_registry import unregister_all
from pugmark.entity_type import BookSchema, EntityTypeSpec
from pugmark.schema_realizer import realize_schema


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    unregister_all()
    from pugmark import entity_registry as er

    er._install_defaults()
    yield


def _spec(name: str, qclass: str | None = "Q1") -> EntityTypeSpec:
    return EntityTypeSpec(
        name=name,
        description=f"description for {name}",
        wikidata_qclass=qclass,
        extraction_prompt_template=f"extract {name}",
        judge_prompt_template=f"judge {name}",
    )


def _book_schema(specs: list[EntityTypeSpec]) -> BookSchema:
    return BookSchema(
        book_id="b",
        proposed_types=specs,
        analyzer_version="v1",
        analyzed_at=datetime.now(),
    )


def test_proposed_only_when_not_registered() -> None:
    """A proposed type with no registered match passes through unchanged."""
    proposed = _spec("recipes", qclass=None)
    realized = realize_schema(_book_schema([proposed]))
    assert {s.name for s in realized} == {"recipes"}
    assert realized[0].description == "description for recipes"


def test_registered_overrides_proposed() -> None:
    """If analyzer proposes 'taxa', the built-in registered spec wins."""
    proposed = _spec("taxa")
    realized = realize_schema(_book_schema([proposed]))
    realized_taxa = next(s for s in realized if s.name == "taxa")
    # Built-in description starts with "Animals + plants…", not our test value.
    assert realized_taxa.description != "description for taxa"


def test_user_override_beats_registered() -> None:
    """An override passed explicitly wins over both registered and proposed."""
    user_taxa = EntityTypeSpec(
        name="taxa",
        description="user-supplied taxa",
        wikidata_qclass="Q16521",
        extraction_prompt_template="user extract",
        judge_prompt_template="user judge",
    )
    proposed = _spec("taxa")
    realized = realize_schema(
        _book_schema([proposed]), overrides={"taxa": user_taxa}
    )
    realized_taxa = next(s for s in realized if s.name == "taxa")
    assert realized_taxa.description == "user-supplied taxa"
    assert realized_taxa.extraction_prompt_template == "user extract"


def test_force_types_filter() -> None:
    """force_types restricts output to a subset."""
    proposed = [_spec("taxa"), _spec("recipes", qclass=None), _spec("people")]
    realized = realize_schema(
        _book_schema(proposed), force_types=["taxa", "recipes"]
    )
    assert {s.name for s in realized} == {"taxa", "recipes"}


def test_force_types_adds_registered_not_proposed() -> None:
    """force_types can add a registered type even if analyzer didn't propose it."""
    proposed = [_spec("taxa")]
    realized = realize_schema(
        _book_schema(proposed), force_types=["taxa", "people"]
    )
    assert {s.name for s in realized} == {"taxa", "people"}
