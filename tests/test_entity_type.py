from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from pugmark.entity_type import BookSchema, EntityTypeSpec


def test_entity_type_spec_defaults() -> None:
    spec = EntityTypeSpec(
        name="people",
        description="Real or fictional persons mentioned in the book",
        wikidata_qclass="Q5",
        extraction_prompt_template="Extract people from {{ chapter_text }}",
        judge_prompt_template="Judge people from {{ chapter_text }}",
    )
    assert spec.name == "people"
    assert spec.min_book_occurrences == 2
    assert spec.min_judge_votes == 2
    assert spec.spec_version == "v1"
    assert spec.examples == []


def test_entity_type_spec_no_qclass() -> None:
    spec = EntityTypeSpec(
        name="factions",
        description="In-book political factions",
        wikidata_qclass=None,
        extraction_prompt_template="...",
        judge_prompt_template="...",
    )
    assert spec.wikidata_qclass is None


def test_entity_type_spec_name_is_normalized_lower() -> None:
    """Name should round-trip lowercase for registry lookups."""
    spec = EntityTypeSpec(
        name="People",
        description="x",
        wikidata_qclass="Q5",
        extraction_prompt_template="x",
        judge_prompt_template="x",
    )
    assert spec.name == "people"


def test_entity_type_spec_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        EntityTypeSpec(
            name="",
            description="x",
            wikidata_qclass=None,
            extraction_prompt_template="x",
            judge_prompt_template="x",
        )


def test_book_schema_holds_proposals() -> None:
    s1 = EntityTypeSpec(
        name="people",
        description="x",
        wikidata_qclass="Q5",
        extraction_prompt_template="x",
        judge_prompt_template="x",
    )
    s2 = EntityTypeSpec(
        name="places",
        description="x",
        wikidata_qclass="Q486972",
        extraction_prompt_template="x",
        judge_prompt_template="x",
    )
    bs = BookSchema(
        book_id="anna-karenina",
        proposed_types=[s1, s2],
        analyzer_version="v1",
        analyzed_at=datetime.now(),
    )
    assert {t.name for t in bs.proposed_types} == {"people", "places"}
