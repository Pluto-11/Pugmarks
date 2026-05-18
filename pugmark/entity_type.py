"""EntityTypeSpec — the per-type configuration that drives extraction.

Either Pugmark's analyzer proposes one of these from book content, or the user
registers one via `pugmark.register_entity_type()`. Specs are immutable value
objects; cache keys include `spec_version` so bumping the spec invalidates
cached extractions cleanly.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class EntityTypeSpec(BaseModel):
    """Description of one entity type the pipeline should extract."""

    name: str = Field(..., min_length=1)
    description: str
    wikidata_qclass: str | None = None
    extraction_prompt_template: str
    judge_prompt_template: str
    min_book_occurrences: int = 2
    min_judge_votes: int = 2
    examples: list[str] = Field(default_factory=list)
    spec_version: str = "v1"

    @field_validator("name", mode="before")
    @classmethod
    def _lowercase(cls, v: object) -> object:
        return v.lower() if isinstance(v, str) else v


class BookSchema(BaseModel):
    """The set of entity types Pugmark will extract from one specific book."""

    book_id: str
    proposed_types: list[EntityTypeSpec]
    analyzer_version: str
    analyzed_at: datetime
