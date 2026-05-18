"""EntityTypeSpec — the per-type configuration that drives extraction.

Either Pugmark's analyzer proposes one of these from book content, or the user
registers one via `pugmark.register_entity_type()`. Specs are immutable value
objects; cache keys include `spec_version` so bumping the spec invalidates
cached extractions cleanly.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityTypeSpec(BaseModel):
    """Description of one entity type the pipeline should extract."""

    model_config = ConfigDict(frozen=True)

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


class BookType(BaseModel):
    """LLM-classified description of what kind of book this is.

    Used to bias the entity-type proposal (e.g., a mid-20th-century hunting
    memoir vs. a contemporary urban thriller call for very different entities).
    Sourced from preface/intro content when present, else first content chapter.
    """

    model_config = ConfigDict(frozen=True)

    genre: str = Field(..., description="e.g., 'natural history memoir', 'urban fantasy'")
    period: str = Field(..., description="e.g., 'mid-20th century', 'contemporary'")
    setting: str = Field(..., description="e.g., 'Indian subcontinent jungles', 'modern Tokyo'")
    themes: list[str] = Field(default_factory=list, description="3-7 high-level themes")
    target_reader: str = Field(default="", description="who this book is written for")
    summary: str = Field(default="", description="1-2 sentence pitch of the book")


class BookSchema(BaseModel):
    """The set of entity types Pugmark will extract from one specific book."""

    model_config = ConfigDict(frozen=True)

    book_id: str
    proposed_types: list[EntityTypeSpec]
    analyzer_version: str
    analyzed_at: datetime
    book_type: BookType | None = None
