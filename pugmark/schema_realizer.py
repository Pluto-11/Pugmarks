"""Schema realizer — merge analyzer proposals + registry + user overrides.

Precedence (highest to lowest):
  1. user overrides (passed via `overrides=`)
  2. registered types (entity_registry._REGISTERED)
  3. analyzer-proposed types (with generic templates)

If `force_types` is given, the output is restricted to that name set; force_types
may include registered names that the analyzer did NOT propose (they will be
added from the registry).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from pugmark.entity_registry import get_registered
from pugmark.entity_type import BookSchema, EntityTypeSpec

logger = logging.getLogger(__name__)


def realize_schema(
    book_schema: BookSchema,
    *,
    overrides: dict[str, EntityTypeSpec] | None = None,
    force_types: Iterable[str] | None = None,
) -> list[EntityTypeSpec]:
    """Realize the final list of EntityTypeSpec for one book."""
    overrides = overrides or {}
    registered = get_registered()

    merged: dict[str, EntityTypeSpec] = {}
    for t in book_schema.proposed_types:
        merged[t.name] = t
    for name, spec in registered.items():
        if name in merged:
            merged[name] = spec  # registered wins over proposed
    for name, spec in overrides.items():
        merged[name.lower()] = spec  # user override wins over both

    if force_types is not None:
        wanted = {n.lower() for n in force_types}
        # Add registered types in `wanted` that aren't in merged yet
        for n in wanted - set(merged.keys()):
            if n in registered:
                merged[n] = registered[n]
            elif n in overrides:
                merged[n] = overrides[n]
            else:
                logger.warning(
                    f"force_type {n!r} not registered or proposed; skipping"
                )
        merged = {k: v for k, v in merged.items() if k in wanted}

    return list(merged.values())
