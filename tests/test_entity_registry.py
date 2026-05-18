from __future__ import annotations

import pytest

from pugmark.entity_registry import (
    get_registered,
    register_entity_type,
    unregister_all,
)
from pugmark.entity_type import EntityTypeSpec


@pytest.fixture(autouse=True)
def _isolate() -> None:
    """Each test starts with the built-in defaults registered, nothing else."""
    unregister_all()
    # built-ins re-register on next import; force it here by re-running setup
    from pugmark import entity_registry as er

    er._install_defaults()
    yield


def test_built_in_defaults_registered() -> None:
    names = set(get_registered().keys())
    assert {"taxa", "people", "places"} <= names


def test_register_new_type() -> None:
    spec = EntityTypeSpec(
        name="wines",
        description="Wine bottles",
        wikidata_qclass="Q282",
        extraction_prompt_template="x",
        judge_prompt_template="x",
    )
    register_entity_type(spec)
    assert "wines" in get_registered()


def test_register_overrides_built_in() -> None:
    custom = EntityTypeSpec(
        name="taxa",
        description="my custom taxa spec",
        wikidata_qclass="Q16521",
        extraction_prompt_template="custom",
        judge_prompt_template="custom",
    )
    register_entity_type(custom)
    assert get_registered()["taxa"].description == "my custom taxa spec"


def test_register_lowercases_name() -> None:
    spec = EntityTypeSpec(
        name="Wines",
        description="x",
        wikidata_qclass=None,
        extraction_prompt_template="x",
        judge_prompt_template="x",
    )
    register_entity_type(spec)
    assert "wines" in get_registered()
    assert "Wines" not in get_registered()
