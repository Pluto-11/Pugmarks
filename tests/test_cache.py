from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from pugmark.cache import Cache


class Toy(BaseModel):
    value: int


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(root=tmp_path)


def test_set_and_get_roundtrip(cache: Cache) -> None:
    cache.set("stage", "abc123", Toy(value=42))
    assert cache.get("stage", "abc123", Toy) == Toy(value=42)


def test_miss_returns_none(cache: Cache) -> None:
    assert cache.get("stage", "missing", Toy) is None


def test_key_isolation_across_stages(cache: Cache) -> None:
    cache.set("a", "k", Toy(value=1))
    cache.set("b", "k", Toy(value=2))
    assert cache.get("a", "k", Toy) == Toy(value=1)
    assert cache.get("b", "k", Toy) == Toy(value=2)


def test_compute_hash_is_stable() -> None:
    h1 = Cache.compute_hash("hello", "v1")
    h2 = Cache.compute_hash("hello", "v1")
    assert h1 == h2


def test_compute_hash_differs_on_version_bump() -> None:
    assert Cache.compute_hash("hello", "v1") != Cache.compute_hash("hello", "v2")


def test_hf_data_path_when_env_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    cache = Cache.from_env()
    root_str = str(cache.root)
    assert "hfhome" in root_str or "/data" in root_str or str(tmp_path) in root_str
