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


def test_hf_home_path_when_env_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SPACE_ID", raising=False)
    hf_home = tmp_path / "hfhome"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    cache = Cache.from_env()
    assert cache.root == hf_home / ".cache" / "pugmark"


def test_space_id_routes_to_data_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPACE_ID", "user/myspace")
    monkeypatch.delenv("HF_HOME", raising=False)
    # Avoid actually mkdir-ing /data on the dev host
    from pathlib import Path as _P
    from unittest.mock import patch as _patch
    with _patch.object(_P, "mkdir"):
        cache = Cache.from_env()
    assert cache.root == _P("/data/.cache/pugmark")


def test_get_returns_none_on_corrupted_file(cache: Cache, tmp_path: Path) -> None:
    cache.set("stage", "key", Toy(value=1))
    (tmp_path / "stage" / "key.json").write_text("not json")
    assert cache.get("stage", "key", Toy) is None


def test_clear_stage_only_removes_that_stage(cache: Cache) -> None:
    cache.set("s1", "k", Toy(value=1))
    cache.set("s2", "k", Toy(value=2))
    cache.clear("s1")
    assert cache.get("s1", "k", Toy) is None
    assert cache.get("s2", "k", Toy) == Toy(value=2)


def test_clear_all_removes_everything(cache: Cache) -> None:
    cache.set("s1", "k", Toy(value=1))
    cache.set("s2", "k", Toy(value=2))
    cache.clear()
    assert cache.get("s1", "k", Toy) is None
    assert cache.get("s2", "k", Toy) is None
