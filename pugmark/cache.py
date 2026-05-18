"""File-based cache with version-keyed invalidation.

Each cached entry is a JSON file at `{root}/{stage}/{hash}.json`. Hash composition
(input + version) ensures version bumps invalidate cleanly with no manual cleanup.

Storage location:
- Local: ~/.cache/pugmark/
- HuggingFace Spaces: /data/.cache/pugmark/ (detected via HF_HOME env var)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Cache:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return cls(root=Path(hf_home) / ".cache" / "pugmark")
        return cls(root=Path.home() / ".cache" / "pugmark")

    @staticmethod
    def compute_hash(*parts: str) -> str:
        h = hashlib.sha256()
        for part in parts:
            h.update(part.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()[:16]

    def _path(self, stage: str, key: str) -> Path:
        return self.root / stage / f"{key}.json"

    def get(self, stage: str, key: str, model_cls: type[T]) -> T | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        return model_cls.model_validate_json(path.read_text())

    def set(self, stage: str, key: str, value: BaseModel) -> None:
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value.model_dump_json(indent=2))

    def clear(self, stage: str | None = None) -> None:
        """Remove cached entries. If stage is given, only that stage; else all."""
        import shutil

        target = self.root / stage if stage else self.root
        if target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
