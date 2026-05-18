"""Versioned prompt loader.

Source of truth: in-repo Jinja2 files at prompts/{name}.{version}.j2
Future: Langfuse fetch overlay (deferred to v2 of this module).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template_text: str

    def render(self, **kwargs: object) -> str:
        return Template(self.template_text).render(**kwargs)


class PromptRegistry:
    _FILENAME_RE = re.compile(r"^(?P<name>.+)\.(?P<version>v\d+)\.j2$")

    def __init__(self, in_repo_dir: Path) -> None:
        self.in_repo_dir = in_repo_dir

    def _candidates(self, name: str) -> list[tuple[str, Path]]:
        """Return list of (version, path) for the given prompt name."""
        out = []
        for path in self.in_repo_dir.glob(f"{name}.*.j2"):
            m = self._FILENAME_RE.match(path.name)
            if m and m.group("name") == name:
                out.append((m.group("version"), path))
        return sorted(out, key=lambda p: int(p[0][1:]))

    def get(self, name: str, version: str | None = None) -> Prompt:
        candidates = self._candidates(name)
        if not candidates:
            raise FileNotFoundError(f"no prompt named {name!r} in {self.in_repo_dir}")
        if version is None:
            chosen_version, chosen_path = candidates[-1]  # highest
        else:
            matches = [c for c in candidates if c[0] == version]
            if not matches:
                raise FileNotFoundError(f"no prompt {name!r} version {version!r}")
            chosen_version, chosen_path = matches[0]
        return Prompt(
            name=name, version=chosen_version, template_text=chosen_path.read_text()
        )
