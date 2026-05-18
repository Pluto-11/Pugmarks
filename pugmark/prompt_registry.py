"""Versioned prompt loader with Langfuse-first + local fallback.

Resolution order, driven by PUGMARK_PROMPTS_SOURCE env:

  local           : in-repo .j2 files only (default in tests / CI / offline dev)
  langfuse_only   : Langfuse only — raise if missing or Langfuse unconfigured
  langfuse_first  : Try Langfuse; if missing / error, fall back to local file

Defaults to 'langfuse_first' if Langfuse credentials are present, else 'local'.

In-process LRU cache means each (name, version) is fetched at most once per
process — Langfuse round-trip cost is amortized across many LLM calls.

Langfuse model:
  - prompt name  = file stem  (e.g. "extract_taxa")
  - label        = version    (e.g. "v1")
  - prompt body  = the .j2 template text; rendering is still done by us
                   via Jinja2 (Langfuse stores it as plain text).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Template

logger = logging.getLogger(__name__)

ALLOWED_SOURCES = ("local", "langfuse_only", "langfuse_first")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template_text: str
    source: str  # "local" or "langfuse"

    def render(self, **kwargs: object) -> str:
        return Template(self.template_text).render(**kwargs)


def _default_source() -> str:
    """Default to langfuse_first if creds are set, else local.

    The env knob PUGMARK_PROMPTS_SOURCE overrides this.
    """
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        return "langfuse_first"
    return "local"


def _resolve_source() -> str:
    src = os.environ.get("PUGMARK_PROMPTS_SOURCE", "").strip() or _default_source()
    if src not in ALLOWED_SOURCES:
        logger.warning(
            f"PUGMARK_PROMPTS_SOURCE={src!r} invalid; falling back to 'local'. "
            f"Valid: {ALLOWED_SOURCES}"
        )
        return "local"
    return src


def _is_langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


@lru_cache(maxsize=64)
def _fetch_langfuse(name: str, version: str) -> str | None:
    """Fetch one prompt body from Langfuse by (name, label=version).

    Returns None when not found or when Langfuse is unconfigured / errors.
    Cached for process lifetime — bump version to invalidate.
    """
    if not _is_langfuse_configured():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("langfuse SDK not installed; cannot fetch prompts")
        return None
    try:
        client = Langfuse()
        prompt = client.get_prompt(name, label=version, max_retries=0)
        if prompt is None:
            return None
        # langfuse v2 exposes the body via `.prompt` (string for text prompts).
        body = getattr(prompt, "prompt", None)
        if isinstance(body, str):
            return body
        # Some shapes wrap text inside `.compile()` or `.get_langchain_prompt()`.
        # For a text-type prompt the raw .prompt attribute is correct.
        logger.warning(f"langfuse prompt {name}@{version} has unexpected shape: {type(body)}")
        return None
    except Exception as e:  # broad: don't kill caller for Langfuse hiccups
        logger.warning(f"langfuse fetch failed for {name}@{version}: {e!r}")
        return None


class PromptRegistry:
    """Loads versioned Jinja2 prompts from Langfuse and/or local files.

    Args:
        in_repo_dir: directory containing `<name>.<version>.j2` files used as
            local source of truth and as the fallback when Langfuse is the
            primary source.
    """

    _FILENAME_RE = re.compile(r"^(?P<name>.+)\.(?P<version>v\d+)\.j2$")

    def __init__(self, in_repo_dir: Path) -> None:
        self.in_repo_dir = in_repo_dir

    # ---- local file discovery (also used by the seed CLI) -------------------

    def _local_candidates(self, name: str) -> list[tuple[str, Path]]:
        """Return [(version, path), ...] for in-repo files matching `name`."""
        out: list[tuple[str, Path]] = []
        for path in self.in_repo_dir.glob(f"{name}.*.j2"):
            m = self._FILENAME_RE.match(path.name)
            if m and m.group("name") == name:
                out.append((m.group("version"), path))
        return sorted(out, key=lambda p: int(p[0][1:]))

    def discover_local(self) -> list[tuple[str, str, Path]]:
        """Walk in-repo dir; return [(name, version, path)] for every .j2 file."""
        found: list[tuple[str, str, Path]] = []
        for path in self.in_repo_dir.glob("*.j2"):
            m = self._FILENAME_RE.match(path.name)
            if m:
                found.append((m.group("name"), m.group("version"), path))
        return sorted(found)

    def _local_get(self, name: str, version: str | None) -> Prompt:
        candidates = self._local_candidates(name)
        if not candidates:
            raise FileNotFoundError(f"no local prompt named {name!r} in {self.in_repo_dir}")
        if version is None:
            chosen_version, chosen_path = candidates[-1]
        else:
            matches = [c for c in candidates if c[0] == version]
            if not matches:
                raise FileNotFoundError(f"no local prompt {name!r} version {version!r}")
            chosen_version, chosen_path = matches[0]
        return Prompt(
            name=name,
            version=chosen_version,
            template_text=chosen_path.read_text(),
            source="local",
        )

    # ---- public ------------------------------------------------------------

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Resolve a prompt according to PUGMARK_PROMPTS_SOURCE.

        For `langfuse_first` and `langfuse_only`, a specific `version` is
        required at the Langfuse side; if None, we resolve the latest local
        version first, then request that version from Langfuse.
        """
        source = _resolve_source()
        # If no explicit version, fall back to the latest local one — we need
        # *some* label to request from Langfuse.
        if version is None:
            try:
                local = self._local_get(name, None)
                effective_version = local.version
            except FileNotFoundError:
                effective_version = "v1"
                local = None  # type: ignore[assignment]
        else:
            effective_version = version
            local = None  # type: ignore[assignment]

        if source == "local":
            return local if local is not None else self._local_get(name, effective_version)

        body = _fetch_langfuse(name, effective_version)
        if body is not None:
            return Prompt(
                name=name,
                version=effective_version,
                template_text=body,
                source="langfuse",
            )

        if source == "langfuse_only":
            raise FileNotFoundError(
                f"PUGMARK_PROMPTS_SOURCE=langfuse_only but {name}@{effective_version} "
                f"not in Langfuse (or Langfuse unconfigured)"
            )

        # langfuse_first → fall back to local
        logger.info(
            f"langfuse miss for {name}@{effective_version}; falling back to local file"
        )
        return local if local is not None else self._local_get(name, effective_version)
