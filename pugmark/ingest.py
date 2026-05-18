"""PDF → Chapter ingestion via PyMuPDF.

list_chapters(pdf) returns metadata about each chapter detected via outline.
load_chapter(pdf, chapter_number) returns a fully populated Chapter.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF

from pugmark.schemas import Chapter

INGEST_VERSION = "v1"

# End-of-line hyphenation: "pan-\nther" → "panther"
_HYPH_RE = re.compile(r"-\n([a-z])")
# Smart quotes/dashes
_SMART_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)
_WHITESPACE_RE = re.compile(r"[ \t]+")


class ChapterInfo(TypedDict):
    number: int
    title: str
    page_start: int
    page_end: int


def _normalize(text: str) -> str:
    text = _HYPH_RE.sub(r"\1", text)
    text = text.translate(_SMART_MAP)
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def list_chapters(pdf_path: Path) -> list[ChapterInfo]:
    """Detect chapters via PDF outline (TOC bookmarks)."""
    doc = fitz.open(pdf_path)
    try:
        toc = doc.get_toc()
    finally:
        doc.close()

    if not toc:
        return []

    out: list[ChapterInfo] = []
    for i, entry in enumerate(toc):
        _level, title, page_start = entry
        if i + 1 < len(toc):
            page_end = toc[i + 1][2] - 1
        else:
            try:
                page_end = doc.page_count
            except (ValueError, AttributeError):
                page_end = page_start
        # We need page_end for the last chapter; reopen briefly to avoid using closed doc
        out.append(
            ChapterInfo(
                number=i + 1, title=title, page_start=page_start, page_end=page_end
            )
        )
    # Fix last chapter's page_end since we closed the doc above
    doc2 = fitz.open(pdf_path)
    try:
        if out:
            out[-1]["page_end"] = doc2.page_count
    finally:
        doc2.close()
    return out


def load_chapter(pdf_path: Path, chapter_number: int) -> Chapter:
    chapters = list_chapters(pdf_path)
    target = next((c for c in chapters if c["number"] == chapter_number), None)
    if target is None:
        raise ValueError(f"chapter {chapter_number} not found in {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        raw_pages: list[str] = []
        normalized_pages: list[str] = []
        for page_num in range(target["page_start"] - 1, target["page_end"]):
            page = doc.load_page(page_num)
            raw = page.get_text("text")
            raw_pages.append(raw)
            normalized_pages.append(_normalize(raw))
    finally:
        doc.close()

    raw_text = "".join(raw_pages)
    # Compute page_offsets as we concatenate normalized pages
    normalized_text_parts: list[str] = []
    page_offsets: list[int] = []
    cursor = 0
    for np_text in normalized_pages:
        page_offsets.append(cursor)
        normalized_text_parts.append(np_text)
        cursor += len(np_text)
    normalized_text = "".join(normalized_text_parts)

    return Chapter(
        book=pdf_path.stem,
        number=target["number"],
        title=target["title"],
        source_pdf=pdf_path,
        page_start=target["page_start"],
        page_end=target["page_end"],
        raw_text=raw_text,
        normalized_text=normalized_text,
        page_offsets=page_offsets,
        ingest_version=INGEST_VERSION,
    )
