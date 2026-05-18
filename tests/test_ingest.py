from __future__ import annotations

from pathlib import Path

from pugmark.ingest import list_chapters, load_chapter
from pugmark.schemas import Chapter

FIXTURE = Path(__file__).parent / "fixtures" / "sample_chapter.pdf"


def test_list_chapters_returns_titles() -> None:
    chapters = list_chapters(FIXTURE)
    assert len(chapters) == 1
    assert chapters[0]["title"] == "The Black Panther of Sivanipalli"
    assert chapters[0]["page_start"] == 1


def test_load_chapter_returns_chapter_with_normalized_text() -> None:
    chapter = load_chapter(FIXTURE, chapter_number=1)
    assert isinstance(chapter, Chapter)
    assert chapter.title == "The Black Panther of Sivanipalli"
    assert "panther" in chapter.normalized_text.lower()
    assert "sambhur" in chapter.normalized_text.lower()
    assert "peepul" in chapter.normalized_text.lower()


def test_chapter_dehyphenates_line_wraps(tmp_path: Path) -> None:
    import fitz

    p = tmp_path / "h.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The pan-\nther stood there.", fontsize=11)
    doc.set_toc([[1, "Test", 1]])
    doc.save(p)
    chapter = load_chapter(p, chapter_number=1)
    assert "panther" in chapter.normalized_text


def test_offset_to_page_maps_page_boundaries() -> None:
    chapter = load_chapter(FIXTURE, chapter_number=1)
    assert chapter.offset_to_page(0) == 1
    # Last char of page 1 should still map to page 1
    boundary = chapter.page_offsets[1]
    assert chapter.offset_to_page(boundary - 1) == 1
    # First char of page 2 should map to page 2
    assert chapter.offset_to_page(boundary) == 2


def test_list_chapters_ignores_subsection_toc_entries(tmp_path: Path) -> None:
    import fitz

    p = tmp_path / "multi_level.pdf"
    doc = fitz.open()
    for _ in range(4):
        page = doc.new_page()
        page.insert_text((72, 72), "x", fontsize=11)
    doc.set_toc(
        [
            [1, "Chapter One", 1],
            [2, "Section 1.1", 2],
            [1, "Chapter Two", 3],
            [2, "Section 2.1", 4],
        ]
    )
    doc.save(p)
    chapters = list_chapters(p)
    assert [c["title"] for c in chapters] == ["Chapter One", "Chapter Two"]
    assert chapters[0]["page_start"] == 1
    assert chapters[0]["page_end"] == 2  # one before next level-1 entry
    assert chapters[1]["page_start"] == 3
    assert chapters[1]["page_end"] == 4  # page_count
