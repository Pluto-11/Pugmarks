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


def test_offset_to_page_resolves_correctly() -> None:
    chapter = load_chapter(FIXTURE, chapter_number=1)
    assert chapter.offset_to_page(0) == 1
    last_offset = len(chapter.normalized_text) - 1
    assert chapter.offset_to_page(last_offset) >= 1
