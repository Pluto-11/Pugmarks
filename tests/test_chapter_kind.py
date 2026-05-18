"""Tests for chapter-kind heuristic classification.

The classifier looks at title patterns + page count + position-in-book to label
each TOC entry as 'content', 'front', or 'back'. This lets the analyzer skip
cover/contents pages and use preface/intro text for book-type classification.
"""
from __future__ import annotations

import pytest

from pugmark.ingest import classify_chapter_kind


def _entries(*items: tuple[str, int, int]) -> list[dict]:
    """Helper: build a list of TOC-like dicts."""
    return [
        {"number": i + 1, "title": t, "page_start": s, "page_end": e}
        for i, (t, s, e) in enumerate(items)
    ]


# ---- title-based front-matter -----------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Cover",
        "Title Page",
        "Copyright",
        "Dedication",
        "Contents",
        "Table of Contents",
        "Foreword",
        "FOREWORD",
        "Preface",
        "Introduction",
        "Prologue",
        "List of Illustrations",
        "Acknowledgments",
        "Acknowledgements",  # British spelling
    ],
)
def test_front_matter_title_patterns(title: str) -> None:
    chapters = _entries(("The Kenneth Anderson Omnibus", 1, 1), (title, 2, 2))
    classified = classify_chapter_kind(chapters)
    # Note: chapter 1 here is itself front matter (cover-like), index 1 is the actual test
    assert classified[1]["kind"] == "front", f"{title!r} should be front-matter"


# ---- title-based back-matter ------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Epilogue",
        "Afterword",
        "Appendix",
        "Appendix A: References",
        "Index",
        "Bibliography",
        "References",
        "Notes",
        "About the Author",
        "Author",
        "Author Biography",
    ],
)
def test_back_matter_title_patterns(title: str) -> None:
    chapters = _entries(
        ("Story One", 5, 50),
        ("Story Two", 51, 100),
        (title, 101, 110),
    )
    classified = classify_chapter_kind(chapters)
    assert classified[-1]["kind"] == "back", f"{title!r} should be back-matter"


# ---- content classification -------------------------------------------------


def test_long_chapter_with_unrecognized_title_is_content() -> None:
    chapters = _entries(
        ("Nine Man-Eaters and One Rogue", 4, 179),
        ("Man-Eaters and Jungle Killers", 180, 301),
    )
    classified = classify_chapter_kind(chapters)
    assert all(c["kind"] == "content" for c in classified)


def test_short_chapter_in_middle_is_still_content() -> None:
    """A short chapter sandwiched between long ones is a real (short) chapter."""
    chapters = _entries(
        ("The Tiger Roars", 5, 100),
        ("Interlude", 101, 102),  # 2 pages but middle position
        ("Tales from the Jungle", 103, 200),
    )
    classified = classify_chapter_kind(chapters)
    assert classified[1]["kind"] == "content"


# ---- combined heuristics ----------------------------------------------------


def test_anderson_omnibus_layout() -> None:
    """The real Anderson Omnibus layout — cover, contents, sub-books, author."""
    chapters = _entries(
        ("The Kenneth Anderson Omnibus", 2, 2),
        ("Contents", 3, 3),
        ("Nine Man-Eaters and One Rogue", 4, 179),
        ("Man-Eaters and Jungle Killers", 180, 301),
        ("The Bond of Love", 1889, 1893),
        ("Author", 1894, 1895),
    )
    classified = classify_chapter_kind(chapters)
    kinds = [c["kind"] for c in classified]
    assert kinds == ["front", "front", "content", "content", "content", "back"]


def test_first_short_chapter_without_pattern_is_tentative_front() -> None:
    """Position-based: first chapter that's very short + unknown title → front."""
    chapters = _entries(
        ("Untitled", 1, 1),  # no known pattern, but 1 page, position 0
        ("Real Chapter", 2, 100),
    )
    classified = classify_chapter_kind(chapters)
    assert classified[0]["kind"] == "front"


def test_last_short_chapter_without_pattern_is_tentative_back() -> None:
    chapters = _entries(
        ("Chapter 1", 1, 100),
        ("Chapter 2", 101, 200),
        ("Unknown End", 201, 202),  # 2 pages, last position
    )
    classified = classify_chapter_kind(chapters)
    assert classified[-1]["kind"] == "back"


def test_empty_input_returns_empty() -> None:
    assert classify_chapter_kind([]) == []


# ---- API contract -----------------------------------------------------------


def test_classifier_does_not_mutate_input() -> None:
    chapters = _entries(("Cover", 1, 1), ("Story", 2, 100))
    original = [dict(c) for c in chapters]
    classify_chapter_kind(chapters)
    assert chapters == original


def test_classifier_preserves_other_fields() -> None:
    chapters = _entries(("Cover", 1, 1), ("Story", 2, 100))
    classified = classify_chapter_kind(chapters)
    assert classified[0]["number"] == 1
    assert classified[0]["title"] == "Cover"
    assert classified[0]["page_start"] == 1
    assert classified[0]["page_end"] == 1
