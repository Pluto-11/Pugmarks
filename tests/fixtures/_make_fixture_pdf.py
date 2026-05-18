"""Run once to generate sample_chapter.pdf for ingest tests.

Usage: uv run python tests/fixtures/_make_fixture_pdf.py
"""
from __future__ import annotations

from pathlib import Path

import fitz

OUT = Path(__file__).parent / "sample_chapter.pdf"

doc = fitz.open()

# Page 1 — chapter title
page = doc.new_page()
page.insert_text((72, 72), "CHAPTER ONE", fontsize=18)
page.insert_text((72, 100), "The Black Panther of Sivanipalli", fontsize=14)
page.insert_text(
    (72, 150),
    "It was a cool evening in the jungle. A panther had been seen near\n"
    "the village three days earlier, and a sambhur was found half-eaten\n"
    "on the edge of the bamboo thicket.",
    fontsize=11,
)

# Page 2
page = doc.new_page()
page.insert_text(
    (72, 72),
    "I sat in a machan beneath a peepul tree, rifle in hand.\n"
    "A jungle fowl crowed somewhere in the distance.",
    fontsize=11,
)

# Add a TOC bookmark
doc.set_toc([[1, "The Black Panther of Sivanipalli", 1]])

doc.save(OUT)
print(f"wrote {OUT}")
