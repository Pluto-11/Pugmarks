"""Tests for the sentence-aware chunker + candidate merger.

These are pure-function tests — no LLM calls, no I/O.
"""
from __future__ import annotations

import pytest

from pugmark.chunking import ChunkRef, chunk_text, merge_candidates
from pugmark.schemas import Candidate

# ============================================================================
# chunk_text
# ============================================================================


def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("") == []


def test_chunk_text_shorter_than_target_returns_single_chunk() -> None:
    text = "A tiger appeared near the peepul tree. The sambhur stood frozen."
    chunks = chunk_text(text, target_tokens=10_000, overlap_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_offset == 0
    assert chunks[0].index == 0


def test_chunk_text_long_text_produces_multiple_chunks() -> None:
    # 200 sentences x ~50 chars = ~10K chars = ~2500 tokens; force a 500-token target
    sentence = "The panther moved silently through the dense jungle near the river. "
    text = sentence * 200
    chunks = chunk_text(text, target_tokens=500, overlap_chars=200)
    assert len(chunks) >= 4, f"expected ≥4 chunks, got {len(chunks)}"
    # Index monotonic from 0
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Offsets are non-decreasing
    offsets = [c.char_offset for c in chunks]
    assert offsets == sorted(offsets)
    # First chunk starts at 0
    assert chunks[0].char_offset == 0


def test_chunk_text_chunks_overlap_by_at_least_overlap_chars() -> None:
    sentence = "The panther moved silently through the dense jungle. "
    text = sentence * 200
    chunks = chunk_text(text, target_tokens=500, overlap_chars=300)
    # Adjacent chunks should overlap — chunk N+1 starts before chunk N ends.
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_end = prev.char_offset + len(prev.text)
        # nxt.char_offset must be < prev_end (overlap)
        assert nxt.char_offset < prev_end, (
            f"chunks {prev.index}→{nxt.index} don't overlap: "
            f"prev ends at {prev_end}, next starts at {nxt.char_offset}"
        )


def test_chunk_text_boundaries_snap_to_sentence_ends() -> None:
    sentence = "Sentence number {n}. "
    text = "".join(sentence.format(n=i) for i in range(500))
    chunks = chunk_text(text, target_tokens=400, overlap_chars=100)
    # Every chunk except possibly the last should end on a period (sentence boundary)
    for c in chunks[:-1]:
        last_char = c.text.rstrip()[-1]
        assert last_char in ".!?", (
            f"chunk {c.index} doesn't end at sentence boundary: ...{c.text[-30:]!r}"
        )


def test_chunk_text_covers_full_text() -> None:
    """Concatenating chunks (deduping overlap) must yield the original text."""
    sentence = "The leopard padded across the dry riverbed. "
    text = sentence * 100
    chunks = chunk_text(text, target_tokens=500, overlap_chars=150)
    # Walk forward, deduping overlapped region
    rebuilt = chunks[0].text
    for c in chunks[1:]:
        # The new chunk starts at c.char_offset; everything we already wrote past that
        # came from the overlap. Append only the non-overlapped tail.
        prev_chunk = chunks[c.index - 1]
        prev_end = prev_chunk.char_offset + len(prev_chunk.text)
        overlap_len = prev_end - c.char_offset
        rebuilt += c.text[overlap_len:]
    assert rebuilt == text


# ============================================================================
# merge_candidates
# ============================================================================


def _cand(surface: str, name: str, char_offset: int, conf: float) -> Candidate:
    return Candidate(
        surface_form=surface,
        proposed_name=name,
        entity_type="taxa",
        context_sentence=f"...{surface}...",
        context_window=f"...{surface}...",
        char_offset=char_offset,
        page=1,
        llm_confidence=conf,
        extractor_version="vtest",
    )


def test_merge_candidates_dedupes_by_lowered_pair() -> None:
    chunk_a = [_cand("Tiger", "tiger", 10, 0.95)]
    chunk_b = [_cand("tiger", "Tiger", 20, 0.80)]
    merged = merge_candidates([chunk_a, chunk_b], chunk_offsets=[0, 100])
    assert len(merged) == 1
    # Kept the higher-confidence record
    assert merged[0].llm_confidence == 0.95


def test_merge_candidates_keeps_distinct_pairs() -> None:
    chunk_a = [_cand("tiger", "tiger", 10, 0.9)]
    chunk_b = [_cand("panther", "panther", 5, 0.85)]
    merged = merge_candidates([chunk_a, chunk_b], chunk_offsets=[0, 100])
    assert len(merged) == 2
    names = {c.proposed_name for c in merged}
    assert names == {"tiger", "panther"}


def test_merge_candidates_rebases_char_offset_to_chapter() -> None:
    # Two chunks: chunk 0 starts at offset 0, chunk 1 starts at chapter offset 500
    chunk_a = [_cand("tiger", "tiger", 10, 0.9)]  # offset 10 in chunk 0
    chunk_b = [_cand("panther", "panther", 25, 0.8)]  # offset 25 in chunk 1
    merged = merge_candidates([chunk_a, chunk_b], chunk_offsets=[0, 500])
    by_name = {c.proposed_name: c for c in merged}
    assert by_name["tiger"].char_offset == 10
    assert by_name["panther"].char_offset == 525


def test_merge_candidates_preserves_first_occurrence_order() -> None:
    chunk_a = [
        _cand("tiger", "tiger", 10, 0.9),
        _cand("peepul", "peepul tree", 50, 0.8),
    ]
    chunk_b = [
        _cand("panther", "panther", 5, 0.85),
        _cand("tiger", "tiger", 15, 0.7),  # duplicate of chunk_a's tiger
    ]
    merged = merge_candidates([chunk_a, chunk_b], chunk_offsets=[0, 200])
    names_in_order = [c.proposed_name for c in merged]
    assert names_in_order == ["tiger", "peepul tree", "panther"]


def test_merge_candidates_empty_input_returns_empty() -> None:
    assert merge_candidates([], chunk_offsets=[]) == []
    assert merge_candidates([[], []], chunk_offsets=[0, 100]) == []


# ============================================================================
# ChunkRef contract
# ============================================================================


def test_chunkref_is_immutable() -> None:
    ref = ChunkRef(text="x", char_offset=0, index=0)
    # frozen dataclass raises FrozenInstanceError on assignment
    with pytest.raises(AttributeError):
        ref.char_offset = 100  # type: ignore[misc]
