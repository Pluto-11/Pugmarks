"""Sentence-aware text chunking + candidate merging for the extract stage.

Why this exists: a chapter's full text can be 100K+ tokens — too big for free-tier
LLMs per-request limits and too coarse for quality entity recall. We split the
chapter into overlapping chunks at sentence boundaries, fan out to the LLM in
parallel, then merge + dedupe.

Pure functions only — no I/O, no LLM, no async. The extract stage composes these.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pugmark.schemas import Candidate

# Approximate: 1 token ≈ 4 chars of English prose (GPT/Gemini-ish tokenizers).
# Used to translate user-facing target_tokens into char counts internally.
CHARS_PER_TOKEN = 4

# Sentence-end matcher: a period/!/? followed by whitespace or end-of-string.
# Captures the position AFTER the terminating punctuation so chunks include it.
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")


@dataclass(frozen=True)
class ChunkRef:
    """A slice of the source text plus where it sits in the original."""

    text: str
    char_offset: int  # offset into the original chapter text
    index: int  # 0-based chunk index


def _sentence_end_positions(text: str) -> list[int]:
    """Indices in `text` where a sentence terminator (.!?) ends.

    Returned indices are the position AFTER the punctuation (and any trailing
    whitespace), i.e. valid chunk-boundary positions.
    """
    return [m.end() for m in _SENTENCE_END_RE.finditer(text)]


def _snap_to_sentence_end(text: str, target_pos: int, slack: int = 200) -> int:
    """Snap `target_pos` to the nearest sentence end within ±slack chars.

    If no sentence end is in range, return target_pos unchanged.
    """
    if target_pos >= len(text):
        return len(text)
    ends = _sentence_end_positions(text)
    if not ends:
        return target_pos
    best = target_pos
    best_dist = slack + 1
    for e in ends:
        d = abs(e - target_pos)
        if d <= slack and d < best_dist:
            best, best_dist = e, d
    return best


def chunk_text(
    text: str,
    *,
    target_tokens: int = 6000,
    overlap_chars: int = 600,
) -> list[ChunkRef]:
    """Split `text` into overlapping sentence-aligned chunks.

    Args:
        text: source text to chunk
        target_tokens: target token budget per chunk (≈ target_tokens * 4 chars)
        overlap_chars: number of chars chunk N+1 starts before chunk N ended

    Returns:
        A list of ChunkRef. Empty list if `text` is empty. Single chunk if
        `text` fits under target. Otherwise, chunks are sentence-aligned at
        their end boundary (where possible) and overlap by ≈ overlap_chars.
    """
    if not text:
        return []
    target_chars = target_tokens * CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return [ChunkRef(text=text, char_offset=0, index=0)]

    chunks: list[ChunkRef] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        # Tentative end: start + target_chars
        target_end = min(start + target_chars, n)
        end = _snap_to_sentence_end(text, target_end, slack=target_chars // 4)
        if end <= start:
            # Defensive: snap failed and went backwards — force forward
            end = target_end
        chunk = ChunkRef(text=text[start:end], char_offset=start, index=idx)
        chunks.append(chunk)
        if end >= n:
            break
        # Next chunk starts overlap_chars before the current end (snapped to sentence)
        next_start = max(start + 1, end - overlap_chars)
        next_start = _snap_to_sentence_end(text, next_start, slack=overlap_chars // 2)
        # Guard: never go backwards or stall
        if next_start <= start:
            next_start = start + max(1, target_chars - overlap_chars)
        start = next_start
        idx += 1

    return chunks


def merge_candidates(
    candidates_per_chunk: list[list[Candidate]],
    chunk_offsets: list[int],
) -> list[Candidate]:
    """Merge candidates from N chunks: rebase offsets, dedupe, keep best confidence.

    Args:
        candidates_per_chunk: parallel list of per-chunk Candidate lists
        chunk_offsets: parallel list — chunk_offsets[i] is the offset of chunk i
            in the original chapter text

    Dedup key: (surface_form.lower().strip(), proposed_name.lower().strip()).
    For duplicates, keeps the record with the highest llm_confidence.
    Order is by first occurrence across chunks.
    """
    if not candidates_per_chunk:
        return []
    if len(candidates_per_chunk) != len(chunk_offsets):
        raise ValueError(
            f"candidates_per_chunk and chunk_offsets length mismatch: "
            f"{len(candidates_per_chunk)} vs {len(chunk_offsets)}"
        )

    by_key: dict[tuple[str, str], Candidate] = {}
    order: list[tuple[str, str]] = []

    for chunk_cands, chunk_offset in zip(candidates_per_chunk, chunk_offsets, strict=True):
        for cand in chunk_cands:
            key = (
                cand.surface_form.lower().strip(),
                cand.proposed_name.lower().strip(),
            )
            rebased = cand.model_copy(update={"char_offset": cand.char_offset + chunk_offset})
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = rebased
                order.append(key)
            elif rebased.llm_confidence > existing.llm_confidence:
                by_key[key] = rebased
            # else: keep existing

    return [by_key[k] for k in order]
