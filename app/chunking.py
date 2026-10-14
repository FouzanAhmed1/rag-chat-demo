"""Chunking with overlap.

Splits text into sentences, then greedily packs sentences into chunks up to
chunk_size characters. When a chunk is full, the trailing sentences (up to
chunk_overlap characters) are carried over into the next chunk, so context is
preserved across chunk boundaries.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Chunk text into pieces of at most chunk_size chars with overlap.

    A single sentence longer than chunk_size is kept whole (chunks never split
    a sentence), so the size limit is a soft cap for well-formed text.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))

    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and current_len + added > chunk_size:
            flush()
            # Carry trailing sentences into the next chunk for overlap.
            kept: list[str] = []
            kept_len = 0
            for prev in reversed(current):
                cost = len(prev) + (1 if kept else 0)
                if kept_len + cost > chunk_overlap:
                    break
                kept.append(prev)
                kept_len += cost
            current = list(reversed(kept))
            current_len = kept_len
        current.append(sentence)
        current_len += len(sentence) + (1 if len(current) > 1 else 0)

    flush()
    return chunks
