"""Tests for chunking: overlap behavior and size limits."""

import pytest

from app.chunking import chunk_text


def _sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} is short." for i in range(n))


def test_overlap_carries_trailing_sentences_into_next_chunk():
    text = _sentences(12)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=60)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        # The last sentence of chunk i must appear inside chunk i+1,
        # because it fits within the 60-char overlap window.
        last_sentence = chunks[i].rsplit(". ", 1)[-1]
        assert last_sentence in chunks[i + 1], (
            f"chunk {i} tail not found in chunk {i + 1}"
        )


def test_chunk_sizes_respected():
    text = _sentences(40)
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert chunks
    for chunk in chunks:
        assert len(chunk) <= 200


def test_single_long_sentence_kept_whole():
    text = "x" * 500 + "."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert chunks == [text]


def test_empty_and_blank_text():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello.", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello.", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        chunk_text("hello.", chunk_size=100, chunk_overlap=-1)
