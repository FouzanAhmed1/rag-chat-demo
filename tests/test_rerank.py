"""Rerank tests: heuristic signals, the RERANKER=none bypass, and the
pluggable Reranker interface."""

import uuid

import pytest

from app import config
from app.embeddings import HashingEmbedder
from app.rag import retrieve
from app.rerank import HeuristicReranker, NoOpReranker, get_reranker, set_reranker
from app.store import Chunk, VectorStore

QUERY = "alpha beta gamma"


def _chunk(name: str, text: str, offset: int = 0) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()), doc_id=name, text=text, source=f"{name}.txt",
        offset=offset,
    )


@pytest.fixture()
def coverage_pair():
    # vector_top repeats one query term heavily (high cosine) but covers
    # only 1/3 of the query; full_match covers every term and has a
    # matching markdown heading.
    vector_top = _chunk("vector-top", " ".join(["alpha"] * 20))
    full_match = _chunk(
        "full-match",
        "# Alpha Beta Gamma Guide\n\nalpha beta gamma explained here.",
    )
    return vector_top, full_match


def test_heuristic_promotes_better_term_coverage(coverage_pair):
    vector_top, full_match = coverage_pair
    reranker = HeuristicReranker()
    hits = [(vector_top, 0.9), (full_match, 0.5)]
    reranked = reranker.rerank(QUERY, hits)
    assert [c.source for c, _ in reranked] == ["full-match.txt", "vector-top.txt"]


def test_heuristic_is_deterministic(coverage_pair):
    vector_top, full_match = coverage_pair
    reranker = HeuristicReranker()
    hits = [(vector_top, 0.9), (full_match, 0.5)]
    first = [(c.id, s) for c, s in reranker.rerank(QUERY, hits)]
    second = [(c.id, s) for c, s in reranker.rerank(QUERY, hits)]
    assert first == second


def test_heading_match_beats_plain_coverage():
    reranker = HeuristicReranker()
    # Same term coverage; the keywords sit in a real markdown heading in one
    # chunk and deep in the body (not the first line) in the other.
    plain = _chunk("plain", "introduction line about taxes\n\nalpha beta gamma deep in the body")
    headed = _chunk("headed", "# Alpha Beta Gamma\n\nintroduction line about taxes")
    assert reranker.score(QUERY, headed) > reranker.score(QUERY, plain)


def test_length_penalty_applies():
    reranker = HeuristicReranker()
    short = _chunk("short", "alpha beta gamma")
    long = _chunk("long", "alpha beta gamma " + "filler words " * 500)
    assert reranker.score(QUERY, short) > reranker.score(QUERY, long)


def test_noop_preserves_order(coverage_pair):
    vector_top, full_match = coverage_pair
    hits = [(vector_top, 0.9), (full_match, 0.5)]
    assert NoOpReranker().rerank(QUERY, hits) == hits


def _synthetic_store() -> VectorStore:
    store = VectorStore()
    embedder = HashingEmbedder(dim=1024)
    texts = [
        ("chunk-a", " ".join(["gadget"] * 20) + " device"),
        ("chunk-b", "gizmo marvel quizzacious"),
        ("chunk-c", "gadget device"),
    ]
    chunks = [
        Chunk(id=str(uuid.uuid4()), doc_id=n, text=t, source=f"{n}.txt")
        for n, t in texts
    ]
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store


def test_reranker_none_preserves_fusion_order(monkeypatch):
    monkeypatch.setattr(config, "RERANKER", "none")
    store = _synthetic_store()
    embedder = HashingEmbedder(dim=1024)
    hits = retrieve("gadget gizmo", store, embedder, alpha=0.5)
    assert [c.source for c, _ in hits] == [
        "chunk-a.txt",
        "chunk-b.txt",
        "chunk-c.txt",
    ]


def test_custom_reranker_plugs_in_via_set_reranker():
    class ReverseReranker:
        def rerank(self, query, hits):
            return list(reversed(hits))

    store = _synthetic_store()
    embedder = HashingEmbedder(dim=1024)
    original = get_reranker()
    set_reranker(ReverseReranker())
    try:
        hits = retrieve(
            "gadget gizmo", store, embedder, alpha=0.5, reranker="heuristic"
        )
        assert [c.source for c, _ in hits] == [
            "chunk-c.txt",
            "chunk-b.txt",
            "chunk-a.txt",
        ]
    finally:
        set_reranker(original)
