"""Hybrid retrieval tests: BM25 + weighted RRF fusion against vector search.

A small synthetic corpus where BM25 and vector cosine disagree on purpose:
chunk-a repeats the common term "gadget" (wins on cosine), while chunk-b
holds the rare term "gizmo" (wins on BM25's IDF). Pure vector ranks
[a, c, b]; pure BM25 ranks [b, a, c].
"""

import uuid

import pytest

from app.embeddings import HashingEmbedder
from app.hybrid import BM25, rrf_fuse
from app.rag import retrieve
from app.store import Chunk, VectorStore

QUERY = "gadget gizmo"


def _synthetic_store() -> VectorStore:
    store = VectorStore()
    embedder = HashingEmbedder(dim=1024)
    texts = [
        ("chunk-a", " ".join(["gadget"] * 20) + " device"),
        ("chunk-b", "gizmo marvel quizzacious"),
        ("chunk-c", "gadget device"),
    ]
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            doc_id=name,
            text=text,
            source=f"{name}.txt",
        )
        for name, text in texts
    ]
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store


@pytest.fixture(scope="module")
def synth():
    store = _synthetic_store()
    return store, HashingEmbedder(dim=1024)


def _order(hits):
    return [c.source for c, _ in hits]


def test_bm25_ranks_rare_term_chunk_first(synth):
    store, _ = synth
    ranked = BM25(store.chunks).ranked(QUERY)
    assert ranked[0][0] == 1  # chunk-b holds the rare term "gizmo"
    assert ranked[0][1] > 0


def test_rrf_fuse_orders_by_weighted_reciprocal_rank():
    # ranker 1: [0, 1, 2]; ranker 2: [2, 0, 1]; k=60.
    fused = rrf_fuse([[0, 1, 2], [2, 0, 1]], [1.0, 1.0], k=60)
    order = [idx for idx, _ in fused]
    # 0: 1/61 + 1/62; 2: 1/63 + 1/61; 1: 1/62 + 1/63
    assert order == [0, 2, 1]


def test_fusion_differs_from_pure_vector(synth):
    store, embedder = synth
    pure_vector = _order(
        store.search(embedder.embed([QUERY])[0], top_k=4, min_score=0.0)
    )
    fused = _order(retrieve(QUERY, store, embedder, alpha=0.5))
    assert pure_vector == ["chunk-a.txt", "chunk-c.txt", "chunk-b.txt"]
    assert fused == ["chunk-a.txt", "chunk-b.txt", "chunk-c.txt"]
    assert fused != pure_vector


def test_alpha_one_reproduces_pure_vector(synth):
    store, embedder = synth
    pure_vector = _order(
        store.search(embedder.embed([QUERY])[0], top_k=4, min_score=0.0)
    )
    fused = _order(retrieve(QUERY, store, embedder, alpha=1.0))
    assert fused == pure_vector


def test_alpha_zero_reproduces_pure_bm25(synth):
    store, embedder = synth
    bm25_order = [store.chunks[i].source for i, _ in BM25(store.chunks).ranked(QUERY)]
    fused = _order(retrieve(QUERY, store, embedder, alpha=0.0))
    assert bm25_order == ["chunk-b.txt", "chunk-a.txt", "chunk-c.txt"]
    assert fused == bm25_order
