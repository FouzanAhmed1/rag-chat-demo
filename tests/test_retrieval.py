"""Retrieval tests: rank the right document top-1, filter junk queries."""

import os
import uuid

import pytest

from app import config
from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.ingest import load_document
from app.store import Chunk, VectorStore

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_docs")


def build_store_from_samples() -> tuple[VectorStore, HashingEmbedder]:
    store = VectorStore()
    embedder = HashingEmbedder(dim=config.EMBED_DIM)
    for name in sorted(os.listdir(SAMPLE_DIR)):
        path = os.path.join(SAMPLE_DIR, name)
        for text, page in load_document(path):
            chunks = []
            for piece in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        doc_id=name,
                        text=piece,
                        source=name,
                        page=page,
                    )
                )
            if chunks:
                store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder


@pytest.fixture(scope="module")
def sample_store():
    return build_store_from_samples()


def _search(store, embedder, query, top_k=4, min_score=0.0):
    q = embedder.embed([query])[0]
    return store.search(q, top_k=top_k, min_score=min_score)


def test_password_query_returns_it_faq_top1(sample_store):
    store, embedder = sample_store
    hits = _search(store, embedder, "How do I reset my forgotten password?")
    assert hits, "expected at least one hit"
    top_chunk, _ = hits[0]
    assert top_chunk.source == "it_faq.md"


def test_return_policy_query_returns_returns_doc_top1(sample_store):
    store, embedder = sample_store
    hits = _search(store, embedder, "What is the return policy for refunds?")
    assert hits, "expected at least one hit"
    top_chunk, _ = hits[0]
    assert top_chunk.source == "returns_and_shipping.md"


def test_pto_query_returns_handbook_top1(sample_store):
    store, embedder = sample_store
    hits = _search(store, embedder, "How many PTO days do full-time employees get?")
    assert hits, "expected at least one hit"
    top_chunk, _ = hits[0]
    assert top_chunk.source == "employee_handbook.md"


def test_min_score_filters_irrelevant_query(sample_store):
    store, embedder = sample_store
    hits = _search(
        store, embedder, "zebra quantum pancakes interstellar", min_score=0.05
    )
    assert hits == []


def test_search_top_k_and_threshold(sample_store):
    store, embedder = sample_store
    hits = _search(store, embedder, "password VPN laptop", top_k=2, min_score=0.0)
    assert len(hits) <= 2
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
