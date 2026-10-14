"""Hybrid retrieval: BM25 keyword scoring fused with vector scores.

- BM25: a small, dependency-free implementation (k1=1.5, b=0.75, standard
  IDF) built over the chunk corpus. Good at exact rare-term matches where
  the hash embedder's cosine similarity can miss.
- RRF: reciprocal rank fusion. fused = sum over rankers of
  weight_r / (k + rank_r), with k=60 by default.

HYBRID_ALPHA blends the two rankers: the vector ranker's fused weight is
alpha and BM25's is (1 - alpha), so 0.0 = pure BM25, 1.0 = pure vector,
0.5 = unweighted RRF (the default).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.store import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by the BM25 index."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Plain BM25 over chunk texts. Built fresh from a chunk list."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens: list[list[str]] = [tokenize(c.text) for c in chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_len = (
            sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        )
        df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            self.tf.append(counts)
            for term in counts:
                df[term] += 1
        n = len(chunks)
        # Standard BM25 IDF with the +1 floor so rare terms stay positive.
        self.idf = {
            term: math.log((n - f + 0.5) / (f + 0.5)) + 1.0
            for term, f in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score of every chunk for the query, in chunk order."""
        if not self.chunks:
            return []
        out = [0.0] * len(self.chunks)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = self.k1 * (
                    1 - self.b + self.b * (self.doc_lens[i] / self.avg_len)
                ) if self.avg_len else self.k1
                out[i] += idf * (tf * (self.k1 + 1)) / (tf + denom)
        return out

    def ranked(self, query: str) -> list[tuple[int, float]]:
        """(chunk index, score) sorted best-first, dropping zero scores."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored


def rrf_fuse(
    rankings: list[list[int]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over chunk-index rankings.

    rankings: one ordered list of chunk indices per ranker (best first).
    weights: per-ranker weight applied to 1/(k + rank); ranks are 1-based.
    Returns (chunk index, fused score) sorted best-first.
    """
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda p: p[1], reverse=True)


def hybrid_retrieve(
    question: str,
    vector_hits: list[tuple[Chunk, float]],
    chunks: list[Chunk],
    top_k: int = 4,
    alpha: float = 0.5,
    rrf_k: int = 60,
    min_score: float = 0.0,
) -> tuple[list[tuple[Chunk, float]], list[dict[str, float | int | None]]]:
    """Fuse a vector ranking with a BM25 ranking via weighted RRF.

    vector_hits: (chunk, cosine score) best-first, as returned by
        VectorStore.search.
    chunks: the full chunk corpus (BM25 is indexed over it).
    alpha: vector weight; BM25 weight is (1 - alpha).
    Returns (fused hits, contributions): hits are (chunk, fused score)
    best-first; contributions is a parallel list of dicts with the per-ranker
    rank (None when the ranker did not surface the chunk) and the raw
    scores, so callers can report which ranker contributed what.
    """
    bm25 = BM25(chunks)
    bm25_ranked = bm25.ranked(question)
    bm25_order = [i for i, _ in bm25_ranked]
    bm25_scores = {i: s for i, s in bm25_ranked}

    id_to_index = {c.id: i for i, c in enumerate(chunks)}
    vector_order = [id_to_index[c.id] for c, _ in vector_hits]
    vector_scores = {id_to_index[c.id]: s for c, s in vector_hits}
    vector_rank = {idx: r for r, idx in enumerate(vector_order, start=1)}
    bm25_rank = {idx: r for r, idx in enumerate(bm25_order, start=1)}

    fused = rrf_fuse([vector_order, bm25_order], [alpha, 1.0 - alpha], k=rrf_k)

    hits: list[tuple[Chunk, float]] = []
    contributions: list[dict[str, float | int | None]] = []
    for idx, fused_score in fused[:top_k]:
        vs = vector_scores.get(idx, 0.0)
        bs = bm25_scores.get(idx, 0.0)
        # Relevance gate: keep a chunk only if at least one active ranker
        # vouched for it (cosine floor for vector, nonzero score for BM25).
        if not ((alpha > 0 and vs >= min_score) or (alpha < 1 and bs > 0)):
            continue
        hits.append((chunks[idx], fused_score))
        contributions.append(
            {
                "vector_rank": vector_rank.get(idx),
                "bm25_rank": bm25_rank.get(idx),
                "vector_score": round(vs, 4),
                "bm25_score": round(bs, 4),
                "fused_score": round(fused_score, 5),
                "vector_weight": alpha,
                "bm25_weight": round(1.0 - alpha, 4),
            }
        )
    return hits, contributions
