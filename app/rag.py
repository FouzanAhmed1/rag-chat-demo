"""Retrieval + answer generation.

- retrieve(): embed the question, search the store, apply the score floor.
- fake_answer(): offline extractive answer with citations (FAKE_LLM=1).
- llm_answer(): real answer via an OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import re
from typing import Any

from app import config
from app.chunking import split_sentences
from app.embeddings import Embedder
from app.hybrid import hybrid_retrieve
from app.store import Chunk, VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "how",
    "what", "when", "where", "why", "which", "who", "to", "of", "in", "on",
    "for", "and", "or", "with", "my", "i", "me", "it", "this", "that", "can",
    "you", "your", "please", "tell", "about",
}


def _keywords(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def retrieve(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    min_score: float | None = None,
    alpha: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Hybrid retrieval: vector cosine + BM25 fused with weighted RRF."""
    hits, _, _ = retrieve_debug(
        question, store, embedder, top_k=top_k, min_score=min_score, alpha=alpha
    )
    return hits


def retrieve_debug(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    min_score: float | None = None,
    alpha: float | None = None,
) -> tuple[list[tuple[Chunk, float]], list[dict[str, Any]], dict[str, Any]]:
    """Like retrieve(), but also returns per-hit ranker contributions and
    a debug dict describing how the ranking was produced."""
    top_k = config.TOP_K if top_k is None else top_k
    min_score = config.MIN_SCORE if min_score is None else min_score
    if alpha is None:
        alpha = config.HYBRID_ALPHA
    alpha = min(1.0, max(0.0, alpha))

    # Search a wider vector pool than top_k so fusion sees every candidate
    # either ranker might surface; the relevance gate lives in hybrid_retrieve.
    query_vec = embedder.embed([question])[0]
    vector_hits = store.search(query_vec, top_k=max(top_k * 4, 10), min_score=0.0)

    hits, contributions = hybrid_retrieve(
        question,
        vector_hits,
        store.chunks,
        top_k=top_k,
        alpha=alpha,
        rrf_k=config.RRF_K,
        min_score=min_score,
    )
    debug = {
        "hybrid_alpha": alpha,
        "rrf_k": config.RRF_K,
        "vector_pool_size": len(vector_hits),
    }
    return hits, contributions, debug


def _best_sentences(chunk_text: str, keywords: list[str], limit: int = 2) -> list[str]:
    scored: list[tuple[int, str]] = []
    for sentence in split_sentences(chunk_text):
        lowered = sentence.lower()
        hits = sum(1 for k in keywords if k in lowered)
        if hits:
            scored.append((hits, sentence))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [s for _, s in scored[:limit]]


def _contributing_rankers(contribution: dict[str, Any] | None) -> list[str]:
    """Which rankers surfaced a hit (used in the debug response)."""
    if not contribution:
        return []
    names = []
    if contribution.get("vector_rank") is not None:
        names.append("vector")
    if contribution.get("bm25_rank") is not None:
        names.append("bm25")
    return names


def fake_answer(
    question: str,
    hits: list[tuple[Chunk, float]],
    contributions: list[dict[str, Any]] | None = None,
    retrieval_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extractive offline answer: best matching sentences plus citations."""
    if not hits:
        return {
            "answer": (
                "I could not find anything relevant in the indexed documents. "
                "Try rephrasing your question or ingest more documents."
            ),
            "citations": [],
            "debug": retrieval_debug or {},
        }
    keywords = _keywords(question)
    picked: list[tuple[str, int]] = []  # (sentence, citation ref)
    for ref, (chunk, _) in enumerate(hits[:2], start=1):
        for sentence in _best_sentences(chunk.text, keywords, limit=2):
            picked.append((sentence, ref))
        if len(picked) >= 3:
            break
    if not picked:
        # Fall back to the start of the best chunk if no sentence matched.
        picked = [(hits[0][0].text[:400].strip(), 1)]

    citations = []
    for i, ((chunk, score), contrib) in enumerate(
        zip(hits, contributions or [{}] * len(hits))
    ):
        citations.append(
            {
                "ref": i + 1,
                "source": chunk.source,
                "page": chunk.page,
                "score": round(score, 3),
                "excerpt": chunk.text[:300].strip(),
                "rankers": _contributing_rankers(contrib),
            }
        )
    lines = ["Here is what I found in your documents:"]
    for sentence, ref in picked[:3]:
        lines.append(f"- {sentence} [{ref}]")
    lines.append("")
    lines.append(
        "Sources: "
        + ", ".join(
            f"[{c['ref']}] {c['source']}"
            + (f" (page {c['page']})" if c["page"] else "")
            for c in citations
        )
    )
    return {
        "answer": "\n".join(lines),
        "citations": citations,
        "debug": retrieval_debug or {},
    }


def llm_answer(
    question: str,
    hits: list[tuple[Chunk, float]],
    contributions: list[dict[str, Any]] | None = None,
    retrieval_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer via an OpenAI-compatible chat completions endpoint."""
    import httpx

    if not config.LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required when FAKE_LLM is not 1")

    context_blocks = []
    citations = []
    for i, ((chunk, score), contrib) in enumerate(
        zip(hits, contributions or [{}] * len(hits)), start=1
    ):
        label = chunk.source + (f" (page {chunk.page})" if chunk.page else "")
        context_blocks.append(f"[{i}] {label}\n{chunk.text}")
        citations.append(
            {
                "ref": i,
                "source": chunk.source,
                "page": chunk.page,
                "score": round(score, 3),
                "excerpt": chunk.text[:300].strip(),
                "rankers": _contributing_rankers(contrib),
            }
        )

    if not hits:
        return {
            "answer": (
                "I could not find anything relevant in the indexed documents. "
                "Try rephrasing your question or ingest more documents."
            ),
            "citations": [],
            "debug": retrieval_debug or {},
        }

    context = "\n\n".join(context_blocks)
    system = (
        "You answer questions using ONLY the context below. "
        "Cite your sources with bracketed numbers like [1], [2] matching the "
        "context blocks. If the context does not contain the answer, say so."
    )
    resp = httpx.post(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return {
        "answer": content,
        "citations": citations,
        "debug": retrieval_debug or {},
    }


def answer_question(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    hits, contributions, debug = retrieve_debug(
        question, store, embedder, top_k=top_k, alpha=alpha
    )
    debug["contributions"] = [
        {"ref": i + 1, "source": chunk.source, **contrib}
        for i, ((chunk, _), contrib) in enumerate(zip(hits, contributions))
    ]
    if config.FAKE_LLM:
        return fake_answer(
            question, hits, contributions=contributions, retrieval_debug=debug
        )
    return llm_answer(
        question, hits, contributions=contributions, retrieval_debug=debug
    )
