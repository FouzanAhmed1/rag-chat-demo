"""Retrieval + answer generation.

- retrieve(): hybrid retrieval (vector cosine + BM25, RRF fusion, rerank).
- fake_answer(): offline extractive answer with citations (FAKE_LLM=1).
- fake_answer_stream(): the same answer, yielded word by word for SSE.
- llm_answer() / llm_answer_stream(): real answers via an OpenAI-compatible
  chat endpoint; see StreamingLLM for the interface a streaming backend
  must implement.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Protocol

from app import config
from app.chunking import split_sentences
from app.embeddings import Embedder
from app.hybrid import hybrid_retrieve
from app.rerank import get_reranker
from app.sessions import Turn, history_prompt_block, rewrite_query
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
    reranker: str | None = None,
) -> list[tuple[Chunk, float]]:
    """Hybrid retrieval: vector cosine + BM25 fused with weighted RRF."""
    hits, _, _ = retrieve_debug(
        question,
        store,
        embedder,
        top_k=top_k,
        min_score=min_score,
        alpha=alpha,
        reranker=reranker,
    )
    return hits


def retrieve_debug(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    min_score: float | None = None,
    alpha: float | None = None,
    reranker: str | None = None,
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

    # Optional rerank stage after fusion; contributions are re-aligned to
    # the reranked order by chunk id. An explicit reranker name overrides
    # the RERANKER env var (handy for tests).
    if reranker is None:
        reranker = config.RERANKER
    if hits and reranker != "none":
        contrib_by_id = {
            chunk.id: contrib for (chunk, _), contrib in zip(hits, contributions)
        }
        reranked = get_reranker(reranker).rerank(question, hits)
        contributions = [contrib_by_id[chunk.id] for chunk, _ in reranked]
        hits = reranked

    debug = {
        "hybrid_alpha": alpha,
        "rrf_k": config.RRF_K,
        "reranker": reranker,
        "retrieval_query": question,
        "vector_pool_size": len(vector_hits),
    }
    return hits, contributions, debug


def _build_citations(
    hits: list[tuple[Chunk, float]],
    contributions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the citation dicts shared by every answer path."""
    citations = []
    for i, ((chunk, score), contrib) in enumerate(
        zip(hits, contributions or [{}] * len(hits)), start=1
    ):
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
    return citations


def _build_context_blocks(hits: list[tuple[Chunk, float]]) -> list[str]:
    return [
        f"[{i}] {chunk.source}"
        + (f" (page {chunk.page})" if chunk.page else "")
        + f"\n{chunk.text}"
        for i, (chunk, _) in enumerate(hits, start=1)
    ]


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
    history: list[Turn] | None = None,
) -> dict[str, Any]:
    """Extractive offline answer: best matching sentences plus citations.

    history is accepted so the session plumbing reaches the generator, and
    is reported in the debug payload; the extractive text itself only ever
    quotes the retrieved chunks.
    """
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

    citations = _build_citations(hits, contributions)
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
    history: list[Turn] | None = None,
) -> dict[str, Any]:
    """Answer via an OpenAI-compatible chat completions endpoint."""
    import httpx

    if not config.LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required when FAKE_LLM is not 1")

    context_blocks = _build_context_blocks(hits)
    citations = _build_citations(hits, contributions)

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
    history_block = history_prompt_block(history or [])
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    if history_block:
        user_content = f"{history_block}\n\n{user_content}"
    resp = httpx.post(
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
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


def _prepare(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    alpha: float | None = None,
    reranker: str | None = None,
    history: list[Turn] | None = None,
    session_id: str | None = None,
) -> tuple[
    list[tuple[Chunk, float]], list[dict[str, Any]], dict[str, Any], list[Turn]
]:
    """Shared retrieval + debug prep for the answer and stream paths."""
    history = history or []
    retrieval_query = rewrite_query(question, history)
    hits, contributions, debug = retrieve_debug(
        retrieval_query,
        store,
        embedder,
        top_k=top_k,
        alpha=alpha,
        reranker=reranker,
    )
    debug["retrieval_query"] = retrieval_query
    debug["history_turns_used"] = len(history)
    debug["session_id"] = session_id
    debug["contributions"] = [
        {"ref": i + 1, "source": chunk.source, **contrib}
        for i, ((chunk, _), contrib) in enumerate(zip(hits, contributions))
    ]
    return hits, contributions, debug, history


def answer_question(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    alpha: float | None = None,
    reranker: str | None = None,
    history: list[Turn] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    hits, contributions, debug, history = _prepare(
        question,
        store,
        embedder,
        top_k=top_k,
        alpha=alpha,
        reranker=reranker,
        history=history,
        session_id=session_id,
    )
    if config.FAKE_LLM:
        return fake_answer(
            question,
            hits,
            contributions=contributions,
            retrieval_debug=debug,
            history=history,
        )
    return llm_answer(
        question,
        hits,
        contributions=contributions,
        retrieval_debug=debug,
        history=history,
    )


class StreamingLLM(Protocol):
    """Interface a real streaming LLM backend must implement.

    A streaming backend receives the user question, the retrieved context
    blocks (already formatted as "[1] source\\ntext"), and the conversation
    history, and yields the answer as text tokens (words, subwords, or
    whatever the model emits) in generation order. The /chat/stream
    endpoint wraps each token in an SSE ``data: {"token": ...}`` event and
    terminates the stream with ``data: [DONE]``.

    A minimal OpenAI-compatible implementation ships below as
    llm_answer_stream(); plug in your own by passing it as the
    ``llm_streamer`` argument to stream_answer().
    """

    def stream_answer(
        self,
        question: str,
        context_blocks: list[str],
        history: list[Turn],
    ) -> Iterator[str]:
        """Yield answer tokens in generation order."""
        ...


def fake_answer_stream(
    question: str,
    hits: list[tuple[Chunk, float]],
    contributions: list[dict[str, Any]] | None = None,
    retrieval_debug: dict[str, Any] | None = None,
    history: list[Turn] | None = None,
) -> Iterator[str]:
    """Yield the fake extractive answer word by word.

    Joining the tokens reproduces fake_answer()["answer"] exactly, so the
    streaming and non-streaming paths can never disagree.
    """
    result = fake_answer(
        question,
        hits,
        contributions=contributions,
        retrieval_debug=retrieval_debug,
        history=history,
    )
    for token in re.findall(r"\S+\s*", result["answer"]):
        yield token


def llm_answer_stream(
    question: str,
    hits: list[tuple[Chunk, float]],
    history: list[Turn] | None = None,
) -> Iterator[str]:
    """Stream tokens from an OpenAI-compatible chat completions endpoint."""
    import json as jsonlib

    import httpx

    if not config.LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required when FAKE_LLM is not 1")

    if not hits:
        yield (
            "I could not find anything relevant in the indexed documents. "
            "Try rephrasing your question or ingest more documents."
        )
        return

    context = "\n\n".join(_build_context_blocks(hits))
    system = (
        "You answer questions using ONLY the context below. "
        "Cite your sources with bracketed numbers like [1], [2] matching the "
        "context blocks. If the context does not contain the answer, say so."
    )
    history_block = history_prompt_block(history or [])
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    if history_block:
        user_content = f"{history_block}\n\n{user_content}"

    with httpx.stream(
        "POST",
        f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "temperature": 0.2,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = jsonlib.loads(payload)["choices"][0]["delta"].get(
                    "content"
                )
            except (KeyError, ValueError):
                continue
            if delta:
                yield delta


def stream_answer(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
    alpha: float | None = None,
    reranker: str | None = None,
    history: list[Turn] | None = None,
    session_id: str | None = None,
    llm_streamer: StreamingLLM | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Iterator[str]]:
    """Prepare a streaming answer.

    Returns (citations, debug, token_iterator). The caller wraps tokens in
    SSE events; joining the tokens gives the full answer text.
    """
    hits, contributions, debug, history = _prepare(
        question,
        store,
        embedder,
        top_k=top_k,
        alpha=alpha,
        reranker=reranker,
        history=history,
        session_id=session_id,
    )
    citations = _build_citations(hits, contributions)
    if config.FAKE_LLM:
        tokens = fake_answer_stream(
            question, hits, contributions, debug, history
        )
    elif llm_streamer is not None:
        tokens = llm_streamer.stream_answer(
            question, _build_context_blocks(hits), history
        )
    else:
        tokens = llm_answer_stream(question, hits, history)
    return citations, debug, tokens
