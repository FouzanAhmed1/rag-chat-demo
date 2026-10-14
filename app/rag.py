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
) -> list[tuple[Chunk, float]]:
    top_k = config.TOP_K if top_k is None else top_k
    min_score = config.MIN_SCORE if min_score is None else min_score
    query_vec = embedder.embed([question])[0]
    return store.search(query_vec, top_k=top_k, min_score=min_score)


def _best_sentences(chunk_text: str, keywords: list[str], limit: int = 2) -> list[str]:
    scored: list[tuple[int, str]] = []
    for sentence in split_sentences(chunk_text):
        lowered = sentence.lower()
        hits = sum(1 for k in keywords if k in lowered)
        if hits:
            scored.append((hits, sentence))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [s for _, s in scored[:limit]]


def fake_answer(
    question: str, hits: list[tuple[Chunk, float]]
) -> dict[str, Any]:
    """Extractive offline answer: best matching sentences plus citations."""
    if not hits:
        return {
            "answer": (
                "I could not find anything relevant in the indexed documents. "
                "Try rephrasing your question or ingest more documents."
            ),
            "citations": [],
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

    citations = [
        {
            "ref": i + 1,
            "source": chunk.source,
            "page": chunk.page,
            "score": round(score, 3),
            "excerpt": chunk.text[:300].strip(),
        }
        for i, (chunk, score) in enumerate(hits)
    ]
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
    return {"answer": "\n".join(lines), "citations": citations}


def llm_answer(
    question: str, hits: list[tuple[Chunk, float]]
) -> dict[str, Any]:
    """Answer via an OpenAI-compatible chat completions endpoint."""
    import httpx

    if not config.LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required when FAKE_LLM is not 1")

    context_blocks = []
    citations = []
    for i, (chunk, score) in enumerate(hits, start=1):
        label = chunk.source + (f" (page {chunk.page})" if chunk.page else "")
        context_blocks.append(f"[{i}] {label}\n{chunk.text}")
        citations.append(
            {
                "ref": i,
                "source": chunk.source,
                "page": chunk.page,
                "score": round(score, 3),
                "excerpt": chunk.text[:300].strip(),
            }
        )

    if not hits:
        return {
            "answer": (
                "I could not find anything relevant in the indexed documents. "
                "Try rephrasing your question or ingest more documents."
            ),
            "citations": [],
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
    return {"answer": content, "citations": citations}


def answer_question(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int | None = None,
) -> dict[str, Any]:
    hits = retrieve(question, store, embedder, top_k=top_k)
    if config.FAKE_LLM:
        return fake_answer(question, hits)
    return llm_answer(question, hits)
