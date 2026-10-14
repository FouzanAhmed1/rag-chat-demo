"""Optional rerank stage applied after hybrid fusion.

Defines a clean Reranker interface (rerank(query, hits) -> hits) plus a
default LLM-free heuristic reranker. To plug in a real cross-encoder (e.g.
sentence-transformers CrossEncoder), subclass Reranker and call
set_reranker() at startup; the README shows a complete snippet.

Controlled by the RERANKER env var: "heuristic" (default) or "none".
"""

from __future__ import annotations

import re
from typing import Protocol

from app import config
from app.store import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "how",
    "what", "when", "where", "why", "which", "who", "to", "of", "in", "on",
    "for", "and", "or", "with", "my", "i", "me", "it", "this", "that", "can",
    "you", "your", "please", "tell", "about",
}


class Reranker(Protocol):
    """Interface for a rerank stage.

    Implementations take the fused (chunk, score) hits and the raw query and
    return them reordered (scores may be replaced). Must be deterministic
    given the same inputs so results are reproducible.
    """

    def rerank(
        self, query: str, hits: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        ...


class HeuristicReranker:
    """Cheap, deterministic, LLM-free reranker.

    Combines four signals, no model needed:
    - query-term coverage: fraction of query keywords present in the chunk
    - heading match: bonus when a query keyword appears in a markdown
      heading or the chunk's first line (titles usually carry the topic)
    - position bonus: slight preference for chunks near the start of their
      document (introductory context), decaying with character offset
    - length penalty: chunks far above ~800 chars are penalized, since long
      chunks dilute the answer with filler
    """

    def __init__(
        self,
        coverage_weight: float = 0.6,
        heading_weight: float = 0.25,
        position_weight: float = 0.15,
    ) -> None:
        self.coverage_weight = coverage_weight
        self.heading_weight = heading_weight
        self.position_weight = position_weight

    def _keywords(self, text: str) -> list[str]:
        return [
            t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
        ]

    def _heading_text(self, chunk: Chunk) -> str:
        lines = chunk.text.splitlines()
        headings = [ln for ln in lines if ln.lstrip().startswith("#")]
        head = " ".join(headings)
        if lines:
            head += " " + lines[0]
        return head.lower()

    def score(self, query: str, chunk: Chunk) -> float:
        keywords = self._keywords(query)
        if not keywords:
            return 0.0
        text = chunk.text.lower()

        coverage = sum(1 for k in keywords if k in text) / len(keywords)

        heading = self._heading_text(chunk)
        heading_match = sum(1 for k in keywords if k in heading) / len(keywords)

        # Position bonus decays with the chunk's character offset in its doc.
        position_bonus = 1.0 / (1.0 + (chunk.offset or 0) / 800.0)

        # Length penalty kicks in above ~800 characters.
        length_penalty = max(0.0, (len(chunk.text) - 800) / 4000.0)

        return (
            self.coverage_weight * coverage
            + self.heading_weight * heading_match
            + self.position_weight * position_bonus
            - length_penalty
        )

    def rerank(
        self, query: str, hits: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        scored = [
            (chunk, self.score(query, chunk), fused_score)
            for chunk, fused_score in hits
        ]
        # Deterministic: break ties by chunk id so equal scores never shuffle.
        scored.sort(key=lambda s: (-s[1], s[0].id))
        return [(chunk, fused_score) for chunk, _, fused_score in scored]


class NoOpReranker:
    """Keeps the fusion order untouched."""

    def rerank(
        self, query: str, hits: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        return list(hits)


_reranker: Reranker = HeuristicReranker()


def set_reranker(reranker: Reranker) -> None:
    """Swap the active reranker (e.g. to plug in a cross-encoder)."""
    global _reranker
    _reranker = reranker


def get_reranker(name: str | None = None) -> Reranker:
    """Return the reranker for a name, defaulting to the RERANKER env var."""
    name = config.RERANKER if name is None else name
    if name == "none":
        return NoOpReranker()
    return _reranker
