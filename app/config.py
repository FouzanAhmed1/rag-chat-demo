"""Environment-driven configuration for rag-chat-demo.

Everything is read from environment variables; nothing secret lives in code.
See .env.example for the full list.
"""

from __future__ import annotations

import os


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Chunking
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 120)

# Retrieval
TOP_K: int = _get_int("TOP_K", 4)
MIN_SCORE: float = _get_float("MIN_SCORE", 0.05)
# Hybrid retrieval blend: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = even RRF.
HYBRID_ALPHA: float = _get_float("HYBRID_ALPHA", 0.5)
# Reciprocal rank fusion constant.
RRF_K: int = _get_int("RRF_K", 60)
# Optional rerank stage after fusion: "heuristic" (default) or "none".
RERANKER: str = os.environ.get("RERANKER", "heuristic").lower()
# Conversation sessions: how many turns of history to keep, and how long an
# idle session lives before it is swept.
SESSION_MAX_TURNS: int = _get_int("SESSION_MAX_TURNS", 6)
SESSION_TTL_SECONDS: int = _get_int("SESSION_TTL_SECONDS", 1800)

# Offline hashing embedder
EMBED_DIM: int = _get_int("EMBED_DIM", 1024)
# "hash" (default, offline), "openai" (OpenAI-compatible embeddings endpoint),
# "sentence-transformers" (needs USE_SENTENCE_TRANSFORMERS=1 and the extra dep)
EMBED_PROVIDER: str = os.environ.get("EMBED_PROVIDER", "hash").lower()
USE_SENTENCE_TRANSFORMERS: bool = os.environ.get("USE_SENTENCE_TRANSFORMERS", "0") == "1"
EMBED_API_KEY: str = os.environ.get("EMBED_API_KEY", "")
EMBED_BASE_URL: str = os.environ.get("EMBED_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# Chat / answer generation. FAKE_LLM=1 (default) answers extractively from
# retrieved chunks, so the demo and the tests run with no network and no key.
FAKE_LLM: bool = os.environ.get("FAKE_LLM", "1") == "1"
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# Storage
DATA_DIR: str = os.environ.get("DATA_DIR", "./data")
SAMPLE_DOCS_DIR: str = os.environ.get("SAMPLE_DOCS_DIR", "./sample_docs")
