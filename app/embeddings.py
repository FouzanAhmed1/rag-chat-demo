"""Swappable embedding interface.

Default: HashingEmbedder, a lightweight TF-style embedder that works fully
offline with no downloads. Optional backends:

- "openai": calls any OpenAI-compatible embeddings endpoint
  (EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL).
- "sentence-transformers": uses a local HF model when
  USE_SENTENCE_TRANSFORMERS=1 and sentence-transformers is installed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Protocol

import numpy as np

from app import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array, rows L2-normalized."""
        ...


class HashingEmbedder:
    """Offline term-frequency embedder with MD5 feature hashing.

    Each token increments a hashed dimension by its count, then the vector is
    L2-normalized so dot products are cosine similarities. Good enough for a
    keyword-driven demo and for deterministic, dependency-free tests.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for token, count in Counter(tokens).items():
                vecs[i, self._index(token)] += float(count)
            norm = float(np.linalg.norm(vecs[i]))
            if norm > 0:
                vecs[i] /= norm
        return vecs


class OpenAICompatibleEmbedder:
    """Embeddings via any OpenAI-compatible /embeddings endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise ValueError("EMBED_API_KEY is required for the openai embedder")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        items = resp.json()["data"]
        items.sort(key=lambda d: d["index"])
        vecs = np.array([d["embedding"] for d in items], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (optional heavy dep)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "sentence-transformers is not installed; pip install "
                "sentence-transformers, or use the default hash embedder"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder() -> Embedder:
    """Pick the embedder from env config."""
    provider = config.EMBED_PROVIDER
    if config.USE_SENTENCE_TRANSFORMERS:
        provider = "sentence-transformers"
    if provider == "hash":
        return HashingEmbedder(dim=config.EMBED_DIM)
    if provider == "openai":
        return OpenAICompatibleEmbedder(
            api_key=config.EMBED_API_KEY,
            base_url=config.EMBED_BASE_URL,
            model=config.EMBED_MODEL,
        )
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown EMBED_PROVIDER: {provider!r}")
