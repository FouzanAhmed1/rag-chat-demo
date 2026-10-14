"""Dependency-light vector store: numpy brute-force cosine similarity.

Rows of the embedding matrix are L2-normalized, so ranking is a single
matrix-vector dot product. The index persists to disk (npz + JSON metadata)
and reloads on startup.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    source: str
    page: int | None = None
    offset: int = 0


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.size else 0

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("chunks and embeddings row counts must match")
        if self.embeddings.size and embeddings.shape[1] != self.dim:
            raise ValueError("embedding dim does not match existing index")
        self.chunks.extend(chunks)
        if self.embeddings.size:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        else:
            self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(
        self, query_vec: np.ndarray, top_k: int = 4, min_score: float = 0.0
    ) -> list[tuple[Chunk, float]]:
        """Return up to top_k (chunk, cosine score) pairs above min_score."""
        if not self.chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        scores = self.embeddings @ q
        order = np.argsort(scores)[::-1]
        hits: list[tuple[Chunk, float]] = []
        for idx in order[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                continue
            hits.append((self.chunks[int(idx)], score))
        return hits

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)

    def save(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(data_dir, "index.npz"), embeddings=self.embeddings
        )
        meta = [asdict(c) for c in self.chunks]
        with open(os.path.join(data_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def load(self, data_dir: str) -> bool:
        npz_path = os.path.join(data_dir, "index.npz")
        json_path = os.path.join(data_dir, "index.json")
        if not (os.path.exists(npz_path) and os.path.exists(json_path)):
            return False
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        loaded = np.load(npz_path)["embeddings"]
        self.chunks = [Chunk(**m) for m in meta]
        self.embeddings = np.asarray(loaded, dtype=np.float32)
        return True
