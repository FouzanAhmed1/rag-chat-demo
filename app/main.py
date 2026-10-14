"""FastAPI app: ingest -> chunk -> embed -> retrieve -> answer.

Endpoints:
    GET  /              single-page chat UI
    GET  /health        status plus index stats
    GET  /documents     list of ingested documents
    POST /ingest        upload a .pdf / .txt / .md file
    POST /chat          ask a question, get an answer with citations
    DELETE /index       clear the index

On startup the persisted index is reloaded and any new files in
SAMPLE_DOCS_DIR are auto-ingested.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app.chunking import chunk_text
from app.embeddings import build_embedder
from app.ingest import SUPPORTED_SUFFIXES, doc_id_for, load_document
from app.rag import answer_question
from app.sessions import sessions
from app.store import Chunk, VectorStore

store = VectorStore()
embedder = build_embedder()
ingested_docs: set[str] = set()


def _chunk_and_add(doc_id: str, source: str, segments: list[tuple[str, int | None]]) -> int:
    """Chunk document segments, embed them, and add them to the store."""
    chunks: list[Chunk] = []
    for text, page in segments:
        offset = 0
        for piece in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=piece,
                    source=source,
                    page=page,
                    offset=offset,
                )
            )
            offset += len(piece)
    if not chunks:
        return 0
    vectors = embedder.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    return len(chunks)


def ingest_path(path: str) -> dict[str, Any]:
    """Ingest one file from disk; idempotent per doc id."""
    doc_id = doc_id_for(path)
    if doc_id in ingested_docs:
        return {"doc_id": doc_id, "chunks_added": 0, "skipped": True}
    segments = load_document(path)
    n = _chunk_and_add(doc_id, source=doc_id, segments=segments)
    if n:
        ingested_docs.add(doc_id)
    return {"doc_id": doc_id, "chunks_added": n, "skipped": False}


def auto_load_sample_docs() -> int:
    total = 0
    sample_dir = config.SAMPLE_DOCS_DIR
    if not os.path.isdir(sample_dir):
        return 0
    for name in sorted(os.listdir(sample_dir)):
        path = os.path.join(sample_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in SUPPORTED_SUFFIXES:
            total += ingest_path(path)["chunks_added"]
    return total


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if store.load(config.DATA_DIR):
        for chunk in store.chunks:
            ingested_docs.add(chunk.doc_id)
    auto_load_sample_docs()
    store.save(config.DATA_DIR)
    yield


app = FastAPI(title="rag-chat-demo", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    debug: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "documents": len(ingested_docs),
        "chunks": len(store.chunks),
        "embedder": type(embedder).__name__,
        "fake_llm": config.FAKE_LLM,
    }


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    return {"documents": sorted(ingested_docs)}


@app.post("/ingest")
async def ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}; use .pdf, .txt, or .md",
        )
    uploads_dir = os.path.join(config.DATA_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    dest = os.path.join(uploads_dir, filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        result = ingest_path(dest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save(config.DATA_DIR)
    result["source"] = filename
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest, x_session_id: str | None = Header(default=None)
) -> dict[str, Any]:
    session_id = req.session_id or x_session_id
    history = sessions.history(session_id) if session_id else []
    result = answer_question(
        req.question,
        store,
        embedder,
        top_k=req.top_k,
        history=history,
        session_id=session_id,
    )
    if session_id:
        sessions.record(session_id, req.question, result["answer"])
    return result


@app.delete("/index")
def clear_index() -> dict[str, Any]:
    store.clear()
    ingested_docs.clear()
    store.save(config.DATA_DIR)
    return {"status": "cleared"}


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "index.html"),
        media_type="text/html",
    )


@app.get("/app.js", include_in_schema=False)
def ui_js() -> FileResponse:
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "app.js"),
        media_type="application/javascript",
    )
