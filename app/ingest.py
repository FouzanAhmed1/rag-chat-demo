"""Document ingestion: PDF (.pdf via pypdf), .txt and .md files.

Each page (PDF) or whole file (txt/md) becomes one or more chunks after the
chunking step in main.py.
"""

from __future__ import annotations

import os

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _read_txt(path: str) -> list[tuple[str, int | None]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [(f.read(), None)]


def _read_pdf(path: str) -> list[tuple[str, int | None]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages: list[tuple[str, int | None]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((text, i + 1))  # 1-based page numbers
    return pages


def load_document(path: str) -> list[tuple[str, int | None]]:
    """Return a list of (text, page) segments for a document."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return _read_txt(path)
    raise ValueError(f"Unsupported file type: {suffix!r} (supported: .pdf, .txt, .md)")


def doc_id_for(path: str) -> str:
    return os.path.basename(path)
