"""API tests via FastAPI TestClient, fully offline (FAKE_LLM=1)."""

import importlib
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "sample_docs")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("rag_data")
    os.environ["FAKE_LLM"] = "1"
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["SAMPLE_DOCS_DIR"] = SAMPLE_DIR

    import app.config
    import app.main

    importlib.reload(app.config)
    importlib.reload(app.main)

    from fastapi.testclient import TestClient

    with TestClient(app.main.app) as test_client:
        yield test_client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["documents"] >= 4  # the four sample docs auto-loaded
    assert body["chunks"] > 0
    assert body["fake_llm"] is True


def test_documents_lists_sample_docs(client):
    res = client.get("/documents")
    assert res.status_code == 200
    docs = res.json()["documents"]
    assert "it_faq.md" in docs
    assert "employee_handbook.md" in docs


def test_ingest_upload_txt(client):
    res = client.post(
        "/ingest",
        files={"file": ("note.txt", b"Packing peanuts are banned from the warehouse.", "text/plain")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "note.txt"
    assert body["chunks_added"] >= 1
    assert body["skipped"] is False


def test_ingest_rejects_unsupported_type(client):
    res = client.post(
        "/ingest",
        files={"file": ("evil.exe", b"not a document", "application/octet-stream")},
    )
    assert res.status_code == 400


def test_chat_returns_answer_with_citations(client):
    res = client.post("/chat", json={"question": "How do I reset my password?"})
    assert res.status_code == 200
    body = res.json()
    assert "answer" in body and body["answer"].strip()
    assert "citations" in body
    assert len(body["citations"]) >= 1
    sources = [c["source"] for c in body["citations"]]
    assert "it_faq.md" in sources
    assert "reset" in body["answer"].lower() or "password" in body["answer"].lower()


def test_chat_no_relevant_docs(client):
    res = client.post("/chat", json={"question": "zebra quantum pancakes interstellar"})
    assert res.status_code == 200
    body = res.json()
    assert "could not find" in body["answer"].lower()
    assert body["citations"] == []


def test_chat_validates_empty_question(client):
    res = client.post("/chat", json={"question": ""})
    assert res.status_code == 422


def test_ui_served_at_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "rag-chat-demo" in res.text
