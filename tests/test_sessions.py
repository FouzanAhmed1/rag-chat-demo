"""Multi-turn session tests: history accumulation, TTL expiry, retrieval
query rewriting, and per-session isolation through the API."""

import importlib
import os
import time

import pytest

from app.sessions import SessionStore, Turn, rewrite_query

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "sample_docs")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("rag_session_data")
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


def test_history_accumulates_in_order():
    store = SessionStore(max_turns=6, ttl_seconds=3600)
    store.record("s1", "q1", "a1")
    store.record("s1", "q2", "a2")
    history = store.history("s1")
    assert [(t.question, t.answer) for t in history] == [("q1", "a1"), ("q2", "a2")]


def test_history_trims_to_max_turns():
    store = SessionStore(max_turns=2, ttl_seconds=3600)
    store.record("s1", "q1", "a1")
    store.record("s1", "q2", "a2")
    store.record("s1", "q3", "a3")
    assert [t.question for t in store.history("s1")] == ["q2", "q3"]


def test_expired_session_is_swept():
    store = SessionStore(max_turns=6, ttl_seconds=3600)
    store.record("s1", "q1", "a1")
    assert store.history("s1") != []
    # Fake the clock: pretend the session went quiet long ago.
    store._sessions["s1"].last_seen = time.time() - 7200
    assert store.history("s1") == []
    assert "s1" not in store._sessions


def test_unknown_session_returns_empty():
    store = SessionStore()
    assert store.history("nope") == []


def test_rewrite_query_without_history():
    assert rewrite_query("how long?", []) == "how long?"


def test_rewrite_query_includes_recent_questions():
    history = [
        Turn("what is the return policy?", "30 days"),
        Turn("does it cover shoes?", "yes"),
    ]
    rewritten = rewrite_query("how do I start one?", history)
    assert "what is the return policy?" in rewritten
    assert "does it cover shoes?" in rewritten
    assert rewritten.endswith("how do I start one?")


def test_answer_question_receives_history():
    from app.embeddings import HashingEmbedder
    from app.rag import answer_question
    from test_hybrid import _synthetic_store

    store = _synthetic_store()
    embedder = HashingEmbedder(dim=1024)
    history = [Turn("what is the gadget policy?", "some earlier answer")]
    result = answer_question(
        "how long does it last?",
        store,
        embedder,
        history=history,
        session_id="sess-1",
        reranker="none",
    )
    assert result["debug"]["history_turns_used"] == 1
    assert "gadget policy" in result["debug"]["retrieval_query"]
    assert result["debug"]["session_id"] == "sess-1"


def test_sessions_are_isolated_via_api(client):
    headers_a = {"X-Session-Id": "session-a"}
    headers_b = {"X-Session-Id": "session-b"}

    r1 = client.post(
        "/chat", json={"question": "How do I reset my password?"}, headers=headers_a
    )
    assert r1.status_code == 200
    assert r1.json()["debug"]["history_turns_used"] == 0

    r2 = client.post(
        "/chat", json={"question": "What about the VPN?"}, headers=headers_a
    )
    assert r2.status_code == 200
    assert r2.json()["debug"]["history_turns_used"] == 1
    assert "reset my password" in r2.json()["debug"]["retrieval_query"]

    # A different session id starts with a clean slate.
    r3 = client.post(
        "/chat", json={"question": "What about the VPN?"}, headers=headers_b
    )
    assert r3.status_code == 200
    assert r3.json()["debug"]["history_turns_used"] == 0


def test_session_id_body_field_also_works(client):
    r1 = client.post(
        "/chat",
        json={"question": "How do I reset my password?", "session_id": "body-sess"},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/chat",
        json={"question": "And the VPN?", "session_id": "body-sess"},
    )
    assert r2.json()["debug"]["history_turns_used"] == 1
