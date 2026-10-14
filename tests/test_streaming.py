"""SSE streaming tests: ordered token events, the [DONE] terminator,
stream/non-stream answer equality, and session recording."""

import importlib
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "sample_docs")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("rag_stream_data")
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


def _parse_sse(text):
    """Return the list of data payloads in stream order."""
    payloads = []
    for part in text.split("\n\n"):
        for line in part.strip().splitlines():
            if line.startswith("data:"):
                payloads.append(line[5:].strip())
    return payloads


def test_fake_answer_stream_matches_fake_answer():
    from test_hybrid import _synthetic_store

    from app.embeddings import HashingEmbedder
    from app.rag import fake_answer, fake_answer_stream, retrieve

    store = _synthetic_store()
    embedder = HashingEmbedder(dim=1024)
    hits = retrieve("gadget gizmo", store, embedder, reranker="none")
    expected = fake_answer("gadget gizmo", hits)["answer"]

    tokens = list(fake_answer_stream("gadget gizmo", hits))
    assert len(tokens) > 5  # the fake LLM really does yield many chunks
    assert "".join(tokens) == expected


def test_stream_endpoint_events_ordered_with_done(client):
    res = client.post(
        "/chat/stream", json={"question": "How do I reset my password?"}
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    payloads = _parse_sse(res.text)
    assert payloads, "expected SSE events"
    assert payloads[-1] == "[DONE]"

    first = json.loads(payloads[0])
    assert "citations" in first
    assert any(c["source"] == "it_faq.md" for c in first["citations"])

    tokens = [json.loads(p)["token"] for p in payloads[1:-1]]
    assert len(tokens) > 5
    streamed_answer = "".join(tokens)

    # The streamed answer must equal the non-streaming answer.
    plain = client.post(
        "/chat", json={"question": "How do I reset my password?"}
    ).json()
    assert streamed_answer == plain["answer"]


def test_stream_records_session_turns(client):
    headers = {"X-Session-Id": "stream-sess-1"}
    r1 = client.post(
        "/chat/stream",
        json={"question": "How do I reset my password?"},
        headers=headers,
    )
    assert r1.status_code == 200
    first_debug = json.loads(_parse_sse(r1.text)[0])["debug"]
    assert first_debug["history_turns_used"] == 0

    r2 = client.post(
        "/chat/stream",
        json={"question": "What about the VPN?"},
        headers=headers,
    )
    assert r2.status_code == 200
    second_debug = json.loads(_parse_sse(r2.text)[0])["debug"]
    assert second_debug["history_turns_used"] == 1
    assert "reset my password" in second_debug["retrieval_query"]


def test_stream_junk_question_still_terminates(client):
    res = client.post(
        "/chat/stream", json={"question": "zebra quantum pancakes interstellar"}
    )
    assert res.status_code == 200
    payloads = _parse_sse(res.text)
    assert payloads[-1] == "[DONE]"
    first = json.loads(payloads[0])
    assert first["citations"] == []
    answer = "".join(json.loads(p)["token"] for p in payloads[1:-1])
    assert "could not find" in answer.lower()
