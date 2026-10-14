# rag-chat-demo

A small, deployable RAG chatbot over your own documents. Upload PDFs, text,
and Markdown files, ask questions in a chat UI, and get answers with clickable
source citations.

It runs fully offline out of the box: a lightweight hashing embedder and an
extractive answer mode mean no API keys, no downloads, and no network calls.
When you are ready for production-quality answers, point it at any
OpenAI-compatible embeddings and chat endpoint with environment variables.

## Architecture

```
ingest  ->  chunk  ->  embed  ->  retrieve  ->  answer
```

1. **Ingest** (`app/ingest.py`): loads `.pdf` (via pypdf, per page), `.txt`,
   and `.md` files. `POST /ingest` accepts uploads; `sample_docs/` is
   auto-loaded at startup.
2. **Chunk** (`app/chunking.py`): splits text into sentences, packs them into
   chunks of at most `CHUNK_SIZE` characters, and carries the trailing
   sentences (up to `CHUNK_OVERLAP` characters) into the next chunk so context
   survives chunk boundaries.
3. **Embed** (`app/embeddings.py`): swappable interface. The default is
   `HashingEmbedder`, an offline TF-style embedder with MD5 feature hashing.
   Optional backends: any OpenAI-compatible embeddings endpoint, or
   sentence-transformers behind a flag.
4. **Retrieve** (`app/store.py`, `app/hybrid.py`, `app/rerank.py`, `app/rag.py`):
   hybrid retrieval fuses numpy brute-force cosine similarity with a
   dependency-free BM25 keyword ranker via weighted reciprocal rank fusion
   (`HYBRID_ALPHA`: 0.0 = pure BM25, 1.0 = pure vector, 0.5 = even blend).
   An optional rerank stage (`RERANKER=heuristic` by default, `none` to
   skip) reorders the fused hits with cheap deterministic signals. The
   index persists to `./data/index.npz` + `index.json` and reloads on
   startup. No vector DB.
5. **Answer** (`app/rag.py`): with `FAKE_LLM=1` (the default) answers are
   extractive, quoting the best-matching sentences with `[n]` citations. Set
   `FAKE_LLM=0` plus `LLM_API_KEY` to get real generated answers from any
   OpenAI-compatible chat endpoint. `POST /chat/stream` streams the same
   answers token by token over server-sent events.
6. **Converse** (`app/sessions.py`): the UI keeps a session id (sent as the
   `X-Session-Id` header), and the server remembers the last
   `SESSION_MAX_TURNS` turns per session so follow-ups like "what about the
   VPN?" inherit the topic of earlier questions. Idle sessions expire after
   `SESSION_TTL_SECONDS`.

The web UI (`app/static/`) is plain HTML and JS, no framework and no CDN,
served by FastAPI at `/`.

## Setup on a fresh machine

Requirements: Python 3.10+ and pip.

```bash
git clone <this-repo>          # or unzip the project folder
cd rag-chat-demo
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and tweak values (chunk size,
retrieval thresholds, model endpoints). Everything works with defaults.

## One-command local run

```bash
uvicorn app.main:app
```

Open http://localhost:8000. The four sample docs (a fictional company,
Meridian Outdoors: handbook, IT FAQ, returns policy, security policy) are
indexed automatically, so you can immediately ask things like:

- "How do I reset my password?"
- "What is the return policy?"
- "How many PTO days do full-time employees get?"

Upload your own files with the file picker in the UI, or via curl:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/your/document.pdf"
```

## Usage examples

Check the index:

```bash
curl http://localhost:8000/health
# {"status":"ok","documents":4,"chunks":12,"embedder":"HashingEmbedder","fake_llm":true}

curl http://localhost:8000/documents
```

Ask a question:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I set up the VPN?"}'
```

Response shape:

```json
{
  "answer": "Here is what I found in your documents:\n- Download the Meridian VPN client ... [1]\n\nSources: [1] it_faq.md",
  "citations": [
    {"ref": 1, "source": "it_faq.md", "page": null, "score": 0.42,
     "excerpt": "Download the Meridian VPN client from the software center ..."}
  ]
}
```

Clear the index and re-ingest:

```bash
curl -X DELETE http://localhost:8000/index
```

### Streaming answers

`POST /chat/stream` accepts the same body as `/chat` and returns
`text/event-stream`. The first event carries citations and debug info, then
one `data: {"token": "..."}` event arrives per token, and the stream ends
with `data: [DONE]`:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'
# data: {"citations": [...], "debug": {...}}
# data: {"token": "Here "}
# data: {"token": "is "}
# ...
# data: [DONE]
```

If a token generator fails mid-stream you get `data: {"error": "..."}`
instead of `[DONE]`; the web UI falls back to the non-streaming `/chat`
endpoint in that case.

The offline fake LLM streams the extractive answer word by word. A real
streaming backend must implement the `StreamingLLM` interface in
`app/rag.py`: a `stream_answer(question, context_blocks, history)` method
that yields tokens in generation order. The default real backend
(`llm_answer_stream`) does this against any OpenAI-compatible
`/chat/completions` endpoint with `"stream": true`.

### Hybrid retrieval

Every question is scored twice: cosine similarity from the embedder and
BM25 keyword scores from a small built-in implementation (no extra
dependencies). The two rankings are fused with reciprocal rank fusion,
`score = sum(weight / (RRF_K + rank))` with `RRF_K=60`:

```bash
# Pure BM25 (great for exact rare-term matches)
HYBRID_ALPHA=0 uvicorn app.main:app
# Even blend (default)
HYBRID_ALPHA=0.5 uvicorn app.main:app
# Pure vector
HYBRID_ALPHA=1 uvicorn app.main:app
```

Each citation in the response lists the `rankers` that surfaced it, and the
`debug` payload shows the per-ranker ranks and raw scores plus the alpha
that was used, so you can see exactly why a chunk won.

### Multi-turn conversation

Send an `X-Session-Id` header (or a `session_id` field in the JSON body)
and the server keeps your last few turns. Follow-up questions are answered
with the conversation history in the prompt, and retrieval uses a query
rewritten from your recent questions, so "how long does that take?" works
right after "what is the return policy?":

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" -H "X-Session-Id: demo-1" \
  -d '{"question": "What is the return policy?"}'
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" -H "X-Session-Id: demo-1" \
  -d '{"question": "How long does that take?"}'
```

Sessions are in-memory only (a restart clears them) and expire after
`SESSION_TTL_SECONDS` of inactivity. The `debug` block in each response
reports `history_turns_used` and the `retrieval_query` that was actually
searched.

### Reranking

After fusion, an optional rerank stage reorders the hits. The default
`RERANKER=heuristic` is a deterministic, LLM-free scorer combining
query-term coverage, markdown heading matches, a small position bonus for
early-document chunks, and a length penalty for very long chunks. Set
`RERANKER=none` to keep the raw fusion order.

To plug in a real cross-encoder (e.g. from sentence-transformers),
implement the `Reranker` interface and register it at startup:

```python
from app.rerank import Reranker, set_reranker

class CrossEncoderReranker(Reranker):
    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, hits):
        # hits: list of (chunk, score); return them reordered
        pairs = [(query, chunk.text) for chunk, _ in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
        return [hit for hit, _ in ranked]

# in app/main.py, or any startup hook:
set_reranker(CrossEncoderReranker())
```

The interface contract is just `rerank(query, hits) -> hits`: same
`(chunk, score)` tuples, best first, deterministic for the same inputs.

## Config / env reference

All settings come from environment variables; see `.env.example` for the full
annotated list. The important ones:

| Variable | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 800 | Max characters per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap characters carried into the next chunk |
| `TOP_K` | 4 | Chunks retrieved per question |
| `MIN_SCORE` | 0.05 | Minimum cosine similarity to count as relevant |
| `HYBRID_ALPHA` | 0.5 | Retrieval blend: 0.0 = pure BM25, 1.0 = pure vector |
| `RRF_K` | 60 | Reciprocal rank fusion constant |
| `RERANKER` | heuristic | Rerank stage: `heuristic` or `none` |
| `SESSION_MAX_TURNS` | 6 | Conversation turns kept per session |
| `SESSION_TTL_SECONDS` | 1800 | Idle seconds before a session expires |
| `EMBED_PROVIDER` | hash | `hash` (offline), `openai`, or `sentence-transformers` |
| `EMBED_DIM` | 1024 | Vector dim for the hash embedder |
| `EMBED_API_KEY` / `EMBED_BASE_URL` / `EMBED_MODEL` | - | OpenAI-compatible embeddings endpoint |
| `USE_SENTENCE_TRANSFORMERS` | 0 | Set to 1 to use a local HF model (needs `pip install sentence-transformers`) |
| `FAKE_LLM` | 1 | 1 = offline extractive answers; 0 = call a real chat model |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | - | OpenAI-compatible chat endpoint (`gpt-4o-mini` default) |
| `DATA_DIR` | ./data | Where the index and uploads are stored |
| `SAMPLE_DOCS_DIR` | ./sample_docs | Folder auto-ingested at startup |

No secrets are ever hard-coded. If you set `EMBED_PROVIDER=openai` you need
`EMBED_API_KEY`; if you set `FAKE_LLM=0` you need `LLM_API_KEY`. Any
OpenAI-compatible provider works (OpenAI, Azure, Ollama, vLLM, Together, ...),
just set the base URL.

## Running the tests

```bash
pytest
```

The suite covers chunking (overlap behavior, size limits, edge cases),
retrieval ranking (each sample-doc query returns the expected document top-1,
junk queries are filtered), hybrid fusion (a synthetic corpus where BM25
outranks vector; alpha 1.0/0.0 reproduce the pure rankings), the heuristic
reranker (term-coverage promotion, determinism, `none` preserves fusion
order), multi-turn sessions (history accumulation, TTL expiry, per-session
isolation), SSE streaming (ordered token events, `[DONE]`, stream equals
non-stream answer), and the API (`/health`, `/documents`, `/ingest` upload
and rejection, `/chat` with citations, the UI at `/`). Everything runs
offline: `FAKE_LLM=1` and the hash embedder are forced in tests.

## Deploy

The repo ships a production-ready `Dockerfile`:

```dockerfile
# builds the image, serves on port 8000
docker build -t rag-chat-demo .
docker run -p 8000:8000 rag-chat-demo
```

**Railway:** `railway init`, then `railway up`. Railway detects the Dockerfile
and deploys it. Set env vars (e.g. `LLM_API_KEY`, `FAKE_LLM=0`) in the Railway
dashboard under Variables. Note: the default `DATA_DIR=./data` is ephemeral on
Railway; attach a volume and set `DATA_DIR` to the mount path if you want the
index to survive restarts.

**Render:** create a new Web Service from this repo, choose the Docker runtime.
Set the port to 8000. Add a persistent disk mounted at `/code/data` and set
`DATA_DIR=/code/data` if you want index persistence. Add env vars in the
dashboard.

**Fly.io:** `fly launch` (it will pick up the Dockerfile), then `fly deploy`.
For index persistence, create a volume (`fly volumes create rag_data`) and set
`DATA_DIR` to the mount path in `fly.toml`. Secrets go via `fly secrets set
LLM_API_KEY=...` rather than env vars.

For real deployments, set `FAKE_LLM=0` with an `LLM_API_KEY`, and consider
`EMBED_PROVIDER=openai` (or `USE_SENTENCE_TRANSFORMERS=1`) for better retrieval
quality than the offline hash embedder.

## Project layout

```
rag-chat-demo/
  app/
    main.py        FastAPI app, routes, startup auto-load
    config.py      env-driven settings
    chunking.py    sentence-aware chunking with overlap
    embeddings.py  swappable embedders (hash default, openai, sentence-transformers)
    store.py       numpy cosine-similarity vector store, persisted to disk
    hybrid.py      dependency-free BM25 + weighted reciprocal rank fusion
    rerank.py      Reranker interface + heuristic reranker (plug in a
                   cross-encoder via set_reranker())
    sessions.py    per-session in-memory conversation history with TTL
    ingest.py      PDF / txt / md loading
    rag.py         retrieval, extractive fake-LLM answers, real LLM answers,
                   SSE token streaming (StreamingLLM interface)
    static/        single-page chat UI (plain HTML/JS, streams tokens)
  sample_docs/     fictional Meridian Outdoors docs, indexed on startup
  tests/           pytest suite (offline)
  Dockerfile       one-command container build
  requirements.txt pinned, dependency-light
  .env.example     every env var documented
```
