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
4. **Retrieve** (`app/store.py`, `app/rag.py`): numpy brute-force cosine
   similarity, top-k with a minimum score floor. The index persists to
   `./data/index.npz` + `index.json` and reloads on startup. No vector DB.
5. **Answer** (`app/rag.py`): with `FAKE_LLM=1` (the default) answers are
   extractive, quoting the best-matching sentences with `[n]` citations. Set
   `FAKE_LLM=0` plus `LLM_API_KEY` to get real generated answers from any
   OpenAI-compatible chat endpoint.

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

## Config / env reference

All settings come from environment variables; see `.env.example` for the full
annotated list. The important ones:

| Variable | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 800 | Max characters per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap characters carried into the next chunk |
| `TOP_K` | 4 | Chunks retrieved per question |
| `MIN_SCORE` | 0.05 | Minimum cosine similarity to count as relevant |
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
junk queries are filtered), and the API (`/health`, `/documents`, `/ingest`
upload and rejection, `/chat` with citations, the UI at `/`). Everything runs
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
    ingest.py      PDF / txt / md loading
    rag.py         retrieval, extractive fake-LLM answers, real LLM answers
    static/        single-page chat UI (plain HTML/JS)
  sample_docs/     fictional Meridian Outdoors docs, indexed on startup
  tests/           pytest suite (offline)
  Dockerfile       one-command container build
  requirements.txt pinned, dependency-light
  .env.example     every env var documented
```
