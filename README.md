# Enterprise RAG Copilot

A production-grade Retrieval-Augmented Generation (RAG) system for asking grounded questions over private documents. Runs entirely on a laptop. No paid APIs required.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-blue)

---

## Features

| Category | What's implemented |
|---|---|
| **Ingestion** | PDF, TXT, Markdown, DOCX, PPTX, HTML; page-aware chunking; per-phase timing |
| **Retrieval** | Dense FAISS + sparse BM25 with Reciprocal Rank Fusion; optional cross-encoder reranking; metadata filtering by filename |
| **Generation** | Ollama (any local model) → flan-t5-base fallback → extractive; multi-turn conversation context; citation enforcement |
| **Groundedness** | Post-generation guard: low-confidence answers overridden to "I don't know"; n-gram overlap evaluation |
| **Evaluation** | Deterministic per-query scores: context relevance, answer groundedness, citation presence |
| **Auth** | X-API-Key header; configurable via env var; timing-safe comparison; rate limiting per IP |
| **Feedback** | Thumbs up/down + optional comment; linked to query_id and conversation_id; SQLite-backed |
| **Document lifecycle** | List, delete, re-ingest; FAISS index rebuilt safely on delete |
| **Conversation** | Multi-turn via conversation_id; TTL expiry; history injected into generation prompt |
| **Observability** | Trace ID per query; per-stage latency; structured JSONL logs; persistent SQLite metrics |
| **Token tracking** | Approximate input/output token counts per query; stored in metrics DB |
| **Streaming** | SSE streaming with phase events (retrieving → generating → evaluating → done) |
| **Frontend** | React + TypeScript + Tailwind; citation cards; eval badges; feedback buttons; document manager |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- [Ollama](https://ollama.ai) (optional but recommended — for best answer quality)

### Backend

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure (copy and edit as needed)
cp .env.example .env

# 3. (Optional) Pull a local LLM
ollama pull llama3

# 4. Start the API
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Docker Compose (full stack)

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## API Reference

All endpoints require `X-API-Key` header when `API_KEY` env var is set.

### `POST /ingest`

Upload a document for indexing.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your-key" \
  -F "file=@report.pdf"
```

Response includes `doc_id`, `chunks_stored`, per-phase timings.

### `POST /query`

Ask a question. Supports conversation history and source filtering.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "query": "What are the main findings?",
    "conversation_id": "abc-123",
    "source_filter": ["report.pdf"]
  }'
```

Response includes `answer`, `citations`, `eval` scores, `confidence`, token counts, and timing breakdown.

### `POST /query/stream`

Same as `/query` but streamed as Server-Sent Events. Events: `retrieving`, `generating`, `evaluating`, `done`.

### `POST /feedback`

Submit thumbs up/down linked to a query.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_id": "...", "rating": 1, "comment": "Very helpful"}'
```

### `GET /documents`

List all ingested documents with metadata (type, size, chunk count, ingest time).

### `DELETE /documents/{doc_id}`

Remove a document from all indexes. Rebuilds FAISS from remaining vectors.

### `GET /health`

Liveness check. Returns index size, model name, hybrid search status.

### `GET /metrics`

Persistent stats: total queries, avg latency, avg groundedness, backend status.

---

## Configuration

All settings are controlled via environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | _(empty)_ | Enable auth by setting a key |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP rate limit; 0 = disabled |
| `GENERATION_MODE` | `auto` | `auto`, `ollama`, `flan-t5`, `extractive` |
| `USE_HYBRID_SEARCH` | `true` | Enable BM25 + dense RRF fusion |
| `BM25_WEIGHT` | `0.3` | Weight for sparse results in fusion |
| `USE_RERANKER` | `false` | Cross-encoder reranking (needs ~90 MB model) |
| `GROUNDEDNESS_THRESHOLD` | `0.25` | Override to "I don't know" below this |
| `SIMILARITY_THRESHOLD` | `0.3` | Min cosine similarity to include a chunk |
| `TOP_K` | `5` | Chunks returned per query |
| `CHUNK_SIZE` | `512` | Chars per chunk |
| `DB_PATH` | `data/rag.db` | SQLite path for metrics, feedback, doc catalog |
| `CONVERSATION_TTL_SECONDS` | `3600` | Conversation expiry |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

**Key design decisions:**

- **Hybrid retrieval**: Dense cosine similarity (FAISS) + sparse BM25, fused with Reciprocal Rank Fusion. Catches both semantic and keyword matches.
- **Groundedness guard**: Post-generation, not just prompt instruction. If answer groundedness < threshold, response is overridden to "I don't know."
- **Local-first**: Everything runs on CPU. No OpenAI API required. Ollama with any open model is the recommended generation path.
- **SQLite for persistence**: Metrics, feedback, and document catalog survive restarts. No external database needed.
- **FAISS deletion**: IndexFlatIP doesn't support in-place deletion. We reconstruct the index from remaining vectors via `reconstruct()`. Correct, simple, O(N).

---

## Testing

```bash
pytest tests/ -v
```

9 test files covering ingestion, chunking, retrieval, generation, evaluation, streaming, and API endpoints (~1400 lines).

---

## Supported Document Formats

| Format | Library | Notes |
|---|---|---|
| PDF | pypdf | Page-wise extraction |
| TXT, Markdown | built-in | Whole file as page 1 |
| DOCX | python-docx | Paragraphs grouped into ~500-word sections |
| PPTX | python-pptx | One page per slide |
| HTML | beautifulsoup4 | Boilerplate tags stripped |

---

## Limitations

- FAISS runs in-process; not suitable for multi-instance horizontal scaling (use Qdrant or Weaviate for that).
- Conversation history is in-memory only; restarts clear sessions.
- flan-t5-base (~250 MB) is a small model — answers are adequate for extraction but not as fluent as large models.
- BM25 index is also in-memory; rebuilt from FAISS metadata on startup.
- No document chunking deduplication across re-ingestion of the same file.

---

## Future Work

- Persistent conversation store (Redis or SQLite)
- Streaming token-level generation from Ollama
- Document chunking deduplication
- Multi-user namespacing / tenant isolation
- Qdrant or Weaviate for production vector storage
- Evaluation harness with ground-truth QA pairs (RAGAS integration)
- Async background ingestion with job status endpoint
- PII detection and redaction before indexing
- CI/CD with pytest + Docker image push
