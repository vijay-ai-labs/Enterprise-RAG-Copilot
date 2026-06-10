# Architecture — Enterprise RAG Copilot

## System Overview

```
Browser (React + TypeScript)
        │
        │  HTTP / SSE
        ▼
   Nginx (port 3000)
        │  reverse proxy
        ▼
 FastAPI Backend (port 8000)
        │
        ├── Auth middleware (X-API-Key)
        ├── Rate limiter (sliding window, per IP)
        │
        ├─ POST /ingest ──────────────────────────────────────┐
        │     Extract → Chunk → Embed → FAISS + BM25 + DocDB  │
        │                                                       │
        ├─ POST /query ────────────────────────────────────────┤
        │     Embed → Hybrid Retrieve → Rerank → Generate      │
        │     → Groundedness Guard → Evaluate → Log → Respond  │
        │                                                       │
        ├─ POST /feedback ─────────────────────────────────────┤
        │     Persist rating + comment to SQLite                │
        │                                                       │
        ├─ GET /documents ─────────────────────────────────────┤
        │     Read document catalog from SQLite                 │
        │                                                       │
        └─ DELETE /documents/{id} ─────────────────────────────┘
              Remove from FAISS + BM25 + SQLite catalog
```

---

## Component Map

```
app/
├── main.py              FastAPI app, all endpoints, middleware wiring
├── config.py            Pydantic Settings (env-var driven)
├── schemas.py           All Pydantic request/response models
│
├── auth.py              X-API-Key dependency + sliding window rate limiter
├── conversation_store.py  In-memory multi-turn history with TTL
├── feedback_store.py    SQLite-backed feedback (rating, comment, chunk_ids)
├── metrics_db.py        SQLite-backed per-query telemetry
├── document_store.py    SQLite-backed document catalog (lifecycle)
│
├── ingestion.py         PDF / TXT / MD / DOCX / PPTX / HTML → (text, page) tuples
├── chunking.py          Sliding-window chunker with whitespace-snap overlap
├── embeddings.py        sentence-transformers singleton (all-MiniLM-L6-v2)
├── vectorstore.py       FAISS IndexFlatIP + JSON metadata; remove_by_filename()
├── bm25_store.py        rank_bm25 sparse index; parallel to FAISS
├── retriever.py         Hybrid RRF fusion; threshold filter; MMR; metadata filter
├── reranker.py          CrossEncoder reranking (optional; off by default)
├── generator.py         Ollama → flan-t5-base → extractive; conversation history
├── evaluator.py         Deterministic scores: relevance, groundedness, citations
└── logger.py            Structured JSONL query logging
```

---

## Query Pipeline (detailed)

```
User query
    │
    ▼ embed_model.encode()
Query vector (384-dim, L2-normalised)
    │
    ├─── Dense retrieval (FAISS)  ──────────────────────────────┐
    │    IndexFlatIP.search(query_vec, fetch_k)                  │
    │    → top-N (chunk, cosine_score) pairs                     │
    │                                                             │
    ├─── Sparse retrieval (BM25)  ──────────────────────────────┤
    │    BM25Okapi.get_scores(tokens)                            │
    │    → top-N (chunk, normalised_score) pairs                 │
    │                                                             │
    └─── RRF Fusion ─────────────────────────────────────────────┘
         score(d) = Σ weight_i / (60 + rank_i(d))
         dense_weight = 1 - bm25_weight (default: 0.7)
         sparse_weight = bm25_weight (default: 0.3)
              │
              ▼ threshold filter (similarity_threshold=0.3)
              ▼ metadata filter (source_filter filenames)
              ▼ dedup by chunk_id
              ▼ optional CrossEncoder reranking
              ▼ top_k results
                   │
                   ▼
              generator.generate(query, chunks, history)
                   │
                   ├── ollama (HTTP to local Ollama)
                   ├── flan-t5-base (HuggingFace transformers)
                   └── extractive (keyword overlap, zero model)
                   │
                   ▼ answer
                   │
              groundedness_guard()
                   │ if answer_groundedness < 0.25 → "I don't know"
                   ▼
              evaluator.evaluate()
                   │ context_relevance, answer_groundedness, citation_presence
                   ▼
              metrics_db.record()  ← persistent SQLite
              log_query()          ← JSONL
              conversation_store.add_turn()  ← in-memory
                   │
                   ▼
              QueryResponse (answer, citations, eval, timings, confidence, tokens)
```

---

## Hybrid Retrieval — Reciprocal Rank Fusion

RRF avoids score normalisation issues when combining heterogeneous ranking signals:

```
RRF_score(d) = dense_weight × 1/(60 + rank_dense(d))
             + sparse_weight × 1/(60 + rank_sparse(d))
```

`k=60` is the standard constant that dampens the impact of top-ranked items. Documents appearing in both lists are rewarded; documents unique to one list still contribute.

**Why not score interpolation?**
FAISS cosine scores and BM25 Okapi scores have incompatible scales. RRF operates on ranks, not scores, making it scale-free.

---

## Groundedness Guard

The prompt instructs the LLM to say "I don't know" — but prompt compliance is not guaranteed. The post-generation guard enforces it:

```python
if answer_groundedness < GROUNDEDNESS_THRESHOLD and "i don't know" not in answer.lower():
    answer = "I don't know based on the provided documents."
```

`answer_groundedness` is computed as the fraction of answer sentences sharing at least one meaningful n-gram (unigram or bigram, stopwords excluded) with the retrieved context. A score of 0.25 means fewer than 1 in 4 answer sentences can be traced to the context — almost certainly hallucination.

---

## Document Deletion

FAISS `IndexFlatIP` does not support in-place vector removal. Deletion rebuilds the index:

1. Identify keep-indices: `[i for i, c in enumerate(chunks) if c.filename != filename]`
2. Extract their vectors: `index.reconstruct(i)` for each keep index
3. Create new `IndexFlatIP`, add extracted vectors
4. Update `_chunks` list to match
5. Persist new index to disk

**Complexity**: O(N) where N = total vectors. Acceptable for small-to-medium collections. For large-scale production, use a vector DB with native delete support (Qdrant, Weaviate, Pinecone).

---

## Persistence

| Store | Backend | Restarts |
|---|---|---|
| FAISS index | `indexes/faiss.index` (binary) | Survives |
| Chunk metadata | `indexes/metadata.json` | Survives |
| BM25 index | In-memory, rebuilt from metadata.json on startup | Survives via rebuild |
| Document catalog | `data/rag.db` (SQLite) | Survives |
| Feedback | `data/rag.db` (SQLite) | Survives |
| Query metrics | `data/rag.db` (SQLite) | Survives |
| Conversation history | In-memory with TTL | Lost on restart |
| Query logs | `logs/queries.jsonl` | Survives |

---

## Authentication Flow

```
Request → RateLimitMiddleware → Router → require_api_key dependency
              │                                    │
              │ if IP exceeded limit               │ if API_KEY env set:
              └─ 429 Too Many Requests             │   compare X-API-Key header
                                                   │   using secrets.compare_digest()
                                                   │   (timing-safe)
                                                   └─ 401 if mismatch / missing
```

If `API_KEY` is not set, all requests are allowed (dev/open mode).

---

## Frontend Architecture

```
App.tsx
├── Header (health status, version)
├── Sidebar
│   ├── DocumentUpload.tsx
│   │   ├── Drag-drop zone (PDF/TXT/MD/DOCX/PPTX/HTML)
│   │   ├── Ingestion timing breakdown
│   │   └── Document list with delete buttons
│   └── System Status panel (index size, model, metrics)
└── ChatPanel.tsx
    ├── Message thread (user + assistant)
    │   ├── CitationCard.tsx (expandable, score bar)
    │   ├── EvalBadge.tsx (quality scores, timing)
    │   └── Feedback buttons (thumbs up/down)
    └── Input bar (Enter to send, Shift+Enter newline)
```

Conversation ID is generated on component mount (`crypto.randomUUID()`). All queries within a session share the same ID, enabling multi-turn context without server-side session management.

---

## Scalability Notes

| Bottleneck | Current approach | Production path |
|---|---|---|
| Vector index | FAISS in-process | Qdrant / Weaviate with horizontal scaling |
| BM25 index | In-memory | Elasticsearch / OpenSearch |
| Embeddings | CPU, single process | GPU inference server (TGI, vLLM) |
| Generation | Ollama on same host | Dedicated inference server or API |
| Database | SQLite | PostgreSQL |
| Auth | Single shared key | OAuth2 / JWT with user namespacing |
