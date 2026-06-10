"""
FastAPI application — Enterprise RAG Copilot v3.0

Endpoints:
  GET    /health                          — liveness + index + cache stats
  GET    /metrics                         — runtime stats (persistent SQLite)
  GET    /metrics/timeseries              — hourly query volume / latency / groundedness
  POST   /ingest                          — upload document (sync)
  POST   /ingest/async                    — upload document (background task)
  GET    /ingest/tasks/{task_id}          — poll background ingest status
  POST   /query                           — grounded answer with conversation history
  POST   /query/stream                    — same as /query, streamed as SSE
  POST   /feedback                        — thumbs up/down rating + optional comment
  GET    /documents                       — list all ingested documents
  DELETE /documents/{doc_id}             — remove document from index
  GET    /conversations                   — list recent conversations
  GET    /conversations/{conversation_id} — full turn history for a conversation
  DELETE /conversations/{conversation_id} — clear a conversation
  POST   /cache/clear                     — flush the semantic query cache
"""

from __future__ import annotations

import json as _json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth import rate_limiter, require_api_key
from app.bm25_store import BM25Store
from app.chunking import chunk_document
from app.config import settings
from app.conversation_store import conversation_store
from app.document_store import DocumentStore
from app.embeddings import EmbeddingModel
from app.evaluator import Evaluator
from app.feedback_store import FeedbackStore
from app.generator import GeneratorChain, approx_tokens, get_backend_status
from app.hyde import generate_hyde_query
from app.ingestion import extract_text
from app.logger import log_query, logger
from app.metrics_db import MetricsDB
from app.query_cache import query_cache
from app.retriever import Retriever
from app.schemas import (
    BackendStatus,
    CacheStats,
    ConversationHistoryResponse,
    ConversationListResponse,
    ConversationSummary,
    ConversationTurn,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    EvalResult,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    IngestTaskStatus,
    IngestTimings,
    LatencyPercentiles,
    MetricsResponse,
    MetricsTimeseriesPoint,
    MetricsTimeseriesResponse,
    QueryRequest,
    QueryResponse,
    QueryTimings,
)
from app.vectorstore import VectorStore

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

vector_store = VectorStore()
bm25_store = BM25Store()
embed_model = EmbeddingModel()
retriever = Retriever(vector_store, bm25_store)
generator = GeneratorChain()
evaluator = Evaluator()

_db_path = settings.db_path
metrics_db = MetricsDB(_db_path)
feedback_store = FeedbackStore(_db_path)
document_store = DocumentStore(_db_path)

# Async ingest task store — in-memory, cleared on restart
_ingest_tasks: dict[str, IngestTaskStatus] = {}

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".pptx", ".html", ".htm"}

_IDK = "I don't know based on the provided documents."


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting Enterprise RAG Copilot v%s", settings.app_version)
    if not settings.api_key:
        logger.warning(
            "⚠️  API_KEY is not set — all endpoints are publicly accessible. "
            "Set API_KEY in your .env file before exposing this service to a network."
        )
    vector_store.load()
    # Try loading persisted BM25 first; fall back to re-seeding from VectorStore
    bm25_store.load()
    if bm25_store.size == 0 and vector_store._chunks:
        bm25_store.add_chunks(vector_store._chunks)
        logger.info("BM25 re-seeded from VectorStore: %d chunks", bm25_store.size)
    yield
    vector_store.save()
    bm25_store.save()
    logger.info("Shutdown: indexes saved.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Copilot",
    version=settings.app_version,
    description=(
        "Production RAG API with hybrid search, multi-turn conversation, "
        "feedback collection, and document lifecycle management."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Rate limiting middleware
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if settings.rate_limit_per_minute > 0:
            client_ip = request.client.host if request.client else "unknown"
            if not rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too many requests",
                        "detail": "Rate limit exceeded. Try again later.",
                    },
                )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _groundedness_guard(
    answer: str,
    groundedness: float,
    chunks_available: bool,
) -> str:
    """Post-generation hard enforcement: override answer when poorly grounded."""
    if (
        chunks_available
        and groundedness < settings.groundedness_threshold
        and _IDK.lower() not in answer.lower()
    ):
        logger.info(
            "Groundedness guard triggered (score=%.3f < threshold=%.3f) — overriding answer.",
            groundedness, settings.groundedness_threshold,
        )
        return _IDK
    return answer


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        index_size=vector_store.size,
        embed_model=embed_model.model_name,
        hybrid_search=bm25_store.available,
        reranker=settings.use_reranker,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def metrics():
    stats = metrics_db.get_summary()
    backend_raw = get_backend_status()
    pcts = metrics_db.get_latency_percentiles()
    cache_stats = query_cache.stats()
    return MetricsResponse(
        index_size=vector_store.size,
        total_queries=stats["total_queries"],
        avg_response_time_ms=stats["avg_response_time_ms"],
        avg_groundedness=stats["avg_groundedness"],
        generator_backend_status=BackendStatus(
            ollama=backend_raw["ollama"],
            flan_t5=backend_raw["flan_t5"],
            openai=backend_raw.get("openai", "not_configured"),
            active_mode=backend_raw["active_mode"],
        ),
        cache=CacheStats(**cache_stats),
        latency_percentiles=LatencyPercentiles(**pcts),
    )


@app.get("/metrics/timeseries", response_model=MetricsTimeseriesResponse, tags=["System"])
async def metrics_timeseries(hours: int = 24, bucket_minutes: int = 60):
    """Hourly buckets of query volume, avg latency, avg groundedness."""
    hours = max(1, min(hours, 168))          # clamp 1h–7d
    bucket_minutes = max(5, min(bucket_minutes, 1440))
    points = metrics_db.get_timeseries(hours=hours, bucket_minutes=bucket_minutes)
    return MetricsTimeseriesResponse(
        points=[MetricsTimeseriesPoint(**p) for p in points],
        hours=hours,
        bucket_minutes=bucket_minutes,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"],
          dependencies=[Depends(require_api_key)])
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD, DOCX, PPTX, HTML).
    Extracts, chunks, embeds, and indexes. Returns per-phase timing and doc_id.
    """
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {settings.max_upload_mb} MB.",
        )

    t_total = time.perf_counter()

    # Phase 1: extract
    t0 = time.perf_counter()
    try:
        pages = extract_text(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    extraction_ms = (time.perf_counter() - t0) * 1000

    if not pages:
        raise HTTPException(status_code=422, detail="No text could be extracted.")

    # Phase 2: chunk
    t0 = time.perf_counter()
    chunks = chunk_document(pages, filename)
    chunking_ms = (time.perf_counter() - t0) * 1000

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks after splitting.")

    # Phase 3: embed
    t0 = time.perf_counter()
    vectors = embed_model.encode([c.text for c in chunks], batch_size=settings.embed_batch_size)
    embedding_ms = (time.perf_counter() - t0) * 1000

    # Phase 4: index (dense + sparse)
    t0 = time.perf_counter()
    vector_store.add_chunks(chunks, vectors)
    bm25_store.add_chunks(chunks)
    if settings.save_index_on_ingest:
        vector_store.save()
        bm25_store.save()
    indexing_ms = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - t_total) * 1000

    # Register in document catalog
    doc_id = document_store.register(
        filename=filename,
        file_type=suffix.lstrip("."),
        pages_extracted=len(pages),
        chunks_stored=len(chunks),
        file_size_bytes=len(file_bytes),
    )

    logger.info(
        "Ingested '%s' (id=%s): %d pages → %d chunks in %.0f ms",
        filename, doc_id, len(pages), len(chunks), total_ms,
    )

    return IngestResponse(
        filename=filename,
        doc_id=doc_id,
        chunks_stored=len(chunks),
        message=f"Successfully ingested '{filename}': {len(chunks)} chunks indexed.",
        pages_extracted=len(pages),
        timings=IngestTimings(
            extraction_ms=round(extraction_ms, 2),
            chunking_ms=round(chunking_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            indexing_ms=round(indexing_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"],
          dependencies=[Depends(require_api_key)])
async def query(request: QueryRequest):
    """
    Ask a question. Supports multi-turn conversation via conversation_id.
    Returns grounded answer, citations, confidence, eval scores, and timing.
    """
    if vector_store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Ingest at least one document first.",
        )

    query_id = str(uuid.uuid4())
    trace_id = query_id  # Same for now; decouple if distributed tracing added
    t_total = time.perf_counter()
    top_k = request.top_k or settings.top_k
    conv_id = request.conversation_id

    # Load conversation history
    history = conversation_store.get_history(conv_id) if conv_id else []

    # Phase 1: embed
    t0 = time.perf_counter()
    original_query_vec = embed_model.encode([request.query])[0]
    query_vec = original_query_vec
    embedding_ms = (time.perf_counter() - t0) * 1000

    # Semantic cache check — skip full pipeline on near-duplicate queries
    cached = query_cache.lookup(original_query_vec)
    if cached:
        total_ms = (time.perf_counter() - t_total) * 1000
        return QueryResponse(
            answer=cached.answer,
            citations=[],
            eval=EvalResult(
                context_relevance=1.0, answer_groundedness=1.0,
                citation_presence=0.0, overall=1.0,
            ),
            generator_backend="cache",
            response_time_ms=round(total_ms, 2),
            query_id=query_id,
            conversation_id=conv_id,
            trace_id=trace_id,
            timings=QueryTimings(
                embedding_ms=round(embedding_ms, 2),
                retrieval_ms=0, generation_ms=0,
                evaluation_ms=0, total_ms=round(total_ms, 2),
            ),
            confidence=1.0,
            input_tokens=0,
            output_tokens=approx_tokens(cached.answer),
        )

    # HyDE: replace query embedding with hypothetical-document embedding
    hyde_query = generate_hyde_query(request.query)
    if hyde_query != request.query:
        query_vec = embed_model.encode([hyde_query])[0]

    # Phase 2: hybrid retrieve
    t0 = time.perf_counter()
    results = retriever.retrieve_from_vec(
        query_vec,
        query=request.query,
        top_k=top_k,
        source_filter=request.source_filter,
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    chunks = [c for c, _ in results]
    scores = [s for _, s in results]
    citations = retriever.to_citations(results)

    # Phase 3: generate (with conversation history)
    t0 = time.perf_counter()
    gen_result = generator.generate(request.query, chunks, history=history)
    generation_ms = (time.perf_counter() - t0) * 1000

    # Phase 4: evaluate
    t0 = time.perf_counter()
    chunk_vecs = vector_store.get_vectors(chunks)
    eval_result = evaluator.evaluate(
        query=request.query,
        answer=gen_result.answer,
        chunks=chunks,
        chunk_scores=scores,
        chunk_vecs=chunk_vecs,
    )
    evaluation_ms = (time.perf_counter() - t0) * 1000

    # Post-generation groundedness guard
    final_answer = _groundedness_guard(
        gen_result.answer,
        eval_result.answer_groundedness,
        chunks_available=bool(chunks),
    )
    if final_answer != gen_result.answer:
        # Re-evaluate overridden answer for accurate scores
        eval_result = evaluator.evaluate(
            query=request.query,
            answer=final_answer,
            chunks=chunks,
            chunk_scores=scores,
            chunk_vecs=chunk_vecs,
        )

    total_ms = (time.perf_counter() - t_total) * 1000

    # Persist conversation turns
    if conv_id:
        conversation_store.add_turn(conv_id, "user", request.query)
        conversation_store.add_turn(conv_id, "assistant", final_answer)

    # Token accounting
    prompt_est = " ".join(c.text for c in chunks) + request.query
    input_tokens = approx_tokens(prompt_est)
    output_tokens = approx_tokens(final_answer)

    timings_dict = {
        "embedding_ms": round(embedding_ms, 2),
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "evaluation_ms": round(evaluation_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    log_query(
        query=request.query,
        retrieved_chunks=len(chunks),
        response_time_ms=total_ms,
        generator_backend=gen_result.backend,
        eval_result=eval_result.model_dump(),
        query_id=query_id,
        timings=timings_dict,
        chunk_ids=[c.chunk_id for c in chunks],
        chunk_scores=scores,
    )

    metrics_db.record(
        query_id=query_id,
        trace_id=trace_id,
        latency_ms=total_ms,
        embedding_ms=embedding_ms,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        evaluation_ms=evaluation_ms,
        generator_backend=gen_result.backend,
        chunks_retrieved=len(chunks),
        groundedness=eval_result.answer_groundedness,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # Store in semantic cache for future identical/similar queries
    query_cache.store(
        query=request.query,
        query_vec=original_query_vec,
        answer=final_answer,
        query_id=query_id,
    )

    return QueryResponse(
        answer=final_answer,
        citations=citations,
        eval=eval_result,
        generator_backend=gen_result.backend,
        response_time_ms=round(total_ms, 2),
        query_id=query_id,
        conversation_id=conv_id,
        trace_id=trace_id,
        timings=QueryTimings(**timings_dict),
        confidence=eval_result.answer_groundedness,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@app.post("/query/stream", tags=["Query"], dependencies=[Depends(require_api_key)])
async def query_stream(request: QueryRequest):
    """
    SSE streaming version of /query.
    Emits: retrieving → token (one per chunk) → evaluating → done.
    Ollama streams token-by-token. flan-t5/extractive yield word-by-word.
    """
    if vector_store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Ingest at least one document first.",
        )

    async def event_generator():
        def emit(payload: dict) -> str:
            return f"data: {_json.dumps(payload)}\n\n"

        query_id = str(uuid.uuid4())
        trace_id = query_id
        t_total = time.perf_counter()
        top_k = request.top_k or settings.top_k
        conv_id = request.conversation_id
        history = conversation_store.get_history(conv_id) if conv_id else []

        try:
            # Phase 1: embed + retrieve
            yield emit({"event": "retrieving"})
            t0 = time.perf_counter()
            query_vec = embed_model.encode([request.query])[0]
            embedding_ms = (time.perf_counter() - t0) * 1000

            # Semantic cache check
            cached = query_cache.lookup(query_vec)
            if cached:
                yield emit({"event": "generating"})
                t0_gen = time.perf_counter()
                for word in cached.answer.split():
                    yield emit({"event": "token", "token": word + " "})
                generation_ms = (time.perf_counter() - t0_gen) * 1000
                total_ms = (time.perf_counter() - t_total) * 1000

                yield emit({"event": "evaluating"})

                if conv_id:
                    conversation_store.add_turn(conv_id, "user", request.query)
                    conversation_store.add_turn(conv_id, "assistant", cached.answer)

                yield emit({
                    "event": "done",
                    "data": {
                        "answer": cached.answer,
                        "citations": [],
                        "eval": {
                            "context_relevance": 1.0,
                            "answer_groundedness": 1.0,
                            "citation_presence": 0.0,
                            "overall": 1.0,
                        },
                        "generator_backend": "cache",
                        "response_time_ms": round(total_ms, 2),
                        "query_id": query_id,
                        "conversation_id": conv_id,
                        "trace_id": trace_id,
                        "timings": {
                            "embedding_ms": round(embedding_ms, 2),
                            "retrieval_ms": 0.0,
                            "generation_ms": round(generation_ms, 2),
                            "evaluation_ms": 0.0,
                            "total_ms": round(total_ms, 2),
                        },
                        "confidence": 1.0,
                        "input_tokens": 0,
                        "output_tokens": approx_tokens(cached.answer),
                    },
                })
                return

            t0 = time.perf_counter()
            results = retriever.retrieve_from_vec(
                query_vec,
                query=request.query,
                top_k=top_k,
                source_filter=request.source_filter,
            )
            retrieval_ms = (time.perf_counter() - t0) * 1000

            chunks = [c for c, _ in results]
            scores = [s for _, s in results]
            citations = retriever.to_citations(results)

            # Phase 2: stream generation token-by-token
            yield emit({"event": "generating"})
            t0 = time.perf_counter()
            tokens: list[str] = []
            gen_backend = "fallback"

            if not chunks:
                yield emit({"event": "token", "token": _IDK})
                tokens.append(_IDK)
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                mode = settings.generation_mode.lower()
                ollama_streamed = False

                # Attempt real Ollama token-by-token streaming first
                if mode in ("auto", "ollama"):
                    try:
                        from app.generator import _build_prompt, stream_ollama
                        prompt = _build_prompt(request.query, chunks, history, max_chars=3800)

                        def _try_stream_ollama():
                            return list(stream_ollama(prompt))

                        tok_list = await loop.run_in_executor(None, _try_stream_ollama)
                        if tok_list:
                            for tok in tok_list:
                                yield emit({"event": "token", "token": tok})
                                tokens.append(tok)
                            gen_backend = "ollama"
                            ollama_streamed = True
                    except Exception:
                        pass  # Ollama unavailable — fall through to generator singleton

                # Non-Ollama path (or Ollama failed): use mockable generator singleton,
                # then yield the full answer word-by-word for incremental UX
                if not ollama_streamed:
                    gen_result = generator.generate(request.query, chunks, history=history)
                    gen_backend = gen_result.backend
                    for word in gen_result.answer.split():
                        tok = word + " "
                        yield emit({"event": "token", "token": tok})
                        tokens.append(tok)

            generation_ms = (time.perf_counter() - t0) * 1000
            raw_answer = "".join(tokens).strip()

            # Phase 3: evaluate
            yield emit({"event": "evaluating"})
            t0 = time.perf_counter()
            chunk_vecs = vector_store.get_vectors(chunks)
            eval_result = evaluator.evaluate(
                query=request.query,
                answer=raw_answer,
                chunks=chunks,
                chunk_scores=scores,
                chunk_vecs=chunk_vecs,
            )
            evaluation_ms = (time.perf_counter() - t0) * 1000

            final_answer = _groundedness_guard(
                raw_answer,
                eval_result.answer_groundedness,
                chunks_available=bool(chunks),
            )
            if final_answer != raw_answer:
                eval_result = evaluator.evaluate(
                    query=request.query,
                    answer=final_answer,
                    chunks=chunks,
                    chunk_scores=scores,
                    chunk_vecs=chunk_vecs,
                )

            total_ms = (time.perf_counter() - t_total) * 1000

            if conv_id:
                conversation_store.add_turn(conv_id, "user", request.query)
                conversation_store.add_turn(conv_id, "assistant", final_answer)

            prompt_est = " ".join(c.text for c in chunks) + request.query
            input_tokens = approx_tokens(prompt_est)
            output_tokens = approx_tokens(final_answer)

            timings_dict = {
                "embedding_ms": round(embedding_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "evaluation_ms": round(evaluation_ms, 2),
                "total_ms": round(total_ms, 2),
            }

            log_query(
                query=request.query,
                retrieved_chunks=len(chunks),
                response_time_ms=total_ms,
                generator_backend=gen_backend,
                eval_result=eval_result.model_dump(),
                query_id=query_id,
                timings=timings_dict,
                chunk_ids=[c.chunk_id for c in chunks],
                chunk_scores=scores,
            )

            metrics_db.record(
                query_id=query_id,
                trace_id=trace_id,
                latency_ms=total_ms,
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                evaluation_ms=evaluation_ms,
                generator_backend=gen_backend,
                chunks_retrieved=len(chunks),
                groundedness=eval_result.answer_groundedness,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Store in semantic cache for future identical/similar queries
            query_cache.store(
                query=request.query,
                query_vec=query_vec,
                answer=final_answer,
                query_id=query_id,
            )

            yield emit({
                "event": "done",
                "data": {
                    "answer": final_answer,
                    "citations": [c.model_dump() for c in citations],
                    "eval": eval_result.model_dump(),
                    "generator_backend": gen_backend,
                    "response_time_ms": round(total_ms, 2),
                    "query_id": query_id,
                    "conversation_id": conv_id,
                    "trace_id": trace_id,
                    "timings": timings_dict,
                    "confidence": eval_result.answer_groundedness,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            })

        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            yield emit({"event": "error", "message": "An unexpected error occurred."})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"],
          dependencies=[Depends(require_api_key)])
async def submit_feedback(request: FeedbackRequest):
    """Store user feedback (thumbs up/down + optional comment) linked to a query."""
    if request.rating not in {-1, 1}:
        raise HTTPException(status_code=400, detail="rating must be 1 (up) or -1 (down).")
    feedback_id = feedback_store.store(
        query_id=request.query_id,
        conversation_id=request.conversation_id,
        rating=request.rating,
        comment=request.comment,
        answer=request.answer,
        chunk_ids=request.chunk_ids,
    )
    return FeedbackResponse(
        feedback_id=feedback_id,
        message="Feedback recorded. Thank you.",
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"],
         dependencies=[Depends(require_api_key)])
async def list_documents():
    """List all ingested documents with metadata."""
    docs = document_store.list_all()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@app.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse, tags=["Documents"],
            dependencies=[Depends(require_api_key)])
async def delete_document(doc_id: str):
    """
    Remove a document from all indexes (dense + sparse) and the document catalog.
    Rebuilds the FAISS index from remaining vectors.
    """
    filename = document_store.delete_by_id(doc_id)
    if filename is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    # Remove from dense index
    removed_dense = vector_store.remove_by_filename(filename)
    # Remove from BM25 index
    removed_bm25 = bm25_store.remove_by_filename(filename)

    if settings.save_index_on_ingest:
        vector_store.save()
        bm25_store.save()

    logger.info(
        "Deleted document '%s' (id=%s): removed %d dense + %d sparse chunks",
        filename, doc_id, removed_dense, removed_bm25,
    )

    return DeleteDocumentResponse(
        doc_id=doc_id,
        filename=filename,
        chunks_removed=removed_dense,
        message=f"Document '{filename}' removed. {removed_dense} chunks deleted from index.",
    )


# ---------------------------------------------------------------------------
# Async background ingestion
# ---------------------------------------------------------------------------

def _run_ingest_task(task_id: str, file_bytes: bytes, filename: str, suffix: str) -> None:
    """Background worker — runs ingest pipeline and updates task status."""
    task = _ingest_tasks[task_id]
    task.status = "processing"
    try:
        pages = extract_text(file_bytes, filename)
        chunks = chunk_document(pages, filename)
        vectors = embed_model.encode([c.text for c in chunks], batch_size=settings.embed_batch_size)
        vector_store.add_chunks(chunks, vectors)
        bm25_store.add_chunks(chunks)
        if settings.save_index_on_ingest:
            vector_store.save()
            bm25_store.save()
        doc_id = document_store.register(
            filename=filename,
            file_type=suffix.lstrip("."),
            pages_extracted=len(pages),
            chunks_stored=len(chunks),
            file_size_bytes=len(file_bytes),
        )
        task.status = "done"
        task.doc_id = doc_id
        task.chunks_stored = len(chunks)
        task.completed_at = time.time()
        logger.info("Async ingest done: '%s' → %d chunks (task=%s)", filename, len(chunks), task_id)
    except Exception as exc:
        task.status = "error"
        task.error = str(exc)
        task.completed_at = time.time()
        logger.error("Async ingest failed (task=%s): %s", task_id, exc)


@app.post("/ingest/async", response_model=IngestTaskStatus, tags=["Ingestion"],
          dependencies=[Depends(require_api_key)])
async def ingest_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a document for background processing.
    Returns a task_id immediately — poll GET /ingest/tasks/{task_id} for status.
    """
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum: {settings.max_upload_mb} MB.")

    task_id = str(uuid.uuid4())
    task = IngestTaskStatus(
        task_id=task_id,
        status="pending",
        filename=filename,
        created_at=time.time(),
    )
    _ingest_tasks[task_id] = task
    background_tasks.add_task(_run_ingest_task, task_id, file_bytes, filename, suffix)
    return task


@app.get("/ingest/tasks/{task_id}", response_model=IngestTaskStatus, tags=["Ingestion"],
         dependencies=[Depends(require_api_key)])
async def get_ingest_task(task_id: str):
    """Poll background ingest task status."""
    task = _ingest_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

@app.get("/conversations", response_model=ConversationListResponse, tags=["Conversations"],
         dependencies=[Depends(require_api_key)])
async def list_conversations(limit: int = 50):
    """List recent active conversations with preview and turn count."""
    limit = max(1, min(limit, 200))
    convs = conversation_store.list_conversations(limit=limit)
    return ConversationListResponse(
        conversations=[ConversationSummary(**c) for c in convs],
        total=len(convs),
    )


@app.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse,
         tags=["Conversations"], dependencies=[Depends(require_api_key)])
async def get_conversation(conversation_id: str):
    """Return full turn history for a conversation."""
    turns = conversation_store.get_full_history(conversation_id)
    if not turns:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        turns=[ConversationTurn(**t) for t in turns],
    )


@app.delete("/conversations/{conversation_id}", tags=["Conversations"],
            dependencies=[Depends(require_api_key)])
async def delete_conversation(conversation_id: str):
    """Clear all turns for a conversation."""
    conversation_store.clear(conversation_id)
    return {"message": f"Conversation '{conversation_id}' cleared."}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@app.post("/cache/clear", tags=["System"], dependencies=[Depends(require_api_key)])
async def clear_cache():
    """Flush the in-memory semantic query cache."""
    count = query_cache.clear()
    return {"message": f"Cache cleared. {count} entries removed."}
