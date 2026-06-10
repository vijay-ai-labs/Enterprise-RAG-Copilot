from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    filename: str
    page: int
    chunk_id: str
    score: float = Field(description="Cosine similarity score (0–1)")
    text_snippet: str = Field(description="First 200 chars of the source chunk")


class Chunk(BaseModel):
    chunk_id: str
    filename: str
    page: int
    text: str
    timestamp: str


class EvalResult(BaseModel):
    context_relevance: float = Field(description="Avg cosine sim: query vs retrieved chunks")
    answer_groundedness: float = Field(description="Fraction of answer sentences grounded in context")
    citation_presence: float = Field(description="1.0 if ≥1 citation found in answer, else 0.0")
    overall: float = Field(description="Weighted composite score")


class IngestTimings(BaseModel):
    extraction_ms: float
    chunking_ms: float
    embedding_ms: float
    indexing_ms: float
    total_ms: float


class IngestResponse(BaseModel):
    filename: str
    doc_id: Optional[str] = None
    chunks_stored: int
    message: str
    pages_extracted: Optional[int] = None
    timings: Optional[IngestTimings] = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    conversation_id: Optional[str] = Field(default=None, max_length=128)
    source_filter: Optional[List[str]] = Field(
        default=None, description="Filter results to these filenames only"
    )


class QueryTimings(BaseModel):
    embedding_ms: float
    retrieval_ms: float
    generation_ms: float
    evaluation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    eval: EvalResult
    generator_backend: str
    response_time_ms: float
    query_id: Optional[str] = None
    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None
    timings: Optional[QueryTimings] = None
    confidence: Optional[float] = Field(
        default=None, description="Answer groundedness score (0–1)"
    )
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class FeedbackRequest(BaseModel):
    query_id: str
    conversation_id: Optional[str] = None
    rating: int = Field(description="1 = thumbs up, -1 = thumbs down", ge=-1, le=1)
    comment: Optional[str] = Field(default=None, max_length=2000)
    answer: Optional[str] = Field(default=None, max_length=10000)
    chunk_ids: Optional[List[str]] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    pages_extracted: int
    chunks_stored: int
    file_size_bytes: int
    ingested_at: float
    status: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_removed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
    embed_model: str
    hybrid_search: bool
    reranker: bool


class BackendStatus(BaseModel):
    ollama: str
    flan_t5: str
    openai: str = "not_configured"
    active_mode: str


class CacheStats(BaseModel):
    enabled: bool
    size: int
    hits: int
    misses: int
    hit_rate: float
    threshold: float
    max_size: int


class LatencyPercentiles(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    count: int


class MetricsResponse(BaseModel):
    index_size: int
    total_queries: int
    avg_response_time_ms: float
    avg_groundedness: float
    generator_backend_status: BackendStatus
    cache: Optional[CacheStats] = None
    latency_percentiles: Optional[LatencyPercentiles] = None


class MetricsTimeseriesPoint(BaseModel):
    timestamp: int
    queries: int
    avg_latency_ms: float
    avg_groundedness: float
    total_tokens: int


class MetricsTimeseriesResponse(BaseModel):
    points: List[MetricsTimeseriesPoint]
    hours: int
    bucket_minutes: int


class ConversationSummary(BaseModel):
    conversation_id: str
    turn_count: int
    last_active: float
    preview: str


class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummary]
    total: int


class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: float


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    turns: List[ConversationTurn]


class IngestTaskStatus(BaseModel):
    task_id: str
    status: str  # pending | processing | done | error
    filename: Optional[str] = None
    doc_id: Optional[str] = None
    chunks_stored: Optional[int] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None
