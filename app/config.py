from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.3   # used in dense-only mode (cosine sim, range 0–1)
    rrf_score_threshold: float = 0.005  # used in hybrid RRF mode (RRF scores ~0.01–0.02)
    fetch_k: int = 20
    use_mmr: bool = False
    mmr_lambda: float = 0.5

    # Hybrid search (dense FAISS + sparse BM25 with RRF fusion)
    use_hybrid_search: bool = True
    bm25_weight: float = 0.3  # 0 = dense-only, 1 = BM25-only

    # Reranking (cross-encoder; ~80 MB model, downloaded once on first query)
    # Gracefully falls back to score-sorted order if download fails.
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Embedding
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 64

    # Generation
    # Modes: auto | openai | ollama | flan-t5 | extractive
    # auto: tries openai → ollama → flan-t5 in order
    generation_mode: str = "auto"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: float = 10.0

    # OpenAI API — set OPENAI_API_KEY to enable
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 1024

    # HyDE (Hypothetical Document Embedding) — generates a hypothetical answer
    # before retrieval to improve recall. Requires OpenAI or Ollama.
    use_hyde: bool = False

    # Semantic query cache — returns cached answers for near-duplicate queries
    cache_enabled: bool = True
    cache_max_size: int = 500
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: float = 3600.0

    # Groundedness guard — answers below this threshold are overridden to "I don't know"
    groundedness_threshold: float = 0.25

    # Storage
    index_dir: str = "indexes"
    log_dir: str = "logs"
    db_path: str = "data/rag.db"
    save_index_on_ingest: bool = True

    # Authentication — set API_KEY env var to enable; all endpoints will require X-API-Key header
    api_key: Optional[str] = None

    # Rate limiting (requests per minute per IP; 0 = disabled)
    rate_limit_per_minute: int = 60

    # Multi-turn conversation
    conversation_ttl_seconds: int = 3600
    max_conversation_turns: int = 10

    # Upload
    max_upload_mb: float = 50.0

    # CORS
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    # Timeouts
    query_timeout_seconds: float = 60.0

    # App
    app_version: str = "3.0.0"


settings = Settings()
