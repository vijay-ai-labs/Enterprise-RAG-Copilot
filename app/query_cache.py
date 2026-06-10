"""
Semantic query cache — returns cached answers for near-duplicate queries.

Uses cosine similarity on L2-normalised query embeddings (same space as
retrieval vectors). A cache hit skips the entire RAG pipeline, cutting
latency by ~95% for repeated or near-identical questions.

Config:
  CACHE_ENABLED              — master toggle (default True)
  CACHE_MAX_SIZE             — max entries before LRU eviction (default 500)
  CACHE_SIMILARITY_THRESHOLD — cosine sim cutoff for a hit (default 0.95)
  CACHE_TTL_SECONDS          — entry expiry (default 3600s)

Cache is in-memory only; cleared on server restart.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.logger import logger


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    query_id: str
    cached_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticQueryCache:
    """Thread-safe in-memory semantic cache with LRU-style eviction."""

    def __init__(self) -> None:
        self._entries: List[CacheEntry] = []
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    def lookup(self, query_vec: np.ndarray) -> Optional[CacheEntry]:
        if not settings.cache_enabled:
            return None

        now = time.time()
        with self._lock:
            self._entries = [
                e for e in self._entries
                if now - e.cached_at < settings.cache_ttl_seconds
            ]
            if not self._entries:
                self._total_misses += 1
                return None

            mat = np.stack([e.embedding for e in self._entries])
            sims = mat @ query_vec  # embeddings are L2-normalised → dot = cosine

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= settings.cache_similarity_threshold:
                entry = self._entries[best_idx]
                entry.hits += 1
                self._total_hits += 1
                logger.info(
                    "Cache HIT (sim=%.4f query_id=%s): %.60s",
                    best_sim, entry.query_id, entry.query,
                )
                return entry

            self._total_misses += 1
            return None

    def store(
        self,
        query: str,
        query_vec: np.ndarray,
        answer: str,
        query_id: str,
    ) -> None:
        if not settings.cache_enabled:
            return
        with self._lock:
            if len(self._entries) >= settings.cache_max_size:
                # Evict bottom 25% by (hits, recency)
                self._entries.sort(key=lambda e: (e.hits, e.cached_at))
                self._entries = self._entries[settings.cache_max_size // 4:]
            self._entries.append(
                CacheEntry(
                    query=query,
                    embedding=query_vec,
                    answer=answer,
                    query_id=query_id,
                )
            )

    def stats(self) -> Dict:
        with self._lock:
            total = self._total_hits + self._total_misses
            return {
                "enabled": settings.cache_enabled,
                "size": len(self._entries),
                "hits": self._total_hits,
                "misses": self._total_misses,
                "hit_rate": round(self._total_hits / max(1, total), 4),
                "threshold": settings.cache_similarity_threshold,
                "max_size": settings.cache_max_size,
            }

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._total_hits = 0
            self._total_misses = 0
        logger.info("Semantic cache cleared (%d entries removed)", count)
        return count


query_cache = SemanticQueryCache()
