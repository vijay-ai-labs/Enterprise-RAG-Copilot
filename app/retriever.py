"""
Retriever: hybrid dense (FAISS) + sparse (BM25) search with Reciprocal Rank Fusion.
Optional cross-encoder reranking after fusion.
Optional metadata filtering by filename.

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranking list i
k=60 is the standard constant.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.embeddings import EmbeddingModel
from app.logger import logger
from app.schemas import Chunk, Citation
from app.vectorstore import VectorStore

_RRF_K = 60


# ---------------------------------------------------------------------------
# MMR (diversity reranking — kept as optional post-step)
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    def trigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}
    ta, tb = trigrams(a), trigrams(b)
    if not ta and not tb:
        return 1.0
    union = len(ta | tb)
    return len(ta & tb) / union if union else 0.0


def _mmr_rerank(
    candidates: List[Tuple[Chunk, float]],
    top_k: int,
    lam: float,
) -> List[Tuple[Chunk, float]]:
    if not candidates:
        return []
    selected: List[Tuple[Chunk, float]] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best_idx, best_score = -1, float("-inf")
        for i, (chunk, rel) in enumerate(remaining):
            if not selected:
                mmr = rel
            else:
                max_sim = max(_text_similarity(chunk.text, s.text) for s, _ in selected)
                mmr = lam * rel - (1.0 - lam) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, i
        if best_idx == -1:
            break
        selected.append(remaining.pop(best_idx))
    return selected


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    bm25_weight: float = 0.3,
    top_k: int = 20,
) -> List[Tuple[Chunk, float]]:
    """
    Reciprocal Rank Fusion of dense and sparse results.
    bm25_weight scales the sparse contribution (0 = dense only, 1 = equal weight).
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    # Dense list contributes with weight (1 - bm25_weight)
    dense_w = 1.0 - bm25_weight
    for rank, (chunk, _) in enumerate(dense_results):
        cid = chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_w / (_RRF_K + rank + 1)
        chunk_map[cid] = chunk

    # Sparse list contributes with weight bm25_weight
    for rank, (chunk, _) in enumerate(sparse_results):
        cid = chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight / (_RRF_K + rank + 1)
        chunk_map[cid] = chunk

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk_map[cid], score) for cid, score in ranked]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, vector_store: VectorStore, bm25_store=None) -> None:
        self._store = vector_store
        self._bm25 = bm25_store
        self._embed = EmbeddingModel()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        query_vec = self._embed.encode([query])[0]
        return self.retrieve_from_vec(
            query_vec,
            query=query,
            top_k=top_k,
            threshold=threshold,
            source_filter=source_filter,
        )

    def retrieve_from_vec(
        self,
        query_vec: np.ndarray,
        query: str = "",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        use_mmr: Optional[bool] = None,
        mmr_lambda: Optional[float] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        k = top_k or settings.top_k
        fk = fetch_k or settings.fetch_k
        apply_mmr = use_mmr if use_mmr is not None else settings.use_mmr
        lam = mmr_lambda if mmr_lambda is not None else settings.mmr_lambda
        fetch_n = max(fk, k * 4)  # Fetch more for fusion/filtering

        # --- Dense retrieval ---
        dense_raw = self._store.search(query_vec, top_k=fetch_n)

        # --- Sparse BM25 retrieval (hybrid) ---
        use_hybrid = (
            settings.use_hybrid_search
            and self._bm25 is not None
            and self._bm25.available
            and query
        )
        if use_hybrid:
            sparse_raw = self._bm25.search(query, top_k=fetch_n)
            fused = _rrf_fuse(
                dense_raw, sparse_raw,
                bm25_weight=settings.bm25_weight,
                top_k=fetch_n,
            )
            # Normalize RRF scores to [0, 1]: max possible is 1/(_RRF_K+1) when
            # a chunk ranks #1 in both lists.  Scaling by (_RRF_K+1) makes scores
            # comparable to cosine similarities and fixes context_relevance metrics.
            fused = [(c, s * (_RRF_K + 1)) for c, s in fused]
            logger.info(
                "Hybrid: %d dense + %d sparse → %d fused",
                len(dense_raw), len(sparse_raw), len(fused),
            )
            if threshold is not None:
                min_score = threshold
            else:
                min_score = settings.similarity_threshold
        else:
            fused = dense_raw
            if threshold is not None:
                min_score = threshold
            else:
                min_score = settings.similarity_threshold

        # --- Threshold filter ---
        filtered = [(c, s) for c, s in fused if s >= min_score]

        # --- Metadata filter ---
        if source_filter:
            lower_filter = {f.lower() for f in source_filter}
            filtered = [
                (c, s) for c, s in filtered
                if c.filename.lower() in lower_filter
            ]

        # --- Dedup by chunk_id ---
        seen: set = set()
        deduped: List[Tuple[Chunk, float]] = []
        for chunk, score in filtered:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduped.append((chunk, score))

        # --- Optional cross-encoder reranking ---
        if settings.use_reranker and query and deduped:
            from app.reranker import rerank
            results = rerank(query, deduped, top_k=k)
        elif apply_mmr and len(deduped) > k:
            results = _mmr_rerank(deduped, k, lam)
        else:
            results = deduped[:k]

        logger.info(
            "Retrieve → %d dense, hybrid=%s, %d filtered, %d returned",
            len(dense_raw), use_hybrid, len(filtered), len(results),
        )
        return results

    def to_citations(self, results: List[Tuple[Chunk, float]]) -> List[Citation]:
        return [
            Citation(
                filename=chunk.filename,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                score=round(score, 4),
                text_snippet=chunk.text[:200],
            )
            for chunk, score in results
        ]
