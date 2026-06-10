"""
Cross-encoder reranking for retrieved candidates.

Uses sentence-transformers CrossEncoder when USE_RERANKER=true.
Falls back to original dense score ordering when unavailable.

The cross-encoder scores each (query, chunk) pair jointly — unlike bi-encoders
that score query and chunk independently. This produces much better relevance
ranking but is ~10–50x slower per chunk. Use after coarse retrieval (top 20–50).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.config import settings
from app.logger import logger
from app.schemas import Chunk

_cross_encoder = None
_load_attempted = False


def _get_cross_encoder():
    global _cross_encoder, _load_attempted
    if _load_attempted:
        return _cross_encoder
    _load_attempted = True

    if not settings.use_reranker:
        return None

    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder: %s", settings.reranker_model)
        _cross_encoder = CrossEncoder(settings.reranker_model)
        logger.info("Cross-encoder loaded.")
    except ImportError:
        logger.warning("sentence-transformers CrossEncoder not available; using score fallback.")
    except Exception as exc:
        logger.warning("Cross-encoder load failed (%s); using score fallback.", exc)

    return _cross_encoder


def rerank(
    query: str,
    candidates: List[Tuple[Chunk, float]],
    top_k: int,
) -> List[Tuple[Chunk, float]]:
    """
    Rerank candidates.
    Returns top_k results sorted by relevance descending.

    With cross-encoder: each (query, chunk_text) pair is scored jointly.
    Without: returns candidates sorted by original score (dense sim or RRF).
    """
    if not candidates:
        return []

    encoder = _get_cross_encoder()
    if encoder is not None:
        try:
            import math
            pairs = [(query, c.text[:512]) for c, _ in candidates]
            ce_scores = encoder.predict(pairs)
            # Sigmoid-normalise raw cross-encoder logits to [0, 1]
            norm_scores = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores]
            ranked = sorted(
                zip([c for c, _ in candidates], norm_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:top_k]
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed (%s); falling back to score order.", exc)

    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
