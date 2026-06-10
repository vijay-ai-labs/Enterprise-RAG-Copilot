"""
BM25 sparse keyword index for hybrid dense+sparse retrieval.
Uses rank_bm25 (BM25Okapi). Gracefully disabled when not installed.

Persistence: save() serialises chunks+corpus to a pickle file alongside the
FAISS index so BM25 state survives restarts independently. On startup, load()
restores the index without requiring a full re-seed from VectorStore chunks.
"""

from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings
from app.logger import logger
from app.schemas import Chunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class BM25Store:
    """Thread-safe BM25 index over ingested chunks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: List[Chunk] = []
        self._corpus: List[List[str]] = []
        self._bm25 = None
        self._available: Optional[bool] = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        with self._lock:
            self._chunks.extend(chunks)
            self._corpus.extend([_tokenize(c.text) for c in chunks])
            self._rebuild_unlocked()

    def remove_by_filename(self, filename: str) -> int:
        with self._lock:
            before = len(self._chunks)
            pairs = [
                (c, t) for c, t in zip(self._chunks, self._corpus)
                if c.filename != filename
            ]
            if pairs:
                chunks, corpus = zip(*pairs)
                self._chunks = list(chunks)
                self._corpus = list(corpus)
            else:
                self._chunks = []
                self._corpus = []
            self._rebuild_unlocked()
            return before - len(self._chunks)

    def _rebuild_unlocked(self) -> None:
        if not self._corpus:
            self._bm25 = None
            return
        if self._available is False:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
            self._available = True
        except ImportError:
            if self._available is None:
                logger.warning(
                    "rank-bm25 not installed; BM25 disabled. "
                    "Install with: pip install rank-bm25"
                )
            self._available = False
            self._bm25 = None
        except Exception as exc:
            logger.warning("BM25 rebuild failed: %s", exc)
            self._bm25 = None

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            try:
                raw_scores = np.array(self._bm25.get_scores(tokens), dtype=np.float32)
                # Sort by rank — do NOT filter by sign.
                # BM25Okapi produces negative IDF for terms appearing in all docs
                # (small corpora). RRF fusion handles the relative weighting;
                # we just need a ranked list.
                top_idx = np.argsort(raw_scores)[::-1][:top_k]
                # Normalise to [0,1] for the score field (informational only).
                s_min, s_max = raw_scores.min(), raw_scores.max()
                if s_max > s_min:
                    norm = (raw_scores - s_min) / (s_max - s_min)
                else:
                    norm = np.zeros_like(raw_scores)
                return [(self._chunks[i], float(norm[i])) for i in top_idx]
            except Exception as exc:
                logger.warning("BM25 search failed: %s", exc)
                return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return Path(settings.index_dir) / "bm25_store.pkl"

    def save(self) -> None:
        """Persist chunks and tokenised corpus to disk."""
        path = self._index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"chunks": self._chunks, "corpus": self._corpus}
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("BM25 index saved: %d chunks → %s", len(payload["chunks"]), path)
        except Exception as exc:
            logger.warning("BM25 save failed: %s", exc)

    def load(self) -> None:
        """Restore chunks and corpus from disk, then rebuild the BM25 index."""
        path = self._index_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            with self._lock:
                self._chunks = payload.get("chunks", [])
                self._corpus = payload.get("corpus", [])
                self._rebuild_unlocked()
            logger.info("BM25 index loaded: %d chunks from %s", len(self._chunks), path)
        except Exception as exc:
            logger.warning("BM25 load failed (%s); will re-seed from VectorStore.", exc)

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def available(self) -> bool:
        return self._bm25 is not None
