"""
FAISS vector store.
Uses IndexFlatIP (inner product on L2-normalised vectors == cosine similarity).
Chunk metadata is stored in a parallel Python list.
Metadata is persisted as JSON. Legacy metadata.pkl files are migrated automatically.

Deletion: FAISS IndexFlatIP does not support in-place removal.
remove_by_filename() rebuilds the index from remaining vectors using reconstruct().
This is O(N) and correct for small-to-medium scale.
"""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.config import settings
from app.logger import logger
from app.schemas import Chunk


class VectorStore:
    def __init__(self, index_dir: str | None = None) -> None:
        self._index_dir = Path(index_dir or settings.index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / "faiss.index"
        self._meta_path = self._index_dir / "metadata.json"
        self._meta_path_legacy = self._index_dir / "metadata.pkl"

        self._index: faiss.Index | None = None
        self._chunks: List[Chunk] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(dimension)

    def _validate_counts(self) -> None:
        n_vec = self._index.ntotal if self._index else 0
        n_meta = len(self._chunks)
        if n_vec != n_meta:
            raise RuntimeError(
                f"Index/metadata mismatch: {n_vec} vectors vs {n_meta} metadata entries. "
                "Delete the index directory to reset."
            )

    def load(self) -> bool:
        with self._lock:
            if not self._index_path.exists():
                logger.info("No existing index — starting fresh.")
                return False

            self._index = faiss.read_index(str(self._index_path))

            if self._meta_path.exists():
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._chunks = [Chunk.model_validate(c) for c in raw]
            elif self._meta_path_legacy.exists():
                with open(self._meta_path_legacy, "rb") as f:
                    self._chunks = pickle.load(f)
                self._validate_counts()
                self.save()
                logger.info("Migrated metadata from pickle to JSON.")
                return True
            else:
                raise RuntimeError(
                    f"FAISS index found at '{self._index_path}' but no metadata file exists. "
                    "Delete the index directory to reset."
                )

            self._validate_counts()
            logger.info("Loaded FAISS index: %d vectors", self._index.ntotal)
            return True

    def save(self) -> None:
        with self._lock:
            if self._index is None:
                return
            faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in self._chunks], f)
            logger.info("Saved FAISS index: %d vectors", self._index.ntotal)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        with self._lock:
            if len(chunks) != vectors.shape[0]:
                raise ValueError("chunks and vectors must have the same length")
            self._ensure_index(vectors.shape[1])
            self._index.add(vectors)
            self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Deletion — rebuilds index from remaining vectors via reconstruct()
    # ------------------------------------------------------------------

    def remove_by_filename(self, filename: str) -> int:
        """Remove all chunks for a filename. Returns count removed. Rebuilds index."""
        with self._lock:
            if not self._chunks or self._index is None:
                return 0

            keep_indices = [
                i for i, c in enumerate(self._chunks) if c.filename != filename
            ]
            removed = len(self._chunks) - len(keep_indices)
            if removed == 0:
                return 0

            if keep_indices:
                dim = self._index.d
                kept_vecs = np.stack(
                    [self._index.reconstruct(i) for i in keep_indices]
                ).astype(np.float32)
                new_index = faiss.IndexFlatIP(dim)
                new_index.add(kept_vecs)
                self._index = new_index
                self._chunks = [self._chunks[i] for i in keep_indices]
            else:
                self._index = None
                self._chunks = []

            logger.info(
                "Removed %d chunks for '%s'; %d remain", removed, filename, len(self._chunks)
            )
            return removed

    def get_vectors(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """Retrieve pre-computed embeddings directly from the FAISS index."""
        with self._lock:
            if self._index is None or not self._chunks:
                return []
            chunk_to_idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}
            vectors = []
            for chunk in chunks:
                idx = chunk_to_idx.get(chunk.chunk_id)
                if idx is not None:
                    vectors.append(self._index.reconstruct(idx))
            return vectors

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []

            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)

            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query_vec, k)

            results: List[Tuple[Chunk, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self._chunks[idx], float(score)))
            return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return self._index.ntotal if self._index else 0

    def chunks_for_filename(self, filename: str) -> List[Chunk]:
        with self._lock:
            return [c for c in self._chunks if c.filename == filename]
