"""
Embedding model wrapper — singleton so the model loads once per process.
Uses sentence-transformers/all-MiniLM-L6-v2 by default (22M params, CPU-friendly).
Vectors are L2-normalised so inner-product == cosine similarity.
"""

from __future__ import annotations

import numpy as np
from typing import List

from app.config import settings
from app.logger import logger


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._model_name = settings.embed_model
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded.")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (N, D)."""
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 norm → cosine via inner product
        )
        return vectors.astype(np.float32)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
