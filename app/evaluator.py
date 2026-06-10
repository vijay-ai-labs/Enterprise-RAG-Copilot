"""
RAG evaluation — no external LLM calls required.

Three scores, each in [0.0, 1.0]:

  context_relevance   — avg cosine similarity between retrieval scores already
                        computed from the vector store (dense or RRF).

  answer_groundedness — hybrid check per answer sentence:
                          1. n-gram overlap with retrieved context (fast, lexical)
                          2. semantic cosine similarity via sentence embeddings
                        A sentence is grounded if EITHER check passes.
                        This catches paraphrased-but-faithful answers that fail
                        pure lexical matching.

  citation_presence   — 1.0 if the answer contains ≥1 citation [filename:pageN].

  overall             — weighted composite (0.3 / 0.5 / 0.2).
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from app.schemas import Chunk, EvalResult

_WEIGHTS = {"context_relevance": 0.3, "answer_groundedness": 0.5, "citation_presence": 0.2}

# Matches [somefile.pdf:page3] or [doc.md:page12]
_CITATION_RE = re.compile(r"\[.+?:page\d+\]", re.IGNORECASE)

# Minimum cosine similarity for a sentence to be considered semantically grounded
_SEM_GROUNDEDNESS_THRESHOLD = 0.35

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "that", "this", "with", "from", "by", "are", "was", "be",
    "as", "has", "have", "had", "not", "do", "did", "does", "i", "you", "we",
    "he", "she", "they", "my", "your", "our", "its", "can", "will", "would",
    "could", "should", "may", "might", "based", "provided", "documents",
}


def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _ngrams(text: str, n: int = 3) -> set:
    tokens = [
        t.lower()
        for t in re.findall(r"\b\w+\b", text)
        if t.lower() not in _STOPWORDS and len(t) > 2
    ]
    if n == 1:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _ngram_grounded(sentence: str, context_ngrams: set) -> bool:
    sent_ngrams = _ngrams(sentence, n=2) | _ngrams(sentence, n=1)
    return bool(sent_ngrams and sent_ngrams & context_ngrams)


def _semantic_grounded(
    sentence_vec: np.ndarray,
    chunk_vecs: List[np.ndarray],
    threshold: float,
) -> bool:
    """True if sentence embedding is within threshold of any chunk embedding."""
    if not chunk_vecs:
        return False
    # chunk_vecs are already L2-normalised from the embedding model
    sims = [float(np.dot(sentence_vec, cv)) for cv in chunk_vecs]
    return max(sims) >= threshold


def _answer_groundedness(
    answer: str,
    chunks: List[Chunk],
    chunk_vecs: Optional[List[np.ndarray]] = None,
) -> float:
    """
    Hybrid groundedness: n-gram overlap OR semantic similarity.

    If chunk_vecs is None, they are computed internally via the singleton
    embedding model (L2-normalised, same space as retrieval vectors).
    Semantic check catches paraphrased-but-faithful sentences that fail
    pure lexical n-gram matching.
    """
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    context_text = " ".join(c.text for c in chunks)
    context_ngrams = _ngrams(context_text, n=2) | _ngrams(context_text, n=1)

    # Resolve chunk vectors — compute if not supplied
    resolved_chunk_vecs: Optional[List[np.ndarray]] = chunk_vecs
    if resolved_chunk_vecs is None and chunks:
        try:
            from app.embeddings import EmbeddingModel
            _embed = EmbeddingModel()
            resolved_chunk_vecs = list(_embed.encode([c.text for c in chunks]))
        except Exception:
            resolved_chunk_vecs = None

    sentence_vecs: Optional[np.ndarray] = None
    use_semantic = resolved_chunk_vecs is not None and len(resolved_chunk_vecs) > 0
    if use_semantic:
        try:
            from app.embeddings import EmbeddingModel
            _embed = EmbeddingModel()
            sentence_vecs = _embed.encode(sentences)
        except Exception:
            use_semantic = False

    grounded = 0
    for idx, sentence in enumerate(sentences):
        if _ngram_grounded(sentence, context_ngrams):
            grounded += 1
        elif use_semantic and sentence_vecs is not None and resolved_chunk_vecs is not None:
            if _semantic_grounded(sentence_vecs[idx], resolved_chunk_vecs, _SEM_GROUNDEDNESS_THRESHOLD):
                grounded += 1

    return grounded / len(sentences)


def _citation_presence(answer: str) -> float:
    return 1.0 if _CITATION_RE.search(answer) else 0.0


class Evaluator:
    """
    Evaluate a single RAG response. No external LLM calls.
    Pass chunk_vecs to enable semantic groundedness scoring.
    """

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: List[Chunk],
        chunk_scores: List[float],
        chunk_vecs: Optional[List[np.ndarray]] = None,
    ) -> EvalResult:
        cr = float(np.mean(chunk_scores)) if chunk_scores else 0.0
        ag = _answer_groundedness(answer, chunks, chunk_vecs=chunk_vecs)
        cp = _citation_presence(answer)

        overall = (
            _WEIGHTS["context_relevance"] * cr
            + _WEIGHTS["answer_groundedness"] * ag
            + _WEIGHTS["citation_presence"] * cp
        )

        return EvalResult(
            context_relevance=round(cr, 4),
            answer_groundedness=round(ag, 4),
            citation_presence=round(cp, 4),
            overall=round(overall, 4),
        )
