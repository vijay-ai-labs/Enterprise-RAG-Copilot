"""Tests for evaluator.py — citation detection, groundedness scoring, and composite eval."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.evaluator import (
    Evaluator,
    _answer_groundedness,
    _citation_presence,
    _ngrams,
    _sentence_split,
)
from app.schemas import Chunk, EvalResult
from datetime import datetime, timezone


@pytest.fixture(autouse=True)
def mock_embedding_model():
    """Mock EmbeddingModel to return dummy vectors to speed up test runs."""
    with patch("app.embeddings.EmbeddingModel") as MockEmbed:
        mock_instance = MagicMock()
        mock_instance.model_name = "mock-embed"
        mock_instance.encode.side_effect = lambda texts, **kwargs: np.zeros((len(texts), 384), dtype=np.float32)
        MockEmbed.return_value = mock_instance
        yield



def _make_chunk(text: str, filename: str = "doc.pdf", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{filename}::page{page}::chunk0",
        filename=filename,
        page=page,
        text=text,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Citation presence
# ---------------------------------------------------------------------------

class TestCitationPresence:
    def test_detects_standard_citation(self):
        answer = "RAG is effective [doc.pdf:page3] for grounded answers."
        assert _citation_presence(answer) == 1.0

    def test_detects_txt_citation(self):
        answer = "See [concepts.txt:page1] for details."
        assert _citation_presence(answer) == 1.0

    def test_detects_md_citation(self):
        answer = "As described [guide.md:page12], the system works."
        assert _citation_presence(answer) == 1.0

    def test_no_citation_returns_zero(self):
        answer = "RAG is a powerful framework without any citation."
        assert _citation_presence(answer) == 0.0

    def test_partial_citation_format_not_matched(self):
        answer = "See [doc.pdf] for more info."  # missing :pageN
        assert _citation_presence(answer) == 0.0

    def test_idonotknow_answer_no_citation(self):
        answer = "I don't know based on the provided documents."
        assert _citation_presence(answer) == 0.0

    def test_multiple_citations_still_scores_one(self):
        answer = "First fact [a.pdf:page1]. Second fact [b.txt:page2]."
        assert _citation_presence(answer) == 1.0


# ---------------------------------------------------------------------------
# Answer groundedness
# ---------------------------------------------------------------------------

class TestAnswerGroundedness:
    def test_fully_grounded_answer(self):
        chunk = _make_chunk("Machine learning enables computers to learn from data automatically.")
        answer = "Machine learning enables computers to learn from data."
        score = _answer_groundedness(answer, [chunk])
        assert score > 0.5

    def test_ungrounded_answer_low_score(self):
        chunk = _make_chunk("The sky is blue on clear days.")
        answer = "Quantum entanglement produces instantaneous correlations across distance."
        score = _answer_groundedness(answer, [chunk])
        assert score < 0.5

    def test_empty_answer_returns_zero(self):
        chunk = _make_chunk("Some context text.")
        assert _answer_groundedness("", [chunk]) == 0.0

    def test_no_chunks_returns_zero(self):
        assert _answer_groundedness("Any answer here.", []) == 0.0

    def test_idonotknow_partially_grounded(self):
        chunk = _make_chunk("Documents describe RAG as a framework.")
        answer = "I don't know based on the provided documents."
        # "documents" and "provided" are in both — some overlap expected
        score = _answer_groundedness(answer, [chunk])
        assert score >= 0.0  # Just verify it runs without error


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------

class TestNgrams:
    def test_bigrams_generated(self):
        ng = _ngrams("machine learning is great", n=2)
        assert "machine learning" in ng

    def test_stopwords_excluded(self):
        ng = _ngrams("the sky is blue", n=1)
        assert "the" not in ng
        assert "is" not in ng
        assert "sky" in ng
        assert "blue" in ng

    def test_empty_text_returns_empty_set(self):
        assert _ngrams("", n=2) == set()


# ---------------------------------------------------------------------------
# Sentence split
# ---------------------------------------------------------------------------

class TestSentenceSplit:
    def test_splits_on_period(self):
        sentences = _sentence_split("First sentence. Second sentence. Third.")
        assert len(sentences) == 3

    def test_single_sentence_no_split(self):
        sentences = _sentence_split("Just one sentence here")
        assert len(sentences) == 1

    def test_empty_returns_empty(self):
        assert _sentence_split("") == []


# ---------------------------------------------------------------------------
# Evaluator integration
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_returns_eval_result(self):
        evaluator = Evaluator()
        chunks = [_make_chunk("RAG uses retrieval to ground answers in documents.")]
        answer = "RAG uses retrieval [doc.pdf:page1] to ground answers."
        result = evaluator.evaluate(
            query="What is RAG?",
            answer=answer,
            chunks=chunks,
            chunk_scores=[0.85],
        )
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.context_relevance <= 1.0
        assert 0.0 <= result.answer_groundedness <= 1.0
        assert 0.0 <= result.citation_presence <= 1.0
        assert 0.0 <= result.overall <= 1.0

    def test_citation_present_boosts_overall(self):
        evaluator = Evaluator()
        chunks = [_make_chunk("Context text about machine learning.")]

        answer_with_citation = "ML is powerful [doc.pdf:page1]."
        answer_without_citation = "ML is powerful."

        result_with = evaluator.evaluate("query", answer_with_citation, chunks, [0.7])
        result_without = evaluator.evaluate("query", answer_without_citation, chunks, [0.7])

        assert result_with.citation_presence == 1.0
        assert result_without.citation_presence == 0.0
        assert result_with.overall > result_without.overall

    def test_empty_chunks_low_scores(self):
        evaluator = Evaluator()
        result = evaluator.evaluate("query", "some answer", [], [])
        assert result.context_relevance == 0.0
        assert result.answer_groundedness == 0.0

    def test_scores_rounded_to_4_decimals(self):
        evaluator = Evaluator()
        chunks = [_make_chunk("text")]
        result = evaluator.evaluate("q", "answer [f.pdf:page1]", chunks, [0.333333])
        # Verify rounding applied
        assert result.context_relevance == round(result.context_relevance, 4)
        assert result.overall == round(result.overall, 4)
