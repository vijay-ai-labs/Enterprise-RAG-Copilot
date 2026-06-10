"""Tests for GENERATION_MODE dispatch: extractive, flan-t5, ollama, auto."""

import re
from datetime import datetime, timezone
from unittest.mock import MagicMock


import app.generator as gen_module
from app.generator import GeneratorChain, _extractive_answer
from app.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str = "Some context.",
    filename: str = "doc.txt",
    page: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=f"{filename}::page{page}::chunk0",
        filename=filename,
        page=page,
        text=text,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Extractive mode — _extractive_answer()
# ---------------------------------------------------------------------------

class TestExtractiveAnswer:
    def test_returns_non_empty_string(self):
        chunks = [_make_chunk("RAG stands for Retrieval-Augmented Generation.")]
        answer = _extractive_answer("What is RAG?", chunks)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_includes_inline_citation(self):
        chunks = [_make_chunk("RAG retrieves documents before generating.", "guide.pdf", 3)]
        answer = _extractive_answer("How does RAG work?", chunks)
        assert "[guide.pdf:page3]" in answer

    def test_empty_chunks_returns_fallback(self):
        answer = _extractive_answer("anything", [])
        assert "don't know" in answer.lower()

    def test_citation_format_matches_evaluator_regex(self):
        """Citation must match the pattern used by Evaluator._citation_presence."""
        chunks = [_make_chunk("The answer is here.", "report.pdf", 5)]
        answer = _extractive_answer("What is the answer?", chunks)
        assert re.search(r"\[.+?:page\d+\]", answer), f"No citation in: {answer!r}"

    def test_deduplicates_identical_sentences_across_chunks(self):
        same_text = "RAG improves accuracy by grounding answers in retrieved documents."
        chunks = [
            _make_chunk(same_text, "a.txt", 1),
            _make_chunk(same_text, "b.txt", 2),
        ]
        answer = _extractive_answer("What does RAG improve?", chunks)
        # Same sentence should appear at most once
        assert answer.count(same_text) <= 1

    def test_selects_sentence_with_most_query_overlap(self):
        # First sentence has no overlap; second has clear overlap
        text = "Unrelated fact here. Machine learning models require training data."
        chunks = [_make_chunk(text)]
        answer = _extractive_answer("What do machine learning models require?", chunks)
        # The better-matched sentence should be selected
        assert "machine learning" in answer.lower() or "training data" in answer.lower()

    def test_multi_chunk_multiple_citations(self):
        chunks = [
            _make_chunk("FAISS enables fast vector search.", "faiss.txt", 1),
            _make_chunk("Sentence transformers produce embeddings.", "embed.txt", 2),
        ]
        answer = _extractive_answer("How does vector search work?", chunks)
        # Should cite at least one source
        assert re.search(r"\[.+?:page\d+\]", answer)


# ---------------------------------------------------------------------------
# Generation mode dispatch — GeneratorChain.generate()
# ---------------------------------------------------------------------------

class TestGenerationModeDispatch:
    def test_extractive_mode_uses_extractive_backend(self, monkeypatch):
        monkeypatch.setattr("app.generator.settings.generation_mode", "extractive")
        chain = GeneratorChain()
        result = chain.generate("test query", [_make_chunk("Some context.")])
        assert result.backend == "extractive"

    def test_flan_t5_mode_uses_flan_backend(self, monkeypatch):
        monkeypatch.setattr("app.generator.settings.generation_mode", "flan-t5")
        monkeypatch.setattr(gen_module, "_try_flan_t5", lambda prompt: "flan answer")

        chain = GeneratorChain()
        result = chain.generate("query", [_make_chunk("context")])
        assert result.backend == "flan-t5"
        assert result.answer == "flan answer"

    def test_ollama_mode_when_reachable(self, monkeypatch):
        monkeypatch.setattr("app.generator.settings.generation_mode", "ollama")
        monkeypatch.setattr(gen_module, "_try_ollama", lambda p: "ollama answer")

        chain = GeneratorChain()
        result = chain.generate("query", [_make_chunk("ctx")])
        assert result.backend == "ollama"
        assert result.answer == "ollama answer"

    def test_ollama_mode_when_unreachable_returns_error_backend(self, monkeypatch):
        monkeypatch.setattr("app.generator.settings.generation_mode", "ollama")
        monkeypatch.setattr(gen_module, "_try_ollama", lambda p: None)

        chain = GeneratorChain()
        result = chain.generate("query", [_make_chunk("ctx")])
        assert result.backend == "ollama-error"

    def test_auto_mode_tries_ollama_first(self, monkeypatch):
        monkeypatch.setattr("app.generator.settings.generation_mode", "auto")
        monkeypatch.setattr(gen_module, "_try_openai", lambda *a, **k: None)
        monkeypatch.setattr(gen_module, "_try_ollama", lambda p: "ollama answer")

        chain = GeneratorChain()
        result = chain.generate("query", [_make_chunk("ctx")])
        assert result.backend == "ollama"

    def test_auto_mode_falls_back_to_flan_when_ollama_down(self, monkeypatch):
        monkeypatch.setattr("app.generator.settings.generation_mode", "auto")
        monkeypatch.setattr(gen_module, "_try_openai", lambda *a, **k: None)
        monkeypatch.setattr(gen_module, "_try_ollama", lambda p: None)
        monkeypatch.setattr(gen_module, "_try_flan_t5", lambda prompt: "flan fallback")

        chain = GeneratorChain()
        result = chain.generate("query", [_make_chunk("ctx")])
        assert result.backend == "flan-t5"

    def test_no_chunks_returns_fallback_regardless_of_mode(self, monkeypatch):
        for mode in ("auto", "ollama", "flan-t5", "extractive"):
            monkeypatch.setattr("app.generator.settings.generation_mode", mode)
            chain = GeneratorChain()
            result = chain.generate("query", [])
            assert result.backend == "fallback"
            assert "don't know" in result.answer.lower()


# ---------------------------------------------------------------------------
# get_backend_status() — does not load flan-t5 if not already cached
# ---------------------------------------------------------------------------

class TestGetBackendStatus:
    def test_returns_required_keys(self, monkeypatch):
        from app.generator import get_backend_status
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=Exception("no ollama")))

        status = get_backend_status()
        assert "ollama" in status
        assert "flan_t5" in status
        assert "active_mode" in status

    def test_flan_not_loaded_when_model_is_none(self, monkeypatch):
        from app.generator import get_backend_status
        monkeypatch.setattr(gen_module, "_flan_model", None)
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=Exception("offline")))

        status = get_backend_status()
        assert status["flan_t5"] == "not_loaded"

    def test_flan_loaded_when_model_exists(self, monkeypatch):
        from app.generator import get_backend_status
        monkeypatch.setattr(gen_module, "_flan_model", MagicMock())
        import requests
        monkeypatch.setattr(requests, "get", MagicMock(side_effect=Exception("offline")))

        status = get_backend_status()
        assert status["flan_t5"] == "loaded"
