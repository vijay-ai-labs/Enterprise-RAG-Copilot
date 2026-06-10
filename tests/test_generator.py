"""Tests for app/generator.py — flan-t5 backend uses AutoModelForSeq2SeqLM (seq2seq)."""

import sys
from unittest.mock import MagicMock, patch
from app.generator import GeneratorChain, _try_flan_t5, _get_flan_model
import app.generator as gen_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "Some context.", filename: str = "doc.txt", page: int = 1):
    from datetime import datetime, timezone
    from app.schemas import Chunk
    return Chunk(
        chunk_id=f"{filename}::page{page}::chunk0",
        filename=filename,
        page=page,
        text=text,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# _get_flan_model — must use AutoModelForSeq2SeqLM (text2text-generation)
# ---------------------------------------------------------------------------

def test_flan_model_uses_seq2seq(monkeypatch):
    """Model must be loaded as AutoModelForSeq2SeqLM, not a causal/text-generation model."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    monkeypatch.setattr(gen_module, "_flan_model", None)
    monkeypatch.setattr(gen_module, "_flan_tokenizer", None)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        model, tokenizer = _get_flan_model()

    mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.assert_called_once_with(
        "google/flan-t5-base"
    )
    assert model is mock_model
    assert tokenizer is mock_tokenizer


def test_flan_model_singleton(monkeypatch):
    """_get_flan_model returns the same objects on repeated calls (no double-load)."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    monkeypatch.setattr(gen_module, "_flan_model", mock_model)
    monkeypatch.setattr(gen_module, "_flan_tokenizer", mock_tokenizer)

    m1, t1 = _get_flan_model()
    m2, t2 = _get_flan_model()
    assert m1 is m2 is mock_model
    assert t1 is t2 is mock_tokenizer


# ---------------------------------------------------------------------------
# _try_flan_t5 — return shape and truncation
# ---------------------------------------------------------------------------

def test_try_flan_t5_returns_string(monkeypatch):
    """_try_flan_t5 must return a plain str decoded from model output."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_model.generate.return_value = MagicMock()
    mock_tokenizer.decode.return_value = "  Flan answer.  "

    monkeypatch.setattr(gen_module, "_flan_model", mock_model)
    monkeypatch.setattr(gen_module, "_flan_tokenizer", mock_tokenizer)

    result = _try_flan_t5("What is RAG?")
    assert result == "Flan answer."


def test_try_flan_t5_truncates_long_prompt(monkeypatch):
    """Prompts longer than _FLAN_MAX_CHARS are truncated before tokenizing."""
    from app.generator import _FLAN_MAX_CHARS
    received_prompts = []

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    def capture_tokenize(prompt, **kw):
        received_prompts.append(prompt)
        return {"input_ids": MagicMock()}

    mock_tokenizer.side_effect = capture_tokenize
    mock_tokenizer.decode.return_value = "ok"
    mock_model.generate.return_value = MagicMock()

    monkeypatch.setattr(gen_module, "_flan_model", mock_model)
    monkeypatch.setattr(gen_module, "_flan_tokenizer", mock_tokenizer)

    _try_flan_t5("x" * 5000)
    assert len(received_prompts[0]) == _FLAN_MAX_CHARS


# ---------------------------------------------------------------------------
# GeneratorChain.generate — mode dispatch and fallback behaviour
# ---------------------------------------------------------------------------

def test_generate_no_chunks_returns_fallback():
    chain = GeneratorChain()
    result = chain.generate("query", [])
    assert result.backend == "fallback"
    assert "don't know" in result.answer.lower()


def test_generate_uses_flan_when_ollama_down(monkeypatch):
    """When Ollama is unreachable, generate falls back to flan-t5."""
    from app.config import settings
    monkeypatch.setattr(settings, "generation_mode", "auto")
    monkeypatch.setattr(gen_module, "_try_openai", lambda *a, **k: None)
    monkeypatch.setattr(gen_module, "_try_ollama", lambda prompt: None)
    monkeypatch.setattr(gen_module, "_try_flan_t5", lambda prompt: "Flan fallback answer.")

    chain = GeneratorChain()
    result = chain.generate("test query", [_make_chunk()])
    assert result.backend == "flan-t5"
    assert result.answer == "Flan fallback answer."


def test_generate_uses_ollama_when_available(monkeypatch):
    """When Ollama returns a response, it is used and flan-t5 is not loaded."""
    from app.config import settings
    monkeypatch.setattr(settings, "generation_mode", "auto")
    monkeypatch.setattr(gen_module, "_try_openai", lambda *a, **k: None)
    monkeypatch.setattr(gen_module, "_try_ollama", lambda prompt: "Ollama answer.")
    monkeypatch.setattr(gen_module, "_flan_model", None)

    chain = GeneratorChain()
    result = chain.generate("test query", [_make_chunk()])
    assert result.backend == "ollama"
    assert result.answer == "Ollama answer."
    # Singleton must not have been initialised (no download triggered)
    assert gen_module._flan_model is None
