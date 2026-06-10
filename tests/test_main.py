"""Tests for app/main.py — error handler, ingest, query endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with isolated VectorStore and mocked heavy singletons."""
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))

    # Prevent real model load
    with patch("app.main.EmbeddingModel") as MockEmbed, \
         patch("app.main.GeneratorChain") as MockGen, \
         patch("app.main.Evaluator") as MockEval:

        mock_embed = MagicMock()
        mock_embed.model_name = "mock-embed"
        mock_embed.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        MockEmbed.return_value = mock_embed

        mock_gen = MagicMock()
        from app.generator import GeneratorResult
        mock_gen.generate.return_value = GeneratorResult(
            answer="Mock answer [doc.txt:page1].", backend="ollama"
        )
        MockGen.return_value = mock_gen

        mock_eval = MagicMock()
        from app.schemas import EvalResult
        mock_eval.evaluate.return_value = EvalResult(
            context_relevance=0.8,
            answer_groundedness=0.7,
            citation_presence=1.0,
            overall=0.77,
        )
        MockEval.return_value = mock_eval

        # Import after monkeypatching so singletons use mocks
        import importlib
        import app.main as main_module
        importlib.reload(main_module)

        # Disable API key auth so tests run without the X-API-Key header
        from app.auth import require_api_key as _require_api_key
        main_module.app.dependency_overrides[_require_api_key] = lambda: None

        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Global exception handler — must NOT leak str(exc)
# ---------------------------------------------------------------------------

def test_500_does_not_leak_exception_detail(client):
    """Global handler returns generic message, not str(exc)."""
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.retriever") as mock_ret:
        mock_vs.size = 1
        mock_em.encode.side_effect = RuntimeError("secret internal error")
        mock_ret.retrieve_from_vec.side_effect = RuntimeError("secret internal error")
        resp = client.post("/query", json={"query": "test"})

    assert resp.status_code == 500
    body = resp.json()
    assert "secret internal error" not in body.get("detail", "")
    assert "unexpected error" in body.get("detail", "").lower()


# ---------------------------------------------------------------------------
# HTTPException responses — 4xx must pass through unchanged
# ---------------------------------------------------------------------------

def test_unsupported_file_type_returns_415(client):
    resp = client.post(
        "/ingest",
        files={"file": ("bad.xyz", b"content", "application/octet-stream")},
    )
    assert resp.status_code == 415
    assert "Unsupported" in resp.json()["detail"]


def test_empty_file_returns_400(client):
    resp = client.post(
        "/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_query_empty_index_returns_400(client):
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.size = 0
        resp = client.post("/query", json={"query": "hello"})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.embed_model") as mock_em:
        mock_vs.size = 42
        mock_em.model_name = "all-MiniLM-L6-v2"
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
