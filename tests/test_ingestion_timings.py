"""Tests for ingestion timing breakdown returned by POST /ingest."""

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
    monkeypatch.setenv("LOG_DIR", str(tmp_path))

    with patch("app.main.EmbeddingModel") as MockEmbed, \
         patch("app.main.GeneratorChain") as MockGen, \
         patch("app.main.Evaluator") as MockEval:

        mock_embed = MagicMock()
        mock_embed.model_name = "mock-embed"
        mock_embed.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        MockEmbed.return_value = mock_embed

        MockGen.return_value = MagicMock()
        MockEval.return_value = MagicMock()

        import importlib
        import app.main as main_module
        importlib.reload(main_module)

        # Disable API key auth so tests run without the X-API-Key header
        from app.auth import require_api_key as _require_api_key
        main_module.app.dependency_overrides[_require_api_key] = lambda: None

        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Timing fields present and well-formed
# ---------------------------------------------------------------------------

def test_ingest_response_has_timings(client):
    content = b"This is a test document with enough content to produce at least one chunk."
    resp = client.post(
        "/ingest",
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "timings" in data, "Response missing 'timings' field"
    t = data["timings"]
    for field in ("extraction_ms", "chunking_ms", "embedding_ms", "indexing_ms", "total_ms"):
        assert field in t, f"Missing timing field: {field}"
        assert isinstance(t[field], (int, float)), f"'{field}' is not numeric"
        assert t[field] >= 0, f"'{field}' is negative"


def test_ingest_response_has_pages_extracted(client):
    content = b"A plaintext document for testing page extraction."
    resp = client.post(
        "/ingest",
        files={"file": ("doc.txt", content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "pages_extracted" in data
    assert data["pages_extracted"] >= 1


def test_ingest_timings_total_covers_components(client):
    """total_ms must be at least as large as the sum of phase timings (minus small overhead)."""
    content = b"Document content for timing consistency check. " * 10
    resp = client.post(
        "/ingest",
        files={"file": ("timing.txt", content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    t = resp.json()["timings"]
    component_sum = t["extraction_ms"] + t["chunking_ms"] + t["embedding_ms"] + t["indexing_ms"]
    # total_ms >= component_sum (allow 10% tolerance for measurement overhead)
    assert t["total_ms"] >= component_sum * 0.9, (
        f"total_ms={t['total_ms']:.2f} less than 90% of component sum {component_sum:.2f}"
    )


# ---------------------------------------------------------------------------
# Upload size validation
# ---------------------------------------------------------------------------

def test_ingest_max_upload_mb_rejected(client, monkeypatch):
    """Files exceeding MAX_UPLOAD_MB must be rejected with 413."""
    import app.config as cfg_module
    # Directly mutate the live settings object shared with app.main
    monkeypatch.setattr(cfg_module.settings, "max_upload_mb", 0.001)  # ~1 KB

    big_content = b"x" * 2048  # 2 KB > 1 KB limit
    resp = client.post(
        "/ingest",
        files={"file": ("big.txt", big_content, "text/plain")},
    )
    assert resp.status_code == 413, f"Expected 413, got {resp.status_code}: {resp.text}"
    assert "large" in resp.json()["detail"].lower()


def test_ingest_within_size_limit_accepted(client):
    """Files within MAX_UPLOAD_MB limit are processed normally."""
    content = b"Small file content that is well within any reasonable limit."
    resp = client.post(
        "/ingest",
        files={"file": ("small.txt", content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Existing contract unchanged
# ---------------------------------------------------------------------------

def test_ingest_response_has_original_fields(client):
    """Existing response fields (filename, chunks_stored, message) still present."""
    content = b"Testing backward compatibility of IngestResponse schema."
    resp = client.post(
        "/ingest",
        files={"file": ("compat.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "filename" in data
    assert "chunks_stored" in data
    assert isinstance(data["chunks_stored"], int)
    assert "message" in data
