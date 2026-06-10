"""Tests for POST /query/stream SSE endpoint and /query timing fields."""

import json
import re
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.generator import GeneratorResult
from app.schemas import Chunk, Citation, EvalResult


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
        mock_embed.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        MockEmbed.return_value = mock_embed

        mock_gen = MagicMock()
        mock_gen.generate.return_value = GeneratorResult(
            answer="Streaming answer [doc.txt:page1].", backend="extractive"
        )
        MockGen.return_value = mock_gen

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = EvalResult(
            context_relevance=0.8,
            answer_groundedness=0.7,
            citation_presence=1.0,
            overall=0.77,
        )
        MockEval.return_value = mock_eval

        import importlib
        import app.main as main_module
        importlib.reload(main_module)

        # Disable API key auth so tests run without the X-API-Key header
        from app.auth import require_api_key as _require_api_key
        main_module.app.dependency_overrides[_require_api_key] = lambda: None

        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk() -> Chunk:
    return Chunk(
        chunk_id="doc.txt::page1::chunk0",
        filename="doc.txt",
        page=1,
        text="Sample context content.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _citation() -> Citation:
    return Citation(
        filename="doc.txt",
        page=1,
        chunk_id="doc.txt::page1::chunk0",
        score=0.9,
        text_snippet="Sample context content.",
    )


def _mock_gen() -> MagicMock:
    m = MagicMock()
    m.generate.return_value = GeneratorResult(
        answer="Streaming answer [doc.txt:page1].", backend="extractive"
    )
    return m


def _mock_eval() -> MagicMock:
    m = MagicMock()
    m.evaluate.return_value = EvalResult(
        context_relevance=0.8,
        answer_groundedness=0.7,
        citation_presence=1.0,
        overall=0.77,
    )
    return m


def _parse_sse_events(raw_text: str) -> list:
    """Parse SSE data lines from raw response body."""
    events = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Tests — empty index guard
# ---------------------------------------------------------------------------

def test_stream_empty_index_returns_400(client):
    with patch("app.main.vector_store") as mock_vs:
        mock_vs.size = 0
        resp = client.post("/query/stream", json={"query": "test"})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Tests — SSE event structure
# ---------------------------------------------------------------------------

def test_stream_emits_retrieving_generating_evaluating_done(client):
    """All four phase events must appear in the event stream."""
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = [(_chunk(), 0.9)]
        mock_ret.to_citations.return_value = [_citation()]

        resp = client.post(
            "/query/stream",
            json={"query": "test question"},
            headers={"Accept": "text/event-stream"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    event_names = [e.get("event") for e in events]

    for expected in ("retrieving", "generating", "evaluating", "done"):
        assert expected in event_names, f"Missing '{expected}' in {event_names}"


def test_stream_phase_event_order(client):
    """Events must arrive in the correct phase order."""
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = []
        mock_ret.to_citations.return_value = []

        resp = client.post("/query/stream", json={"query": "order test"})

    events = _parse_sse_events(resp.text)
    phase_events = [
        e["event"] for e in events
        if e.get("event") in ("retrieving", "generating", "evaluating", "done")
    ]
    assert phase_events == ["retrieving", "generating", "evaluating", "done"], (
        f"Event order mismatch: {phase_events}"
    )


def test_stream_done_event_contains_full_response(client):
    """The 'done' event data must contain all QueryResponse fields."""
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = []
        mock_ret.to_citations.return_value = []

        resp = client.post("/query/stream", json={"query": "full response test"})

    events = _parse_sse_events(resp.text)
    done_events = [e for e in events if e.get("event") == "done"]
    assert len(done_events) == 1, f"Expected 1 'done' event, got {len(done_events)}"

    data = done_events[0]["data"]
    for field in ("answer", "citations", "eval", "generator_backend",
                  "response_time_ms", "query_id", "timings"):
        assert field in data, f"Missing field '{field}' in done event data"


def test_stream_timings_have_all_phases(client):
    """Timings in the done event must include all 5 phase keys."""
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = []
        mock_ret.to_citations.return_value = []

        resp = client.post("/query/stream", json={"query": "timings test"})

    events = _parse_sse_events(resp.text)
    done = next(e for e in events if e.get("event") == "done")
    timings = done["data"]["timings"]
    for key in ("embedding_ms", "retrieval_ms", "generation_ms", "evaluation_ms", "total_ms"):
        assert key in timings, f"Missing timing key: {key}"
        assert timings[key] >= 0


def test_stream_query_id_is_uuid_v4(client):
    """query_id in the done event must be a valid UUID v4."""
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = []
        mock_ret.to_citations.return_value = []

        resp = client.post("/query/stream", json={"query": "uuid test"})

    events = _parse_sse_events(resp.text)
    done = next(e for e in events if e.get("event") == "done")
    query_id = done["data"]["query_id"]
    assert uuid_pattern.match(query_id), f"query_id is not UUID v4: {query_id!r}"


# ---------------------------------------------------------------------------
# Tests — /query (non-streaming) timing fields
# ---------------------------------------------------------------------------

def test_query_response_has_timing_breakdown(client):
    """POST /query must return query_id and timings alongside existing fields."""
    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = []
        mock_ret.to_citations.return_value = []

        resp = client.post("/query", json={"query": "timing test"})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "query_id" in data
    assert "timings" in data
    timings = data["timings"]
    for key in ("embedding_ms", "retrieval_ms", "generation_ms", "evaluation_ms", "total_ms"):
        assert key in timings, f"Missing timing key: {key}"
        assert timings[key] >= 0


def test_stream_query_cache_hit(client):
    """POST /query/stream must return cached answers for near-identical queries."""
    from app.query_cache import query_cache
    query_cache.clear()

    with patch("app.main.vector_store") as mock_vs, \
         patch("app.main.retriever") as mock_ret, \
         patch("app.main.embed_model") as mock_em, \
         patch("app.main.generator", _mock_gen()), \
         patch("app.main.evaluator", _mock_eval()):

        mock_vs.size = 1
        mock_em.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        mock_ret.retrieve_from_vec.return_value = []
        mock_ret.to_citations.return_value = []

        # Run first query to populate cache
        resp1 = client.post("/query/stream", json={"query": "cache test"})
        assert resp1.status_code == 200
        events1 = _parse_sse_events(resp1.text)
        done1 = next(e for e in events1 if e.get("event") == "done")

        # Second identical query should hit the cache
        resp2 = client.post("/query/stream", json={"query": "cache test"})
        assert resp2.status_code == 200
        events2 = _parse_sse_events(resp2.text)
        done2 = next(e for e in events2 if e.get("event") == "done")

        assert done2["data"]["generator_backend"] == "cache"
        assert done2["data"]["answer"] == done1["data"]["answer"]

