"""Tests for VectorStore and Retriever — search, threshold, edge cases."""

import numpy as np
import pytest
from app.schemas import Chunk
from app.vectorstore import VectorStore


def _make_chunk(text: str, page: int = 1, filename: str = "test.txt") -> Chunk:
    from datetime import datetime, timezone
    return Chunk(
        chunk_id=f"{filename}::page{page}::chunk0",
        filename=filename,
        page=page,
        text=text,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    """Returns a unit vector with 1.0 at position idx (simulates normalised embedding)."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# VectorStore unit tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_empty_store_returns_nothing(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        result = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
        assert result == []

    def test_size_zero_on_empty_store(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        assert store.size == 0

    def test_add_and_search_returns_chunk(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        dim = 4
        chunk = _make_chunk("Machine learning is powerful.")
        vec = _unit_vec(dim, 0).reshape(1, -1)
        store.add_chunks([chunk], vec)

        query = _unit_vec(dim, 0)
        results = store.search(query, top_k=1)
        assert len(results) == 1
        retrieved_chunk, score = results[0]
        assert retrieved_chunk.text == "Machine learning is powerful."
        assert score > 0.9  # should be ~1.0 for identical vectors

    def test_top_k_limits_results(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        dim = 8
        chunks = [_make_chunk(f"chunk {i}", page=i) for i in range(5)]
        vecs = np.eye(5, dim, dtype=np.float32)  # 5 orthogonal unit vectors
        store.add_chunks(chunks, vecs)

        results = store.search(_unit_vec(dim, 0), top_k=3)
        assert len(results) <= 3

    def test_results_sorted_by_score_descending(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        chunks = [_make_chunk(f"doc {i}", page=i) for i in range(3)]
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # identical to query → score 1.0
            [0.8, 0.6, 0.0, 0.0],  # partial match
            [0.0, 0.0, 1.0, 0.0],  # orthogonal → score 0.0
        ], dtype=np.float32)
        store.add_chunks(chunks, vecs)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load_roundtrip(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        dim = 4
        chunk = _make_chunk("Persistent chunk.")
        vec = _unit_vec(dim, 1).reshape(1, -1)
        store.add_chunks([chunk], vec)
        store.save()

        store2 = VectorStore(index_dir=str(tmp_path))
        loaded = store2.load()
        assert loaded is True
        assert store2.size == 1

        results = store2.search(_unit_vec(dim, 1), top_k=1)
        assert len(results) == 1
        assert results[0][0].text == "Persistent chunk."

    def test_chunks_and_vectors_length_mismatch_raises(self, tmp_path):
        store = VectorStore(index_dir=str(tmp_path))
        chunks = [_make_chunk("a"), _make_chunk("b")]
        vecs = np.eye(1, 4, dtype=np.float32)  # only 1 vector for 2 chunks
        with pytest.raises(ValueError, match="same length"):
            store.add_chunks(chunks, vecs)

    def test_save_and_load_uses_json_metadata(self, tmp_path):
        """Saved metadata must be JSON, not pickle."""
        import json as _json
        store = VectorStore(index_dir=str(tmp_path))
        chunk = _make_chunk("JSON roundtrip.")
        store.add_chunks([chunk], _unit_vec(4, 0).reshape(1, -1))
        store.save()

        meta_json = tmp_path / "metadata.json"
        assert meta_json.exists(), "metadata.json must be written"
        with open(meta_json) as f:
            raw = _json.load(f)
        assert isinstance(raw, list)
        assert raw[0]["text"] == "JSON roundtrip."

    def test_load_detects_index_metadata_mismatch(self, tmp_path):
        """load() raises RuntimeError when vector count != metadata count."""
        import json as _json
        import faiss as _faiss

        # Write a 2-vector FAISS index but only 1 metadata entry
        dim = 4
        idx = _faiss.IndexFlatIP(dim)
        vecs = np.eye(2, dim, dtype=np.float32)
        idx.add(vecs)
        _faiss.write_index(idx, str(tmp_path / "faiss.index"))

        # Write only 1 metadata entry
        from datetime import datetime, timezone
        meta = [
            {
                "chunk_id": "f::page1::chunk0",
                "filename": "f.txt",
                "page": 1,
                "text": "only one",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
        with open(tmp_path / "metadata.json", "w") as f:
            _json.dump(meta, f)

        store2 = VectorStore(index_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="mismatch"):
            store2.load()

    def test_load_migrates_legacy_pickle(self, tmp_path):
        """Load auto-migrates metadata.pkl → metadata.json and validates counts."""
        import pickle as _pickle
        import faiss as _faiss

        # Write FAISS index with 1 vector
        dim = 4
        idx = _faiss.IndexFlatIP(dim)
        idx.add(_unit_vec(dim, 0).reshape(1, -1))
        _faiss.write_index(idx, str(tmp_path / "faiss.index"))

        # Write legacy pickle metadata
        chunk = _make_chunk("pickle legacy")
        with open(tmp_path / "metadata.pkl", "wb") as f:
            _pickle.dump([chunk], f)

        store = VectorStore(index_dir=str(tmp_path))
        loaded = store.load()
        assert loaded is True
        assert store.size == 1
        assert (tmp_path / "metadata.json").exists(), "Should have migrated to JSON"

    def test_get_vectors_retrieves_correct_vectors(self, tmp_path):
        """get_vectors() must return the same vectors that were stored in FAISS in O(1)."""
        store = VectorStore(index_dir=str(tmp_path))
        chunks = [_make_chunk("chunk a", filename="a.txt"), _make_chunk("chunk b", filename="b.txt")]
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        store.add_chunks(chunks, vecs)

        # Retrieve vectors for both chunks
        retrieved_vecs = store.get_vectors(chunks)
        assert len(retrieved_vecs) == 2
        np.testing.assert_array_almost_equal(retrieved_vecs[0], vecs[0])
        np.testing.assert_array_almost_equal(retrieved_vecs[1], vecs[1])

        # Try with single chunk
        retrieved_vecs_single = store.get_vectors([chunks[1]])
        assert len(retrieved_vecs_single) == 1
        np.testing.assert_array_almost_equal(retrieved_vecs_single[0], vecs[1])

        # Try with nonexistent chunk
        nonexistent = _make_chunk("nonexistent")
        retrieved_vecs_none = store.get_vectors([nonexistent])
        assert len(retrieved_vecs_none) == 0


# ---------------------------------------------------------------------------
# Retriever threshold tests (using real EmbeddingModel would be slow — mock it)
# ---------------------------------------------------------------------------

class TestRetriever:
    def test_threshold_filters_low_scores(self, tmp_path, monkeypatch):
        from app.retriever import Retriever

        store = VectorStore(index_dir=str(tmp_path))

        chunks = [
            _make_chunk("highly relevant content", page=1),
            _make_chunk("totally unrelated stuff", page=2),
        ]
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # will score ~1.0 with query
            [0.0, 1.0, 0.0, 0.0],  # will score ~0.0 with query
        ], dtype=np.float32)
        store.add_chunks(chunks, vecs)

        # Monkeypatch embed so no real model is loaded
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        monkeypatch.setattr(
            "app.retriever.EmbeddingModel.encode",
            lambda self, texts, **kw: np.array([query_vec], dtype=np.float32),
        )

        retriever = Retriever(store)
        results = retriever.retrieve("relevant query", top_k=5, threshold=0.5)

        # Only the first chunk should pass threshold
        assert len(results) == 1
        assert results[0][0].text == "highly relevant content"

    def test_to_citations_format(self, tmp_path, monkeypatch):
        from app.retriever import Retriever

        store = VectorStore(index_dir=str(tmp_path))
        dim = 4
        chunk = _make_chunk("Citation test content.", page=7, filename="report.pdf")
        vec = _unit_vec(dim, 0).reshape(1, -1)
        store.add_chunks([chunk], vec)

        query_vec = _unit_vec(dim, 0)
        monkeypatch.setattr(
            "app.retriever.EmbeddingModel.encode",
            lambda self, texts, **kw: np.array([query_vec], dtype=np.float32),
        )

        retriever = Retriever(store)
        results = retriever.retrieve("test", top_k=1, threshold=0.0)
        citations = retriever.to_citations(results)

        assert len(citations) == 1
        c = citations[0]
        assert c.filename == "report.pdf"
        assert c.page == 7
        assert len(c.text_snippet) <= 200
        assert 0.0 <= c.score <= 1.0
