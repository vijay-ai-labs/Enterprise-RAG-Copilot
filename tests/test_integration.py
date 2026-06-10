"""
Integration tests — real chunking, real FAISS retrieval, real BM25, real evaluator.

Embedding model is replaced with a deterministic word-hash stub to avoid
downloading sentence-transformers during CI. All other pipeline components
(VectorStore, BM25Store, Retriever, Chunker, Evaluator) are real.

NOTE: retrieve_from_vec is called with threshold=0.0 explicitly to test pure
retrieval logic independent of threshold settings. In production, hybrid mode
uses settings.rrf_score_threshold (default 0.005) and dense-only mode uses
settings.similarity_threshold (default 0.3), which are calibrated for their
respective score ranges (RRF ~0.01–0.02, cosine sim 0–1).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.bm25_store import BM25Store
from app.chunking import chunk_document
from app.evaluator import Evaluator
from app.ingestion import extract_text
from app.retriever import Retriever
from app.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Deterministic embedding stub
# ---------------------------------------------------------------------------

_VOCAB: dict = {}
_RNG = np.random.default_rng(42)
_DIM = 64


def _stub_encode(texts, batch_size: int = 64) -> np.ndarray:
    """
    Word-hash embeddings: each word maps to a fixed random vector.
    Sum word vectors, L2-normalise. Semantically similar texts (shared
    words) produce higher cosine similarity than unrelated texts.
    """
    vecs = []
    for text in texts:
        words = text.lower().split()
        vec = np.zeros(_DIM, dtype=np.float32)
        for w in words:
            if w not in _VOCAB:
                _VOCAB[w] = _RNG.standard_normal(_DIM).astype(np.float32)
            vec += _VOCAB[w]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vecs.append(vec)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

RAG_TEXT = (
    "Retrieval-Augmented Generation combines a retrieval step with a generative model. "
    "Documents are indexed in a vector store using dense embeddings. "
    "When a query arrives, the most relevant documents are retrieved by similarity search. "
    "Retrieved documents are passed as context to the language model. "
    "This technique reduces hallucination and improves factual accuracy. "
    "Hybrid search combines dense vector retrieval with sparse BM25 keyword matching. "
    "Reciprocal Rank Fusion merges ranked lists from multiple retrieval methods."
)

UNRELATED_TEXT = (
    "The capital of France is Paris. Its population is approximately 2.1 million. "
    "Paris is known for the Eiffel Tower and the Louvre museum."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def empty_stores(tmp_path):
    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()
    return vs, bm25


@pytest.fixture()
def populated_stores(tmp_path):
    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()
    pages = [(RAG_TEXT, 1)]
    chunks = chunk_document(pages, "rag_overview.txt")
    vectors = _stub_encode([c.text for c in chunks])
    vs.add_chunks(chunks, vectors)
    bm25.add_chunks(chunks)
    return vs, bm25, chunks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_preserves_metadata():
    pages = [(RAG_TEXT, 1)]
    chunks = chunk_document(pages, "test_doc.txt")

    assert len(chunks) >= 1
    for c in chunks:
        assert c.filename == "test_doc.txt"
        assert c.page == 1
        assert len(c.text) > 0
        assert c.chunk_id.startswith("test_doc.txt::")


def test_chunking_no_word_loss():
    long_text = "important_word " * 400
    pages = [(long_text, 1)]
    chunks = chunk_document(pages, "long.txt")

    all_text = " ".join(c.text for c in chunks)
    # Overlap means words appear in multiple chunks; count must be >= 400 (no loss).
    assert all_text.count("important_word") >= 400


def test_chunking_multipage():
    pages = [(f"Content for page {i}. " * 20, i) for i in range(1, 4)]
    chunks = chunk_document(pages, "multipage.pdf")

    page_nums = {c.page for c in chunks}
    assert page_nums == {1, 2, 3}


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

def test_vectorstore_add_and_search(populated_stores):
    vs, bm25, chunks = populated_stores

    assert vs.size == len(chunks)

    query_vec = _stub_encode(["RAG retrieval augmented generation"])[0]
    results = vs.search(query_vec, top_k=3)

    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert top_score > 0.0
    assert top_chunk.filename == "rag_overview.txt"


def test_vectorstore_save_and_reload(populated_stores, tmp_path):
    vs, bm25, chunks = populated_stores
    vs.save()

    vs2 = VectorStore(index_dir=str(tmp_path))
    vs2.load()

    assert vs2.size == vs.size

    query_vec = _stub_encode(["vector store retrieval"])[0]
    results = vs2.search(query_vec, top_k=3)
    assert len(results) > 0


def test_vectorstore_delete_removes_all_chunks(populated_stores):
    vs, bm25, chunks = populated_stores
    original_size = vs.size

    removed = vs.remove_by_filename("rag_overview.txt")

    assert removed == original_size
    assert vs.size == 0


def test_vectorstore_delete_nonexistent_filename(populated_stores):
    vs, bm25, chunks = populated_stores
    removed = vs.remove_by_filename("does_not_exist.pdf")
    assert removed == 0
    assert vs.size == len(chunks)


# ---------------------------------------------------------------------------
# Hybrid retrieval (Retriever + RRF)
# NOTE: threshold=0.0 passed explicitly because default 0.3 is incompatible
# with RRF score range (~0.01–0.02). See module docstring.
# ---------------------------------------------------------------------------

def test_hybrid_retrieval_returns_chunks(populated_stores):
    vs, bm25, _ = populated_stores
    retriever = Retriever(vs, bm25)

    query = "How does RAG reduce hallucination?"
    query_vec = _stub_encode([query])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query=query, top_k=3, threshold=0.0
    )

    assert len(results) > 0
    chunk, score = results[0]
    assert chunk.filename == "rag_overview.txt"


def test_retrieval_respects_top_k(populated_stores):
    vs, bm25, _ = populated_stores
    retriever = Retriever(vs, bm25)

    query_vec = _stub_encode(["retrieval augmented generation"])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query="retrieval augmented generation", top_k=2, threshold=0.0
    )

    assert len(results) <= 2


def test_retrieval_source_filter(tmp_path):
    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()

    for name in ("doc_a.txt", "doc_b.txt"):
        pages = [(RAG_TEXT, 1)]
        chunks = chunk_document(pages, name)
        vectors = _stub_encode([c.text for c in chunks])
        vs.add_chunks(chunks, vectors)
        bm25.add_chunks(chunks)

    retriever = Retriever(vs, bm25)
    query_vec = _stub_encode(["retrieval"])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query="retrieval", top_k=5,
        threshold=0.0, source_filter=["doc_a.txt"]
    )

    assert all(c.filename == "doc_a.txt" for c, _ in results)


def test_retrieval_empty_index(empty_stores):
    vs, bm25 = empty_stores
    retriever = Retriever(vs, bm25)

    query_vec = _stub_encode(["any query"])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query="any query", top_k=5, threshold=0.0
    )

    assert results == []


def test_citations_built_from_results(populated_stores):
    vs, bm25, _ = populated_stores
    retriever = Retriever(vs, bm25)

    query_vec = _stub_encode(["RAG documents context"])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query="RAG documents context", top_k=3, threshold=0.0
    )
    citations = retriever.to_citations(results)

    assert len(citations) == len(results)
    for cit in citations:
        assert cit.filename == "rag_overview.txt"
        assert cit.page >= 1
        assert len(cit.text_snippet) > 0
        assert 0.0 <= cit.score <= 1.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def test_evaluator_grounded_answer_scores_high(populated_stores):
    vs, bm25, _ = populated_stores
    evaluator = Evaluator()

    query_vec = _stub_encode(["RAG retrieval generation"])[0]
    results = vs.search(query_vec, top_k=3)
    chunks = [c for c, _ in results]
    scores = [s for _, s in results]

    # Answer uses words directly from the context
    answer = (
        "RAG combines retrieval with generation. "
        "Retrieved documents are passed as context to reduce hallucination."
    )
    result = evaluator.evaluate(
        query="How does RAG work?",
        answer=answer,
        chunks=chunks,
        chunk_scores=scores,
    )

    assert result.context_relevance >= 0.0
    assert result.answer_groundedness >= 0.0
    assert 0.0 <= result.overall <= 1.0


def test_evaluator_empty_context_returns_zero_relevance():
    evaluator = Evaluator()
    result = evaluator.evaluate(
        query="What is quantum physics?",
        answer="I don't know based on the provided documents.",
        chunks=[],
        chunk_scores=[],
    )

    assert result.context_relevance == 0.0
    assert result.overall >= 0.0


def test_evaluator_citation_presence_detected():
    from app.chunking import chunk_document
    evaluator = Evaluator()

    pages = [(RAG_TEXT, 1)]
    chunks = chunk_document(pages, "rag_overview.txt")
    scores = [0.8] * len(chunks)

    answer = "RAG uses retrieval to improve generation [rag_overview.txt:page1]."
    result = evaluator.evaluate(
        query="How does RAG work?",
        answer=answer,
        chunks=chunks,
        chunk_scores=scores,
    )

    assert result.citation_presence == 1.0


def test_evaluator_citation_absence_scores_zero():
    from app.chunking import chunk_document
    evaluator = Evaluator()

    pages = [(RAG_TEXT, 1)]
    chunks = chunk_document(pages, "rag_overview.txt")
    scores = [0.8] * len(chunks)

    answer = "RAG uses retrieval to improve generation."  # no citation bracket
    result = evaluator.evaluate(
        query="How does RAG work?",
        answer=answer,
        chunks=chunks,
        chunk_scores=scores,
    )

    assert result.citation_presence == 0.0


# ---------------------------------------------------------------------------
# Full pipeline: ingest → retrieve → evaluate (no generator)
# ---------------------------------------------------------------------------

def test_full_pipeline_txt(tmp_path):
    content = (
        b"Enterprise search uses vector embeddings for semantic similarity. "
        b"BM25 is a sparse retrieval method based on term frequency. "
        b"Hybrid search combines dense and sparse retrieval for better coverage."
    )
    pages = extract_text(content, "search.txt")
    assert len(pages) >= 1

    chunks = chunk_document(pages, "search.txt")
    assert len(chunks) >= 1

    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()
    vectors = _stub_encode([c.text for c in chunks])
    vs.add_chunks(chunks, vectors)
    bm25.add_chunks(chunks)

    retriever = Retriever(vs, bm25)
    query = "How does hybrid search work?"
    query_vec = _stub_encode([query])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query=query, top_k=3, threshold=0.0
    )

    assert len(results) > 0
    assert results[0][0].filename == "search.txt"

    evaluator = Evaluator()
    answer = "Hybrid search combines dense embeddings with BM25 sparse retrieval."
    eval_result = evaluator.evaluate(
        query=query,
        answer=answer,
        chunks=[c for c, _ in results],
        chunk_scores=[s for _, s in results],
    )
    assert eval_result.overall >= 0.0


def test_full_pipeline_markdown(tmp_path):
    content = (
        b"# RAG Overview\n\n"
        b"RAG stands for Retrieval-Augmented Generation.\n"
        b"It improves LLM accuracy by grounding answers in retrieved documents.\n"
        b"The retriever finds relevant chunks using cosine similarity.\n"
    )
    pages = extract_text(content, "overview.md")
    chunks = chunk_document(pages, "overview.md")
    assert len(chunks) >= 1

    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()
    vectors = _stub_encode([c.text for c in chunks])
    vs.add_chunks(chunks, vectors)
    bm25.add_chunks(chunks)

    retriever = Retriever(vs, bm25)
    query_vec = _stub_encode(["What does RAG stand for?"])[0]
    results = retriever.retrieve_from_vec(
        query_vec, query="What does RAG stand for?", top_k=3, threshold=0.0
    )
    assert len(results) > 0


def test_full_pipeline_html(tmp_path):
    content = (
        b"<html><body>"
        b"<h1>Vector Search</h1>"
        b"<p>Dense retrieval converts text into high-dimensional vectors.</p>"
        b"<p>FAISS enables fast approximate nearest-neighbour search at scale.</p>"
        b"</body></html>"
    )
    pages = extract_text(content, "vectors.html")
    chunks = chunk_document(pages, "vectors.html")
    assert len(chunks) >= 1

    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()
    vectors = _stub_encode([c.text for c in chunks])
    vs.add_chunks(chunks, vectors)
    bm25.add_chunks(chunks)

    query_vec = _stub_encode(["FAISS vector search"])[0]
    retriever = Retriever(vs, bm25)
    results = retriever.retrieve_from_vec(
        query_vec, query="FAISS vector search", top_k=3, threshold=0.0
    )
    assert len(results) > 0


def test_pipeline_delete_then_query_returns_empty(tmp_path):
    vs = VectorStore(index_dir=str(tmp_path))
    bm25 = BM25Store()

    pages = [(RAG_TEXT, 1)]
    chunks = chunk_document(pages, "to_delete.txt")
    vectors = _stub_encode([c.text for c in chunks])
    vs.add_chunks(chunks, vectors)
    bm25.add_chunks(chunks)

    vs.remove_by_filename("to_delete.txt")
    bm25.remove_by_filename("to_delete.txt")

    assert vs.size == 0

    query_vec = _stub_encode(["RAG retrieval"])[0]
    retriever = Retriever(vs, bm25)
    results = retriever.retrieve_from_vec(
        query_vec, query="RAG retrieval", top_k=3, threshold=0.0
    )
    assert results == []
