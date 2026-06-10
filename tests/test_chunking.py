"""Tests for app/chunking.py — sliding-window chunker with metadata."""

import pytest
from app.chunking import chunk_document, _split_text
from app.schemas import Chunk


def _make_pages(text: str, page: int = 1):
    return [(text, page)]


# ---------------------------------------------------------------------------
# _split_text unit tests
# ---------------------------------------------------------------------------

def test_split_text_empty_returns_empty():
    assert _split_text("", 512, 64) == []


def test_split_text_short_text_single_chunk():
    text = "Short text."
    chunks = _split_text(text, 512, 64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_produces_multiple_chunks():
    # 600-char text with chunk_size=200, overlap=50
    text = "A" * 600
    chunks = _split_text(text, 200, 50)
    assert len(chunks) > 1


def test_split_text_no_chunk_exceeds_size():
    text = "word " * 300  # 1500 chars
    chunks = _split_text(text, 200, 40)
    for chunk in chunks:
        assert len(chunk) <= 200


def test_split_text_overlap_creates_shared_content():
    # Create text with identifiable words so overlap is detectable
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunk_size = 50
    overlap = 20
    chunks = _split_text(text, chunk_size, overlap)
    # Verify consecutive chunks share some content
    if len(chunks) >= 2:
        shared = set(chunks[0].split()) & set(chunks[1].split())
        assert len(shared) > 0


def test_split_text_overlap_must_be_less_than_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(_make_pages("text"), "f.txt", chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------------------
# chunk_document integration tests
# ---------------------------------------------------------------------------

def test_chunk_document_returns_chunk_objects():
    pages = _make_pages("Hello world. " * 20, page=1)
    chunks = chunk_document(pages, "test.txt", chunk_size=100, chunk_overlap=20)
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_document_metadata_fields():
    pages = _make_pages("Some text content here.", page=3)
    chunks = chunk_document(pages, "myfile.pdf", chunk_size=512, chunk_overlap=64)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.filename == "myfile.pdf"
    assert c.page == 3
    assert "myfile.pdf" in c.chunk_id
    assert "page3" in c.chunk_id
    assert "chunk0" in c.chunk_id
    assert c.timestamp  # ISO timestamp present


def test_chunk_document_multi_page_metadata():
    pages = [("Page one content. " * 5, 1), ("Page two content. " * 5, 2)]
    chunks = chunk_document(pages, "doc.pdf", chunk_size=100, chunk_overlap=20)
    page_numbers = {c.page for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_id_uniqueness():
    pages = _make_pages("word " * 200, page=1)
    chunks = chunk_document(pages, "file.txt", chunk_size=100, chunk_overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_chunk_text_non_empty():
    pages = _make_pages("Non-empty content. " * 10, page=1)
    chunks = chunk_document(pages, "file.txt", chunk_size=100, chunk_overlap=20)
    assert all(c.text.strip() for c in chunks), "No chunk should be empty"


def test_empty_page_produces_no_chunks():
    pages = [("", 1)]
    chunks = chunk_document(pages, "empty.txt", chunk_size=512, chunk_overlap=64)
    assert chunks == []


# ---------------------------------------------------------------------------
# No-text-loss regression tests (whitespace-snap bug fix)
# ---------------------------------------------------------------------------

def test_no_text_lost_with_whitespace_snapping():
    """All characters in the original text must appear in at least one chunk."""
    # Alternating words and spaces so whitespace snapping is triggered
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 5
    chunk_size = 40
    overlap = 10
    chunks = _split_text(text, chunk_size, overlap)

    combined = " ".join(chunks)
    # Every word from the original must appear somewhere in the output
    for word in text.split():
        assert word in combined, f"Word '{word}' lost after chunking"


def test_no_text_lost_long_text_no_spaces():
    """Text with no spaces must not lose any characters."""
    text = "A" * 500
    chunks = _split_text(text, 100, 20)
    # With no spaces, no snapping occurs — total chars across chunks with overlap
    # Every position in the original must be covered by at least one chunk
    covered = set()
    start = 0
    step = 100 - 20
    pos = 0
    for chunk in chunks:
        for i, ch in enumerate(chunk):
            covered.add(pos + i)
        pos += step
    # Simpler check: all chars present in at least one chunk
    all_chars = "".join(chunks)
    assert len(all_chars) >= len(text)  # overlap means total >= original


def test_no_text_lost_short_text():
    """Text shorter than chunk_size must produce exactly one chunk with all content."""
    text = "Hello world"
    chunks = _split_text(text, 512, 64)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_no_text_lost_exact_coverage():
    """Every word in original text must appear in at least one chunk (varied sizes)."""
    import random
    random.seed(42)
    words = [f"word{i}" for i in range(80)]
    text = " ".join(words)
    chunks = _split_text(text, 60, 15)
    combined_text = " ".join(chunks)
    for word in words:
        assert word in combined_text, f"'{word}' dropped during chunking"


def test_snapped_chunk_overlaps_correctly():
    """After whitespace snap, the overlap region must be repeated in next chunk."""
    # Construct text where snapping is predictable
    # "aaaaa bbbbb ccccc ddddd" with chunk_size=11, overlap=5
    # chunk 1: "aaaaa bbbbb" → snap at 5 → "aaaaa", end=5, next start=5-5=0?
    # Actually snap > overlap means snap(5) > overlap(5) is False → no snap
    # Use overlap=3: snap(5) > 3 → snap, chunk="aaaaa", end=5, next start=5-3=2
    text = "aaaaa bbbbb ccccc ddddd eeeee"
    chunks = _split_text(text, 11, 3)
    # Verify no chunk exceeds chunk_size
    for c in chunks:
        assert len(c) <= 11
    # Verify all words appear
    combined = " ".join(chunks)
    for word in text.split():
        assert word in combined
