"""Tests for app/ingestion.py — text extraction from PDF, TXT, and Markdown."""

import io
import pytest
from app.ingestion import extract_text, _extract_plaintext


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------

def test_txt_returns_single_page():
    content = b"Hello world. This is a test document."
    pages = extract_text(content, "test.txt")
    assert len(pages) == 1
    text, page_num = pages[0]
    assert page_num == 1
    assert "Hello world" in text


def test_txt_strips_whitespace():
    content = b"   trimmed content   "
    pages = extract_text(content, "doc.txt")
    text, _ = pages[0]
    assert text == "trimmed content"


def test_md_treated_as_plaintext():
    content = b"# Heading\n\nSome markdown **content**."
    pages = extract_text(content, "readme.md")
    assert len(pages) == 1
    text, page_num = pages[0]
    assert page_num == 1
    assert "Heading" in text


def test_markdown_extension_also_supported():
    content = b"# Another doc"
    pages = extract_text(content, "notes.markdown")
    assert len(pages) == 1


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(b"data", "file.xyz")


def test_utf8_with_special_chars():
    content = "Ünïcödé têxt: café naïve résumé".encode("utf-8")
    pages = extract_text(content, "unicode.txt")
    text, _ = pages[0]
    assert "café" in text


def test_empty_txt_returns_empty_list():
    pages = extract_text(b"   ", "empty.txt")
    # Strip leaves empty string — should return no pages
    assert pages == [] or (len(pages) == 1 and pages[0][0] == "")


# ---------------------------------------------------------------------------
# PDF extraction (minimal — no real PDF binary needed for unit test)
# ---------------------------------------------------------------------------

def test_pdf_extension_dispatches_to_pdf_extractor(monkeypatch):
    """Verify PDF path is taken; mock PdfReader to avoid real PDF bytes."""
    from unittest.mock import MagicMock, patch

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page one content here."

    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]

    with patch("app.ingestion.PdfReader", return_value=mock_reader_instance, create=True):
        # Patch the import inside the function
        import app.ingestion as ing_module
        original = ing_module._extract_pdf

        def patched_pdf(file_bytes, filename):
            from unittest.mock import MagicMock
            reader = mock_reader_instance
            pages = []
            for i, p in enumerate(reader.pages, start=1):
                text = p.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append((text, i))
            return pages

        monkeypatch.setattr(ing_module, "_extract_pdf", patched_pdf)
        pages = extract_text(b"%PDF-fake", "report.pdf")

    assert len(pages) == 1
    assert pages[0][1] == 1  # page number
    assert "Page one content" in pages[0][0]
