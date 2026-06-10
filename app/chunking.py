"""
Chunking: sliding-window text splitter with overlap.
Each chunk preserves metadata: filename, page, chunk_id, timestamp.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

from app.config import settings
from app.schemas import Chunk

# (text, page_number)
PageText = Tuple[str, int]


def chunk_document(
    pages: List[PageText],
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be less than chunk_size ({size})")

    chunks: List[Chunk] = []
    ts = datetime.now(timezone.utc).isoformat()

    for page_text, page_num in pages:
        page_chunks = _split_text(page_text, size, overlap)
        for idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{filename}::page{page_num}::chunk{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    filename=filename,
                    page=page_num,
                    text=chunk_text,
                    timestamp=ts,
                )
            )

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of at most `chunk_size` characters,
    stepping by (chunk_size - overlap) so consecutive chunks share context.
    Splits prefer whitespace boundaries to avoid cutting mid-word.
    """
    if not text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        # Snap end to nearest whitespace (unless we're at the final chunk).
        # Advance start from the snapped end so no text is skipped.
        if end < len(text):
            snap = chunk.rfind(" ")
            if snap != -1 and snap > overlap:
                chunk = chunk[:snap]
                end = start + snap

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks
