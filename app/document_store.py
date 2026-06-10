"""
SQLite-backed document catalog for lifecycle management.
Tracks document-level metadata separate from chunk-level vector metadata.
Supports: register (ingest), list, delete, re-ingest (upsert).
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from app.logger import logger


class DocumentStore:
    """Tracks ingested documents with their metadata."""

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id           TEXT PRIMARY KEY,
                    filename         TEXT NOT NULL UNIQUE,
                    file_type        TEXT NOT NULL,
                    pages_extracted  INTEGER NOT NULL DEFAULT 0,
                    chunks_stored    INTEGER NOT NULL DEFAULT 0,
                    file_size_bytes  INTEGER NOT NULL DEFAULT 0,
                    ingested_at      REAL NOT NULL,
                    status           TEXT NOT NULL DEFAULT 'ready'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_filename ON documents(filename)")

    def register(
        self,
        *,
        filename: str,
        file_type: str,
        pages_extracted: int,
        chunks_stored: int,
        file_size_bytes: int,
    ) -> str:
        """Upsert a document record. Returns doc_id."""
        doc_id = str(uuid.uuid4())
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO documents
                   (doc_id, filename, file_type, pages_extracted, chunks_stored,
                    file_size_bytes, ingested_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'ready')
                   ON CONFLICT(filename) DO UPDATE SET
                     doc_id          = excluded.doc_id,
                     file_type       = excluded.file_type,
                     pages_extracted = excluded.pages_extracted,
                     chunks_stored   = excluded.chunks_stored,
                     file_size_bytes = excluded.file_size_bytes,
                     ingested_at     = excluded.ingested_at,
                     status          = 'ready'
                """,
                (
                    doc_id, filename, file_type, pages_extracted,
                    chunks_stored, file_size_bytes, time.time(),
                ),
            )
        logger.info(
            "Document registered: '%s' id=%s chunks=%d", filename, doc_id, chunks_stored
        )
        return doc_id

    def delete_by_id(self, doc_id: str) -> Optional[str]:
        """Delete by doc_id. Returns filename if found, None if not."""
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT filename FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        logger.info("Document deleted: id=%s filename=%s", doc_id, row["filename"])
        return row["filename"]

    def list_all(self) -> List[Dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_filename(self, filename: str) -> Optional[Dict]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename=?", (filename,)
            ).fetchone()
        return dict(row) if row else None
