"""
SQLite-backed feedback storage.
Persists thumbs-up/down ratings and optional comments linked to query/conversation IDs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from app.logger import logger


class FeedbackStore:
    """Durable feedback store backed by SQLite."""

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
                CREATE TABLE IF NOT EXISTS feedback (
                    id              TEXT PRIMARY KEY,
                    query_id        TEXT NOT NULL,
                    conversation_id TEXT,
                    rating          INTEGER NOT NULL,
                    comment         TEXT,
                    answer          TEXT,
                    chunk_ids       TEXT,
                    created_at      REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_id ON feedback(query_id)")

    def store(
        self,
        *,
        query_id: str,
        conversation_id: Optional[str],
        rating: int,
        comment: Optional[str],
        answer: Optional[str],
        chunk_ids: Optional[List[str]],
    ) -> str:
        feedback_id = str(uuid.uuid4())
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO feedback
                   (id, query_id, conversation_id, rating, comment, answer, chunk_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    feedback_id,
                    query_id,
                    conversation_id,
                    rating,
                    comment,
                    answer,
                    json.dumps(chunk_ids or []),
                    time.time(),
                ),
            )
        logger.info(
            "Feedback stored id=%s rating=%d query_id=%s",
            feedback_id, rating, query_id,
        )
        return feedback_id

    def list_recent(self, limit: int = 100) -> List[dict]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
