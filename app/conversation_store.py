"""
SQLite-backed multi-turn conversation store with TTL-based expiry.
Thread-safe. Persists conversation history across server restarts.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.logger import logger


@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationStore:
    """SQLite-backed store for multi-turn conversations with TTL expiry."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._path = Path(db_path or settings.db_path)
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
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_id "
                "ON conversation_turns(conversation_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_ts "
                "ON conversation_turns(created_at)"
            )

    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        now = time.time()
        max_rows = settings.max_conversation_turns * 2  # user + assistant per turn

        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO conversation_turns
                   (conversation_id, role, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, role, content, now),
            )
            # Keep only the most recent max_rows turns per conversation
            conn.execute(
                """DELETE FROM conversation_turns
                   WHERE conversation_id = ?
                   AND id NOT IN (
                       SELECT id FROM conversation_turns
                       WHERE conversation_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?
                   )""",
                (conversation_id, conversation_id, max_rows),
            )

    def get_history(self, conversation_id: str) -> List[Turn]:
        cutoff = time.time() - settings.conversation_ttl_seconds
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """SELECT role, content, created_at
                   FROM conversation_turns
                   WHERE conversation_id = ? AND created_at > ?
                   ORDER BY created_at ASC""",
                (conversation_id, cutoff),
            ).fetchall()
        return [
            Turn(role=r["role"], content=r["content"], timestamp=r["created_at"])
            for r in rows
        ]

    def clear(self, conversation_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?",
                (conversation_id,),
            )

    def purge_expired(self) -> int:
        """Delete all turns older than TTL. Returns count removed."""
        cutoff = time.time() - settings.conversation_ttl_seconds
        with self._lock, self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_turns WHERE created_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
        if deleted:
            logger.info("Purged %d expired conversation turns", deleted)
        return deleted

    def list_conversations(self, limit: int = 50) -> list:
        """
        Return summary of recent conversations (newest first).
        Each entry: conversation_id, turn_count, first_user_message, last_active.
        """
        cutoff = time.time() - settings.conversation_ttl_seconds
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    conversation_id,
                    COUNT(*) AS turn_count,
                    MAX(created_at) AS last_active,
                    MIN(CASE WHEN role='user' THEN content END) AS first_message
                FROM conversation_turns
                WHERE created_at > ?
                GROUP BY conversation_id
                ORDER BY last_active DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        return [
            {
                "conversation_id": r["conversation_id"],
                "turn_count": r["turn_count"],
                "last_active": r["last_active"],
                "preview": (r["first_message"] or "")[:120],
            }
            for r in rows
        ]

    def get_full_history(self, conversation_id: str) -> list:
        """Return all turns for a conversation regardless of TTL (for history view)."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """SELECT role, content, created_at
                   FROM conversation_turns
                   WHERE conversation_id = ?
                   ORDER BY created_at ASC""",
                (conversation_id,),
            ).fetchall()
        return [
            {"role": r["role"], "content": r["content"], "timestamp": r["created_at"]}
            for r in rows
        ]


conversation_store = ConversationStore()
