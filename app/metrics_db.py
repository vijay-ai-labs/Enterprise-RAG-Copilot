"""
SQLite-backed persistent metrics store.
Replaces the in-memory metrics_store for durable per-query telemetry.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict

class MetricsDB:
    """Persist per-query metrics to SQLite so they survive restarts."""

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
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id          TEXT NOT NULL,
                    trace_id          TEXT,
                    latency_ms        REAL NOT NULL,
                    embedding_ms      REAL,
                    retrieval_ms      REAL,
                    generation_ms     REAL,
                    evaluation_ms     REAL,
                    generator_backend TEXT,
                    chunks_retrieved  INTEGER,
                    groundedness      REAL,
                    input_tokens      INTEGER,
                    output_tokens     INTEGER,
                    created_at        REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON query_metrics(created_at)"
            )

    def record(
        self,
        *,
        query_id: str,
        trace_id: str = "",
        latency_ms: float,
        embedding_ms: float = 0,
        retrieval_ms: float = 0,
        generation_ms: float = 0,
        evaluation_ms: float = 0,
        generator_backend: str = "",
        chunks_retrieved: int = 0,
        groundedness: float = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO query_metrics
                   (query_id, trace_id, latency_ms, embedding_ms, retrieval_ms,
                    generation_ms, evaluation_ms, generator_backend, chunks_retrieved,
                    groundedness, input_tokens, output_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query_id, trace_id, latency_ms, embedding_ms, retrieval_ms,
                    generation_ms, evaluation_ms, generator_backend, chunks_retrieved,
                    groundedness, input_tokens, output_tokens, time.time(),
                ),
            )

    def get_summary(self) -> Dict:
        with self._lock, self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)             AS total_queries,
                    AVG(latency_ms)      AS avg_latency_ms,
                    AVG(groundedness)    AS avg_groundedness,
                    AVG(chunks_retrieved) AS avg_chunks
                FROM query_metrics
            """).fetchone()
        return {
            "total_queries": row["total_queries"] or 0,
            "avg_response_time_ms": round(row["avg_latency_ms"] or 0.0, 2),
            "avg_groundedness": round(row["avg_groundedness"] or 0.0, 4),
            "avg_chunks_retrieved": round(row["avg_chunks"] or 0.0, 2),
        }

    def get_timeseries(self, hours: int = 24, bucket_minutes: int = 60) -> list:
        """
        Returns hourly buckets of query count, avg latency, avg groundedness.
        bucket_minutes: width of each bucket (default 60 = hourly).
        hours: how far back to look (default 24).
        """
        bucket_secs = bucket_minutes * 60
        cutoff = time.time() - hours * 3600
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    CAST((created_at - ?) / ? AS INTEGER) AS bucket,
                    COUNT(*) AS queries,
                    AVG(latency_ms) AS avg_latency_ms,
                    AVG(groundedness) AS avg_groundedness,
                    SUM(input_tokens + output_tokens) AS total_tokens
                FROM query_metrics
                WHERE created_at >= ?
                GROUP BY bucket
                ORDER BY bucket ASC
                """,
                (cutoff, bucket_secs, cutoff),
            ).fetchall()

        result = []
        for row in rows:
            bucket_start = cutoff + row["bucket"] * bucket_secs
            result.append({
                "timestamp": round(bucket_start),
                "queries": row["queries"],
                "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
                "avg_groundedness": round(row["avg_groundedness"] or 0, 4),
                "total_tokens": row["total_tokens"] or 0,
            })
        return result

    def get_latency_percentiles(self) -> Dict:
        """Returns P50, P95, P99 latency from all stored queries."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT latency_ms FROM query_metrics ORDER BY latency_ms"
            ).fetchall()
        if not rows:
            return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "count": 0}
        vals = [r["latency_ms"] for r in rows]
        n = len(vals)

        def pct(p: float) -> float:
            idx = min(int(n * p / 100), n - 1)
            return round(vals[idx], 2)

        return {"p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99), "count": n}
