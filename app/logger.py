import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

_log_path: Path | None = None


def _get_log_path() -> Path:
    global _log_path
    if _log_path is None:
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = log_dir / "queries.jsonl"
    return _log_path


def log_query(
    query: str,
    retrieved_chunks: int,
    response_time_ms: float,
    generator_backend: str,
    eval_result: Dict[str, Any],
    *,
    query_id: Optional[str] = None,
    timings: Optional[Dict[str, float]] = None,
    chunk_ids: Optional[List[str]] = None,
    chunk_scores: Optional[List[float]] = None,
) -> None:
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "response_time_ms": round(response_time_ms, 2),
        "generator_backend": generator_backend,
        "eval": eval_result,
        "timings": timings,
        "chunk_ids": chunk_ids,
        "chunk_scores": [round(s, 4) for s in chunk_scores] if chunk_scores else None,
    }
    # Drop None values to keep log records lean
    record = {k: v for k, v in record.items() if v is not None}
    try:
        with open(_get_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logging.warning("Failed to write query log: %s", exc)


# Standard Python logger for app-level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_copilot")
