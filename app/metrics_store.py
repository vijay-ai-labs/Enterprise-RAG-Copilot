"""In-process query metrics accumulator. Resets on server restart."""

import threading

_lock = threading.Lock()
_total_queries: int = 0
_total_time_ms: float = 0.0


def record_query(response_time_ms: float) -> None:
    global _total_queries, _total_time_ms
    with _lock:
        _total_queries += 1
        _total_time_ms += response_time_ms


def get_stats() -> dict:
    with _lock:
        if _total_queries == 0:
            return {"total_queries": 0, "avg_response_time_ms": 0.0}
        return {
            "total_queries": _total_queries,
            "avg_response_time_ms": round(_total_time_ms / _total_queries, 2),
        }
