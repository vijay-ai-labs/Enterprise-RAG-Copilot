"""
HyDE (Hypothetical Document Embedding) — pre-retrieval query expansion.

Generates a short hypothetical answer to the user's query, then embeds THAT
instead of the raw question. Answer-space embeddings align better with chunk
embeddings, improving recall especially for factoid and multi-hop queries.

Priority: OpenAI → Ollama → original query (fallback).
Disabled when use_hyde=False in config (default off to avoid latency cost
when no LLM backend is configured).
"""

from __future__ import annotations

import requests

from app.config import settings
from app.logger import logger

_HYDE_PROMPT = (
    "Write a short, factual passage (2-3 sentences) that would directly answer "
    "the following question. Write as if you are certain of the answer.\n\n"
    "Question: {query}\n\nPassage:"
)


def generate_hyde_query(query: str) -> str:
    """
    Return a hypothetical document for the query.
    Falls back to the original query string on any failure.
    """
    if not settings.use_hyde:
        return query

    if settings.openai_api_key:
        try:
            return _openai_hyde(query)
        except Exception as exc:
            logger.warning("HyDE via OpenAI failed: %s", exc)

    try:
        return _ollama_hyde(query)
    except Exception as exc:
        logger.warning("HyDE via Ollama failed: %s — using original query", exc)

    return query


def _openai_hyde(query: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai not installed") from exc

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        max_tokens=200,
        temperature=0.3,
    )
    text = (response.choices[0].message.content or "").strip()
    logger.info("HyDE generated %d chars via OpenAI", len(text))
    return text or query


def _ollama_hyde(query: str) -> str:
    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={
            "model": settings.ollama_model,
            "prompt": _HYDE_PROMPT.format(query=query),
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 200},
        },
        timeout=settings.ollama_timeout_seconds,
    )
    resp.raise_for_status()
    text = resp.json().get("response", "").strip()
    logger.info("HyDE generated %d chars via Ollama", len(text))
    return text or query
