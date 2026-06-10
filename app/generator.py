"""
Answer generator with configurable backend and multi-turn conversation support.

Backends:
  auto      — Try OpenAI → Ollama → flan-t5-base in order
  openai    — Require OpenAI API key
  ollama    — Require Ollama; return error message if unreachable
  flan-t5   — Always use flan-t5-base (no Ollama call)
  extractive — Synthesize from top chunks via keyword overlap; zero model calls

The prompt enforces grounded answers:
  - Answer ONLY from provided context
  - Cite every fact as [filename:pageN]
  - Respond "I don't know" when context is insufficient

Conversation history is injected so the model can answer follow-up questions
without repeating context from prior turns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Generator, List, Optional

import requests

from app.config import settings
from app.logger import logger
from app.schemas import Chunk

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer questions using ONLY the context documents provided. "
    "For every fact you state, add a citation in the format [filename:pageN] immediately after it. "
    'If the context does not contain enough information, respond exactly with: '
    '"I don\'t know based on the provided documents." '
    "Never fabricate information. Be concise and accurate."
)

# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def _build_openai_messages(query: str, chunks: List[Chunk], history: list = None) -> list:
    context_text = _build_context(chunks)
    conv_block = _build_conversation_block(history or [])
    user_content = (
        f"Context documents:\n\n{context_text}\n\n"
        f"{conv_block}"
        f"Question: {query}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _try_openai(query: str, chunks: List[Chunk], history: list = None) -> Optional[str]:
    if not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed. Run: pip install openai")
        return None

    client = OpenAI(api_key=settings.openai_api_key)
    messages = _build_openai_messages(query, chunks, history)

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            max_tokens=settings.openai_max_tokens,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception as exc:
        logger.warning("OpenAI API call failed: %s", exc)
        return None


def stream_openai(query: str, chunks: List[Chunk], history: list = None) -> Generator[str, None, None]:
    """Stream tokens from OpenAI chat completions API."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai not installed")

    client = OpenAI(api_key=settings.openai_api_key)
    messages = _build_openai_messages(query, chunks, history)

    stream = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        max_tokens=settings.openai_max_tokens,
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

_PROMPT_TEMPLATE = """\
You are a precise question-answering assistant. Answer the question using ONLY the context below.
For every fact you state, add a citation in the format [filename:pageN] immediately after it.
If the context does not contain enough information to answer, respond exactly with:
"I don't know based on the provided documents."

Context:
{context}
{conversation_block}
Question: {question}

Answer:"""

_IDK = "I don't know based on the provided documents."




@dataclass
class GeneratorResult:
    answer: str
    backend: str  # "ollama" | "flan-t5" | "extractive" | "fallback" | "ollama-error"


def _build_context(chunks: List[Chunk]) -> str:
    parts = []
    for chunk in chunks:
        label = f"[{chunk.filename}:page{chunk.page}]"
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _build_conversation_block(history: list) -> str:
    if not history:
        return ""
    lines = ["[Conversation history]"]
    for turn in history:
        prefix = "User" if turn.role == "user" else "Assistant"
        # Truncate long history turns to keep prompt manageable
        lines.append(f"{prefix}: {turn.content[:400]}")
    return "\n".join(lines) + "\n\n"


def _build_prompt(query: str, chunks: List[Chunk], history: list = None, max_chars: int = 3800) -> str:
    context = _build_context(chunks)
    conv_block = _build_conversation_block(history or [])
    prompt = _PROMPT_TEMPLATE.format(
        context=context,
        conversation_block=conv_block,
        question=query,
    )
    return prompt[:max_chars]


# flan-t5-base tokenizer max is 512 tokens (~1800 chars at ~3.5 chars/token).
# Using 4000 chars caused silent truncation of most RAG context.
_FLAN_MAX_CHARS = 1800


# ---------------------------------------------------------------------------
# Extractive mode — zero model calls
# ---------------------------------------------------------------------------

def _extractive_answer(query: str, chunks: List[Chunk]) -> str:
    if not chunks:
        return _IDK

    query_tokens = {t.lower() for t in re.findall(r"\b\w+\b", query) if len(t) > 2}
    selected: List[str] = []
    seen_sentences: set = set()

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
        best_sentence, best_overlap = "", -1

        for sent in sentences:
            s = sent.strip()
            if not s or s in seen_sentences:
                continue
            tokens = {t.lower() for t in re.findall(r"\b\w+\b", s) if len(t) > 2}
            overlap = len(query_tokens & tokens)
            if overlap > best_overlap:
                best_overlap, best_sentence = overlap, s

        if best_sentence:
            citation = f"[{chunk.filename}:page{chunk.page}]"
            selected.append(f"{best_sentence} {citation}")
            seen_sentences.add(best_sentence)

    return " ".join(selected) if selected else _IDK


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _try_ollama(prompt: str) -> Optional[str]:
    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    try:
        resp = requests.post(url, json=payload, timeout=settings.ollama_timeout_seconds)
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except requests.exceptions.ConnectionError:
        logger.info("Ollama not reachable at %s — falling back.", settings.ollama_url)
        return None
    except Exception as exc:
        logger.warning("Ollama request failed: %s — falling back.", exc)
        return None


def stream_ollama(prompt: str) -> Generator[str, None, None]:
    """
    Stream tokens from Ollama one chunk at a time.
    Yields individual token strings. Raises on connection failure so caller
    can fall back to non-streaming path.
    """
    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    with requests.post(url, json=payload, stream=True,
                       timeout=settings.ollama_timeout_seconds) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done", False):
                break


def stream_answer(
    query: str,
    chunks: List[Chunk],
    history: list = None,
) -> Generator[str, None, None]:
    """
    Yield answer tokens using the best available backend.
    Claude and Ollama stream natively. Other backends generate the full answer
    then yield word-by-word so the SSE client still receives incremental updates.
    """
    mode = settings.generation_mode.lower()

    if mode == "extractive":
        answer = _extractive_answer(query, chunks)
        for word in answer.split():
            yield word + " "
        return

    # OpenAI streaming
    if mode in ("openai", "auto") and settings.openai_api_key:
        try:
            yielded_any = False
            for token in stream_openai(query, chunks, history):
                yield token
                yielded_any = True
            if yielded_any:
                return
        except Exception as exc:
            logger.warning("OpenAI stream failed (%s); trying Ollama.", exc)
        if mode == "openai":
            yield "OpenAI API is configured as the required backend but failed."
            return

    if mode in ("ollama", "auto"):
        prompt = _build_prompt(query, chunks, history, max_chars=3800)
        try:
            yielded_any = False
            for token in stream_ollama(prompt):
                yield token
                yielded_any = True
            if yielded_any:
                return
        except Exception as exc:
            logger.warning("Ollama stream failed (%s); falling back to flan-t5.", exc)
        if mode == "ollama":
            yield "Ollama is configured as the required backend but is unreachable."
            return

    # flan-t5 fallback (or explicit flan-t5 mode): generate full answer, yield word-by-word
    flan_prompt = _build_prompt(query, chunks, history, max_chars=_FLAN_MAX_CHARS)
    answer = _try_flan_t5(flan_prompt)
    for word in answer.split():
        yield word + " "


# ---------------------------------------------------------------------------
# flan-t5-base
# ---------------------------------------------------------------------------

_flan_model = None
_flan_tokenizer = None


def _get_flan_model():
    global _flan_model, _flan_tokenizer
    if _flan_model is not None:
        return _flan_model, _flan_tokenizer
    logger.info("Loading google/flan-t5-base (~250 MB on first call)…")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers required: pip install transformers torch") from exc

    _flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    _flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    _flan_model.eval()
    logger.info("flan-t5-base loaded.")
    return _flan_model, _flan_tokenizer


def _try_flan_t5(prompt: str) -> str:
    import torch
    model, tokenizer = _get_flan_model()
    if len(prompt) > _FLAN_MAX_CHARS:
        prompt = prompt[:_FLAN_MAX_CHARS]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        # 256 new tokens is ample for grounded RAG answers and ~halves CPU
        # generation latency vs 512 on the flan-t5 fallback path.
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Backend status probe
# ---------------------------------------------------------------------------

def get_backend_status() -> dict:
    ollama_ok = False
    try:
        resp = requests.get(settings.ollama_url.rstrip("/"), timeout=2.0)
        ollama_ok = resp.status_code < 500
    except Exception:
        pass
    return {
        "ollama": "reachable" if ollama_ok else "unreachable",
        "flan_t5": "loaded" if _flan_model is not None else "not_loaded",
        "openai": "configured" if settings.openai_api_key else "not_configured",
        "active_mode": settings.generation_mode,
    }


# ---------------------------------------------------------------------------
# Approximate token counting
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """~1.3 tokens per word for English text."""
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class GeneratorChain:
    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        history: list = None,
    ) -> GeneratorResult:
        if not chunks:
            return GeneratorResult(answer=_IDK, backend="fallback")

        mode = settings.generation_mode.lower()

        if mode == "extractive":
            return GeneratorResult(answer=_extractive_answer(query, chunks), backend="extractive")

        if mode == "flan-t5":
            prompt = _build_prompt(query, chunks, history, max_chars=_FLAN_MAX_CHARS)
            return GeneratorResult(answer=_try_flan_t5(prompt), backend="flan-t5")

        if mode == "openai":
            answer = _try_openai(query, chunks, history)
            if answer:
                return GeneratorResult(answer=answer, backend="openai")
            return GeneratorResult(
                answer="OpenAI API is configured as the required backend but failed.",
                backend="openai-error",
            )

        if mode == "ollama":
            prompt = _build_prompt(query, chunks, history)
            answer = _try_ollama(prompt)
            if answer:
                return GeneratorResult(answer=answer, backend="ollama")
            return GeneratorResult(
                answer="Ollama is configured as the required backend but is unreachable.",
                backend="ollama-error",
            )

        # auto — OpenAI → Ollama → flan-t5 fallback
        if settings.openai_api_key:
            answer = _try_openai(query, chunks, history)
            if answer:
                return GeneratorResult(answer=answer, backend="openai")

        ollama_prompt = _build_prompt(query, chunks, history, max_chars=3800)
        answer = _try_ollama(ollama_prompt)
        if answer:
            return GeneratorResult(answer=answer, backend="ollama")
        flan_prompt = _build_prompt(query, chunks, history, max_chars=_FLAN_MAX_CHARS)
        return GeneratorResult(answer=_try_flan_t5(flan_prompt), backend="flan-t5")
