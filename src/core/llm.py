"""
LLM call wrapper with streaming and retry support.
"""

import logging
import os
import time
from typing import Iterator, Optional

from openai import OpenAI

log = logging.getLogger(__name__)


def _build_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    resolved_key = api_key or os.getenv("LLM_API_KEY", "")
    if not resolved_key:
        raise ValueError("LLM API key not set. Use LLM_API_KEY env var or API setting.")
    resolved_base = base_url or os.getenv("LLM_BASE_URL")
    return OpenAI(api_key=resolved_key, base_url=resolved_base)


def stream_chat(
    messages: list,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> Iterator[str]:
    client = _build_client(api_key=api_key, base_url=base_url)
    resolved_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = (2 ** attempt) * 1.0
                log.warning(
                    "LLM call attempt %d/%d failed (%s), retrying in %.1fs…",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
    raise last_exc


def chat_complete(
    messages: list,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    client = _build_client(api_key=api_key, base_url=base_url)
    resolved_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = (2 ** attempt) * 1.0
                log.warning(
                    "LLM call attempt %d/%d failed (%s), retrying in %.1fs…",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
    raise last_exc
