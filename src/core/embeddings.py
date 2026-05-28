"""
Embedding model caller with retry support.
"""

import logging
import time
from typing import List, Optional

from openai import OpenAI

log = logging.getLogger(__name__)


def get_embeddings(
    texts: List[str],
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_retries: int = 3,
) -> List[List[float]]:
    if not texts:
        return []

    if provider == "hash":
        return _hash_embeddings(texts)

    import os
    resolved_key = api_key or os.getenv("EMBEDDING_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Embedding API key not set. Use EMBEDDING_API_KEY env var or API setting."
        )
    resolved_model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    resolved_base = base_url or os.getenv("EMBEDDING_BASE_URL")

    client = OpenAI(api_key=resolved_key, base_url=resolved_base)

    try:
        batch = batch_size or 25
        all_embeddings = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            all_embeddings.extend(
                _embed_with_retry(
                    client, resolved_model, chunk, max_retries=max_retries,
                )
            )
        return all_embeddings
    except Exception as exc:
        raise RuntimeError(f"Embedding failed: {exc}") from exc


def _embed_with_retry(
    client, model: str, texts: List[str], max_retries: int = 3,
) -> List[List[float]]:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = (2 ** attempt) * 0.5
                log.warning(
                    "Embedding attempt %d/%d failed (%s), retrying in %.1fs…",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
    raise last_exc


def _hash_embeddings(texts: List[str], dim: int = 1024) -> List[List[float]]:
    import hashlib
    embeddings = []
    for text in texts:
        seed = abs(int(hashlib.sha256(text.encode()).hexdigest(), 16))
        emb = []
        for i in range(dim):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            emb.append((seed % 1000) / 500.0 - 1.0)
        embeddings.append(emb)
    return embeddings
