"""
Embedding wrapper: support OpenAI and local embeddings
"""
from typing import List, Optional
import hashlib


DEFAULT_DIM = 1536


def _hash_vector(text: str, dim: int = 1536) -> List[float]:
    h = hashlib.sha256(text.encode('utf-8')).digest()
    out = []
    for i in range(dim):
        out.append((h[i % len(h)] / 255.0) * (1.0 - (i / dim)))
    return out


def get_embeddings(texts: List[str], api_key: Optional[str] = None, provider: str = "openai", model: Optional[str] = None) -> List[List[float]]:
    """Return embeddings for a list of texts.

    provider: 'openai' | 'local' | 'hash'
    model: provider-specific model name
    """
    if not texts:
        return []

    # OpenAI provider
    if provider == "openai":
        try:
            import openai

            if api_key:
                openai.api_key = api_key
            use_model = model or "text-embedding-3-small"
            resp = openai.Embedding.create(input=texts, model=use_model)
            return [d["embedding"] for d in resp["data"]]
        except Exception:
            pass

    # Local provider using sentence-transformers if available
    if provider == "local":
        try:
            from sentence_transformers import SentenceTransformer

            use_model = model or "all-MiniLM-L6-v2"
            m = SentenceTransformer(use_model)
            embs = m.encode(texts, show_progress_bar=False)
            return [list(map(float, e)) for e in embs]
        except Exception:
            pass

    # fallback deterministic hash-based vectors
    dim = DEFAULT_DIM
    return [_hash_vector(t, dim=dim) for t in texts]
