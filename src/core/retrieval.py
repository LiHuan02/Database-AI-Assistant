"""
Retrieval and RAG orchestration: query -> retrieve -> construct prompt
"""
from typing import List, Dict
from pathlib import Path
from core.embeddings import get_embeddings
from db.chroma_client import ChromaClient
import json


def retrieve_relevant(library_name: str, query: str, k: int = 5, data_root: str = "data", api_key: str = None, embedding_provider: str = "openai", embedding_model: str = None) -> List[Dict]:
    """Retrieve top-k relevant chunks for `query` from the specified library.

    Attempts to use chroma collection; if unavailable, falls back to simple cosine-similarity over stored embeddings in index.json.
    Returns list of dicts: {id, text, meta, score}
    """
    data_root = Path(data_root)
    chroma = ChromaClient(data_root=str(data_root))
    collection = chroma.get_collection(library_name)

    if collection is not None:
        try:
            # compute embedding for query using selected embedding provider
            q_emb = get_embeddings([query], api_key=api_key, provider=embedding_provider, model=embedding_model)[0]
            # chroma query with query_embeddings when available
            if hasattr(collection, "query"):
                try:
                    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"]) or {}
                except TypeError:
                    # some chroma versions expect query_texts
                    res = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"]) or {}
            else:
                res = {}

            docs = []
            docs_list = res.get("documents", [[]])[0]
            metas_list = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for i, d in enumerate(docs_list):
                docs.append({"id": None, "text": d, "meta": metas_list[i] if i < len(metas_list) else {}, "score": dists[i] if i < len(dists) else None})
            return docs
        except Exception:
            pass

    # fallback: look for data/chroma/<library>/index.json
    idx_path = data_root / "chroma" / library_name / "index.json"
    if not idx_path.exists():
        # fallback to scanning raw docs: return empty
        return []
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    docs = idx.get("docs", [])
    if not docs:
        return []

    # compute embedding for query and do cosine similarity
    q_emb = get_embeddings([query], api_key=api_key, provider=embedding_provider, model=embedding_model)[0]

    def cos(a, b):
        import math

        num = sum(x * y for x, y in zip(a, b))
        den1 = math.sqrt(sum(x * x for x in a))
        den2 = math.sqrt(sum(y * y for y in b))
        if den1 == 0 or den2 == 0:
            return 0.0
        return num / (den1 * den2)

    scored = []
    for d in docs:
        emb = d.get("embedding")
        if not emb:
            score = 0.0
        else:
            score = cos(q_emb, emb)
        scored.append({"id": d.get("id"), "text": d.get("text"), "meta": d.get("meta"), "score": score})

    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored[:k]
