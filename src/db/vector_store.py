"""Vector store management: build and delete per-library vector stores.

This module centralizes vector store creation/removal. It delegates embedding
computation to `core.embeddings.get_embeddings` via `core.ingest.ingest_documents`.
"""
from pathlib import Path
from typing import Optional
import shutil
import logging

from core import ingest as ingest_module
from db.chroma_client import ChromaClient

log = logging.getLogger(__name__)


def build_vector_store(library_name: str, data_root: str = "data", api_key: Optional[str] = None, embedding_provider: str = "openai", embedding_model: Optional[str] = None):
    """Scan `data/libraries/<library>/docs` and ingest documents into chroma collection.

    Returns the same dict as `ingest_documents`.
    """
    data_root_path = Path(data_root)
    docs_dir = data_root_path / "libraries" / library_name / "docs"
    if not docs_dir.exists():
        return {"inserted": 0, "error": "no docs directory"}
    file_paths = [str(p) for p in docs_dir.iterdir() if p.is_file()]

    # delegate to core.ingest which handles embeddings/upsert
    try:
        res = ingest_module.ingest_documents(file_paths, library_name, data_root=data_root, api_key=api_key, embedding_provider=embedding_provider, embedding_model=embedding_model)
        return res
    except Exception:
        log.exception("build_vector_store failed for %s", library_name)
        return {"inserted": 0, "error": "exception during build"}


def delete_vector_store(library_name: str, data_root: str = "data"):
    """Delete persisted chroma directory for the library (cleanup)."""
    data_root_path = Path(data_root)
    persist_dir = data_root_path / "chroma" / library_name
    try:
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        # also attempt to drop collection via chroma client if available
        cc = ChromaClient(data_root=str(data_root))
        client = cc._client_for(library_name)
        if client is not None:
            try:
                col = client.get_collection(library_name)
                # some chroma clients support delete_collection
                if hasattr(client, 'delete_collection'):
                    client.delete_collection(library_name)
            except Exception:
                pass
        return True
    except Exception:
        log.exception("delete_vector_store failed for %s", library_name)
        return False
