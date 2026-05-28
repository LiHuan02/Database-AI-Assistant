"""Chroma vector store sync and search — 每个知识库独立存储。"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from core import ingest as ingest_module
from core.embeddings import get_embeddings
from db.chroma_client import ChromaClient

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def sync_vector_store(
    library_name: str,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """增量同步：只处理新增、修改和删除的文档。"""
    docs_dir = Path(data_root) / "libraries" / library_name / "docs"
    if not docs_dir.exists():
        ChromaClient(library_name, data_root=data_root).reset_collection()
        return {"synced": False, "inserted": 0, "deleted": 0}

    manifest = _docs_manifest(library_name, data_root)
    manifest_path = _manifest_path(library_name, data_root)
    old_manifest = _read_manifest(manifest_path)

    if manifest == old_manifest and (not manifest or _collection_has_data(library_name, data_root)):
        return {"synced": False, "inserted": 0, "deleted": 0}

    if not manifest and old_manifest:
        return _full_rebuild(library_name, data_root, api_key, embedding_provider,
                             embedding_model, embedding_base_url, batch_size, manifest_path, manifest)

    if not old_manifest:
        return _full_rebuild(library_name, data_root, api_key, embedding_provider,
                             embedding_model, embedding_base_url, batch_size, manifest_path, manifest)

    old_docs = set(old_manifest.keys())
    new_docs = set(manifest.keys())

    added = new_docs - old_docs
    removed = old_docs - new_docs
    possibly_modified = {
        name for name in (new_docs & old_docs)
        if old_manifest[name] != manifest[name]
    }

    if not added and not removed and not possibly_modified:
        return {"synced": False, "inserted": 0, "deleted": 0}

    if removed or possibly_modified:
        if len(removed) + len(possibly_modified) > len(old_docs) * 0.6:
            return _full_rebuild(library_name, data_root, api_key, embedding_provider,
                                 embedding_model, embedding_base_url, batch_size,
                                 manifest_path, manifest)

    total_inserted = 0
    total_deleted = 0
    embedding_params = {
        "api_key": api_key,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "batch_size": batch_size,
    }

    for doc_name in removed:
        ingest_module.remove_document_from_store(doc_name, library_name, data_root=data_root)
        total_deleted += 1

    for doc_name in possibly_modified:
        file_path = str(docs_dir / doc_name)
        ingest_module.remove_document_from_store(doc_name, library_name, data_root=data_root)
        result = ingest_module.add_document_to_store(
            file_path, library_name, data_root=data_root, **embedding_params,
        )
        total_inserted += result.get("inserted", 0)
        total_deleted += 1

    for doc_name in added:
        file_path = str(docs_dir / doc_name)
        result = ingest_module.add_document_to_store(
            file_path, library_name, data_root=data_root, **embedding_params,
        )
        total_inserted += result.get("inserted", 0)

    _write_manifest(manifest_path, manifest)
    log.info(
        "Incremental sync '%s': +%d chunks, -%d docs",
        library_name, total_inserted, total_deleted,
    )
    return {"synced": True, "inserted": total_inserted, "deleted": total_deleted}


def _full_rebuild(library_name, data_root, api_key, embedding_provider,
                  embedding_model, embedding_base_url, batch_size,
                  manifest_path, manifest):
    result = build_vector_store(
        library_name,
        data_root=data_root,
        api_key=api_key,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        batch_size=batch_size,
    )
    _write_manifest(manifest_path, manifest)
    result["synced"] = True
    return result


def build_vector_store(
    library_name: str,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """Full rebuild from all docs in directory."""
    docs_dir = Path(data_root) / "libraries" / library_name / "docs"
    if not docs_dir.exists():
        ChromaClient(library_name, data_root=data_root).reset_collection()
        return {"inserted": 0}

    file_paths = [
        str(path)
        for path in sorted(docs_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    ChromaClient(library_name, data_root=data_root).reset_collection()
    if not file_paths:
        return {"inserted": 0}

    try:
        return ingest_module.ingest_documents(
            file_paths,
            library_name,
            data_root=data_root,
            api_key=api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            batch_size=batch_size,
        )
    except Exception:
        log.exception("build_vector_store failed: %s", library_name)
        return {"inserted": 0, "error": "build_vector_store failed"}


def search_vector_store(
    library_name: str,
    query: str,
    k: int = 5,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> List[Dict]:
    """Search Chroma vector store."""
    if not query.strip():
        return []
    query_embedding = get_embeddings(
        [query],
        api_key=api_key,
        provider=embedding_provider,
        model=embedding_model,
        base_url=embedding_base_url,
        batch_size=batch_size,
    )[0]
    collection = ChromaClient(library_name, data_root=data_root).get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]
    return [
        {
            "id": ids[i] if i < len(ids) else None,
            "text": text,
            "meta": metadatas[i] if i < len(metadatas) else {},
            "score": distances[i] if i < len(distances) else None,
        }
        for i, text in enumerate(docs)
    ]


def delete_vector_store(library_name: str, data_root: str = "data"):
    return ChromaClient(library_name, data_root=data_root).delete_collection()


# ── manifest helpers ──────────────────────────────────────────

def _docs_manifest(library_name: str, data_root: str):
    docs_dir = Path(data_root) / "libraries" / library_name / "docs"
    if not docs_dir.exists():
        return {}
    manifest = {}
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            stat = path.stat()
            manifest[path.name] = {"size": stat.st_size, "mtime": stat.st_mtime}
    return manifest


def _manifest_path(library_name: str, data_root: str) -> Path:
    return Path(data_root) / "libraries" / library_name / "manifest.json"


def _read_manifest(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def _write_manifest(path: Path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _collection_has_data(library_name: str, data_root: str) -> bool:
    try:
        return ChromaClient(library_name, data_root=data_root).get_collection().count() > 0
    except Exception:
        return False
