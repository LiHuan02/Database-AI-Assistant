"""基于 Chroma 的向量库同步与检索。"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

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
    """文档目录变化时重建 Chroma collection。"""
    manifest = _docs_manifest(library_name, data_root)
    manifest_path = _manifest_path(library_name, data_root)
    old_manifest = _read_manifest(manifest_path)
    if manifest == old_manifest and (not manifest or _collection_has_data(library_name, data_root)):
        return {"synced": False, "inserted": None}

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
    """按当前 docs 目录全量重建 Chroma collection。"""
    docs_dir = Path(data_root) / "libraries" / library_name / "docs"
    if not docs_dir.exists():
        ChromaClient(data_root=data_root).reset_collection(library_name)
        return {"inserted": 0}

    file_paths = [
        str(path)
        for path in sorted(docs_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    ChromaClient(data_root=data_root).reset_collection(library_name)
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
        log.exception("构建向量库失败：%s", library_name)
        return {"inserted": 0, "error": "构建向量库失败"}


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
    """只在 Chroma 向量库中检索，不直接扫描原始文档。"""
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
    collection = ChromaClient(data_root=data_root).get_collection(library_name)
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
            "id": ids[index] if index < len(ids) else None,
            "text": text,
            "meta": metadatas[index] if index < len(metadatas) else {},
            "score": distances[index] if index < len(distances) else None,
        }
        for index, text in enumerate(docs)
    ]


def delete_vector_store(library_name: str, data_root: str = "data"):
    """删除指定知识库对应的 Chroma collection。"""
    return ChromaClient(data_root=data_root).delete_collection(library_name)


def _collection_has_data(library_name: str, data_root: str) -> bool:
    try:
        return ChromaClient(data_root=data_root).get_collection(library_name).count() > 0
    except Exception:
        return False


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
    return Path(data_root) / "chroma" / "_manifests" / f"{ChromaClient(data_root=data_root).collection_name(library_name)}.json"


def _read_manifest(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def _write_manifest(path: Path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
