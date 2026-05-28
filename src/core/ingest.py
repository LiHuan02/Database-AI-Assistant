"""
Document ingestion: load, split, embed, and write to Chroma.
Supports batch and single-document (incremental) modes.
"""

import logging
from pathlib import Path
from typing import List, Optional

from core.embeddings import get_embeddings
from db.chroma_client import ChromaClient
from loaders.docx_loader import load_docx
from loaders.md_loader import load_md
from loaders.pdf_loader import load_pdf
from loaders.txt_loader import load_txt
from utils.text_splitter import split_text

log = logging.getLogger(__name__)

LOADER_MAP = {
    ".txt": load_txt,
    ".md": load_md,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def _load_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    loader = LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(file_path)


def _build_chunks(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    api_key, provider, model, base_url, batch_size,
):
    embeddings = get_embeddings(
        texts,
        api_key=api_key,
        provider=provider,
        model=model,
        base_url=base_url,
        batch_size=batch_size,
    )
    return {"ids": ids, "embeddings": embeddings, "metadatas": metadatas, "documents": texts}


# ── batch ingest (used for full rebuild) ──────────────────────

def ingest_documents(
    file_paths: List[str],
    library_name: str,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    all_ids = []
    all_embeddings = []
    all_metadatas = []
    all_documents = []

    for file_path in file_paths:
        doc_name = Path(file_path).name
        raw_text = _load_file(file_path)
        chunks = split_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{library_name}:{doc_name}:{i}")
            all_documents.append(chunk)
            all_metadatas.append({
                "library": library_name,
                "doc_name": doc_name,
                "chunk_index": i,
                "source": file_path,
            })

    if not all_ids:
        return {"inserted": 0}

    total = 0
    collection = ChromaClient(library_name, data_root=data_root).get_collection()
    _batch = batch_size or 32
    for offset in range(0, len(all_ids), _batch):
        slice_ids = all_ids[offset:offset + _batch]
        slice_docs = all_documents[offset:offset + _batch]
        slice_meta = all_metadatas[offset:offset + _batch]
        chunk_data = _build_chunks(
            slice_docs, slice_meta, slice_ids,
            api_key, embedding_provider, embedding_model, embedding_base_url, batch_size,
        )
        collection.add(**chunk_data)
        total += len(slice_ids)

    return {"inserted": total}


# ── incremental operations ────────────────────────────────────

def add_document_to_store(
    file_path: str,
    library_name: str,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    doc_name = Path(file_path).name
    raw_text = _load_file(file_path)
    chunks = split_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    ids = []
    docs = []
    metas = []
    for i, chunk in enumerate(chunks):
        ids.append(f"{library_name}:{doc_name}:{i}")
        docs.append(chunk)
        metas.append({
            "library": library_name,
            "doc_name": doc_name,
            "chunk_index": i,
            "source": file_path,
        })

    if not ids:
        return {"inserted": 0}

    collection = ChromaClient(library_name, data_root=data_root).get_collection()
    total = 0
    _batch = batch_size or 32
    for offset in range(0, len(ids), _batch):
        slice_ids = ids[offset:offset + _batch]
        slice_docs = docs[offset:offset + _batch]
        slice_meta = metas[offset:offset + _batch]
        chunk_data = _build_chunks(
            slice_docs, slice_meta, slice_ids,
            api_key, embedding_provider, embedding_model, embedding_base_url, batch_size,
        )
        collection.add(**chunk_data)
        total += len(slice_ids)

    log.info("Added document '%s' to library '%s': %d chunks", doc_name, library_name, total)
    return {"inserted": total}


def remove_document_from_store(
    doc_name: str,
    library_name: str,
    data_root: str = "data",
):
    collection = ChromaClient(library_name, data_root=data_root).get_collection()
    try:
        collection.delete(where={"doc_name": doc_name})
        log.info("Removed document '%s' from library '%s'", doc_name, library_name)
    except Exception:
        log.exception("Failed to delete chunks for document '%s'", doc_name)


def replace_document_in_store(
    file_path: str,
    library_name: str,
    data_root: str = "data",
    **kwargs,
):
    doc_name = Path(file_path).name
    remove_document_from_store(doc_name, library_name, data_root=data_root)
    return add_document_to_store(file_path, library_name, data_root=data_root, **kwargs)
