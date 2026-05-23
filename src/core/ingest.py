"""
Document ingestion: loaders -> split -> embed -> upsert to Chroma collection
"""
from pathlib import Path
from typing import List
import os
import uuid

from utils.text_splitter import split_text
from core.embeddings import get_embeddings
from db.chroma_client import ChromaClient


def _load_file(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".md":
        from loaders.md_loader import load_md

        return load_md(path)
    if ext == ".pdf":
        from loaders.pdf_loader import load_pdf

        return load_pdf(path)
    if ext in (".docx",):
        from loaders.docx_loader import load_docx

        return load_docx(path)
    if ext in (".txt",):
        from loaders.txt_loader import load_txt

        return load_txt(path)
    # fallback: try to read as text
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def ingest_documents(file_paths: List[str], library_name: str, data_root: str = "data", api_key: str = None, embedding_provider: str = "openai", embedding_model: str = None):
    """Ingest given file paths into the per-library vector store.

    Steps:
    - load file
    - split into chunks
    - compute embeddings
    - upsert into chroma collection (per-library)
    If chromadb is not available, fallback to storing chunk texts under data/chroma/<library>/index.json
    """
    data_root = Path(data_root)
    chroma = ChromaClient(data_root=str(data_root))
    collection = chroma.get_collection(library_name)

    all_ids = []
    all_texts = []
    all_metadatas = []

    for path in file_paths:
        text = _load_file(path)
        if not text:
            continue
        chunks = split_text(text)
        for i, c in enumerate(chunks):
            uid = str(uuid.uuid4())
            meta = {"source": os.path.basename(path), "chunk_index": i}
            all_ids.append(uid)
            all_texts.append(c)
            all_metadatas.append(meta)

    if not all_texts:
        return {"inserted": 0}

    embeddings = get_embeddings(all_texts, api_key=api_key, provider=embedding_provider, model=embedding_model)

    if collection is None:
        # fallback: persist index file
        idx_dir = data_root / "chroma" / library_name
        idx_dir.mkdir(parents=True, exist_ok=True)
        import json

        idx_path = idx_dir / "index.json"
        existing = {"docs": []}
        if idx_path.exists():
            try:
                existing = json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {"docs": []}
        for i in range(len(all_texts)):
            existing["docs"].append({"id": all_ids[i], "text": all_texts[i], "meta": all_metadatas[i], "embedding": embeddings[i]})
        idx_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"inserted": len(all_texts), "fallback": True}

    # try to add to chroma collection
    try:
        # chroma collection API may accept add/upsert
        if hasattr(collection, "add"):
            collection.add(ids=all_ids, documents=all_texts, metadatas=all_metadatas, embeddings=embeddings)
        elif hasattr(collection, "upsert"):
            collection.upsert(ids=all_ids, documents=all_texts, metadatas=all_metadatas, embeddings=embeddings)
        else:
            # try create_documents
            collection.add(ids=all_ids, documents=all_texts, metadatas=all_metadatas, embeddings=embeddings)
        return {"inserted": len(all_texts), "fallback": False}
    except Exception:
        # last-resort fallback to index file
        idx_dir = data_root / "chroma" / library_name
        idx_dir.mkdir(parents=True, exist_ok=True)
        import json

        idx_path = idx_dir / "index.json"
        existing = {"docs": []}
        if idx_path.exists():
            try:
                existing = json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {"docs": []}
        for i in range(len(all_texts)):
            existing["docs"].append({"id": all_ids[i], "text": all_texts[i], "meta": all_metadatas[i], "embedding": embeddings[i]})
        idx_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"inserted": len(all_texts), "fallback": True}

# Note: build_vector_store moved to `db.vector_store.build_vector_store` to
# separate concerns: core handles ingestion and embeddings, db handles store lifecycle.
