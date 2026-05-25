"""文档入库：读取文件、切分文本、生成向量、写入 Chroma。"""

from pathlib import Path
from typing import List, Optional
import os
import uuid

from core.embeddings import get_embeddings
from db.chroma_client import ChromaClient
from utils.text_splitter import split_text


def load_file(path: str) -> str:
    """根据扩展名读取文档内容。"""
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext == ".md":
        from loaders.md_loader import load_md

        return load_md(path)
    if ext == ".pdf":
        from loaders.pdf_loader import load_pdf

        return load_pdf(path)
    if ext == ".docx":
        from loaders.docx_loader import load_docx

        return load_docx(path)
    if ext == ".txt":
        from loaders.txt_loader import load_txt

        return load_txt(path)
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def ingest_documents(
    file_paths: List[str],
    library_name: str,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """把文档块写入指定知识库的 Chroma collection。"""
    ids, texts, metadatas = [], [], []
    for path in file_paths:
        content = load_file(path)
        if not content:
            continue
        source = os.path.basename(path)
        for index, chunk in enumerate(split_text(content)):
            chunk = chunk.strip()
            if not chunk:
                continue
            ids.append(f"{source}:{index}:{uuid.uuid4()}")
            texts.append(chunk)
            metadatas.append({"source": source, "path": str(Path(path)), "chunk_index": index})

    collection = ChromaClient(data_root=data_root).get_collection(library_name)
    if not texts:
        return {"inserted": 0}

    embeddings = get_embeddings(
        texts,
        api_key=api_key,
        provider=embedding_provider,
        model=embedding_model,
        base_url=embedding_base_url,
        batch_size=batch_size,
    )
    for start in range(0, len(texts), 100):
        end = start + 100
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
    return {"inserted": len(texts)}


def build_vector_store(*args, **kwargs):
    """兼容旧导入路径。"""
    from db.vector_store import build_vector_store as _build_vector_store

    return _build_vector_store(*args, **kwargs)
