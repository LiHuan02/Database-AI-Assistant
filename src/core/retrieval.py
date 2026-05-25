"""检索入口：先同步文档到 Chroma，再只查询 Chroma 向量库。"""

from typing import Dict, List, Optional

from db.vector_store import search_vector_store, sync_vector_store


def retrieve_relevant(
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
    """检索相关上下文；query 应该是 LLM 生成的摘要。"""
    sync_vector_store(
        library_name,
        data_root=data_root,
        api_key=api_key,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        batch_size=batch_size,
    )
    return search_vector_store(
        library_name,
        query,
        k=k,
        data_root=data_root,
        api_key=api_key,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        batch_size=batch_size,
    )
