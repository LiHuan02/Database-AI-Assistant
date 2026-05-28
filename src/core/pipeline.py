"""
RAG pipeline: memory → retrieval → LLM answer with streaming.
"""

from typing import Iterator, Optional

from core.llm import stream_chat
from core.memory import build_memory
from utils.config import get_library_config


def stream_lcel_pipeline(
    library_name: str,
    chat_id: str,
    user_input: str,
    chat_manager,
    *,
    data_root: str = "data",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Iterator[str]:
    lib_cfg = get_library_config(library_name, data_root=data_root)

    prompt = build_memory(
        library_name,
        chat_id,
        user_input,
        chat_manager,
        data_root=data_root,
        api_key=api_key,
        base_url=base_url,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        top_k=lib_cfg.top_k,
        enable_multi_query=True,
    )

    messages = [{"role": "user", "content": prompt}]

    yield from stream_chat(
        messages,
        api_key=api_key,
        base_url=base_url,
        provider=llm_provider,
        model=llm_model,
    )
