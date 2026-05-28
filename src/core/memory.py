"""
Memory / retrieval strategy: build search query from chat history,
optionally decompose into multiple sub-queries (multi-query retrieval).
"""

from typing import List

from core.llm import chat_complete
from db.vector_store import search_vector_store


MULTI_QUERY_PROMPT = """你是一个检索优化助手。用户提出了一个问题。
请生成 3 个不同角度的检索查询，用中文表达，以帮助从知识库中找到更多相关信息。
每个查询占一行，不要编号。

用户问题：{question}

检索查询："""


def build_memory(
    library_name: str,
    chat_id: str,
    user_input: str,
    chat_manager,
    data_root: str = "data",
    api_key: str = None,
    base_url: str = None,
    embedding_api_key: str = None,
    embedding_base_url: str = None,
    embedding_provider: str = None,
    embedding_model: str = None,
    llm_provider: str = None,
    llm_model: str = None,
    top_k: int = 5,
    enable_multi_query: bool = True,
):
    history = chat_manager.get_messages(library_name, chat_id)
    history_text = _history_text(history)

    if enable_multi_query and len(user_input) > 6:
        sub_queries = _generate_sub_queries(
            user_input, history_text,
            api_key=api_key, base_url=base_url,
            provider=llm_provider, model=llm_model,
        )
        queries = [user_input] + sub_queries
    else:
        queries = [user_input]

    seen_chunks = set()
    all_chunks = []

    for query in queries:
        chunks = search_vector_store(
            library_name,
            query,
            k=top_k,
            data_root=data_root,
            api_key=embedding_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
        )
        for chunk in chunks:
            key = chunk.get("id") or chunk.get("text", "")[:80]
            if key not in seen_chunks:
                seen_chunks.add(key)
                all_chunks.append(chunk)

    all_chunks.sort(key=lambda c: c.get("score", 1.0))

    retrieved = "\n".join(
        f"[来源: {c.get('meta', {}).get('doc_name', '未知')}]\n{c.get('text', '')}"
        for c in all_chunks[:top_k * 2]
    )

    return (
        f"您正在与一个专业的AI助手对话。\n"
        f"以下是知识库中可能相关的文档内容，请基于这些内容回答用户问题。\n"
        f"━━━ 检索上下文 ━━━\n"
        f"{retrieved}\n"
        f"━━━ 对话历史 ━━━\n"
        f"{history_text}\n"
        f"━━━ 当前问题 ━━━\n"
        f"{user_input}"
    )


def _history_text(messages: list) -> str:
    if not messages:
        return "（无历史对话）"
    lines = []
    for m in messages[-6:]:
        role = m.get("role", "assistant")
        name = "用户" if role == "user" else "助手"
        lines.append(f"{name}: {m.get('text', '')}")
    return "\n".join(lines)


def _generate_sub_queries(
    question: str,
    history_text: str,
    api_key=None,
    base_url=None,
    provider=None,
    model=None,
) -> List[str]:
    try:
        prompt = MULTI_QUERY_PROMPT.format(question=question)
        if history_text and history_text != "（无历史对话）":
            prompt = prompt.replace("用户问题：{question}",
                                    f"对话历史：\n{history_text}\n\n用户问题：{question}")
        resp = chat_complete(
            [{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            model=model,
        )
        lines = [line.strip() for line in resp.strip().split("\n") if line.strip()]
        lines = [line.lstrip("0123456789.、) ）").strip() for line in lines]
        return lines[:3]
    except Exception:
        return []
