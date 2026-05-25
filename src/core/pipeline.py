"""RAG 对话管道：摘要 -> Chroma 检索 -> 回答。"""

from typing import Iterator, List, Optional, Tuple
import logging

from langchain_core.runnables import RunnableLambda

from core.llm import generate_reply, stream_reply
from db.vector_store import search_vector_store, sync_vector_store

log = logging.getLogger(__name__)


def lcel_pipeline(
    library_name: str,
    chat_id: Optional[str],
    user_question: str,
    chat_manager,
    data_root: str = "data",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    k: int = 5,
) -> Tuple[str, List[dict]]:
    """非流式执行完整 RAG 流程。"""
    env = prepare_rag_env(
        library_name,
        chat_id,
        user_question,
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
        k=k,
    )
    messages = _build_answer_messages(env)
    answer = generate_reply(
        messages,
        model=env.get("llm_model"),
        api_key=env.get("llm_api_key"),
        provider=env.get("llm_provider"),
        base_url=env.get("llm_base_url"),
    )
    return answer, env.get("contexts", [])


def stream_lcel_pipeline(
    library_name: str,
    chat_id: Optional[str],
    user_question: str,
    chat_manager,
    data_root: str = "data",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    k: int = 5,
) -> Iterator[str]:
    """流式执行 RAG 流程，逐段产出最终回答。"""
    env = prepare_rag_env(
        library_name,
        chat_id,
        user_question,
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
        k=k,
    )
    messages = _build_answer_messages(env)
    yield from stream_reply(
        messages,
        model=env.get("llm_model"),
        api_key=env.get("llm_api_key"),
        provider=env.get("llm_provider"),
        base_url=env.get("llm_base_url"),
    )


def prepare_rag_env(
    library_name: str,
    chat_id: Optional[str],
    user_question: str,
    chat_manager,
    data_root: str = "data",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    k: int = 5,
) -> dict:
    """使用 LangChain Runnable 管道符组织摘要和检索阶段。"""
    chain = (
        RunnableLambda(_load_history)
        | RunnableLambda(_summarize_for_retrieval)
        | RunnableLambda(_sync_and_retrieve)
    )
    return chain.invoke(
        {
            "library_name": library_name,
            "chat_id": chat_id,
            "user_question": user_question,
            "chat_manager": chat_manager,
            "data_root": data_root,
            "llm_api_key": api_key,
            "llm_base_url": base_url,
            "embedding_api_key": embedding_api_key,
            "embedding_base_url": embedding_base_url,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "k": k,
        }
    )


def _load_history(env: dict) -> dict:
    manager = env.get("chat_manager")
    if not manager or not env.get("chat_id"):
        env["history"] = []
        return env
    try:
        env["history"] = manager.get_messages(env["library_name"], env["chat_id"]) or []
    except Exception as exc:
        log.exception("读取历史对话失败")
        raise RuntimeError(f"读取历史对话失败：{exc}") from exc
    return env


def _summarize_for_retrieval(env: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "你负责把当前问题和历史对话改写成适合向量检索的一段中文摘要。"
                "只输出摘要本身，保留关键实体、约束、时间、指标和用户意图，不要回答问题。"
            ),
        }
    ]
    for item in env.get("history", [])[-12:]:
        role = item.get("role") if item.get("role") in {"user", "assistant"} else "user"
        messages.append({"role": role, "content": item.get("text", "")})
    messages.append({"role": "user", "content": f"当前问题：{env.get('user_question', '')}"})
    try:
        summary = generate_reply(
            messages,
            model=env.get("llm_model"),
            api_key=env.get("llm_api_key"),
            provider=env.get("llm_provider"),
            base_url=env.get("llm_base_url"),
        ).strip()
    except Exception as exc:
        log.exception("生成检索摘要失败")
        raise RuntimeError(f"生成检索摘要失败：{exc}") from exc
    env["retrieval_summary"] = summary or env.get("user_question", "")
    return env


def _sync_and_retrieve(env: dict) -> dict:
    try:
        sync_vector_store(
            env["library_name"],
            data_root=env.get("data_root", "data"),
            api_key=env.get("embedding_api_key"),
            embedding_provider=env.get("embedding_provider"),
            embedding_model=env.get("embedding_model"),
            embedding_base_url=env.get("embedding_base_url"),
        )
        env["contexts"] = search_vector_store(
            env["library_name"],
            env.get("retrieval_summary", ""),
            k=env.get("k", 5),
            data_root=env.get("data_root", "data"),
            api_key=env.get("embedding_api_key"),
            embedding_provider=env.get("embedding_provider"),
            embedding_model=env.get("embedding_model"),
            embedding_base_url=env.get("embedding_base_url"),
        )
    except Exception as exc:
        log.exception("向量库同步或检索失败")
        raise RuntimeError(f"向量库同步或检索失败：{exc}") from exc
    return env


def _build_answer_messages(env: dict) -> List[dict]:
    context_text = _format_contexts(env.get("contexts", []))
    messages = [
        {
            "role": "system",
            "content": (
                "你是中文知识库问答助手。优先依据检索上下文回答。"
                "如果上下文为空或不足以确认，请明确说明当前知识库没有足够依据。"
            ),
        },
        {"role": "system", "content": f"检索摘要：{env.get('retrieval_summary', '')}"},
        {"role": "system", "content": f"检索上下文：\n{context_text or '（无检索上下文）'}"},
    ]
    for item in env.get("history", [])[-8:]:
        role = item.get("role") if item.get("role") in {"user", "assistant"} else "user"
        messages.append({"role": role, "content": item.get("text", "")})
    messages.append({"role": "user", "content": env.get("user_question", "")})
    return messages


def _format_contexts(contexts: List[dict]) -> str:
    parts = []
    for index, context in enumerate(contexts, start=1):
        source = context.get("meta", {}).get("source", "")
        text = context.get("text", "")
        parts.append(f"[{index}] 来源：{source}\n{text}")
    return "\n\n".join(parts)
