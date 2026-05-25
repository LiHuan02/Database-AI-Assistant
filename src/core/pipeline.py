"""RAG 对话管道：摘要 -> Chroma 检索 -> 回答。"""

from typing import List, Optional, Tuple
import logging

from langchain_core.runnables import RunnableLambda

from core.llm import generate_reply
from core.retrieval import retrieve_relevant

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
    """执行完整对话流程并返回 `(回答文本, 检索上下文)`。

    流程固定为：
    当前问题 + 历史对话 + system prompt -> LLM 生成检索摘要
    -> 摘要进入 Chroma 向量库检索上下文
    -> 上下文 + 历史对话 + 当前问题 -> LLM 生成最终回答。
    """
    try:
        chain = (
            RunnableLambda(_load_inputs)
            | RunnableLambda(_summarize_for_retrieval)
            | RunnableLambda(_retrieve_contexts)
            | RunnableLambda(_answer_question)
        )
        result = chain.invoke(
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
        return result.get("assistant_text", ""), result.get("contexts", [])
    except Exception as exc:
        log.exception("RAG 管道执行失败")
        return f"无法生成回答：{exc}", []


def _load_inputs(env: dict) -> dict:
    env["history"] = _get_history(env.get("chat_manager"), env["library_name"], env.get("chat_id"))
    return env


def _summarize_for_retrieval(env: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "你负责把用户当前问题和历史对话改写成适合向量检索的一段中文摘要。"
                "只输出摘要本身，保留关键实体、约束、时间、指标和用户意图；不要回答问题。"
            ),
        }
    ]
    for item in env.get("history", [])[-12:]:
        messages.append({"role": item.get("role", "user"), "content": item.get("text", "")})
    messages.append({"role": "user", "content": f"当前问题：{env.get('user_question', '')}"})
    try:
        summary = generate_reply(
            messages,
            model=env.get("llm_model"),
            api_key=env.get("llm_api_key"),
            provider=env.get("llm_provider"),
            base_url=env.get("llm_base_url"),
        ).strip()
    except Exception:
        log.exception("生成检索摘要失败，使用当前问题作为检索摘要")
        summary = env.get("user_question", "")
    env["retrieval_summary"] = summary or env.get("user_question", "")
    return env


def _retrieve_contexts(env: dict) -> dict:
    env["contexts"] = retrieve_relevant(
        env["library_name"],
        env.get("retrieval_summary", ""),
        k=env.get("k", 5),
        data_root=env.get("data_root", "data"),
        api_key=env.get("embedding_api_key"),
        embedding_provider=env.get("embedding_provider"),
        embedding_model=env.get("embedding_model"),
        embedding_base_url=env.get("embedding_base_url"),
    )
    return env


def _answer_question(env: dict) -> dict:
    messages = _build_answer_messages(
        user_question=env.get("user_question", ""),
        history=env.get("history", []),
        contexts=env.get("contexts", []),
        retrieval_summary=env.get("retrieval_summary", ""),
    )
    env["assistant_text"] = generate_reply(
        messages,
        model=env.get("llm_model"),
        api_key=env.get("llm_api_key"),
        provider=env.get("llm_provider"),
        base_url=env.get("llm_base_url"),
    )
    return env


def _get_history(chat_manager, library_name: str, chat_id: Optional[str]) -> List[dict]:
    if not chat_manager or not chat_id:
        return []
    try:
        return chat_manager.get_messages(library_name, chat_id) or []
    except Exception:
        log.exception("读取历史对话失败")
        return []


def _build_answer_messages(
    user_question: str,
    history: List[dict],
    contexts: List[dict],
    retrieval_summary: str,
) -> List[dict]:
    context_text = _format_contexts(contexts)
    messages = [
        {
            "role": "system",
            "content": (
                "你是中文知识库问答助手。优先依据检索上下文回答；"
                "如果上下文为空或不足以确认，请明确说明当前知识库没有足够依据。"
            ),
        },
        {"role": "system", "content": f"检索摘要：{retrieval_summary}"},
        {"role": "system", "content": f"检索上下文：\n{context_text or '（无检索上下文）'}"},
    ]
    for item in history[-8:]:
        role = item.get("role") if item.get("role") in {"user", "assistant"} else "user"
        messages.append({"role": role, "content": item.get("text", "")})
    messages.append({"role": "user", "content": user_question})
    return messages


def _format_contexts(contexts: List[dict]) -> str:
    parts = []
    for index, context in enumerate(contexts, start=1):
        source = context.get("meta", {}).get("source", "")
        text = context.get("text", "")
        parts.append(f"[{index}] 来源：{source}\n{text}")
    return "\n\n".join(parts)
