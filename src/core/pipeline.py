"""LCEL 风格对话流水线：Summarize -> Retrieve -> Answer

提供一个 `lcel_pipeline` 函数，接收库名、会话 id、用户问题，返回助手回复和检索到的上下文列表。
"""
from typing import List, Tuple, Optional
import logging

from core.llm import generate_reply
from core.retrieval import retrieve_relevant

log = logging.getLogger(__name__)


def _summarize_history(history: List[dict], llm_provider: str, llm_model: Optional[str], api_key: Optional[str]) -> str:
    if not history:
        return ""
    # build summarize prompt
    system = "请将下面的对话摘要为简短要点，便于检索相关文档。保留关键实体与问题意图。"
    messages = [{"role": "system", "content": system}]
    for m in history[-20:]:
        messages.append({"role": m.get("role", "user"), "content": m.get("text", "")})
    try:
        summary = generate_reply(messages, model=llm_model or "gpt-3.5-turbo", api_key=api_key, provider=llm_provider)
        return summary or ""
    except Exception:
        log.exception("summarize failed")
        return " ".join([m.get("text", "") for m in history[-5:]])


class SimpleRunnable:
    """A tiny runnable abstraction that supports `|` chaining.

    Each runnable accepts and returns an `env` dict. This is NOT LangChain's
    Runnable, but provides a similar `|` chaining interface so steps can be
    composed simply.
    """

    def __init__(self, fn):
        self.fn = fn

    def __or__(self, other):
        if isinstance(other, SimpleRunnable):
            return PipelineRunnable([self, other])
        if isinstance(other, PipelineRunnable):
            return PipelineRunnable([self] + other.runnables)
        raise TypeError("Can only chain SimpleRunnable or PipelineRunnable")

    def __call__(self, env: dict) -> dict:
        return self.fn(env)


class PipelineRunnable:
    def __init__(self, runnables: List[SimpleRunnable]):
        self.runnables = runnables

    def __or__(self, other):
        if isinstance(other, SimpleRunnable):
            return PipelineRunnable(self.runnables + [other])
        if isinstance(other, PipelineRunnable):
            return PipelineRunnable(self.runnables + other.runnables)
        raise TypeError("Can only chain SimpleRunnable or PipelineRunnable")

    def __call__(self, env: dict) -> dict:
        cur = env
        for r in self.runnables:
            try:
                cur = r(cur) or cur
            except Exception:
                log.exception('runnable step failed')
        return cur


def _summarize_step(env: dict) -> dict:
    history = env.get('history', [])
    llm_provider = env.get('llm_provider')
    llm_model = env.get('llm_model')
    api_key = env.get('api_key')
    summary = _summarize_history(history, llm_provider, llm_model, api_key)
    env['summary'] = summary
    return env


def _retrieve_step(env: dict) -> dict:
    # build query from question + summary
    question = env.get('question', '')
    q = question
    if env.get('summary'):
        q = question + '\n对话摘要: ' + env.get('summary')
    contexts = retrieve_relevant(env.get('library'), q, k=env.get('k', 5), data_root=env.get('data_root', 'data'), api_key=env.get('api_key'), embedding_provider=env.get('embedding_provider', 'openai'), embedding_model=env.get('embedding_model'))
    env['contexts'] = contexts
    return env


def _answer_step(env: dict) -> dict:
    # construct messages and call generate_reply
    system_prompt = env.get('system_prompt') or '你是一个中文优先的问答助手；使用提供的上下文来回答用户的问题，必要时请给出来源。'
    messages = [{"role": "system", "content": system_prompt}]
    if env.get('summary'):
        messages.append({"role": "system", "content": f"对话摘要: {env.get('summary')}"})
    for c in env.get('contexts', []):
        src = c.get('meta', {}).get('source', '')
        txt = c.get('text', '')
        messages.append({"role": "system", "content": f"来源:{src}; 内容:{txt}"})
    for h in env.get('history', [])[-8:]:
        messages.append({"role": h.get('role'), "content": h.get('text')})
    messages.append({"role": "user", "content": env.get('question', '')})
    try:
        assistant_text = generate_reply(messages, model=env.get('llm_model') or 'gpt-3.5-turbo', api_key=env.get('api_key'), provider=env.get('llm_provider') or 'openai')
    except Exception:
        log.exception('answer step failed')
        assistant_text = '（回退）无法生成回答'
    env['assistant_text'] = assistant_text
    return env


# build a composable runnable pipeline
summarize_r = SimpleRunnable(_summarize_step)
retrieve_r = SimpleRunnable(_retrieve_step)
answer_r = SimpleRunnable(_answer_step)

composed_pipeline = summarize_r | retrieve_r | answer_r


def lcel_pipeline(
    library_name: str,
    chat_id: Optional[str],
    user_question: str,
    chat_manager,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    k: int = 5,
) -> Tuple[str, List[dict]]:
    """Run LCEL pipeline using the SimpleRunnable composed pipeline.

    Returns (assistant_text, contexts)
    """
    try:
        history = []
        if chat_manager and chat_id:
            try:
                history = chat_manager.get_messages(library_name, chat_id) or []
            except Exception:
                log.exception("getting history failed")

        env = {
            'library': library_name,
            'chat_id': chat_id,
            'question': user_question,
            'history': history,
            'data_root': data_root,
            'api_key': api_key,
            'embedding_provider': embedding_provider,
            'embedding_model': embedding_model,
            'llm_provider': llm_provider,
            'llm_model': llm_model,
            'k': k,
        }

        out_env = composed_pipeline(env)
        return out_env.get('assistant_text', ''), out_env.get('contexts', [])
    except Exception as e:
        log.exception("lcel pipeline failed")
        return (f"（回退）无法生成回答：{e}", [])


def lcel_pipeline(
    library_name: str,
    chat_id: Optional[str],
    user_question: str,
    chat_manager,
    data_root: str = "data",
    api_key: Optional[str] = None,
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    k: int = 5,
) -> Tuple[str, List[dict]]:
    """Run LCEL pipeline and return (assistant_text, contexts_list).

    - summarize recent history via LLM
    - retrieve relevant chunks from vector DB using query embedding
    - call LLM with system prompt + retrieved contexts + history + user question
    """
    try:
        history = []
        if chat_manager and chat_id:
            try:
                history = chat_manager.get_messages(library_name, chat_id) or []
            except Exception:
                log.exception("getting history failed")

        summary = _summarize_history(history, llm_provider, llm_model, api_key)

        # build retrieval query combining question and summary
        query = user_question
        if summary:
            query = user_question + "\n对话摘要: " + summary

        contexts = retrieve_relevant(library_name, query, k=k, data_root=data_root, api_key=api_key, embedding_provider=embedding_provider, embedding_model=embedding_model)

        # construct final messages
        system_prompt = "你是一个中文优先的问答助手；使用提供的上下文来回答用户的问题，必要时请给出来源。"
        messages = [{"role": "system", "content": system_prompt}]
        if summary:
            messages.append({"role": "system", "content": f"对话摘要: {summary}"})

        # include retrieved contexts as system messages
        for c in contexts:
            src = c.get("meta", {}).get("source", "")
            txt = c.get("text", "")
            messages.append({"role": "system", "content": f"来源:{src}; 内容:{txt}"})

        # include recent history
        for h in history[-8:]:
            messages.append({"role": h.get("role"), "content": h.get("text")})

        messages.append({"role": "user", "content": user_question})

        assistant_text = generate_reply(messages, model=llm_model or "gpt-3.5-turbo", api_key=api_key, provider=llm_provider)

        return assistant_text, contexts
    except Exception as e:
        log.exception("lcel pipeline failed")
        # fallback simple echo
        return f"（回退）无法生成回答：{e}", []
