"""
LLM abstraction supporting multiple providers: OpenAI, Anthropic, Local (transformers)
Provides `generate_reply(messages, model, api_key, provider)` returning assistant text.
"""
from typing import List, Dict, Optional


def generate_reply(messages: List[Dict], model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, provider: str = "openai") -> str:
    """Generate reply from messages.

    messages: list of {role, content}
    provider: 'openai' | 'anthropic' | 'local'
    """
    # Try direct provider implementations first
    if provider in ("openai", "azure_openai", "openrouter"):
        try:
            import openai

            if api_key:
                openai.api_key = api_key
            # For provider variants, user should pass appropriate model string
            resp = openai.ChatCompletion.create(model=model, messages=messages)
            return resp["choices"][0]["message"]["content"]
        except Exception:
            pass
    if provider == "anthropic":
        try:
            # anthropic usage if installed
            from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

            client = Anthropic(api_key=api_key)
            # anthropic expects a string prompt; we join messages naively
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            resp = client.completions.create(model=model, prompt=prompt, max_tokens=512)
            return resp.get("completion", "")
        except Exception:
            pass
    # local transformers fallback
    try:
        from transformers import pipeline

        pipe = pipeline("text-generation", model=model)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        out = pipe(prompt, max_length=512, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", "")
    except Exception:
        pass

    # last resort: simple echo or summary
    user_msgs = [m.get("content") for m in messages if m.get("role") == "user"]
    if user_msgs:
        return "（本地回退）我已收到你的问题：" + user_msgs[-1]
    return "（回退）无法生成回答"