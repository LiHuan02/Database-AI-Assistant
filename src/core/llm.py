"""大模型调用封装，支持 OpenAI 兼容接口的 base_url。"""

from typing import Dict, List, Optional

from utils.config import get_llm_config

OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "openrouter",
    "deepseek",
    "siliconflow",
    "dashscope",
    "zhipu",
    "moonshot",
    "baichuan",
    "compatible",
}


def generate_reply(
    messages: List[Dict],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """调用大模型生成回复。"""
    config = get_llm_config(provider=provider, model=model, api_key=api_key, base_url=base_url)
    provider_name = (config.provider or "openai").lower()

    if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
        return _openai_compatible_chat(messages, config.model, config.api_key, config.base_url)
    if provider_name == "anthropic":
        return _anthropic_chat(messages, config.model, config.api_key)
    if provider_name == "local":
        return _local_chat(messages, config.model)
    raise ValueError(f"不支持的大模型厂商：{config.provider}")


def _openai_compatible_chat(
    messages: List[Dict],
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> str:
    import openai

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def _anthropic_chat(messages: List[Dict], model: str, api_key: Optional[str]) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    system_parts = [item.get("content", "") for item in messages if item.get("role") == "system"]
    chat_messages = [
        {"role": item.get("role", "user"), "content": item.get("content", "")}
        for item in messages
        if item.get("role") in {"user", "assistant"}
    ]
    response = client.messages.create(
        model=model,
        system="\n".join(system_parts) if system_parts else None,
        messages=chat_messages,
        max_tokens=1024,
    )
    return "".join(getattr(part, "text", "") for part in getattr(response, "content", []))


def _local_chat(messages: List[Dict], model: str) -> str:
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model)
    prompt = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in messages)
    output = pipe(prompt, max_new_tokens=512, do_sample=False)
    if isinstance(output, list) and output:
        return output[0].get("generated_text", "")
    return ""
