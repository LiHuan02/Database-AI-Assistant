"""从环境变量读取应用配置。"""

from dataclasses import dataclass
from typing import Optional
import os

from dotenv import load_dotenv


load_dotenv()


def get_env(key: str, default=None):
    value = os.getenv(key)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


def get_llm_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ModelConfig:
    resolved_provider = provider or get_env("LLM_PROVIDER", "openai")
    return ModelConfig(
        provider=resolved_provider,
        model=model or get_env("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key or get_env("LLM_API_KEY") or get_env("OPENAI_API_KEY"),
        base_url=base_url or get_env("LLM_BASE_URL") or _provider_base_url(resolved_provider),
    )


def get_embedding_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ModelConfig:
    resolved_provider = provider or get_env("EMBEDDING_PROVIDER", "openai")
    return ModelConfig(
        provider=resolved_provider,
        model=model or get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=api_key or get_env("EMBEDDING_API_KEY") or get_env("OPENAI_API_KEY"),
        base_url=base_url or get_env("EMBEDDING_BASE_URL") or _provider_base_url(resolved_provider),
    )


def _provider_base_url(provider: str) -> Optional[str]:
    """常见 OpenAI 兼容厂商的默认 base_url，.env 中的配置优先。"""
    mapping = {
        "openai": None,
        "openrouter": "https://openrouter.ai/api/v1",
        "deepseek": "https://api.deepseek.com",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "moonshot": "https://api.moonshot.cn/v1",
        "baichuan": "https://api.baichuan-ai.com/v1",
    }
    return mapping.get((provider or "").lower())
