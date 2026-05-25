"""向量模型封装：LLM 和 Embedding 使用完全独立的厂商、密钥和 base_url。"""

from typing import Iterable, List, Optional
import hashlib
import logging

from utils.config import get_embedding_config, get_env

log = logging.getLogger(__name__)

DEFAULT_DIM = 1536
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


def get_embeddings(
    texts: List[str],
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """返回文本向量。

    兼容接口的批量上限差异很大，默认每批 25 条，避免 DashScope 等厂商报
    `batch size is invalid, it should not be larger than 25`。
    """
    clean_texts = _normalize_texts(texts)
    if not clean_texts:
        return []

    config = get_embedding_config(provider=provider, model=model, api_key=api_key, base_url=base_url)
    provider_name = (config.provider or "openai").lower()
    resolved_batch_size = max(1, int(batch_size or get_env("EMBEDDING_BATCH_SIZE", 25)))

    if provider_name == "hash":
        return [_hash_vector(text, dim=DEFAULT_DIM) for text in clean_texts]

    if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
        return _openai_compatible_embeddings(
            clean_texts,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            batch_size=resolved_batch_size,
        )

    if provider_name == "local":
        try:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(config.model or "all-MiniLM-L6-v2")
            embeddings = encoder.encode(clean_texts, show_progress_bar=False)
            return [list(map(float, item)) for item in embeddings]
        except Exception:
            log.exception("本地向量模型调用失败")
            raise

    raise ValueError(f"不支持的向量模型厂商：{config.provider}")


def _openai_compatible_embeddings(
    texts: List[str],
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    batch_size: int,
) -> List[List[float]]:
    import openai

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)

    vectors: List[List[float]] = []
    for batch in _batched(texts, batch_size):
        # OpenAI 兼容接口期望 input 是 str 或 list[str]，这里保证永远传 list[str]。
        resp = client.embeddings.create(input=list(batch), model=model)
        vectors.extend([item.embedding for item in resp.data])
    return vectors


def _normalize_texts(texts: Iterable[str]) -> List[str]:
    normalized = []
    for item in texts or []:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _batched(items: List[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _hash_vector(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] / 255.0) * (1.0 - (i / dim)) for i in range(dim)]
