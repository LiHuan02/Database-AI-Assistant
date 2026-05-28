"""
Configuration loader.

Priority (highest to lowest):
  1. Streamlit secrets (st.secrets)
  2. Environment variables (.env via python-dotenv)
  3. Hard-coded defaults
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    batch_size: int = 25


@dataclass
class LibraryConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5


def _try_st_secrets(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


def _resolve(key: str, env_key: str, default: str = "") -> str:
    val = _try_st_secrets(key, default)
    if val != default:
        return val
    return os.getenv(env_key, default)


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        provider=_resolve("LLM_PROVIDER", "LLM_PROVIDER", "openai"),
        model=_resolve("LLM_MODEL", "LLM_MODEL", "gpt-4o-mini"),
        api_key=_resolve("LLM_API_KEY", "LLM_API_KEY", ""),
        base_url=_resolve("LLM_BASE_URL", "LLM_BASE_URL", ""),
    )


def get_embedding_config() -> EmbeddingConfig:
    batch_str = _resolve("EMBEDDING_BATCH_SIZE", "EMBEDDING_BATCH_SIZE", "25")
    try:
        batch = int(batch_str)
    except (ValueError, TypeError):
        batch = 25
    return EmbeddingConfig(
        provider=_resolve("EMBEDDING_PROVIDER", "EMBEDDING_PROVIDER", "openai"),
        model=_resolve("EMBEDDING_MODEL", "EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=_resolve("EMBEDDING_API_KEY", "EMBEDDING_API_KEY", ""),
        base_url=_resolve("EMBEDDING_BASE_URL", "EMBEDDING_BASE_URL", ""),
        batch_size=batch,
    )


def get_library_config(library_name: str, data_root: str = "data") -> LibraryConfig:
    meta_path = Path(data_root) / "libraries" / library_name / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg = meta.get("config", {})
            return LibraryConfig(
                chunk_size=cfg.get("chunk_size", 1000),
                chunk_overlap=cfg.get("chunk_overlap", 200),
                top_k=cfg.get("top_k", 5),
            )
        except Exception:
            pass
    return LibraryConfig()


def save_library_config(
    library_name: str,
    cfg: LibraryConfig,
    data_root: str = "data",
):
    meta_path = Path(data_root) / "libraries" / library_name / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta["config"] = {
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "top_k": cfg.top_k,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
