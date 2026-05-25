"""Chroma 向量库客户端封装。"""

from pathlib import Path
from typing import Optional
import hashlib
import logging
import os

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)


class ChromaClient:
    """按知识库管理 Chroma PersistentClient 和 Collection。"""

    def __init__(self, data_root: str = "data", chroma_settings: Optional[dict] = None):
        self.data_root = Path(data_root)
        self.chroma_settings = chroma_settings or {}
        configured_dir = os.getenv("CHROMA_PERSIST_DIRECTORY") if str(self.data_root) == "data" else None
        persist_dir = configured_dir or str(self.data_root / "chroma")
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            settings = Settings(anonymized_telemetry=False, **self.chroma_settings)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir), settings=settings)
        return self._client

    def collection_name(self, library_name: str) -> str:
        """Chroma collection 名称只能包含有限字符，这里统一做稳定映射。"""
        digest = hashlib.sha1(library_name.encode("utf-8")).hexdigest()[:12]
        return f"library_{digest}"

    def get_collection(self, library_name: str):
        name = self.collection_name(library_name)
        return self.client.get_or_create_collection(name=name, metadata={"library": library_name})

    def reset_collection(self, library_name: str):
        name = self.collection_name(library_name)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        return self.client.get_or_create_collection(name=name, metadata={"library": library_name})

    def delete_collection(self, library_name: str) -> bool:
        name = self.collection_name(library_name)
        try:
            self.client.delete_collection(name)
            return True
        except Exception:
            log.exception("删除 Chroma collection 失败：%s", library_name)
            return False


def clear_chroma_system_cache():
    """测试或批处理结束后释放 Chroma 全局缓存，降低 Windows 文件锁概率。"""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
