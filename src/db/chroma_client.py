"""Chroma 向量库客户端封装 — 每个知识库独立 PersistentClient。"""

from pathlib import Path
import logging
import os

import chromadb

log = logging.getLogger(__name__)


class ChromaClient:
    """每个知识库拥有独立的 Chroma PersistentClient（独立 SQLite 文件）。"""

    def __init__(self, library_name: str, data_root: str = "data"):
        self.library_name = library_name
        persist_dir = Path(data_root) / "libraries" / library_name / "chroma"
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def collection_name(self) -> str:
        """每个 Chroma 实例只服务一个知识库，使用固定名称即可。"""
        return "documents"

    def get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name(),
            metadata={"library": self.library_name},
        )

    def reset_collection(self):
        name = self.collection_name()
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        return self.client.get_or_create_collection(
            name=name,
            metadata={"library": self.library_name},
        )

    def delete_collection(self) -> bool:
        name = self.collection_name()
        try:
            self.client.delete_collection(name)
            return True
        except Exception:
            log.exception("删除 Chroma collection 失败：%s", self.library_name)
            return False


def clear_chroma_system_cache():
    """测试或批处理结束后释放 Chroma 全局缓存，降低 Windows 文件锁概率。"""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
