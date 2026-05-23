"""
Manage chat sessions per library: create/delete/list sessions, store conversation IDs
"""

from typing import List, Dict
import uuid
from pathlib import Path
import json
from core.memory import MemoryStore
from db.meta_store import MetaStore


class ChatManager:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.memory = MemoryStore(data_root=data_root)
        self.meta = MetaStore(db_path=str(self.data_root / "meta.db"))

    def _chats_dir(self, library_name: str) -> Path:
        d = self.data_root / "libraries" / library_name / "chats"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_chats(self, library_name: str) -> List[Dict]:
        try:
            return self.meta.list_chats(library_name)
        except Exception:
            return self.memory.list_chats(library_name)

    def create_chat(self, library_name: str, title: str) -> str:
        chat_id = str(uuid.uuid4())
        path = self._chats_dir(library_name) / f"{chat_id}.json"
        data = {"id": chat_id, "library": library_name, "title": title, "messages": []}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            self.meta.add_chat(chat_id, library_name, title)
        except Exception:
            pass
        return chat_id

    def delete_chat(self, library_name: str, chat_id: str) -> None:
        path = self._chats_dir(library_name) / f"{chat_id}.json"
        if path.exists():
            path.unlink()
        try:
            self.meta.delete_chat(chat_id)
        except Exception:
            pass

    def append_user_message(self, library_name: str, chat_id: str, text: str):
        self.memory.append_message(library_name, chat_id, "user", text)

    def append_assistant_message(self, library_name: str, chat_id: str, text: str):
        self.memory.append_message(library_name, chat_id, "assistant", text)

    def get_messages(self, library_name: str, chat_id: str):
        return self.memory.get_messages(library_name, chat_id)
