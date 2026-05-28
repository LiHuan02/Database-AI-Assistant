"""
Chat session manager: create, list, rename, delete conversations.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from db.meta_store import MetaStore

log = logging.getLogger(__name__)


class ChatManager:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.store = MetaStore(str(self.data_root / "meta.db"))

    def _chat_dir(self, library_name: str):
        path = self.data_root / "libraries" / library_name / "chats"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _chat_file(self, library_name: str, chat_id: str):
        return self._chat_dir(library_name) / f"{chat_id}.json"

    def create_chat(self, library_name: str, title: str) -> str:
        chat_id = str(uuid.uuid4())
        self.store.add_chat(chat_id, library_name, title)
        self._chat_file(library_name, chat_id).write_text(
            json.dumps([], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return chat_id

    def list_chats(self, library_name: str):
        return self.store.list_chats(library_name)

    def delete_chat(self, library_name: str, chat_id: str):
        self.store.delete_chat(chat_id)
        chat_file = self._chat_file(library_name, chat_id)
        if chat_file.exists():
            chat_file.unlink()

    def rename_chat(self, library_name: str, chat_id: str, new_title: str):
        self.store.add_chat(chat_id, library_name, new_title)

    def get_messages(self, library_name: str, chat_id: str) -> list:
        chat_file = self._chat_file(library_name, chat_id)
        if not chat_file.exists():
            return []
        data = json.loads(chat_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("messages", [])
        return data

    def _append_message(self, library_name: str, chat_id: str, role: str, text: str):
        messages = self.get_messages(library_name, chat_id)
        messages.append({"role": role, "text": text})
        self._chat_file(library_name, chat_id).write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_user_message(self, library_name: str, chat_id: str, text: str):
        self._append_message(library_name, chat_id, "user", text)

    def append_assistant_message(self, library_name: str, chat_id: str, text: str):
        self._append_message(library_name, chat_id, "assistant", text)

    def search_chats(self, library_name: str, query: str) -> list:
        results = []
        for chat_info in self.list_chats(library_name):
            chat_id = chat_info["id"]
            messages = self.get_messages(library_name, chat_id)
            for msg in messages:
                if query.lower() in msg.get("text", "").lower():
                    results.append({
                        "chat_id": chat_id,
                        "title": chat_info.get("title"),
                        "role": msg["role"],
                        "snippet": msg["text"][:200],
                    })
        return results

    def export_chat_markdown(
        self, library_name: str, chat_id: str,
    ) -> str:
        messages = self.get_messages(library_name, chat_id)
        chats = self.list_chats(library_name)
        title = next(
            (c["title"] for c in chats if c["id"] == chat_id),
            chat_id,
        )
        lines = [f"# {title}", "", f"*知识库: {library_name}*", "", "---", ""]
        for msg in messages:
            role = msg.get("role", "assistant")
            name = "**用户**" if role == "user" else "**AI 助手**"
            lines.append(f"### {name}")
            lines.append("")
            lines.append(msg.get("text", ""))
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)
