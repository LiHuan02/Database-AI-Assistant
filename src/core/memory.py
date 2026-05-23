"""
Session memory support: store/retrieve messages for a chat (in-memory or persistent)
"""
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class MemoryStore:
    """Persistent per-chat memory stored as JSON under `data/libraries/<library>/chats/<chat_id>.json`."""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def _chat_path(self, library_name: str, chat_id: str) -> Path:
        d = self.data_root / "libraries" / library_name / "chats"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{chat_id}.json"

    def append_message(self, library_name: str, chat_id: str, role: str, text: str):
        path = self._chat_path(library_name, chat_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"id": chat_id, "library": library_name, "messages": []}
        ts = datetime.utcnow().isoformat() + 'Z'
        data.setdefault("messages", []).append({"role": role, "text": text, "ts": ts})
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_messages(self, library_name: str, chat_id: str) -> List[Dict]:
        path = self._chat_path(library_name, chat_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("messages", [])

    def list_chats(self, library_name: str) -> List[Dict]:
        d = Path(self.data_root) / "libraries" / library_name / "chats"
        if not d.exists():
            return []
        out = []
        for p in d.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append({"id": data.get("id"), "title": data.get("title", "")})
            except Exception:
                continue
        return out
