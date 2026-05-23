"""
Session memory support: store/retrieve messages for a chat (in-memory or persistent)
"""

class MemoryStore:
    def __init__(self):
        self._store = {}

    def append_message(self, chat_id: str, role: str, text: str):
        self._store.setdefault(chat_id, []).append({"role": role, "text": text})

    def get_messages(self, chat_id: str):
        return self._store.get(chat_id, [])
