"""
Manage chat sessions per library: create/delete/list sessions, store conversation IDs
"""

from typing import List, Dict

class ChatManager:
    def list_chats(self, library_name: str) -> List[Dict]:
        return []

    def create_chat(self, library_name: str, title: str) -> str:
        return "chat-id-placeholder"

    def delete_chat(self, chat_id: str) -> None:
        pass
