"""
Simple metadata and chat persistence (sqlite or JSON wrapper)
"""

import sqlite3
from pathlib import Path

class MetaStore:
    def __init__(self, db_path: str = "meta.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            library TEXT,
            title TEXT
        )
        """)
        self.conn.commit()

    def add_chat(self, chat_id: str, library: str, title: str) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO chats (id, library, title) VALUES (?, ?, ?)", (chat_id, library, title))
        self.conn.commit()

    def delete_chat(self, chat_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        self.conn.commit()

    def list_chats(self, library: str):
        cur = self.conn.cursor()
        cur.execute("SELECT id, title FROM chats WHERE library = ?", (library,))
        return [{"id": row[0], "title": row[1]} for row in cur.fetchall()]
