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
