"""
Chroma client wrapper: connect and return collection for a given library name
"""
from pathlib import Path
from typing import Optional

try:
    from chromadb import Client
    from chromadb.config import Settings
except Exception:
    Client = None


class ChromaClient:
    """Wrapper that supports per-library persistent directories under a data root.

    If chromadb is not installed the methods degrade gracefully.
    """

    def __init__(self, data_root: str = "data", chroma_settings: Optional[dict] = None):
        self.data_root = Path(data_root)
        self.chroma_settings = chroma_settings or {}
        self._clients = {}

    def _client_for(self, library_name: str):
        # Create or reuse a chromadb Client configured to persist under data/chroma/<library>
        if Client is None:
            return None
        if library_name in self._clients:
            return self._clients[library_name]
        persist_dir = str(self.data_root / "chroma" / library_name)
        settings = {**self.chroma_settings, "persist_directory": persist_dir}
        try:
            client = Client(Settings(**settings))
        except Exception:
            # fallback to default client
            client = Client()
        self._clients[library_name] = client
        return client

    def get_collection(self, library_name: str, collection_name: Optional[str] = None):
        """Return a chroma collection object for the library. If chromadb not available, return None."""
        client = self._client_for(library_name)
        if client is None:
            return None
        name = collection_name or library_name
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(name)
