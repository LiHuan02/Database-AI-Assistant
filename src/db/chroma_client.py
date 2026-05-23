"""
Chroma client wrapper: connect and return collection for a given library name
"""

from chromadb import Client

class ChromaClient:
    def __init__(self, persist_directory: str = None):
        self.client = Client()

    def get_collection(self, name: str):
        return self.client.get_collection(name)
