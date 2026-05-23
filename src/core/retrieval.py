"""
Retrieval and RAG orchestration: query -> retrieve -> construct prompt
"""

def retrieve_relevant(library_name: str, query: str, k: int = 5):
    """Return top-k relevant document chunks for query from library."""
    raise NotImplementedError
