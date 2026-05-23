"""
Embedding wrapper: support OpenAI and local embeddings
"""

def get_embedding_client(kind: str = "openai", api_key: str = None):
    """Return an embeddings function/object depending on configuration."""
    raise NotImplementedError
