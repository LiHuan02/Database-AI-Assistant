"""
Text splitting helper (wrapper for langchain splitter)
"""

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Return list of text chunks. Placeholder implementation."""
    # naive splitter
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - chunk_overlap
    return chunks
