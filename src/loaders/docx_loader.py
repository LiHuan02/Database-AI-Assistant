"""
DOCX loader using python-docx
"""

from docx import Document

def load_docx(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        parts.append(p.text)
    return "\n".join(parts)
