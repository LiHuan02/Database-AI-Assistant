"""
Text splitter using RecursiveCharacterTextSplitter with semantic boundaries.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


_SEPARATORS_ZH = [
    "\n\n\n",
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
]

_SEPARATORS_EN = [
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
]


def _pick_separators(text: str, chunk_overlap: int) -> list[str]:
    has_cjk = any("一" <= c <= "鿿" for c in text[:2000])
    return _SEPARATORS_ZH if has_cjk else _SEPARATORS_EN


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    separators = _pick_separators(text, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )
    return splitter.split_text(text)