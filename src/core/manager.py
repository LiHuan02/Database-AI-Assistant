"""
Library (collection) manager: create/delete/list libraries (Chroma collections)
"""

from typing import List
class LibraryManager:
    """Compatibility wrapper that delegates to `core.library_manager.LibraryManager`.

    This allows older imports from `core.manager` to continue working.
    """

    def __init__(self, data_root: str = "data"):
        from core.library_manager import LibraryManager as LM

        self._lm = LM(data_root=data_root)

    def list_libraries(self) -> List[str]:
        return [l.get("name") for l in self._lm.list_libraries()]

    def create_library(self, name: str, description: str = "") -> None:
        self._lm.create_library(name, description=description)

    def delete_library(self, name: str) -> None:
        self._lm.delete_library(name)
