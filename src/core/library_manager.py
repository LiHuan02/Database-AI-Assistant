"""
LibraryManager: filesystem-based knowledge base management
- create/delete/list libraries
- each library lives under data/libraries/<name>
- store simple metadata in meta.json
"""
from pathlib import Path
import json
from typing import List, Dict


class LibraryManager:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        (self.data_root / "libraries").mkdir(parents=True, exist_ok=True)

    def _lib_path(self, name: str) -> Path:
        return self.data_root / "libraries" / name

    def create_library(self, name: str, description: str = "") -> Dict:
        p = self._lib_path(name)
        p.mkdir(parents=True, exist_ok=True)
        (p / "docs").mkdir(exist_ok=True)
        (p / "chats").mkdir(exist_ok=True)
        meta = {"name": name, "description": description}
        try:
            (p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return meta

    def delete_library(self, name: str) -> bool:
        p = self._lib_path(name)
        if not p.exists():
            return False
        # careful remove
        try:
            import shutil

            shutil.rmtree(p)
            return True
        except Exception:
            return False

    def list_libraries(self) -> List[Dict]:
        out = []
        base = self.data_root / "libraries"
        if not base.exists():
            return out
        for d in base.iterdir():
            if d.is_dir():
                meta = {"name": d.name}
                try:
                    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                    meta.update(m)
                except Exception:
                    pass
                out.append(meta)
        return out
