from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.retrieval import retrieve_relevant
from core.embeddings import _batched
from db.chroma_client import clear_chroma_system_cache
from db.vector_store import build_vector_store, sync_vector_store


def test_chroma_ingest_and_retrieve():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data_root:
        lib = "测试知识库"
        docs_dir = Path(data_root) / "libraries" / lib / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "sample.txt").write_text(
            "这是一个测试文档。我们正在测试向量化和检索功能。关键词：数据库。",
            encoding="utf-8",
        )

        res = build_vector_store(lib, data_root=data_root, embedding_provider="hash")
        assert res["inserted"] > 0

        results = retrieve_relevant(
            lib,
            "数据库",
            k=5,
            data_root=data_root,
            embedding_provider="hash",
        )
        assert results
        assert results[0]["meta"]["source"] == "sample.txt"
    clear_chroma_system_cache()


def test_sync_removes_deleted_documents():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data_root:
        lib = "testlib"
        docs_dir = Path(data_root) / "libraries" / lib / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        doc_path = docs_dir / "sample.txt"
        doc_path.write_text("数据库 文档", encoding="utf-8")

        sync_vector_store(lib, data_root=data_root, embedding_provider="hash")
        doc_path.unlink()
        sync_vector_store(lib, data_root=data_root, embedding_provider="hash")

        results = retrieve_relevant(
            lib,
            "数据库",
            k=5,
            data_root=data_root,
            embedding_provider="hash",
        )
        assert results == []
    clear_chroma_system_cache()


def test_embedding_batches_are_limited_to_25():
    batches = list(_batched([str(i) for i in range(60)], 25))
    assert [len(batch) for batch in batches] == [25, 25, 10]
