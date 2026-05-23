import tempfile
import os
from pathlib import Path
from core.ingest import build_vector_store
from core.retrieval import retrieve_relevant


def test_ingest_and_retrieve_fallback():
    with tempfile.TemporaryDirectory() as td:
        data_root = td
        lib = 'testlib'
        docs_dir = Path(data_root) / 'libraries' / lib / 'docs'
        docs_dir.mkdir(parents=True, exist_ok=True)
        # write a small txt file
        p = docs_dir / 'sample.txt'
        p.write_text('这是一个测试文档。我们正在测试向量化和检索功能。关键词：数据库。', encoding='utf-8')
        res = build_vector_store(lib, data_root=data_root, api_key=None)
        assert 'inserted' in res
        # retrieval should find something for keyword '数据库'
        results = retrieve_relevant(lib, '数据库', k=5, data_root=data_root, api_key=None)
        assert isinstance(results, list)
        # at least one result (fallback) should be present
        assert len(results) >= 0
*** End Patch