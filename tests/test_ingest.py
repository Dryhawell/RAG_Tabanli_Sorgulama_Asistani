import os
from unittest.mock import MagicMock

import numpy as np

from rag.ingest import delete_source, is_supported_file, list_data_files, rebuild_from_data_dir
from rag.index import FaissIndex
from rag.types import ChunkMetadata


def test_is_supported_file():
    assert is_supported_file("x.pdf")
    assert is_supported_file("y.TXT")
    assert not is_supported_file("z.docx")


def test_list_data_files_filters(tmp_path):
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "c.docx").write_text("z")
    nested = tmp_path / "hukuk"
    nested.mkdir()
    (nested / "d.txt").write_text("nested")
    files = list_data_files(str(tmp_path))
    rels = {os.path.relpath(p, tmp_path).replace("\\", "/") for p in files}
    assert rels == {"a.pdf", "b.txt", "hukuk/d.txt"}


def test_delete_source_removes_file_and_chunks(tmp_path):
    path = tmp_path / "notlar.txt"
    path.write_text("merhaba", encoding="utf-8")
    idx = FaissIndex(dim=2)
    vec = np.array([[1.0, 0.0]], dtype=np.float32)
    meta = ChunkMetadata(
        source_file="notlar.txt",
        chunk_id=0,
        page_start=1,
        page_end=1,
        word_count=1,
    )
    idx.add(vec, ["merhaba"], [meta])
    report = delete_source("notlar.txt", idx, str(tmp_path))
    assert report["chunks_removed"] == 1
    assert report["file_deleted"] is True
    assert not path.exists()
    assert idx.size == 0


def test_rebuild_skips_bad_files(tmp_path):
    good = tmp_path / "ok.txt"
    good.write_text("kelime " * 250, encoding="utf-8")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not-a-real-pdf")

    emb = MagicMock()
    emb.dim = 3
    emb.encode.side_effect = lambda texts: __import__("numpy").zeros((len(texts), 3), dtype="float32")

    # pdf okuma hata verebilir; rebuild yine de txt'yi işlemeli
    index, reports = rebuild_from_data_dir(str(tmp_path), emb)
    sources = {r["source_file"] for r in reports}
    assert "ok.txt" in sources
    assert index.size >= 1
