import os
from unittest.mock import MagicMock

from rag.ingest import is_supported_file, list_data_files, rebuild_from_data_dir


def test_is_supported_file():
    assert is_supported_file("x.pdf")
    assert is_supported_file("y.TXT")
    assert not is_supported_file("z.docx")


def test_list_data_files_filters(tmp_path):
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "c.docx").write_text("z")
    (tmp_path / "subdir").mkdir()
    files = list_data_files(str(tmp_path))
    names = {os.path.basename(p) for p in files}
    assert names == {"a.pdf", "b.txt"}


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
