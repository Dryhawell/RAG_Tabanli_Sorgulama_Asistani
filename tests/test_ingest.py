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
    emb.model_name = "mock-emb"
    emb.encode.side_effect = lambda texts: __import__("numpy").zeros((len(texts), 3), dtype="float32")

    # pdf okuma hata verebilir; rebuild yine de txt'yi işlemeli
    index, reports = rebuild_from_data_dir(str(tmp_path), emb)
    sources = {r["source_file"] for r in reports}
    assert "ok.txt" in sources
    assert index.size >= 1


def test_rebuild_delta_skips_unchanged_and_removes_deleted(tmp_path):
    from rag.ingest import rebuild_delta_from_data_dir

    data = tmp_path / "data"
    data.mkdir()
    meta = tmp_path / "meta"
    meta.mkdir()
    a = data / "a.txt"
    b = data / "b.txt"
    a.write_text("kelime " * 250, encoding="utf-8")
    b.write_text("metin " * 250, encoding="utf-8")

    emb = MagicMock()
    emb.dim = 3
    emb.model_name = "mock-emb"
    emb.encode.side_effect = lambda texts: __import__("numpy").zeros(
        (len(texts), 3), dtype="float32"
    )

    index = FaissIndex(dim=3, embedding_model="mock-emb")
    man = str(meta / "ingest_manifest.json")
    index, reports1, summary1 = rebuild_delta_from_data_dir(
        str(data),
        emb,
        index,
        manifest_path=man,
        meta_path=str(meta / "sources.json"),
        chunk_size_words=50,
        overlap_ratio=0.1,
    )
    assert summary1["updated"] == 2
    assert summary1["unchanged"] == 0
    assert emb.encode.call_count >= 2
    size_after = index.size

    emb.encode.reset_mock()
    index, reports2, summary2 = rebuild_delta_from_data_dir(
        str(data),
        emb,
        index,
        manifest_path=man,
        meta_path=str(meta / "sources.json"),
        chunk_size_words=50,
        overlap_ratio=0.1,
    )
    assert summary2["unchanged"] == 2
    assert summary2["updated"] == 0
    assert emb.encode.call_count == 0
    assert index.size == size_after
    assert all(r.get("action") == "unchanged" for r in reports2)

    a.write_text("yeni " * 250, encoding="utf-8")
    b.unlink()
    emb.encode.reset_mock()
    index, reports3, summary3 = rebuild_delta_from_data_dir(
        str(data),
        emb,
        index,
        manifest_path=man,
        meta_path=str(meta / "sources.json"),
        chunk_size_words=50,
        overlap_ratio=0.1,
    )
    assert summary3["updated"] == 1
    assert summary3["removed"] == 1
    assert "b.txt" in summary3["removed_sources"]
    assert emb.encode.call_count >= 1
    actions = {r["source_file"]: r.get("action") for r in reports3}
    assert actions.get("a.txt") == "updated"
    assert actions.get("b.txt") == "removed"


def test_stable_chunk_uid_and_diff():
    from rag.chunking import diff_chunk_fingerprints, stable_chunk_uid

    u1 = stable_chunk_uid("a.txt", "hello world")
    u2 = stable_chunk_uid("a.txt", "hello world")
    u3 = stable_chunk_uid("a.txt", "hello worlds")
    assert u1 == u2 and u1 != u3
    d = diff_chunk_fingerprints([u1], [u1, u3])
    assert d["unchanged"] == 1
    assert d["added"] == 1
    assert d["identical"] is False


def test_rebuild_delta_stores_chunk_uids(tmp_path):
    from rag.ingest import load_ingest_manifest, rebuild_delta_from_data_dir

    data = tmp_path / "data"
    data.mkdir()
    meta = tmp_path / "meta"
    meta.mkdir()
    (data / "a.txt").write_text("kelime " * 250, encoding="utf-8")
    emb = MagicMock()
    emb.dim = 3
    emb.model_name = "mock-emb"
    emb.encode.side_effect = lambda texts: __import__("numpy").zeros(
        (len(texts), 3), dtype="float32"
    )
    index = FaissIndex(dim=3, embedding_model="mock-emb")
    man = str(meta / "ingest_manifest.json")
    rebuild_delta_from_data_dir(
        str(data),
        emb,
        index,
        manifest_path=man,
        meta_path=str(meta / "sources.json"),
        chunk_size_words=50,
        overlap_ratio=0.1,
    )
    doc = load_ingest_manifest(man)
    assert doc["sources"]["a.txt"].get("chunk_uids")
    assert isinstance(doc["sources"]["a.txt"]["chunk_uids"], list)


def test_rebuild_delta_selective_reembed(tmp_path):
    from rag.ingest import load_ingest_manifest, rebuild_delta_from_data_dir

    data = tmp_path / "data"
    data.mkdir()
    meta = tmp_path / "meta"
    meta.mkdir()
    path = data / "doc.txt"
    # İki ayrı paragraf → en az iki chunk
    path.write_text(
        ("alpha " * 80) + "\n\n" + ("beta " * 80),
        encoding="utf-8",
    )
    emb = MagicMock()
    emb.dim = 3
    emb.model_name = "mock-emb"
    emb.encode.side_effect = lambda texts: __import__("numpy").zeros(
        (len(texts), 3), dtype="float32"
    )
    index = FaissIndex(dim=3, embedding_model="mock-emb")
    man = str(meta / "ingest_manifest.json")
    kwargs = dict(
        manifest_path=man,
        meta_path=str(meta / "sources.json"),
        chunk_size_words=40,
        overlap_ratio=0.05,
    )
    index, _, s1 = rebuild_delta_from_data_dir(str(data), emb, index, **kwargs)
    assert s1["updated"] == 1
    first_calls = emb.encode.call_count
    first_size = index.size
    uids = load_ingest_manifest(man)["sources"]["doc.txt"]["chunk_uids"]
    assert len(uids) >= 2

    # Sadece ikinci paragrafı değiştir → seçici encode (tamamı değil)
    path.write_text(
        ("alpha " * 80) + "\n\n" + ("gamma " * 80),
        encoding="utf-8",
    )
    emb.encode.reset_mock()
    emb.encode.side_effect = lambda texts: __import__("numpy").zeros(
        (len(texts), 3), dtype="float32"
    )
    index, reports, s2 = rebuild_delta_from_data_dir(str(data), emb, index, **kwargs)
    assert s2["updated"] == 1
    sel = next(r for r in reports if r.get("source_file") == "doc.txt")
    assert sel.get("selective") is True
    assert emb.encode.call_count >= 1
    # Seçici: encode edilen chunk sayısı tümünden küçük olmalı
    encoded_n = emb.encode.call_args.args[0]
    assert len(encoded_n) < first_calls or len(encoded_n) < len(uids)
    assert index.size >= 1
    assert index.size == first_size or abs(index.size - first_size) <= len(uids)
