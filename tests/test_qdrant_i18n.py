import numpy as np
import pytest

from rag.i18n import get_language, set_language, t
from rag.store import create_index, vector_backend
from rag.types import ChunkMetadata


def _meta(name: str, cid: int = 0) -> ChunkMetadata:
    return ChunkMetadata(
        source_file=name,
        chunk_id=cid,
        page_start=1,
        page_end=1,
        word_count=3,
        folder="genel",
        tags=["public"],
    )


def test_faiss_backend_factory(monkeypatch):
    monkeypatch.setattr("rag.store.VECTOR_BACKEND", "faiss")
    assert vector_backend() == "faiss"
    idx = create_index(dim=4, embedding_model="fake")
    assert idx.dim == 4
    vecs = np.eye(2, 4, dtype=np.float32)
    idx.add(vecs, ["a b c", "d e f"], [_meta("a.txt", 0), _meta("b.txt", 1)])
    hits = idx.search(vecs[0], top_k=1)
    assert hits and hits[0].metadata.source_file == "a.txt"


def test_qdrant_memory_index(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setattr("rag.store.VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr("rag.store.QDRANT_URL", "")
    monkeypatch.setattr("rag.store.QDRANT_PATH", "")
    monkeypatch.setattr("rag.store.QDRANT_COLLECTION", "test_rag")

    idx = create_index(dim=4, embedding_model="fake", backend="qdrant")
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    idx.add(vecs, ["alpha text here", "beta text here"], [_meta("a.txt", 0), _meta("b.txt", 1)])
    assert idx.size == 2
    assert idx.list_sources() == ["a.txt", "b.txt"]

    hits = idx.search(vecs[0], top_k=2)
    assert hits[0].metadata.source_file == "a.txt"
    assert hits[0].score > 0.9

    docstore = tmp_path / "doc.json"
    marker = tmp_path / "q.marker"
    idx.save(str(marker), str(docstore))
    assert docstore.exists()

    removed = idx.remove_source("a.txt")
    assert removed == 1
    assert idx.list_sources() == ["b.txt"]


def test_qdrant_payload_chunk_uid_lookup(monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setattr("rag.store.VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr("rag.store.QDRANT_URL", "")
    monkeypatch.setattr("rag.store.QDRANT_PATH", "")
    monkeypatch.setattr("rag.store.QDRANT_COLLECTION", "test_rag_uid")

    idx = create_index(dim=4, embedding_model="fake", backend="qdrant")
    m1 = _meta("doc.txt", 0)
    m1.chunk_uid = "uid-keep"
    m2 = _meta("doc.txt", 1)
    m2.chunk_uid = "uid-drop"
    vecs = np.eye(2, 4, dtype=np.float32)
    idx.add(vecs, ["keep text here now", "drop text here now"], [m1, m2])

    # memory map temizlenmiş gibi payload üzerinden bul
    idx._id_to_meta.clear()
    found = idx.ids_for_chunk_uids("doc.txt", ["uid-keep", "uid-drop"], use_payload=True)
    assert len(found) == 2
    removed = idx.remove_ids(idx.ids_for_chunk_uids("doc.txt", ["uid-drop"], use_payload=True))
    assert removed == 1
    left = idx.ids_for_chunk_uids("doc.txt", ["uid-keep"], use_payload=True)
    assert len(left) == 1
    assert idx.ids_for_source("doc.txt", use_payload=True) == left


def test_i18n_tr_en():
    set_language("tr")
    assert get_language() == "tr"
    assert "Asistan" in t("app_title")
    assert t("login_btn") == "Giriş yap"
    set_language("en")
    assert t("login_btn") == "Sign in"
    assert "Assistant" in t("app_title")
    assert t("no_answer").startswith("This information")
    set_language("tr")
