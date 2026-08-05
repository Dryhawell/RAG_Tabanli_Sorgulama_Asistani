"""Dual-write / vector store migration tests."""

import numpy as np
import pytest

from rag.store import DualWriteIndex, create_index, migrate_vector_store, vector_backend
from rag.types import ChunkMetadata


def _meta(name: str, cid: int = 0, uid: str | None = None) -> ChunkMetadata:
    return ChunkMetadata(
        source_file=name,
        chunk_id=cid,
        page_start=1,
        page_end=1,
        word_count=3,
        folder="",
        tags=[],
        chunk_uid=uid,
    )


def test_dual_write_index_forwards_add(monkeypatch):
    monkeypatch.delenv("RAG_VECTOR_DUAL_WRITE", raising=False)
    primary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    secondary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    dual = DualWriteIndex(primary, secondary, secondary_backend="faiss")
    vecs = np.eye(1, 3, dtype=np.float32)
    dual.add(vecs, ["hello world text"], [_meta("a.txt", 0, "u1")])
    assert dual.size == 1
    assert secondary.size == 1
    assert dual.list_sources() == ["a.txt"]


def test_create_index_dual_write_env(monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv("RAG_VECTOR_DUAL_WRITE", "qdrant")
    monkeypatch.setattr("rag.store.VECTOR_BACKEND", "faiss")
    monkeypatch.setattr("rag.store.QDRANT_URL", "")
    monkeypatch.setattr("rag.store.QDRANT_PATH", "")
    monkeypatch.setattr("rag.store.QDRANT_COLLECTION", "dual_test")
    idx = create_index(dim=4, embedding_model="m", backend="faiss")
    assert isinstance(idx, DualWriteIndex)
    vecs = np.eye(1, 4, dtype=np.float32)
    idx.add(vecs, ["alpha beta gamma"], [_meta("x.txt", 0, "uid-x")])
    assert idx.size == 1
    assert idx.secondary.size == 1


def test_migrate_faiss_to_qdrant(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    monkeypatch.setattr("rag.store.QDRANT_URL", "")
    monkeypatch.setattr("rag.store.QDRANT_PATH", "")
    monkeypatch.setattr("rag.store.QDRANT_COLLECTION", "mig_test")
    monkeypatch.delenv("RAG_VECTOR_DUAL_WRITE", raising=False)

    src = create_index(dim=4, embedding_model="m", backend="faiss", dual_write="")
    vecs = np.eye(2, 4, dtype=np.float32)
    src.add(
        vecs,
        ["one two three", "four five six"],
        [_meta("a.txt", 0, "u0"), _meta("b.txt", 1, "u1")],
    )
    idx_path = str(tmp_path / "faiss.index")
    doc_path = str(tmp_path / "docstore.json")
    src.save(idx_path, doc_path)

    report = migrate_vector_store(
        source_backend="faiss",
        target_backend="qdrant",
        index_path=idx_path,
        docstore_path=doc_path,
        verify=True,
        collection="mig_test",
    )
    assert report["copied"] == 2
    assert report["verify"]["sources_match"] is True
    assert report["verify"]["size_match"] is True
    assert report["ok"] is True
    assert vector_backend() in {"faiss", "qdrant"}
