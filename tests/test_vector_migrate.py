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


def test_dual_write_lag_report_ok():
    primary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    secondary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    dual = DualWriteIndex(primary, secondary, secondary_backend="faiss")
    vecs = np.eye(1, 3, dtype=np.float32)
    dual.add(vecs, ["hello world text"], [_meta("a.txt", 0, "u1")])
    from rag.store import dual_write_lag_report

    report = dual_write_lag_report(dual)
    assert report["dual_write"] is True
    assert report["lag"] == 0
    assert report["ok"] is True
    assert report["primary_size"] == 1
    assert report["secondary_size"] == 1


def test_dual_write_lag_when_secondary_behind():
    primary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    secondary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    dual = DualWriteIndex(primary, secondary, secondary_backend="faiss")
    vecs = np.eye(1, 3, dtype=np.float32)
    dual.add(vecs, ["hello world text"], [_meta("a.txt", 0, "u1")])
    # secondary'yi manuel olarak geride bırak
    dual.secondary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    from rag.store import dual_write_lag_report

    report = dual_write_lag_report(dual)
    assert report["lag"] == 1
    assert report["ok"] is False
    assert "a.txt" in report["missing_sources"]


def test_dual_write_catch_up_replays_missing_sources(monkeypatch, tmp_path):
    from rag.store import dual_write_catch_up, dual_write_lag_report

    metrics = []

    def fake_record(kind, values=None, **kwargs):
        metrics.append({"kind": kind, "values": values or {}})

    monkeypatch.setattr("rag.metrics.record_metric", fake_record)
    primary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    secondary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    dual = DualWriteIndex(primary, secondary, secondary_backend="faiss")
    vecs = np.eye(2, 3, dtype=np.float32)
    dual.add(
        vecs,
        ["hello world text", "another chunk here"],
        [_meta("a.txt", 0, "u0"), _meta("b.txt", 0, "u1")],
    )
    dual.secondary = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    assert dual_write_lag_report(dual)["ok"] is False
    report = dual_write_catch_up(dual)
    assert report["ok"] is True
    assert set(report["fixed_sources"]) == {"a.txt", "b.txt"}
    assert report["copied_chunks"] == 2
    assert report["lag_after"]["ok"] is True
    assert dual.secondary.size == 2
    assert sorted(dual.secondary.list_sources()) == ["a.txt", "b.txt"]
    kinds = [m["kind"] for m in metrics]
    assert "vector_dual_write_catch_up" in kinds
    assert any(m["values"].get("result") == "fixed" for m in metrics)
    assert any(m["values"].get("result") == "done" for m in metrics)


def test_write_cutover_env(tmp_path):
    from rag.store import write_cutover_env

    path = write_cutover_env(
        target_backend="qdrant",
        path=str(tmp_path / "cutover.env"),
        clear_dual_write=True,
    )
    text = open(path, encoding="utf-8").read()
    assert "RAG_VECTOR_BACKEND=qdrant" in text
    assert "RAG_VECTOR_DUAL_WRITE=" in text


def test_migrate_vector_cutover_cli(tmp_path, monkeypatch):
    from rag.cli import build_parser

    monkeypatch.setattr("rag.cli.INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr("rag.cli.DOCSTORE_PATH", str(tmp_path / "docstore.json"))
    monkeypatch.setattr("rag.cli.METADATA_DIR", str(tmp_path))
    # boş faiss index oluştur
    idx = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    idx.save(str(tmp_path / "faiss.index"), str(tmp_path / "docstore.json"))
    parser = build_parser()
    out = str(tmp_path / "out.env")
    args = parser.parse_args(
        ["migrate-vector", "--cutover", "--target", "qdrant", "--write-env", out]
    )
    rc = args.func(args)
    assert rc == 0
    assert open(out, encoding="utf-8").read().count("RAG_VECTOR_BACKEND=qdrant") == 1


def test_migrate_vector_lag_report_cli(tmp_path, monkeypatch, capsys):
    from rag.cli import build_parser

    monkeypatch.setattr("rag.cli.INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr("rag.cli.DOCSTORE_PATH", str(tmp_path / "docstore.json"))
    idx = create_index(dim=3, embedding_model="m", backend="faiss", dual_write="")
    idx.save(str(tmp_path / "faiss.index"), str(tmp_path / "docstore.json"))
    parser = build_parser()
    args = parser.parse_args(["migrate-vector", "--lag-report"])
    rc = args.func(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dual_write" in out
