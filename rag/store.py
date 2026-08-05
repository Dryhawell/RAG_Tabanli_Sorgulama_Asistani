"""Vektör deposu factory: FAISS (varsayılan) veya Qdrant (+ dual-write / migrate)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from app.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    VECTOR_BACKEND,
)
from rag.index import FaissIndex
from rag.types import ChunkMetadata, RetrievedChunk

VectorIndex = Union[FaissIndex, Any]


def vector_backend() -> str:
    return (VECTOR_BACKEND or "faiss").strip().lower()


def dual_write_backend() -> str:
    """İkincil backend (ör. RAG_VECTOR_DUAL_WRITE=qdrant). Boş = kapalı."""
    return (
        os.environ.get("RAG_VECTOR_DUAL_WRITE", "").strip()
        or os.environ.get("RAG_VECTOR_MIGRATION_TARGET", "").strip()
    ).lower()


def create_index(
    dim: int,
    embedding_model: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    collection: Optional[str] = None,
    dual_write: Optional[str] = None,
) -> VectorIndex:
    kind = (backend or vector_backend()).lower()
    primary = _create_single(
        dim,
        embedding_model,
        backend=kind,
        collection=collection,
    )
    secondary_kind = (dual_write if dual_write is not None else dual_write_backend()).lower()
    if secondary_kind and secondary_kind != kind:
        try:
            secondary = _create_single(
                dim,
                embedding_model,
                backend=secondary_kind,
                collection=collection,
            )
            return DualWriteIndex(primary, secondary, secondary_backend=secondary_kind)
        except Exception:
            return primary
    return primary


def _create_single(
    dim: int,
    embedding_model: Optional[str],
    *,
    backend: str,
    collection: Optional[str] = None,
) -> VectorIndex:
    kind = (backend or "faiss").lower()
    if kind == "qdrant":
        from rag.qdrant_index import QdrantIndex

        url = (QDRANT_URL or "").strip() or None
        path = None if url else ((QDRANT_PATH or "").strip() or None)
        return QdrantIndex(
            dim=dim,
            embedding_model=embedding_model,
            collection=collection or QDRANT_COLLECTION,
            url=url,
            path=path,
            api_key=(QDRANT_API_KEY or "").strip() or None,
        )
    return FaissIndex(dim=dim, embedding_model=embedding_model)


def load_index(
    index_path: str,
    docstore_path: str,
    *,
    backend: Optional[str] = None,
    dim: Optional[int] = None,
    embedding_model: Optional[str] = None,
    dual_write: Optional[str] = None,
) -> VectorIndex:
    kind = (backend or vector_backend()).lower()
    if kind == "qdrant":
        from rag.qdrant_index import QdrantIndex

        if os.path.isfile(docstore_path):
            url = (QDRANT_URL or "").strip() or None
            path = None if url else ((QDRANT_PATH or "").strip() or None)
            primary = QdrantIndex.load(
                index_path,
                docstore_path,
                url=url,
                path=path,
                api_key=(QDRANT_API_KEY or "").strip() or None,
                collection=QDRANT_COLLECTION,
            )
        else:
            primary = create_index(
                dim=dim or 384,
                embedding_model=embedding_model,
                backend="qdrant",
                dual_write="",
            )
    elif os.path.exists(index_path) and os.path.exists(docstore_path):
        primary = FaissIndex.load(index_path, docstore_path)
    else:
        primary = FaissIndex(dim=dim or 384, embedding_model=embedding_model)

    secondary_kind = (dual_write if dual_write is not None else dual_write_backend()).lower()
    if secondary_kind and secondary_kind != kind:
        try:
            secondary = _create_single(
                getattr(primary, "dim", dim or 384),
                getattr(primary, "embedding_model", embedding_model),
                backend=secondary_kind,
            )
            return DualWriteIndex(primary, secondary, secondary_backend=secondary_kind)
        except Exception:
            return primary
    return primary


class DualWriteIndex:
    """Primary üzerinden okur; yazmaları secondary'ye de iletir (best-effort)."""

    def __init__(
        self,
        primary: VectorIndex,
        secondary: VectorIndex,
        *,
        secondary_backend: str = "",
    ):
        self.primary = primary
        self.secondary = secondary
        self.secondary_backend = secondary_backend
        self._secondary_errors: List[str] = []

    @property
    def dim(self) -> int:
        return int(getattr(self.primary, "dim", 0))

    @property
    def embedding_model(self) -> Optional[str]:
        return getattr(self.primary, "embedding_model", None)

    @embedding_model.setter
    def embedding_model(self, value: Optional[str]) -> None:
        self.primary.embedding_model = value
        try:
            self.secondary.embedding_model = value
        except Exception:
            pass

    @property
    def size(self) -> int:
        return int(getattr(self.primary, "size", 0))

    def _sec(self, fn_name: str, *args, **kwargs) -> None:
        try:
            fn = getattr(self.secondary, fn_name)
            fn(*args, **kwargs)
        except Exception as exc:
            self._secondary_errors.append(f"{fn_name}:{type(exc).__name__}")

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[ChunkMetadata]):
        self.primary.add(embeddings, texts, metas)
        self._sec("add", embeddings, texts, metas)

    def remove_ids(self, ids: List[int]) -> int:
        removed = self.primary.remove_ids(ids)
        self._sec("remove_ids", ids)
        return removed

    def remove_source(self, source_file: str) -> int:
        removed = self.primary.remove_source(source_file)
        self._sec("remove_source", source_file)
        return removed

    def replace_source(
        self,
        source_file: str,
        embeddings: np.ndarray,
        texts: List[str],
        metas: List[ChunkMetadata],
    ) -> int:
        removed = self.primary.replace_source(source_file, embeddings, texts, metas)
        self._sec("replace_source", source_file, embeddings, texts, metas)
        return removed

    def ids_for_source(self, source_file: str) -> List[int]:
        return self.primary.ids_for_source(source_file)

    def ids_for_chunk_uids(self, source_file: str, uids: Sequence[str]):
        return self.primary.ids_for_chunk_uids(source_file, uids)

    def list_sources(self) -> List[str]:
        return self.primary.list_sources()

    def list_folders(self) -> List[str]:
        return self.primary.list_folders()

    def list_tags(self) -> List[str]:
        return self.primary.list_tags()

    def search(self, query: np.ndarray, top_k: int = 6) -> List[RetrievedChunk]:
        return self.primary.search(query, top_k=top_k)

    def save(self, index_path: str, docstore_path: str):
        self.primary.save(index_path, docstore_path)
        try:
            # secondary için ayrı sidecar (qdrant marker)
            sec_doc = docstore_path
            if sec_doc.endswith(".json"):
                sec_doc = sec_doc[:-5] + f".{self.secondary_backend or 'secondary'}.json"
            else:
                sec_doc = docstore_path + f".{self.secondary_backend or 'secondary'}"
            sec_idx = index_path + f".{self.secondary_backend or 'secondary'}"
            self.secondary.save(sec_idx, sec_doc)
        except Exception as exc:
            self._secondary_errors.append(f"save:{type(exc).__name__}")
        try:
            report_dual_write_lag(self)
        except Exception:
            pass

    def lag_report(self) -> Dict[str, Any]:
        return dual_write_lag_report(self)


def dual_write_lag_report(index: Any) -> Dict[str, Any]:
    """Primary vs secondary size/sources farkı + hata özeti."""
    if not isinstance(index, DualWriteIndex):
        return {
            "dual_write": False,
            "ok": True,
            "reason": "not_dual_write",
        }
    primary_size = int(getattr(index.primary, "size", 0) or 0)
    secondary_size = int(getattr(index.secondary, "size", 0) or 0)
    try:
        primary_sources = set(index.primary.list_sources())
    except Exception:
        primary_sources = set()
    try:
        secondary_sources = set(index.secondary.list_sources())
    except Exception:
        secondary_sources = set()
    lag = primary_size - secondary_size
    missing = sorted(primary_sources - secondary_sources)
    extra = sorted(secondary_sources - primary_sources)
    errors = list(getattr(index, "_secondary_errors", []) or [])
    ok = lag == 0 and not missing and not extra and not errors
    return {
        "dual_write": True,
        "ok": ok,
        "primary_backend": type(index.primary).__name__,
        "secondary_backend": index.secondary_backend or type(index.secondary).__name__,
        "primary_size": primary_size,
        "secondary_size": secondary_size,
        "lag": lag,
        "primary_sources": len(primary_sources),
        "secondary_sources": len(secondary_sources),
        "missing_sources": missing,
        "extra_sources": extra,
        "secondary_errors": errors[-20:],
        "secondary_error_count": len(errors),
    }


def report_dual_write_lag(index: Any) -> Dict[str, Any]:
    """Lag raporunu JSONL + Prometheus'a yazar."""
    report = dual_write_lag_report(index)
    if not report.get("dual_write"):
        return report
    try:
        from rag.metrics import record_metric

        record_metric(
            "vector_dual_write_lag",
            values={
                "lag": report.get("lag"),
                "primary_size": report.get("primary_size"),
                "secondary_size": report.get("secondary_size"),
                "secondary_error_count": report.get("secondary_error_count"),
                "ok": report.get("ok"),
                "secondary_backend": report.get("secondary_backend"),
                "missing_sources": len(report.get("missing_sources") or []),
            },
        )
    except Exception:
        pass
    return report


def write_cutover_env(
    *,
    target_backend: str,
    path: str,
    clear_dual_write: bool = True,
) -> str:
    """Cutover için önerilen env satırlarını dosyaya yazar."""
    target = (target_backend or "qdrant").strip().lower() or "qdrant"
    lines = [
        f"# RAG vector cutover — generated",
        f"RAG_VECTOR_BACKEND={target}",
    ]
    if clear_dual_write:
        lines.append("RAG_VECTOR_DUAL_WRITE=")
        lines.append("RAG_VECTOR_MIGRATION_TARGET=")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _sources_needing_catch_up(index: "DualWriteIndex") -> List[str]:
    lag = dual_write_lag_report(index)
    needed = list(lag.get("missing_sources") or [])
    try:
        primary_sources = list(index.primary.list_sources())
    except Exception:
        primary_sources = []
    for src in primary_sources:
        if src in needed:
            continue
        try:
            p_n = len(index.primary.ids_for_source(src))
        except Exception:
            continue
        try:
            s_n = len(index.secondary.ids_for_source(src))
        except Exception:
            s_n = -1
        if p_n != s_n:
            needed.append(src)
    return needed


def dual_write_catch_up(
    index: Any,
    *,
    sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Primary'deki eksik/uyumsuz kaynakları secondary'ye yeniden yazar."""
    if not isinstance(index, DualWriteIndex):
        return {"ok": False, "error": "not_dual_write", "dual_write": False}
    lag_before = dual_write_lag_report(index)
    to_fix = (
        [str(s) for s in sources if str(s).strip()]
        if sources is not None
        else _sources_needing_catch_up(index)
    )
    fixed: List[str] = []
    failed: List[Dict[str, Any]] = []
    copied_chunks = 0
    text_map = getattr(index.primary, "_id_to_text", {}) or {}
    meta_map = getattr(index.primary, "_id_to_meta", {}) or {}
    secondary_backend = index.secondary_backend or type(index.secondary).__name__
    total = len(to_fix)

    def _progress(*, result: str, source: str = "", chunks: int = 0) -> None:
        remaining = max(0, total - len(fixed) - len(failed))
        try:
            from rag.metrics import record_metric

            record_metric(
                "vector_dual_write_catch_up",
                values={
                    "result": result,
                    "source": source,
                    "copied_chunks": chunks,
                    "fixed": len(fixed),
                    "failed": len(failed),
                    "remaining": remaining,
                    "total": total,
                    "lag": lag_before.get("lag"),
                    "secondary_backend": secondary_backend,
                },
            )
        except Exception:
            pass

    _progress(result="start")

    for src in to_fix:
        try:
            ids = list(index.primary.ids_for_source(src))
        except Exception as exc:
            failed.append({"source": src, "error": f"ids:{type(exc).__name__}"})
            _progress(result="failed", source=src)
            continue
        texts: List[str] = []
        metas: List[Any] = []
        rows: List[np.ndarray] = []
        for cid in ids:
            text = text_map.get(cid)
            meta = meta_map.get(cid)
            vec = _reconstruct_vector(index.primary, int(cid))
            if text is None or meta is None or vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            texts.append(text)
            metas.append(meta)
            rows.append(arr)
        if not texts:
            failed.append({"source": src, "error": "empty_or_unreconstructable"})
            _progress(result="failed", source=src)
            continue
        emb = np.vstack(rows)
        try:
            index.secondary.replace_source(src, emb, texts, metas)
        except Exception:
            try:
                index.secondary.remove_source(src)
                index.secondary.add(emb, texts, metas)
            except Exception as exc:
                failed.append({"source": src, "error": type(exc).__name__})
                index._secondary_errors.append(f"catch_up:{type(exc).__name__}")
                _progress(result="failed", source=src)
                continue
        fixed.append(src)
        copied_chunks += len(texts)
        _progress(result="fixed", source=src, chunks=len(texts))

    lag_after = report_dual_write_lag(index)
    ok = bool(lag_after.get("ok")) and not failed
    _progress(result="done" if ok else "incomplete", chunks=copied_chunks)
    return {
        "ok": ok,
        "dual_write": True,
        "requested_sources": to_fix,
        "fixed_sources": fixed,
        "copied_chunks": copied_chunks,
        "failed": failed,
        "lag_before": lag_before,
        "lag_after": lag_after,
    }

def _reconstruct_vector(index: VectorIndex, cid: int) -> Optional[np.ndarray]:
    try:
        if hasattr(index, "idmap"):
            vec = index.idmap.reconstruct(int(cid))
            return np.asarray(vec, dtype=np.float32)
    except Exception:
        pass
    try:
        # Qdrant: retrieve with vectors
        client = getattr(index, "client", None)
        if client is None:
            return None
        points = client.retrieve(
            collection_name=index.collection,
            ids=[int(cid)],
            with_vectors=True,
            with_payload=False,
        )
        if not points:
            return None
        vec = points[0].vector
        if isinstance(vec, dict):
            vec = next(iter(vec.values()))
        return np.asarray(vec, dtype=np.float32)
    except Exception:
        return None


def migrate_vector_store(
    *,
    source_backend: str,
    target_backend: str,
    index_path: str,
    docstore_path: str,
    verify: bool = True,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Kaynak indeksi hedefe kopyalar (vektör reconstruct + add)."""
    src_kind = (source_backend or "faiss").strip().lower()
    dst_kind = (target_backend or "qdrant").strip().lower()
    if src_kind == dst_kind:
        return {"ok": False, "error": "same_backend"}

    src = load_index(
        index_path,
        docstore_path,
        backend=src_kind,
        dual_write="",
    )
    dst = create_index(
        dim=getattr(src, "dim", 384),
        embedding_model=getattr(src, "embedding_model", None),
        backend=dst_kind,
        collection=collection,
        dual_write="",
    )

    id_map = getattr(src, "_id_to_meta", {}) or {}
    text_map = getattr(src, "_id_to_text", {}) or {}
    copied = 0
    skipped = 0
    for cid, meta in sorted(id_map.items(), key=lambda x: int(x[0])):
        text = text_map.get(cid)
        vec = _reconstruct_vector(src, int(cid))
        if text is None or vec is None:
            skipped += 1
            continue
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        dst.add(vec, [text], [meta])
        copied += 1

    dst.embedding_model = getattr(src, "embedding_model", None)
    # hedef path
    if dst_kind == "qdrant":
        out_doc = docstore_path
        if out_doc.endswith(".json"):
            out_doc = out_doc[:-5] + ".qdrant.json"
        out_idx = index_path + ".qdrant"
    else:
        out_idx = index_path
        out_doc = docstore_path
    dst.save(out_idx, out_doc)

    report: Dict[str, Any] = {
        "ok": True,
        "source": src_kind,
        "target": dst_kind,
        "copied": copied,
        "skipped": skipped,
        "source_size": getattr(src, "size", None),
        "target_size": getattr(dst, "size", None),
        "index_path": out_idx,
        "docstore_path": out_doc,
    }
    if verify:
        src_sources = set(src.list_sources())
        dst_sources = set(dst.list_sources())
        report["verify"] = {
            "sources_match": src_sources == dst_sources,
            "source_count": len(src_sources),
            "target_count": len(dst_sources),
            "size_match": int(getattr(src, "size", -1)) == int(getattr(dst, "size", -2)),
        }
        report["ok"] = bool(
            report["verify"]["sources_match"] and report["verify"]["size_match"]
        )
    return report
