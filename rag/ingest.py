"""Doküman ingest / rebuild yardımcıları."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import (
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE_WORDS,
    DATA_DIR,
    METADATA_DIR,
    SUPPORTED_EXTENSIONS,
)
from rag.chunking import chunk_fingerprints, chunk_pages, diff_chunk_fingerprints
from rag.embed import Embedder
from rag.index import FaissIndex
from rag.store import create_index
from rag.meta_store import (
    delete_source_meta,
    get_source_meta,
    normalize_folder,
    normalize_tags,
    relative_source_name,
    upsert_source_meta,
)
from rag.readers import read_document


def is_supported_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def list_data_files(data_dir: str, *, skip_user_namespaces: bool = True) -> List[str]:
    """data_dir altındaki desteklenen dosyaları (alt klasörler dahil) listeler.

    skip_user_namespaces: paylaşımlı data/ altında `users/` kişisel alanlarını atlar.
    """
    if not os.path.isdir(data_dir):
        return []
    files: List[str] = []
    abs_root = os.path.abspath(data_dir)
    for root, dirs, names in os.walk(data_dir):
        if skip_user_namespaces and os.path.abspath(root) == abs_root and "users" in dirs:
            dirs.remove("users")
        for name in sorted(names):
            path = os.path.join(root, name)
            if os.path.isfile(path) and is_supported_file(path):
                files.append(path)
    files.sort()
    return files


def resolve_source_identity(
    path: str,
    data_dir: str = DATA_DIR,
    *,
    folder: Optional[str] = None,
    tags: Optional[Sequence[str] | str] = None,
    meta_path: Optional[str] = None,
) -> Tuple[str, str, List[str]]:
    """(source_file, folder, tags) üretir.

    folder verilmezse göreli yolun üst klasöründen çıkarılır; sidecar varsa
    etiketler birleştirilir.
    """
    source_file = relative_source_name(path, data_dir)
    derived_folder = normalize_folder(os.path.dirname(source_file))
    folder_n = normalize_folder(folder) if folder is not None else derived_folder

    existing = get_source_meta(source_file, path=meta_path)
    tag_list = normalize_tags(tags) if tags is not None else list(existing.get("tags") or [])
    if folder is None and existing.get("folder"):
        # Sidecar klasör bilgisini koru (dosya kökteyse)
        folder_n = existing["folder"] or folder_n
    return source_file, folder_n, tag_list


def ingest_path(
    path: str,
    index: FaissIndex,
    embedder: Embedder,
    *,
    replace_existing: bool = True,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    data_dir: str = DATA_DIR,
    folder: Optional[str] = None,
    tags: Optional[Sequence[str] | str] = None,
    meta_path: Optional[str] = None,
) -> Dict:
    """Tek bir dosyayı oku, chunk'la, embed et ve indekse ekle/yenile."""
    from rag.otel import set_span_attrs, start_span

    with start_span(
        "rag.ingest_path",
        attributes={"rag.source_path": os.path.basename(path)},
    ) as span:
        if not is_supported_file(path):
            raise ValueError(f"Desteklenmeyen dosya türü: {path}")

        source_file, folder_n, tag_list = resolve_source_identity(
            path, data_dir, folder=folder, tags=tags, meta_path=meta_path
        )
        set_span_attrs(span, {"rag.source_file": source_file})
        _, pages = read_document(path)
        chunk_texts, metas = chunk_pages(
            source_file=source_file,
            pages=pages,
            chunk_size_words=chunk_size_words,
            overlap_ratio=overlap_ratio,
            folder=folder_n,
            tags=tag_list,
        )
        if not chunk_texts:
            return {
                "source_file": source_file,
                "folder": folder_n,
                "tags": tag_list,
                "chunks_added": 0,
                "chunks_removed": 0,
                "skipped": True,
                "reason": "Boş veya parçalanabilir metin yok",
                "chunk_uids": [],
            }

        if index.embedding_model and index.embedding_model != embedder.model_name:
            raise ValueError(
                f"Embedding modeli uyuşmuyor: indeks={index.embedding_model}, "
                f"seçili={embedder.model_name}. Lütfen indeksi yeniden oluşturun."
            )

        uids = chunk_fingerprints(chunk_texts, source_file=source_file)
        vecs = embedder.encode(chunk_texts)
        removed = 0
        if replace_existing:
            removed = index.replace_source(source_file, vecs, chunk_texts, metas)
        else:
            index.add(vecs, chunk_texts, metas)
        index.embedding_model = embedder.model_name
        upsert_source_meta(source_file, folder=folder_n, tags=tag_list, path=meta_path)

        result = {
            "source_file": source_file,
            "folder": folder_n,
            "tags": tag_list,
            "chunks_added": len(chunk_texts),
            "chunks_removed": removed,
            "skipped": False,
            "reason": None,
            "chunk_uids": uids,
            "selective": False,
        }
        set_span_attrs(span, {"rag.chunks_added": len(chunk_texts)})
        return result


def _source_has_chunk_uids(index: FaissIndex, source_file: str) -> bool:
    ids = index.ids_for_source(source_file)
    if not ids:
        return True
    for cid in ids:
        meta = getattr(index, "_id_to_meta", {}).get(cid)
        if meta is None or not getattr(meta, "chunk_uid", None):
            return False
    return True


def apply_selective_chunk_update(
    index: FaissIndex,
    embedder: Embedder,
    *,
    source_file: str,
    chunk_texts: List[str],
    metas: List,
    previous_uids: Sequence[str],
    folder: str = "",
    tags: Optional[Sequence[str]] = None,
    meta_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Sadece eklenen/silinen chunk UID'lerini encode/remove eder."""
    from rag.otel import set_span_attrs, start_span

    with start_span(
        "rag.ingest_selective",
        attributes={"rag.source_file": source_file},
    ) as span:
        cur_uids = chunk_fingerprints(chunk_texts, source_file=source_file)
        for m, uid in zip(metas, cur_uids):
            if getattr(m, "chunk_uid", None) != uid:
                try:
                    m.chunk_uid = uid
                except Exception:
                    pass
        cdiff = diff_chunk_fingerprints(previous_uids, cur_uids)
        prev_set = set(previous_uids or [])
        cur_set = set(cur_uids)
        removed_uids = prev_set - cur_set
        added_uids = cur_set - prev_set

        if not _source_has_chunk_uids(index, source_file):
            # Legacy kaynak → full replace
            vecs = embedder.encode(chunk_texts) if chunk_texts else __import__("numpy").zeros((0, embedder.dim), dtype="float32")
            removed = index.replace_source(source_file, vecs, chunk_texts, metas)
            upsert_source_meta(source_file, folder=folder, tags=list(tags or []), path=meta_path)
            out = {
                "source_file": source_file,
                "chunks_added": len(chunk_texts),
                "chunks_removed": removed,
                "skipped": False,
                "selective": False,
                "reason": "legacy_full_replace",
                "chunk_uids": cur_uids,
                "chunk_diff": cdiff,
            }
            set_span_attrs(span, {"rag.chunks_added": len(chunk_texts), "rag.selective": False})
            return out

        removed_ids = index.ids_for_chunk_uids(source_file, removed_uids) if removed_uids else []
        # Orphan UID'ler (manifest dışı ama indekste)
        for cid in list(index.ids_for_source(source_file)):
            meta = getattr(index, "_id_to_meta", {}).get(cid)
            uid = getattr(meta, "chunk_uid", None) if meta else None
            if uid and uid not in cur_set and cid not in removed_ids:
                removed_ids.append(cid)
        removed_n = index.remove_ids(removed_ids) if removed_ids else 0

        uid_to_i = {u: i for i, u in enumerate(cur_uids)}
        add_texts: List[str] = []
        add_metas: List = []
        for uid in sorted(added_uids, key=lambda u: uid_to_i.get(u, 0)):
            i = uid_to_i[uid]
            add_texts.append(chunk_texts[i])
            add_metas.append(metas[i])
        if add_texts:
            vecs = embedder.encode(add_texts)
            index.add(vecs, add_texts, add_metas)
        index.embedding_model = embedder.model_name
        upsert_source_meta(source_file, folder=folder, tags=list(tags or []), path=meta_path)
        out = {
            "source_file": source_file,
            "folder": folder,
            "tags": list(tags or []),
            "chunks_added": len(add_texts),
            "chunks_removed": removed_n,
            "skipped": False,
            "selective": True,
            "reason": None,
            "chunk_uids": cur_uids,
            "chunk_diff": cdiff,
        }
        set_span_attrs(
            span,
            {
                "rag.chunks_added": len(add_texts),
                "rag.chunks_removed": removed_n,
                "rag.selective": True,
            },
        )
        return out


def delete_source(
    source_file: str,
    index: FaissIndex,
    data_dir: str,
    *,
    meta_path: Optional[str] = None,
) -> Dict:
    """Kaynağı diskten, indeksten ve sidecar meta'dan kaldırır."""
    removed_chunks = index.remove_source(source_file)
    path = os.path.join(data_dir, source_file)
    deleted_file = False
    if os.path.isfile(path):
        os.remove(path)
        deleted_file = True
        # boş klasörleri temizlemeyi dene
        parent = os.path.dirname(path)
        while parent and os.path.abspath(parent).startswith(os.path.abspath(data_dir)):
            if parent == os.path.abspath(data_dir):
                break
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)
    delete_source_meta(source_file, path=meta_path)
    return {
        "source_file": source_file,
        "chunks_removed": removed_chunks,
        "file_deleted": deleted_file,
    }


def ensure_data_path(
    filename: str,
    data_dir: str = DATA_DIR,
    folder: str = "",
) -> str:
    """data_dir[/folder]/filename yolunu hazırlar."""
    folder_n = normalize_folder(folder)
    base = os.path.join(data_dir, folder_n) if folder_n else data_dir
    os.makedirs(base, exist_ok=True)
    # yalnızca dosya adını kullan
    safe_name = os.path.basename(filename)
    return os.path.join(base, safe_name)


def rebuild_from_data_dir(
    data_dir: str,
    embedder: Embedder,
    *,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    meta_path: Optional[str] = None,
    skip_user_namespaces: bool = True,
) -> Tuple[FaissIndex, List[Dict]]:
    """data/ altındaki desteklenen dosyalardan indeksi sıfırdan kurar."""
    from rag.otel import start_span

    with start_span("rag.rebuild_full", attributes={"rag.data_dir": data_dir}):
        index = create_index(dim=embedder.dim, embedding_model=embedder.model_name)
        reports: List[Dict] = []
        for path in list_data_files(data_dir, skip_user_namespaces=skip_user_namespaces):
            try:
                report = ingest_path(
                    path,
                    index,
                    embedder,
                    replace_existing=False,
                    chunk_size_words=chunk_size_words,
                    overlap_ratio=overlap_ratio,
                    data_dir=data_dir,
                    meta_path=meta_path,
                )
                reports.append(report)
            except Exception as exc:
                reports.append(
                    {
                        "source_file": relative_source_name(path, data_dir),
                        "chunks_added": 0,
                        "chunks_removed": 0,
                        "skipped": True,
                        "reason": str(exc),
                    }
                )
        index.embedding_model = embedder.model_name
        return index, reports


def ingest_manifest_path(base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "ingest_manifest.json")


def load_ingest_manifest(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or ingest_manifest_path()
    if not os.path.isfile(p):
        return {"version": 1, "sources": {}, "updated_at": None}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "sources": {}, "updated_at": None}
        sources = data.get("sources")
        if not isinstance(sources, dict):
            sources = {}
        return {
            "version": int(data.get("version") or 1),
            "sources": sources,
            "updated_at": data.get("updated_at"),
        }
    except Exception:
        return {"version": 1, "sources": {}, "updated_at": None}


def save_ingest_manifest(manifest: Dict[str, Any], path: Optional[str] = None) -> str:
    p = path or ingest_manifest_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    out = {
        "version": int(manifest.get("version") or 1),
        "sources": manifest.get("sources") or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return p


def file_content_fingerprint(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return {
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        "sha256": h.hexdigest(),
    }


def _source_fingerprint_entry(
    path: str,
    *,
    embedding_model: str,
    chunk_size_words: int,
    overlap_ratio: float,
) -> Dict[str, Any]:
    fp = file_content_fingerprint(path)
    return {
        **fp,
        "embedding_model": embedding_model,
        "chunk_size_words": int(chunk_size_words),
        "overlap_ratio": float(overlap_ratio),
    }


def _fingerprint_matches(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> bool:
    if not isinstance(prev, dict):
        return False
    keys = (
        "sha256",
        "size",
        "embedding_model",
        "chunk_size_words",
        "overlap_ratio",
    )
    for k in keys:
        if prev.get(k) != cur.get(k):
            return False
    return True


def rebuild_delta_from_data_dir(
    data_dir: str,
    embedder: Embedder,
    index: FaissIndex,
    *,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    meta_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    skip_user_namespaces: bool = True,
) -> Tuple[FaissIndex, List[Dict], Dict[str, Any]]:
    """Değişmeyen kaynakları atlayarak incremental rebuild.

    - Diskte yok ama indeks/manifest'te varsa kaldırır
    - Fingerprint aynıysa skip
    - Yeni/değişen dosyaları `ingest_path(replace_existing=True)` ile yeniler
    """
    from rag.otel import set_span_attrs, start_span

    man_path = manifest_path or ingest_manifest_path()
    with start_span(
        "rag.rebuild_delta",
        attributes={"rag.data_dir": data_dir, "rag.manifest": man_path},
    ) as span:
        manifest = load_ingest_manifest(man_path)
        sources_prev: Dict[str, Any] = dict(manifest.get("sources") or {})
        paths = list_data_files(data_dir, skip_user_namespaces=skip_user_namespaces)
        current: Dict[str, str] = {}
        for path in paths:
            src = relative_source_name(path, data_dir)
            current[src] = path

        reports: List[Dict] = []
        removed_sources: List[str] = []
        indexed = set()
        try:
            indexed.update(index.list_sources())
        except Exception:
            pass
        indexed.update(sources_prev.keys())
        for src in sorted(indexed):
            if src in current:
                continue
            chunks = 0
            try:
                chunks = index.remove_source(src)
            except Exception:
                chunks = 0
            delete_source_meta(src, path=meta_path)
            sources_prev.pop(src, None)
            removed_sources.append(src)
            reports.append(
                {
                    "source_file": src,
                    "chunks_added": 0,
                    "chunks_removed": chunks,
                    "skipped": True,
                    "action": "removed",
                    "reason": "source_missing_on_disk",
                }
            )

        unchanged = 0
        updated = 0
        chunk_skipped = 0
        for src, path in sorted(current.items()):
            cur_fp = _source_fingerprint_entry(
                path,
                embedding_model=embedder.model_name,
                chunk_size_words=chunk_size_words,
                overlap_ratio=overlap_ratio,
            )
            prev_fp = sources_prev.get(src)
            if _fingerprint_matches(prev_fp if isinstance(prev_fp, dict) else None, cur_fp):
                unchanged += 1
                reports.append(
                    {
                        "source_file": src,
                        "chunks_added": 0,
                        "chunks_removed": 0,
                        "skipped": True,
                        "action": "unchanged",
                        "reason": "fingerprint_match",
                    }
                )
                continue
            # Dosya değişti: chunk-level diff — aynı chunk seti ise encode atla
            prev_chunks = []
            if isinstance(prev_fp, dict):
                prev_chunks = list(prev_fp.get("chunk_uids") or [])
            try:
                _, pages = read_document(path)
                chunk_texts, metas = chunk_pages(
                    source_file=src,
                    pages=pages,
                    chunk_size_words=chunk_size_words,
                    overlap_ratio=overlap_ratio,
                )
                cur_uids = chunk_fingerprints(chunk_texts, source_file=src)
                cdiff = diff_chunk_fingerprints(prev_chunks, cur_uids)
                if prev_chunks and cdiff.get("identical"):
                    sources_prev[src] = {
                        **cur_fp,
                        "chunks": len(cur_uids),
                        "chunk_uids": cur_uids,
                    }
                    chunk_skipped += 1
                    reports.append(
                        {
                            "source_file": src,
                            "chunks_added": 0,
                            "chunks_removed": 0,
                            "skipped": True,
                            "action": "chunk_unchanged",
                            "reason": "chunk_fingerprints_match",
                            "chunk_diff": cdiff,
                        }
                    )
                    continue
                # Dosya değişti ve chunk seti farklı → seçici re-embed
                _sf, folder_n, tag_list = resolve_source_identity(
                    path, data_dir, meta_path=meta_path
                )
                for m in metas:
                    m.folder = folder_n
                    m.tags = list(tag_list)
                if prev_chunks and _source_has_chunk_uids(index, src):
                    report = apply_selective_chunk_update(
                        index,
                        embedder,
                        source_file=src,
                        chunk_texts=chunk_texts,
                        metas=metas,
                        previous_uids=prev_chunks,
                        folder=folder_n,
                        tags=tag_list,
                        meta_path=meta_path,
                    )
                else:
                    report = ingest_path(
                        path,
                        index,
                        embedder,
                        replace_existing=True,
                        chunk_size_words=chunk_size_words,
                        overlap_ratio=overlap_ratio,
                        data_dir=data_dir,
                        meta_path=meta_path,
                    )
                    report = dict(report)
                    report["chunk_diff"] = cdiff
                report = dict(report)
                report["action"] = "updated" if prev_fp else "added"
                report["chunk_diff"] = report.get("chunk_diff") or cdiff
                reports.append(report)
                if not report.get("skipped"):
                    sources_prev[src] = {
                        **cur_fp,
                        "chunks": len(list(report.get("chunk_uids") or cur_uids)),
                        "chunk_uids": list(report.get("chunk_uids") or cur_uids),
                    }
                    updated += 1
                elif prev_fp is None:
                    sources_prev[src] = {
                        **cur_fp,
                        "chunks": 0,
                        "chunk_uids": cur_uids,
                    }
            except Exception as exc:
                reports.append(
                    {
                        "source_file": src,
                        "chunks_added": 0,
                        "chunks_removed": 0,
                        "skipped": True,
                        "action": "error",
                        "reason": str(exc),
                    }
                )

        index.embedding_model = embedder.model_name
        manifest["sources"] = sources_prev
        save_ingest_manifest(manifest, man_path)
        summary = {
            "unchanged": unchanged,
            "updated": updated,
            "removed": len(removed_sources),
            "chunk_skipped": chunk_skipped,
            "removed_sources": removed_sources,
            "total_files": len(current),
            "manifest_path": man_path,
            "index_size": getattr(index, "size", None),
        }
        set_span_attrs(
            span,
            {
                "rag.unchanged": unchanged,
                "rag.updated": updated,
                "rag.removed": len(removed_sources),
                "rag.chunk_skipped": chunk_skipped,
            },
        )
        return index, reports, summary
