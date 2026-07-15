"""Doküman ingest / rebuild yardımcıları."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import (
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE_WORDS,
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
)
from rag.chunking import chunk_pages
from rag.embed import Embedder
from rag.index import FaissIndex
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


def list_data_files(data_dir: str) -> List[str]:
    """data_dir altındaki desteklenen dosyaları (alt klasörler dahil) listeler."""
    if not os.path.isdir(data_dir):
        return []
    files: List[str] = []
    for root, _dirs, names in os.walk(data_dir):
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
) -> Tuple[str, str, List[str]]:
    """(source_file, folder, tags) üretir.

    folder verilmezse göreli yolun üst klasöründen çıkarılır; sidecar varsa
    etiketler birleştirilir.
    """
    source_file = relative_source_name(path, data_dir)
    derived_folder = normalize_folder(os.path.dirname(source_file))
    folder_n = normalize_folder(folder) if folder is not None else derived_folder

    existing = get_source_meta(source_file)
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
) -> Dict:
    """Tek bir dosyayı oku, chunk'la, embed et ve indekse ekle/yenile."""
    if not is_supported_file(path):
        raise ValueError(f"Desteklenmeyen dosya türü: {path}")

    source_file, folder_n, tag_list = resolve_source_identity(
        path, data_dir, folder=folder, tags=tags
    )
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
        }

    if index.embedding_model and index.embedding_model != embedder.model_name:
        raise ValueError(
            f"Embedding modeli uyuşmuyor: indeks={index.embedding_model}, "
            f"seçili={embedder.model_name}. Lütfen indeksi yeniden oluşturun."
        )

    vecs = embedder.encode(chunk_texts)
    removed = 0
    if replace_existing:
        removed = index.replace_source(source_file, vecs, chunk_texts, metas)
    else:
        index.add(vecs, chunk_texts, metas)
    index.embedding_model = embedder.model_name
    upsert_source_meta(source_file, folder=folder_n, tags=tag_list)

    return {
        "source_file": source_file,
        "folder": folder_n,
        "tags": tag_list,
        "chunks_added": len(chunk_texts),
        "chunks_removed": removed,
        "skipped": False,
        "reason": None,
    }


def delete_source(
    source_file: str,
    index: FaissIndex,
    data_dir: str,
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
    delete_source_meta(source_file)
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
) -> Tuple[FaissIndex, List[Dict]]:
    """data/ altındaki desteklenen dosyalardan indeksi sıfırdan kurar."""
    index = FaissIndex(dim=embedder.dim, embedding_model=embedder.model_name)
    reports: List[Dict] = []
    for path in list_data_files(data_dir):
        try:
            report = ingest_path(
                path,
                index,
                embedder,
                replace_existing=False,
                chunk_size_words=chunk_size_words,
                overlap_ratio=overlap_ratio,
                data_dir=data_dir,
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
