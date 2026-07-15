"""Doküman ingest / rebuild yardımcıları."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from app.config import (
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE_WORDS,
    SUPPORTED_EXTENSIONS,
)
from rag.chunking import chunk_pages
from rag.embed import Embedder
from rag.index import FaissIndex
from rag.readers import read_document


def is_supported_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def list_data_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    files = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and is_supported_file(path):
            files.append(path)
    return files


def ingest_path(
    path: str,
    index: FaissIndex,
    embedder: Embedder,
    *,
    replace_existing: bool = True,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
) -> Dict:
    """Tek bir dosyayı oku, chunk'la, embed et ve indekse ekle/yenile."""
    if not is_supported_file(path):
        raise ValueError(f"Desteklenmeyen dosya türü: {path}")

    source_name, pages = read_document(path)
    chunk_texts, metas = chunk_pages(
        source_file=source_name,
        pages=pages,
        chunk_size_words=chunk_size_words,
        overlap_ratio=overlap_ratio,
    )
    if not chunk_texts:
        return {
            "source_file": source_name,
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
        removed = index.replace_source(source_name, vecs, chunk_texts, metas)
    else:
        index.add(vecs, chunk_texts, metas)
    index.embedding_model = embedder.model_name

    return {
        "source_file": source_name,
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
    """Kaynağı diskten ve indeksten kaldırır."""
    removed_chunks = index.remove_source(source_file)
    path = os.path.join(data_dir, source_file)
    deleted_file = False
    if os.path.isfile(path):
        os.remove(path)
        deleted_file = True
    return {
        "source_file": source_file,
        "chunks_removed": removed_chunks,
        "file_deleted": deleted_file,
    }


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
            )
            reports.append(report)
        except Exception as exc:
            reports.append(
                {
                    "source_file": os.path.basename(path),
                    "chunks_added": 0,
                    "chunks_removed": 0,
                    "skipped": True,
                    "reason": str(exc),
                }
            )
    index.embedding_model = embedder.model_name
    return index, reports
