"""Vektör deposu factory: FAISS (varsayılan) veya Qdrant."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from app.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    VECTOR_BACKEND,
)
from rag.index import FaissIndex

VectorIndex = Union[FaissIndex, Any]


def vector_backend() -> str:
    return (VECTOR_BACKEND or "faiss").strip().lower()


def create_index(
    dim: int,
    embedding_model: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    collection: Optional[str] = None,
) -> VectorIndex:
    kind = (backend or vector_backend()).lower()
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
) -> VectorIndex:
    kind = (backend or vector_backend()).lower()
    if kind == "qdrant":
        from rag.qdrant_index import QdrantIndex

        if os.path.isfile(docstore_path):
            url = (QDRANT_URL or "").strip() or None
            path = None if url else ((QDRANT_PATH or "").strip() or None)
            return QdrantIndex.load(
                index_path,
                docstore_path,
                url=url,
                path=path,
                api_key=(QDRANT_API_KEY or "").strip() or None,
                collection=QDRANT_COLLECTION,
            )
        return create_index(
            dim=dim or 384,
            embedding_model=embedding_model,
            backend="qdrant",
        )

    if os.path.exists(index_path) and os.path.exists(docstore_path):
        return FaissIndex.load(index_path, docstore_path)
    return FaissIndex(dim=dim or 384, embedding_model=embedding_model)
