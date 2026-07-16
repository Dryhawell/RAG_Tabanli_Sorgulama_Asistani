"""Uçtan uca retrieval: dense → hybrid → metadata filtre → rerank."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.config import (
    DEFAULT_TOP_K,
    HYBRID_ALPHA,
    NO_ANSWER_THRESHOLD,
    RERANK_CANDIDATES,
)
from rag.hybrid import BM25Index, hybrid_search
from rag.index import FaissIndex
from rag.meta_store import normalize_folder, normalize_tags
from rag.rerank import LexicalReranker, rerank_chunks
from rag.types import RetrievedChunk


def _matches_filters(
    chunk: RetrievedChunk,
    *,
    source_filter: Optional[Sequence[str]],
    folder_filter: Optional[Sequence[str]],
    tag_filter: Optional[Sequence[str]],
    tag_mode: str = "any",
) -> bool:
    meta = chunk.metadata
    if source_filter and meta.source_file not in set(source_filter):
        return False

    if folder_filter:
        allowed_folders = {normalize_folder(f) for f in folder_filter}
        folder = normalize_folder(meta.folder)
        if folder not in allowed_folders:
            return False

    if tag_filter:
        wanted = {t.lower() for t in normalize_tags(list(tag_filter))}
        have = {t.lower() for t in (meta.tags or [])}
        if not wanted:
            return True
        if tag_mode == "all":
            if not wanted.issubset(have):
                return False
        else:
            if have.isdisjoint(wanted):
                return False
    return True


def apply_metadata_filters(
    chunks: Sequence[RetrievedChunk],
    *,
    source_filter: Optional[Sequence[str]] = None,
    folder_filter: Optional[Sequence[str]] = None,
    tag_filter: Optional[Sequence[str]] = None,
    tag_mode: str = "any",
) -> List[RetrievedChunk]:
    return [
        c
        for c in chunks
        if _matches_filters(
            c,
            source_filter=source_filter,
            folder_filter=folder_filter,
            tag_filter=tag_filter,
            tag_mode=tag_mode,
        )
    ]


def retrieve(
    index: FaissIndex,
    query_vec: np.ndarray,
    query_text: str,
    *,
    bm25: Optional[BM25Index] = None,
    top_k: int = DEFAULT_TOP_K,
    use_hybrid: bool = True,
    hybrid_alpha: float = HYBRID_ALPHA,
    use_reranker: bool = False,
    reranker=None,
    candidate_k: Optional[int] = None,
    source_filter: Optional[Sequence[str]] = None,
    folder_filter: Optional[Sequence[str]] = None,
    tag_filter: Optional[Sequence[str]] = None,
    tag_mode: str = "any",
    threshold: float = NO_ANSWER_THRESHOLD,
) -> Tuple[List[RetrievedChunk], float]:
    """Döner: (final_chunks, gate_score).

    gate_score, no-answer eşiği için dense/hybrid skorlarının max'ıdır
    (rerank skoru farklı ölçekte olabileceği için gate'te kullanılmaz).
    """
    if index.size == 0:
        return [], 0.0

    cand = candidate_k or max(top_k * 3, RERANK_CANDIDATES if use_reranker else top_k)
    # filtre varsa daha fazla aday çek
    if source_filter or folder_filter or tag_filter:
        cand = max(cand, min(index.size, top_k * 8))
    cand = min(cand, index.size)

    dense_hits = index.search(query_vec, top_k=cand)
    if use_hybrid and bm25 is not None:
        hits = hybrid_search(
            index,
            query_vec,
            query_text,
            bm25,
            top_k=cand,
            alpha=hybrid_alpha,
        )
    else:
        hits = dense_hits

    filt_kwargs = dict(
        source_filter=source_filter or None,
        folder_filter=folder_filter or None,
        tag_filter=tag_filter or None,
        tag_mode=tag_mode,
    )
    if source_filter or folder_filter or tag_filter:
        hits = apply_metadata_filters(hits, **filt_kwargs)
        dense_hits = apply_metadata_filters(dense_hits, **filt_kwargs)

    # No-answer eşiği için yalnızca dense (cosine) skoru kullan.
    # Hybrid füzyon min-max normalize edildiği için her zaman ~1 üretebilir.
    gate = dense_hits[0].score if dense_hits else 0.0

    if not hits or gate < threshold:
        return hits[:top_k], gate

    if use_reranker:
        engine = reranker or LexicalReranker()
        hits = rerank_chunks(query_text, hits, top_k=top_k, reranker=engine)
    else:
        hits = hits[:top_k]

    return hits, gate
