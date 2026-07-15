"""Uçtan uca retrieval: dense → hybrid → rerank."""

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
from rag.rerank import LexicalReranker, rerank_chunks
from rag.types import RetrievedChunk


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
    threshold: float = NO_ANSWER_THRESHOLD,
) -> Tuple[List[RetrievedChunk], float]:
    """Döner: (final_chunks, gate_score).

    gate_score, no-answer eşiği için dense/hybrid skorlarının max'ıdır
    (rerank skoru farklı ölçekte olabileceği için gate'te kullanılmaz).
    """
    if index.size == 0:
        return [], 0.0

    cand = candidate_k or max(top_k * 3, RERANK_CANDIDATES if use_reranker else top_k)
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
        gate = max(
            dense_hits[0].score if dense_hits else 0.0,
            hits[0].score if hits else 0.0,
        )
    else:
        hits = dense_hits
        gate = dense_hits[0].score if dense_hits else 0.0

    if source_filter:
        allowed = set(source_filter)
        hits = [h for h in hits if h.metadata.source_file in allowed]
        dense_filtered = [h for h in dense_hits if h.metadata.source_file in allowed]
        gate = dense_filtered[0].score if dense_filtered else (hits[0].score if hits else 0.0)
        if use_hybrid and hits:
            gate = max(gate, hits[0].score)

    if not hits or gate < threshold:
        return hits[:top_k], gate

    if use_reranker:
        engine = reranker or LexicalReranker()
        hits = rerank_chunks(query_text, hits, top_k=top_k, reranker=engine)
    else:
        hits = hits[:top_k]

    return hits, gate
