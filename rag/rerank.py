"""Aday chunk'ları sorguya göre yeniden sıralama (cross-encoder veya lexikal yedek)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from app.config import DEFAULT_RERANKER_MODEL
from rag.hybrid import tokenize
from rag.types import RetrievedChunk


class LexicalReranker:
    """Model indirmeden çalışan basit örtüşme tabanlı reranker (test/fallback)."""

    def __init__(self, model_name: str = "lexical"):
        self.model_name = model_name

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(passages)
        scores = []
        for p in passages:
            tokens = tokenize(p)
            if not tokens:
                scores.append(0.0)
                continue
            overlap = sum(1 for t in tokens if t in q)
            scores.append(overlap / (len(tokens) ** 0.5))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        raw = self.model.predict(pairs)
        return [float(x) for x in raw]


def get_reranker(model_name: Optional[str] = None, prefer_cross_encoder: bool = True):
    name = model_name or DEFAULT_RERANKER_MODEL
    if not prefer_cross_encoder or name == "lexical":
        return LexicalReranker(model_name="lexical")
    try:
        return CrossEncoderReranker(model_name=name)
    except Exception:
        return LexicalReranker(model_name="lexical")


def rerank_chunks(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_k: int,
    reranker=None,
) -> List[RetrievedChunk]:
    """Chunk listesini yeniden skorlayıp top_k döndürür."""
    if not chunks:
        return []
    engine = reranker or LexicalReranker()
    scores = engine.score(query, [c.text for c in chunks])
    ranked = sorted(
        zip(chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]
    out: List[RetrievedChunk] = []
    for chunk, score in ranked:
        out.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=float(score),
                text=chunk.text,
                metadata=chunk.metadata,
            )
        )
    return out
