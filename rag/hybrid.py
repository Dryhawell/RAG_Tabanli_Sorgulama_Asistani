"""Basit BM25 + dense (vektör) hibrit retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rag.types import RetrievedChunk

_TOKEN_RE = re.compile(r"[\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: Dict[int, List[str]] = {}
        self.doc_len: Dict[int, int] = {}
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self.n_docs = 0

    def clear(self):
        self.doc_tokens.clear()
        self.doc_len.clear()
        self.df = Counter()
        self.avgdl = 0.0
        self.n_docs = 0

    def _recompute_stats(self):
        self.df = Counter()
        total_len = 0
        for tokens in self.doc_tokens.values():
            total_len += len(tokens)
            for term in set(tokens):
                self.df[term] += 1
        self.n_docs = len(self.doc_tokens)
        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0

    def add(self, doc_id: int, text: str):
        tokens = tokenize(text)
        # eski varsa df yeniden kurulacak
        self.doc_tokens[doc_id] = tokens
        self.doc_len[doc_id] = len(tokens)
        self._recompute_stats()

    def add_many(self, items: Sequence[Tuple[int, str]]):
        for doc_id, text in items:
            tokens = tokenize(text)
            self.doc_tokens[doc_id] = tokens
            self.doc_len[doc_id] = len(tokens)
        self._recompute_stats()

    def remove(self, doc_ids: Sequence[int]):
        changed = False
        for doc_id in doc_ids:
            if doc_id in self.doc_tokens:
                del self.doc_tokens[doc_id]
                self.doc_len.pop(doc_id, None)
                changed = True
        if changed:
            self._recompute_stats()

    def score(self, query: str) -> Dict[int, float]:
        q_tokens = tokenize(query)
        if not q_tokens or self.n_docs == 0:
            return {}
        scores: Dict[int, float] = {}
        for doc_id, tokens in self.doc_tokens.items():
            tf = Counter(tokens)
            dl = self.doc_len.get(doc_id, 0)
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                # idf (BM25+ soft)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
            if s:
                scores[doc_id] = s
        return scores


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse_scores(
    dense: Dict[int, float],
    sparse: Dict[int, float],
    alpha: float = 0.65,
) -> Dict[int, float]:
    """alpha * dense + (1-alpha) * sparse; skorlar min-max normalize edilir."""
    dense_n = _normalize_scores(dense)
    sparse_n = _normalize_scores(sparse)
    keys = set(dense_n) | set(sparse_n)
    out: Dict[int, float] = {}
    for k in keys:
        out[k] = alpha * dense_n.get(k, 0.0) + (1.0 - alpha) * sparse_n.get(k, 0.0)
    return out


def hybrid_search(
    index,
    query_vec: np.ndarray,
    query_text: str,
    bm25: BM25Index,
    top_k: int = 6,
    alpha: float = 0.65,
    candidate_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """FAISS adayları + BM25 skorlarını birleştirerek nihai sıralama üretir."""
    if index.size == 0:
        return []

    cand = candidate_k or max(top_k * 4, top_k)
    dense_hits = index.search(query_vec, top_k=min(cand, index.size))
    dense_scores = {h.chunk_id: h.score for h in dense_hits}

    sparse_scores = bm25.score(query_text)
    # BM25 adaylarını da ekle (docstore üzerinden)
    fused = fuse_scores(dense_scores, sparse_scores, alpha=alpha)
    if not fused:
        return dense_hits[:top_k]

    ranked_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]
    # Metadata/text için önce dense hit map, yoksa index store
    by_id = {h.chunk_id: h for h in dense_hits}
    out: List[RetrievedChunk] = []
    for cid in ranked_ids:
        if cid in by_id:
            hit = by_id[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(fused[cid]),
                    text=hit.text,
                    metadata=hit.metadata,
                )
            )
            continue
        meta = index._id_to_meta.get(cid)
        text = index._id_to_text.get(cid)
        if meta is None or text is None:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                score=float(fused[cid]),
                text=text,
                metadata=meta,
            )
        )
    return out


def build_bm25_from_index(index) -> BM25Index:
    bm25 = BM25Index()
    bm25.add_many([(cid, text) for cid, text in index._id_to_text.items()])
    return bm25
