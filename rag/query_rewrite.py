"""Sorgu yeniden yazma ve HyDE (Hypothetical Document Embeddings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

GenerateFn = Callable[[str], str]

HYDE_PROMPT = """Kullanıcının sorusuna yanıt veren kısa bir paragraf yaz.
Bu paragraf retrieval için hipotetik bir geçittir; gerçekçi üslup kullan.
Yalnızca paragrafı yaz, başka açıklama ekleme.

Soru: {question}

Paragraf:"""

EXPAND_PROMPT = """Aşağıdaki soruyu aynı anlamı koruyarak 2 alternatif ifadeyle yeniden yaz.
Her satıra yalnızca bir ifade yaz. Numara veya madde işareti kullanma.

Soru: {question}

Alternatifler:"""


@dataclass
class RewriteResult:
    original: str
    mode: str
    retrieval_texts: List[str] = field(default_factory=list)
    bm25_query: str = ""
    hypothetical: Optional[str] = None
    expansions: List[str] = field(default_factory=list)


def _clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-•*0123456789.). ").strip()
        if line:
            out.append(line)
    return out


def generate_hyde_passage(question: str, generate_fn: GenerateFn) -> str:
    raw = (generate_fn(HYDE_PROMPT.format(question=question)) or "").strip()
    # ilk paragrafı al
    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return parts[0] if parts else raw


def expand_query(question: str, generate_fn: GenerateFn, *, n: int = 2) -> List[str]:
    raw = generate_fn(EXPAND_PROMPT.format(question=question)) or ""
    alts = [a for a in _clean_lines(raw) if a.lower() != question.lower()]
    return alts[:n]


def rewrite_query(
    question: str,
    *,
    mode: str = "none",
    generate_fn: Optional[GenerateFn] = None,
) -> RewriteResult:
    """mode: none | hyde | expand | hyde+expand"""
    q = (question or "").strip()
    mode = (mode or "none").strip().lower()
    result = RewriteResult(original=q, mode=mode, bm25_query=q, retrieval_texts=[q])
    if not q or mode in {"", "none", "off", "0"}:
        return result
    if generate_fn is None:
        return result

    if "hyde" in mode:
        hypo = generate_hyde_passage(q, generate_fn)
        if hypo:
            result.hypothetical = hypo
            result.retrieval_texts = [hypo]

    if "expand" in mode:
        alts = expand_query(q, generate_fn)
        result.expansions = alts
        # BM25 için orijinal + genişletmeler
        result.bm25_query = " ".join([q] + alts)
        if "hyde" not in mode:
            result.retrieval_texts = [q] + alts
        elif result.hypothetical:
            # HyDE + expand: hipotetik + orijinal + expand
            result.retrieval_texts = [result.hypothetical, q] + alts

    return result


def embed_rewrite(
    embedder,
    rewrite: RewriteResult,
) -> np.ndarray:
    """Retrieval metinlerinin ortalama embedding'ini döndürür (1, dim)."""
    texts = rewrite.retrieval_texts or [rewrite.original]
    vecs = embedder.encode(texts)
    if vecs.ndim == 1:
        return vecs.reshape(1, -1)
    mean = vecs.mean(axis=0, keepdims=True).astype(np.float32)
    # L2 normalize (cosine/IP için)
    norm = np.linalg.norm(mean, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return mean / norm
