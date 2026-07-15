"""Basit retrieval/eval yardımcıları."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from rag.hybrid import BM25Index, hybrid_search
from rag.index import FaissIndex


@dataclass
class EvalCase:
    question: str
    # Beklenen kaynak dosya adı (opsiyonel)
    expected_source: Optional[str] = None
    # Yanıt dokümanda olmamalıysa True
    expect_no_answer: bool = False


@dataclass
class EvalResult:
    question: str
    passed: bool
    gate_score: float
    top_sources: List[str]
    reason: str


def evaluate_cases(
    index: FaissIndex,
    embedder,
    cases: Sequence[EvalCase],
    *,
    bm25: Optional[BM25Index] = None,
    top_k: int = 6,
    threshold: float = 0.30,
    use_hybrid: bool = True,
    alpha: float = 0.65,
) -> List[EvalResult]:
    if bm25 is None:
        from rag.hybrid import build_bm25_from_index

        bm25 = build_bm25_from_index(index)

    results: List[EvalResult] = []
    for case in cases:
        qvec = embedder.encode([case.question])
        dense = index.search(qvec, top_k=top_k)
        if use_hybrid:
            hits = hybrid_search(
                index, qvec, case.question, bm25, top_k=top_k, alpha=alpha
            )
            gate = max(
                dense[0].score if dense else 0.0,
                hits[0].score if hits else 0.0,
            )
        else:
            hits = dense
            gate = dense[0].score if dense else 0.0

        top_sources = [h.metadata.source_file for h in hits]
        answered = bool(hits) and gate >= threshold

        if case.expect_no_answer:
            passed = not answered
            reason = "beklenen: yok" + (" | model cevap üretir gibi" if answered else " | OK")
        elif case.expected_source:
            passed = answered and case.expected_source in top_sources
            reason = (
                f"beklenen kaynak={case.expected_source}; top={top_sources[:3]}; gate={gate:.3f}"
            )
        else:
            passed = answered
            reason = f"gate={gate:.3f}; top={top_sources[:3]}"

        results.append(
            EvalResult(
                question=case.question,
                passed=passed,
                gate_score=gate,
                top_sources=top_sources,
                reason=reason,
            )
        )
    return results


def summarize(results: Sequence[EvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": (passed / total) if total else 0.0,
    }
