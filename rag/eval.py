"""Retrieval eval / regression yardımcıları."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Union

from rag.chunking import chunk_pages
from rag.hybrid import BM25Index, build_bm25_from_index
from rag.index import FaissIndex
from rag.readers import read_document
from rag.retrieve import retrieve


@dataclass
class EvalCase:
    question: str
    # Beklenen kaynak dosya adı (opsiyonel)
    expected_source: Optional[str] = None
    # Yanıt dokümanda olmamalıysa True
    expect_no_answer: bool = False
    id: Optional[str] = None


@dataclass
class EvalResult:
    question: str
    passed: bool
    gate_score: float
    top_sources: List[str]
    reason: str
    case_id: Optional[str] = None


def load_cases(path: str) -> List[EvalCase]:
    """JSON eval setini yükler.

    Desteklenen format:
    [
      {"id": "...", "question": "...", "expected_source": "a.txt", "expect_no_answer": false},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Eval seti bir JSON listesi olmalıdır")

    cases: List[EvalCase] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        if not q:
            continue
        cases.append(
            EvalCase(
                id=item.get("id"),
                question=q,
                expected_source=item.get("expected_source") or None,
                expect_no_answer=bool(item.get("expect_no_answer", False)),
            )
        )
    return cases


def build_index_from_fixture_dir(fixture_dir: str, embedder) -> FaissIndex:
    """evals/fixtures benzeri klasörden küçük bir indeks kurar."""
    index = FaissIndex(dim=embedder.dim, embedding_model=getattr(embedder, "model_name", None))
    if not os.path.isdir(fixture_dir):
        raise FileNotFoundError(f"Fixture klasörü yok: {fixture_dir}")

    for name in sorted(os.listdir(fixture_dir)):
        path = os.path.join(fixture_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in {".txt", ".pdf"}:
            continue
        source_name, pages = read_document(path, enable_ocr=False, enable_layout=False)
        texts, metas = chunk_pages(
            source_file=source_name,
            pages=pages,
            chunk_size_words=200,
            overlap_ratio=0.1,
            min_chunk_words=20,
        )
        if not texts:
            continue
        vecs = embedder.encode(texts)
        index.add(vecs, texts, metas)
    return index


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
    use_reranker: bool = False,
    reranker=None,
) -> List[EvalResult]:
    if bm25 is None:
        bm25 = build_bm25_from_index(index)

    results: List[EvalResult] = []
    for case in cases:
        qvec = embedder.encode([case.question])
        hits, gate = retrieve(
            index,
            qvec,
            case.question,
            bm25=bm25,
            top_k=top_k,
            use_hybrid=use_hybrid,
            hybrid_alpha=alpha,
            use_reranker=use_reranker,
            reranker=reranker,
            threshold=threshold,
        )

        top_sources = [h.metadata.source_file for h in hits]
        answered = bool(hits) and gate >= threshold

        if case.expect_no_answer:
            # Negatif sorular: dense skor eşiğin altında kalmalı
            passed = gate < threshold
            reason = (
                f"beklenen: yok | gate={gate:.3f}"
                + (" | OK" if passed else " | skor yüksek (yanlış pozitif riski)")
            )
        elif case.expected_source:
            # Pozitif: hit@k (kaynak top listede) — retrieval regression metriği
            matched = any(
                s == case.expected_source or os.path.basename(s) == case.expected_source
                for s in top_sources
            )
            passed = matched
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
                case_id=case.id,
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


def results_to_dict(results: Sequence[EvalResult], summary: Optional[dict] = None) -> dict:
    return {
        "summary": summary or summarize(results),
        "results": [asdict(r) for r in results],
    }


def run_regression(
    fixture_dir: str,
    cases_path: str,
    embedder,
    *,
    top_k: int = 4,
    threshold: float = 0.30,
    use_hybrid: bool = True,
    alpha: float = 0.55,
    use_reranker: bool = False,
    reranker=None,
    min_accuracy: float = 1.0,
) -> dict:
    """Fixture + cases ile regression çalıştırır; rapor dict döner."""
    cases = load_cases(cases_path)
    index = build_index_from_fixture_dir(fixture_dir, embedder)
    if index.size == 0:
        raise RuntimeError(f"Fixture indeks boş: {fixture_dir}")
    results = evaluate_cases(
        index,
        embedder,
        cases,
        top_k=top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        alpha=alpha,
        use_reranker=use_reranker,
        reranker=reranker,
    )
    summary = summarize(results)
    summary["min_accuracy"] = min_accuracy
    summary["ok"] = summary["accuracy"] + 1e-9 >= min_accuracy
    return results_to_dict(results, summary)
