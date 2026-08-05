"""Yanıt kalitesi için LLM-as-judge ve deterministik heuristic judge."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from rag.llm import LLMProvider, generate_answer

NO_ANSWER_PHRASE = "bu bilgi dokümanda bulunmamaktadır"

JUDGE_PROMPT = """Sen bir RAG yanıt denetçisisin. Yalnızca verilen bağlam ve yanıta bak.
JSON dışında hiçbir şey yazma.

Kurallar:
- Yanıt bağlamda olmayan iddialar içeriyorsa grounded=false.
- Yanıt "Bu bilgi dokümanda bulunmamaktadır" ise ve bağlam boş/yetersizse grounded=true.
- score: 0.0–1.0 arasında bağlılık skoru.

Çıktı formatı:
{{"grounded": true/false, "score": 0.0, "reason": "kısa gerekçe"}}

Soru:
{question}

Bağlam:
{context}

Yanıt:
{answer}
"""


@dataclass
class JudgeCase:
    question: str
    answer: str
    contexts: List[str]
    expect_grounded: bool = True
    expect_no_answer: bool = False
    id: Optional[str] = None


@dataclass
class JudgeResult:
    case_id: Optional[str]
    question: str
    passed: bool
    grounded: bool
    score: float
    reason: str
    mode: str


def load_judge_cases(path: str) -> List[JudgeCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Judge seti bir JSON listesi olmalıdır")

    cases: List[JudgeCase] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue
        ctx = item.get("contexts") or item.get("context") or []
        if isinstance(ctx, str):
            ctx = [ctx]
        cases.append(
            JudgeCase(
                id=item.get("id"),
                question=q,
                answer=a,
                contexts=[str(c) for c in ctx if str(c).strip()],
                expect_grounded=bool(item.get("expect_grounded", True)),
                expect_no_answer=bool(item.get("expect_no_answer", False)),
            )
        )
    return cases


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]{3,}", (text or "").lower())}


def _overlap_ratio(answer: str, contexts: Sequence[str]) -> float:
    a_tok = _tokenize(answer)
    if not a_tok:
        return 0.0
    c_tok: set[str] = set()
    for c in contexts:
        c_tok |= _tokenize(c)
    if not c_tok:
        return 0.0
    return len(a_tok & c_tok) / max(1, len(a_tok))


def heuristic_judge(case: JudgeCase, *, min_overlap: float = 0.12) -> JudgeResult:
    """Model çağrısı olmadan bağlılık kontrolü (CI / hızlı regression)."""
    answer_l = (case.answer or "").lower()
    is_no = NO_ANSWER_PHRASE in answer_l

    if case.expect_no_answer:
        passed = is_no
        return JudgeResult(
            case_id=case.id,
            question=case.question,
            passed=passed,
            grounded=is_no,
            score=1.0 if is_no else 0.0,
            reason="no-answer bekleniyordu" if passed else "no-answer ifadesi yok",
            mode="heuristic",
        )

    if is_no:
        # Pozitif soruda no-answer genelde grounded=false beklentisiyle çakışır
        grounded = False
        passed = grounded == case.expect_grounded
        return JudgeResult(
            case_id=case.id,
            question=case.question,
            passed=passed,
            grounded=grounded,
            score=0.0,
            reason="yanıt no-answer; bağlamlı yanıt bekleniyordu",
            mode="heuristic",
        )

    overlap = _overlap_ratio(case.answer, case.contexts)
    grounded = overlap >= min_overlap and bool(case.contexts)
    passed = grounded == case.expect_grounded
    return JudgeResult(
        case_id=case.id,
        question=case.question,
        passed=passed,
        grounded=grounded,
        score=round(overlap, 4),
        reason=f"token overlap={overlap:.2%} (eşik={min_overlap:.0%})",
        mode="heuristic",
    )


def _parse_judge_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # ```json ... ``` sarmalıysa çıkar
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Judge çıktısı obje değil")
    return data


def llm_judge(
    case: JudgeCase,
    *,
    provider: LLMProvider = "ollama",
    model_name: str = "phi3:mini",
    generate_fn: Optional[Callable[[str], str]] = None,
) -> JudgeResult:
    """LLM ile grounded/score değerlendirmesi. generate_fn test için enjekte edilebilir."""
    context = "\n---\n".join(case.contexts) if case.contexts else "(boş)"
    prompt = JUDGE_PROMPT.format(
        question=case.question,
        context=context,
        answer=case.answer,
    )
    if generate_fn is None:
        raw = generate_answer(provider=provider, model_name=model_name, prompt=prompt)
    else:
        raw = generate_fn(prompt)

    try:
        data = _parse_judge_json(raw)
        grounded = bool(data.get("grounded"))
        score = float(data.get("score", 1.0 if grounded else 0.0))
        reason = str(data.get("reason") or "llm-judge")
    except Exception as exc:
        return JudgeResult(
            case_id=case.id,
            question=case.question,
            passed=False,
            grounded=False,
            score=0.0,
            reason=f"judge parse hatası: {exc}",
            mode="llm",
        )

    if case.expect_no_answer:
        is_no = NO_ANSWER_PHRASE in case.answer.lower()
        passed = is_no and grounded
    else:
        passed = grounded == case.expect_grounded

    return JudgeResult(
        case_id=case.id,
        question=case.question,
        passed=passed,
        grounded=grounded,
        score=max(0.0, min(1.0, score)),
        reason=reason,
        mode="llm",
    )


def evaluate_judge_cases(
    cases: Sequence[JudgeCase],
    *,
    mode: str = "heuristic",
    provider: LLMProvider = "ollama",
    model_name: str = "phi3:mini",
    generate_fn: Optional[Callable[[str], str]] = None,
    min_overlap: float = 0.12,
) -> List[JudgeResult]:
    results: List[JudgeResult] = []
    for case in cases:
        if mode == "llm":
            results.append(
                llm_judge(
                    case,
                    provider=provider,
                    model_name=model_name,
                    generate_fn=generate_fn,
                )
            )
        else:
            results.append(heuristic_judge(case, min_overlap=min_overlap))
    return results


def summarize_judge(results: Sequence[JudgeResult]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = (passed / total) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": accuracy,
    }


def run_judge_file(
    cases_path: str,
    *,
    mode: str = "heuristic",
    provider: LLMProvider = "ollama",
    model_name: str = "phi3:mini",
    min_accuracy: float = 1.0,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    from rag.otel import set_span_attrs, start_span

    with start_span(
        "rag.judge.run",
        attributes={
            "rag.judge.mode": mode,
            "rag.judge.cases_path": cases_path,
            "rag.judge.min_accuracy": float(min_accuracy),
        },
    ) as span:
        cases = load_judge_cases(cases_path)
        set_span_attrs(span, {"rag.judge.case_count": len(cases)})
        results = evaluate_judge_cases(
            cases,
            mode=mode,
            provider=provider,
            model_name=model_name,
            generate_fn=generate_fn,
        )
        summary = summarize_judge(results)
        summary["min_accuracy"] = min_accuracy
        summary["ok"] = summary["accuracy"] >= min_accuracy
        summary["mode"] = mode
        set_span_attrs(
            span,
            {
                "rag.judge.accuracy": float(summary["accuracy"]),
                "rag.judge.ok": bool(summary["ok"]),
                "rag.judge.passed": int(summary["passed"]),
                "rag.judge.failed": int(summary["failed"]),
            },
        )
        return {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
