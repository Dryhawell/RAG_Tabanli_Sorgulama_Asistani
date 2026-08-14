"""Çok adımlı plan üretimi ve planlı agent çalıştırma."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from rag.agent import AgentResult, run_agent_loop, run_heuristic_tools, tools_context_block
from rag.memory import AgentMemory
from rag.tools import execute_tool, extract_final, parse_tool_calls

GenerateFn = Callable[[str], str]

PLAN_PROMPT = """Kullanıcı sorusunu 2-4 kısa adıma böl.
Yalnızca şu formatta yaz:

PLAN
1. ...
2. ...
END_PLAN

Soru: {question}
{memory}

Plan:"""

STEP_PROMPT = """Aşağıdaki plan adımını uygula. Gerekirse TOOL_CALL kullan, bitince FINAL yaz.

Soru: {question}
Bellek:
{memory}

Tüm plan:
{plan_text}

Şu anki adım ({step_i}/{step_n}): {step}

Önceki adım sonuçları:
{prior}

Yanıt (TOOL_CALL veya FINAL):
"""


@dataclass
class PlanResult:
    plan: List[str] = field(default_factory=list)
    step_results: List[str] = field(default_factory=list)
    answer: str = ""
    tool_traces: List[Dict] = field(default_factory=list)
    agent: Optional[AgentResult] = None


def parse_plan(text: str) -> List[str]:
    raw = text or ""
    m = re.search(r"PLAN\s*(.*?)\s*END_PLAN", raw, re.DOTALL | re.IGNORECASE)
    block = m.group(1) if m else raw
    steps = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\).\:\-]\s*", "", line)
        line = re.sub(r"^[\-\*]\s*", "", line)
        if line and not line.upper().startswith("PLAN"):
            steps.append(line)
    return steps[:6]


def make_plan(
    question: str,
    generate_fn: GenerateFn,
    *,
    memory: Optional[AgentMemory] = None,
) -> List[str]:
    mem = memory.context_block() if memory else ""
    raw = generate_fn(PLAN_PROMPT.format(question=question, memory=mem or "(yok)"))
    steps = parse_plan(raw or "")
    if not steps:
        # yedek minimal plan
        steps = [
            "Gerekli araçları çalıştır",
            "Doküman bağlamıyla yanıtı birleştir",
        ]
    return steps


def heuristic_plan(question: str) -> List[str]:
    q = (question or "").lower()
    steps = []
    if any(k in q for k in ("hesapla", "kaç eder", "+", "*", "/")):
        steps.append("Hesap makinesi ile sayısal işlemi yap")
    if any(k in q for k in ("bugün", "tarih", "gün")):
        steps.append("Takvim aracından tarihi al")
    if any(k in q for k in ("web", "araştır", "güncel")):
        steps.append("Web araması yap")
    if any(k in q for k in ("tablo", "karşılaştır", "özet")):
        steps.append("İlgili tablo/doküman parçalarını incele")
    steps.append("Sonuçları birleştirip yanıtla")
    # tekilleştir
    seen = set()
    out = []
    for s in steps:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def run_planned_agent(
    question: str,
    generate_fn: GenerateFn,
    *,
    memory: Optional[AgentMemory] = None,
    max_steps: int = 4,
    use_llm_plan: bool = True,
) -> PlanResult:
    """Plan üret → adım adım tool/FINAL → birleşik yanıt."""
    mem = memory or AgentMemory()
    if use_llm_plan:
        try:
            plan = make_plan(question, generate_fn, memory=mem)
        except Exception:
            plan = heuristic_plan(question)
    else:
        plan = heuristic_plan(question)

    result = PlanResult(plan=plan)
    prior_parts: List[str] = []
    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))

    for i, step in enumerate(plan[:max_steps]):
        prompt = STEP_PROMPT.format(
            question=question,
            memory=mem.context_block() or "(yok)",
            plan_text=plan_text,
            step_i=i + 1,
            step_n=len(plan),
            step=step,
            prior="\n".join(prior_parts) or "(yok)",
        )
        raw = (generate_fn(prompt) or "").strip()
        for name, args in parse_tool_calls(raw):
            out = execute_tool(name, args)
            result.tool_traces.append({"name": name, "args": args, "result": out, "step": step})
            prior_parts.append(f"[{step}] {name}: {out}")
        final = extract_final(raw)
        if final:
            result.step_results.append(final)
            prior_parts.append(f"[{step}] {final}")
        elif not parse_tool_calls(raw) and raw:
            result.step_results.append(raw)
            prior_parts.append(f"[{step}] {raw}")

    # son birleştirme
    if result.step_results:
        result.answer = result.step_results[-1]
    elif result.tool_traces:
        result.answer = tools_context_block(result.tool_traces)
    else:
        # fallback klasik agent
        agent = run_agent_loop(question, generate_fn, max_steps=max_steps, seed_tools=True)
        result.agent = agent
        result.answer = agent.answer
        result.tool_traces.extend(agent.tool_traces)

    mem.last_plan = plan
    return result
