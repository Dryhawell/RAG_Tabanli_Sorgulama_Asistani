"""Basit agent döngüsü: araç çağrıları + FINAL yanıt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rag.tools import (
    TOOL_SPECS,
    execute_tool,
    extract_final,
    heuristic_tool_plan,
    list_tools,
    parse_tool_calls,
)

GenerateFn = Callable[[str], str]

AGENT_PROMPT = """Sen yardımcı bir asistansın. Gerekirse araç çağır, sonra nihai yanıt ver.

Kullanılabilir araçlar:
{tool_list}

Araç çağırmak için tek satır yaz:
TOOL_CALL {{"name": "calculator", "args": {{"expression": "2+2"}}}}

İşin bitince:
FINAL <yanıt metni>

Soru: {question}
{extra}

Şimdi yanıtla (TOOL_CALL veya FINAL):
"""


@dataclass
class AgentResult:
    answer: str
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    mode: str = "heuristic"  # heuristic | llm


def _tool_list_text() -> str:
    lines = []
    for name, spec in TOOL_SPECS.items():
        lines.append(f"- {name}: {spec['description']} args={spec['args']}")
    return "\n".join(lines)


def run_heuristic_tools(question: str) -> AgentResult:
    plans = heuristic_tool_plan(question)
    traces = []
    parts = []
    for name, args in plans:
        out = execute_tool(name, args)
        traces.append({"name": name, "args": args, "result": out})
        parts.append(f"[{name}] {out}")
    answer = "\n".join(parts) if parts else ""
    return AgentResult(answer=answer, tool_traces=traces, steps=len(traces), mode="heuristic")


def run_agent_loop(
    question: str,
    generate_fn: GenerateFn,
    *,
    max_steps: int = 3,
    seed_tools: bool = True,
) -> AgentResult:
    """LLM ile TOOL_CALL / FINAL döngüsü. seed_tools: önce heuristic çalıştır."""
    traces: List[Dict[str, Any]] = []
    extra_blocks: List[str] = []

    if seed_tools:
        seeded = run_heuristic_tools(question)
        traces.extend(seeded.tool_traces)
        if seeded.answer:
            extra_blocks.append("Ön araç sonuçları:\n" + seeded.answer)

    history = ""
    final_answer = ""
    steps = 0
    for step in range(max_steps):
        steps = step + 1
        extra = "\n".join(extra_blocks + ([history] if history else []))
        prompt = AGENT_PROMPT.format(
            tool_list=_tool_list_text(),
            question=question,
            extra=extra,
        )
        raw = (generate_fn(prompt) or "").strip()
        final = extract_final(raw)
        calls = parse_tool_calls(raw)
        if final and not calls:
            final_answer = final
            break
        if not calls:
            # MODEL doğrudan cevap verdiyse FINAL say
            final_answer = final or raw
            break
        for name, args in calls:
            out = execute_tool(name, args)
            traces.append({"name": name, "args": args, "result": out})
            history += f"\nTOOL_RESULT {name}: {out}\n"
        if final:
            final_answer = final
            break
    else:
        final_answer = final_answer or history.strip() or "Araçlar çalıştı; nihai yanıt üretilemedi."

    return AgentResult(
        answer=final_answer.strip(),
        tool_traces=traces,
        steps=steps,
        mode="llm",
    )


def tools_context_block(traces: Sequence[Dict[str, Any]]) -> str:
    if not traces:
        return ""
    lines = ["Araç sonuçları:"]
    for t in traces:
        lines.append(f"- {t.get('name')}: {t.get('result')}")
    return "\n".join(lines)
