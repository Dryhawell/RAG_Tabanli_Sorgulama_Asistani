"""Agent kısa bellek: oturum notları, özet, çalışma belleği."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

GenerateFn = Callable[[str], str]

SUMMARY_PROMPT = """Aşağıdaki sohbet geçmişini 3-5 cümlede özetle.
Kullanıcı tercihleri ve önemli olguları koru.

Geçmiş:
{history}

Özet:"""


@dataclass
class AgentMemory:
    summary: str = ""
    notes: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    last_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "notes": list(self.notes),
            "facts": list(self.facts),
            "last_plan": list(self.last_plan),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AgentMemory":
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            summary=str(data.get("summary") or ""),
            notes=[str(x) for x in (data.get("notes") or []) if str(x).strip()],
            facts=[str(x) for x in (data.get("facts") or []) if str(x).strip()],
            last_plan=[str(x) for x in (data.get("last_plan") or []) if str(x).strip()],
        )

    def context_block(self, *, max_notes: int = 8) -> str:
        parts = []
        if self.summary:
            parts.append(f"Sohbet özeti: {self.summary}")
        if self.facts:
            parts.append("Bilinen olgular:\n- " + "\n- ".join(self.facts[-max_notes:]))
        if self.notes:
            parts.append("Notlar:\n- " + "\n- ".join(self.notes[-max_notes:]))
        if self.last_plan:
            parts.append("Son plan:\n- " + "\n- ".join(self.last_plan))
        return "\n".join(parts)


def load_memory_from_session(session: Optional[dict]) -> AgentMemory:
    if not session:
        return AgentMemory()
    return AgentMemory.from_dict(session.get("agent_memory"))


def save_memory_to_session(session: dict, memory: AgentMemory) -> dict:
    session["agent_memory"] = memory.to_dict()
    return session


def extract_facts_heuristic(messages: Sequence[dict], *, limit: int = 10) -> List[str]:
    """Kullanıcı mesajlarından kaba olgu adayları (tercih / sayı / dosya)."""
    facts: List[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if re.search(r"\b(tercih|unutma|hatırla|önemli)\b", text, re.I):
            facts.append(text[:200])
        elif re.search(r"\d+\s*(gün|tl|%|adet)", text, re.I):
            facts.append(text[:200])
    # tekilleştir sıra koru
    seen = set()
    out = []
    for f in facts:
        key = f.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= limit:
            break
    return out


def summarize_history(
    messages: Sequence[dict],
    generate_fn: GenerateFn,
    *,
    max_messages: int = 8,
) -> str:
    recent = list(messages)[-max_messages:]
    if not recent:
        return ""
    lines = []
    for m in recent:
        role = m.get("role") or "?"
        content = (m.get("content") or "").strip().replace("\n", " ")
        if content:
            lines.append(f"{role}: {content[:300]}")
    if not lines:
        return ""
    return (generate_fn(SUMMARY_PROMPT.format(history="\n".join(lines))) or "").strip()


def update_memory_after_turn(
    memory: AgentMemory,
    *,
    messages: Sequence[dict],
    plan: Optional[List[str]] = None,
    note: Optional[str] = None,
    generate_fn: Optional[GenerateFn] = None,
    refresh_summary: bool = False,
) -> AgentMemory:
    memory.facts = extract_facts_heuristic(messages) or memory.facts
    if plan:
        memory.last_plan = list(plan)
    if note and note.strip():
        memory.notes.append(note.strip()[:300])
        memory.notes = memory.notes[-20:]
    if refresh_summary and generate_fn is not None:
        try:
            memory.summary = summarize_history(messages, generate_fn)
        except Exception:
            pass
    return memory
