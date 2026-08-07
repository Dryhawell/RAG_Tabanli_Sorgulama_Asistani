"""Basit araçlar: hesap makinesi, takvim, web arama."""

from __future__ import annotations

import ast
import json
import operator
import re
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# Güvenli aritmetik
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if not op:
            raise ValueError("Desteklenmeyen işlem")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if not op:
            raise ValueError("Desteklenmeyen işlem")
        return op(_eval_node(node.operand))
    raise ValueError("İfade yalnızca sayılar ve + - * / ** içerebilir")


def tool_calculator(expression: str) -> str:
    expr = (expression or "").strip()
    if not expr:
        return "Hata: boş ifade"
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
        return f"{expr} = {value}"
    except Exception as exc:
        return f"Hesaplama hatası: {exc}"


_TR_WEEKDAYS = [
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
]
_TR_MONTHS = [
    "",
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
]


def _fmt_tr(d: date) -> str:
    return f"{d.day} {_TR_MONTHS[d.month]} {d.year} {_TR_WEEKDAYS[d.weekday()]}"


def tool_calendar(
    action: str = "today",
    *,
    days: int = 0,
    date_str: Optional[str] = None,
) -> str:
    """action: today | add_days | weekday | parse"""
    action = (action or "today").strip().lower()
    today = date.today()
    if action in {"today", "bugun", "bugün"}:
        return f"Bugün: {_fmt_tr(today)} (ISO {today.isoformat()})"
    if action in {"add_days", "offset"}:
        target = today + timedelta(days=int(days))
        return f"{days:+d} gün → {_fmt_tr(target)} (ISO {target.isoformat()})"
    if action in {"weekday", "gun"}:
        base = today
        if date_str:
            base = date.fromisoformat(date_str[:10])
        return f"{base.isoformat()} → {_TR_WEEKDAYS[base.weekday()]}"
    if action == "parse" and date_str:
        base = date.fromisoformat(date_str[:10])
        return _fmt_tr(base)
    return f"Bilinmeyen takvim eylemi: {action}"


def tool_web_search(query: str, *, max_results: int = 3, timeout: float = 8.0) -> str:
    """DuckDuckGo Instant Answer API (anahtarsız, sınırlı)."""
    q = (query or "").strip()
    if not q:
        return "Hata: boş arama"
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=timeout,
            headers={"User-Agent": "RAG-Assistant/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Web arama hatası: {exc}"

    parts: List[str] = []
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        src = data.get("AbstractURL") or data.get("AbstractSource") or ""
        parts.append(f"Özet: {abstract}" + (f" ({src})" if src else ""))
    related = data.get("RelatedTopics") or []
    n = 0
    for item in related:
        if n >= max_results:
            break
        if isinstance(item, dict) and item.get("Text"):
            parts.append(f"- {item['Text']}" + (f" <{item.get('FirstURL')}>" if item.get("FirstURL") else ""))
            n += 1
        elif isinstance(item, dict) and item.get("Topics"):
            for sub in item["Topics"]:
                if n >= max_results:
                    break
                if isinstance(sub, dict) and sub.get("Text"):
                    parts.append(f"- {sub['Text']}")
                    n += 1
    if not parts:
        answer = (data.get("Answer") or "").strip()
        if answer:
            parts.append(f"Yanıt: {answer}")
    if not parts:
        return f"Web aramada sonuç yok: {q}"
    return f"Arama: {q}\n" + "\n".join(parts)


TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "calculator": {
        "description": "Matematiksel ifade hesapla (ör. 12*1.2+3)",
        "args": {"expression": "str"},
        "fn": lambda args: tool_calculator(str(args.get("expression") or "")),
    },
    "calendar": {
        "description": "Bugünün tarihi / gün ekle / haftanın günü",
        "args": {"action": "today|add_days|weekday", "days": "int", "date_str": "YYYY-MM-DD"},
        "fn": lambda args: tool_calendar(
            str(args.get("action") or "today"),
            days=int(args.get("days") or 0),
            date_str=args.get("date_str"),
        ),
    },
    "web_search": {
        "description": "Kısa web araması (DuckDuckGo)",
        "args": {"query": "str"},
        "fn": lambda args: tool_web_search(str(args.get("query") or "")),
    },
}


def list_tools() -> List[str]:
    return sorted(TOOL_SPECS.keys())


def execute_tool(name: str, args: Optional[Dict[str, Any]] = None) -> str:
    spec = TOOL_SPECS.get(name)
    if not spec:
        return f"Bilinmeyen araç: {name}"
    try:
        return str(spec["fn"](args or {}))
    except Exception as exc:
        return f"Araç hatası ({name}): {exc}"


_TOOL_CALL_RE = re.compile(r"TOOL_CALL\s*", re.IGNORECASE)


def _extract_json_object(text: str, start: int) -> Optional[Tuple[str, int]]:
    """start konumundan itibaren dengeli {...} JSON nesnesini alır."""
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1], j + 1
    return None


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Metinden TOOL_CALL {...} bloklarını çıkarır."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    pos = 0
    raw = text or ""
    while True:
        m = _TOOL_CALL_RE.search(raw, pos)
        if not m:
            break
        extracted = _extract_json_object(raw, m.end())
        if not extracted:
            break
        blob, next_pos = extracted
        pos = next_pos
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = str(payload.get("name") or "").strip()
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if name:
            out.append((name, args))
    return out


def extract_final(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"FINAL\s*[:\-]?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def heuristic_tool_plan(question: str) -> List[Tuple[str, Dict[str, Any]]]:
    """LLM olmadan basit niyet → araç eşlemesi."""
    q = (question or "").strip().lower()
    plans: List[Tuple[str, Dict[str, Any]]] = []
    # matematik
    if re.search(r"[\d\s\+\-\*/\(\)\.]+", q) and re.search(r"[\+\-\*/]", q):
        expr = re.findall(r"[\d\.\s\+\-\*/\(\)]+", question)
        if expr:
            candidate = max(expr, key=len).strip()
            if re.search(r"\d", candidate) and re.search(r"[\+\-\*/]", candidate):
                plans.append(("calculator", {"expression": candidate}))
    if any(k in q for k in ("kaç eder", "hesapla", "topla", "çarp", "böl")):
        nums = re.findall(r"[\d\.]+", question)
        if len(nums) >= 2 and ("+" in question or "topla" in q):
            plans.append(("calculator", {"expression": "+".join(nums[:5])}))
    # takvim
    if any(k in q for k in ("bugün", "bugun", "tarih", "hangi gün", "kaç gün sonra")):
        if "kaç gün sonra" in q or "gun sonra" in q:
            m = re.search(r"(\d+)\s*gün", q)
            days = int(m.group(1)) if m else 0
            plans.append(("calendar", {"action": "add_days", "days": days}))
        else:
            plans.append(("calendar", {"action": "today"}))
    # web
    if any(k in q for k in ("internette", "web'de", "webde", "araştır", "google", "güncel")):
        plans.append(("web_search", {"query": question}))
    return plans
