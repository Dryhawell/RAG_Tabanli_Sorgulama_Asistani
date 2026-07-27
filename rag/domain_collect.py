"""Otomatik domain eğitim çifti toplama (sohbet, audit, metrik)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import METADATA_DIR

DEFAULT_DOMAIN_PAIRS_PATH = os.path.join(METADATA_DIR, "domain_training", "pairs.jsonl")


def _norm_anchor(text: str) -> str:
    return (text or "").strip().lower()


def _pair_key(anchor: str, positive: str) -> Tuple[str, str]:
    pos = (positive or "").strip()
    preview = pos[:160].lower()
    return _norm_anchor(anchor), preview


def dedupe_pairs(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for row in pairs:
        anchor = str(row.get("anchor") or "").strip()
        positive = str(row.get("positive") or "").strip()
        if len(anchor) < 4 or len(positive) < 12:
            continue
        key = _pair_key(anchor, positive)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def merge_pairs(
    primary: Sequence[Dict[str, Any]],
    extra: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return dedupe_pairs(list(primary) + list(extra))


def load_domain_pairs(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("anchor") and row.get("positive"):
                rows.append(row)
    return rows


def append_domain_pairs(
    pairs: Sequence[Dict[str, Any]],
    path: str,
    *,
    dedupe_existing: bool = True,
) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing = load_domain_pairs(path) if dedupe_existing else []
    merged = merge_pairs(existing, pairs)
    new_count = len(merged) - len(existing)
    with open(path, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return max(0, new_count)


def collect_from_chats(
    chat_root: str,
    *,
    min_gate_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Sohbet JSON'larından (soru, top chunk) çiftleri."""
    pairs: List[Dict[str, Any]] = []
    if not os.path.isdir(chat_root):
        return pairs

    for dirpath, _, filenames in os.walk(chat_root):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    session = json.load(f)
            except Exception:
                continue
            messages = session.get("messages") or []
            for i, msg in enumerate(messages):
                if msg.get("role") != "user":
                    continue
                anchor = str(msg.get("content") or "").strip()
                if len(anchor) < 4:
                    continue
                assistant = messages[i + 1] if i + 1 < len(messages) else None
                if not assistant or assistant.get("role") != "assistant":
                    continue
                sources = assistant.get("sources") or []
                if not sources:
                    continue
                top = sources[0]
                score = float(top.get("score") or 0)
                if score < min_gate_score:
                    continue
                positive = str(top.get("text") or "").strip()
                if len(positive) < 12:
                    continue
                pairs.append(
                    {
                        "anchor": anchor,
                        "positive": positive[:900],
                        "source": "chat",
                        "session_id": session.get("id"),
                        "source_file": top.get("source_file"),
                        "score": score,
                    }
                )
    return pairs


def collect_from_audit(
    audit_path: str,
    *,
    min_gate_score: float = 0.35,
) -> List[Dict[str, Any]]:
    """Audit query olaylarından soru çiftleri (chunk metni yoksa atlanır)."""
    from rag.audit import read_audit

    pairs: List[Dict[str, Any]] = []
    if not os.path.isfile(audit_path):
        return pairs
    for row in read_audit(path=audit_path, event="query", limit=5000):
        details = row.get("details") or {}
        if details.get("no_answer"):
            continue
        gate = float(details.get("gate_score") or 0)
        if gate < min_gate_score:
            continue
        anchor = str(details.get("question") or "").strip()
        positive = str(details.get("top_chunk") or "").strip()
        if len(anchor) < 4 or len(positive) < 12:
            continue
        pairs.append(
            {
                "anchor": anchor,
                "positive": positive[:900],
                "source": "audit",
                "gate_score": gate,
                "source_file": details.get("top_source"),
            }
        )
    return pairs


def collect_from_metrics(
    metrics_path: str,
    *,
    min_gate_score: float = 0.35,
) -> List[Dict[str, Any]]:
    """Metrik JSONL'den zengin query kayıtları."""
    from rag.metrics import read_metrics

    pairs: List[Dict[str, Any]] = []
    if not os.path.isfile(metrics_path):
        return pairs
    for row in read_metrics(path=metrics_path, kind="query", limit=5000):
        values = row.get("values") or {}
        if values.get("no_answer"):
            continue
        gate = float(values.get("gate_score") or 0)
        if gate < min_gate_score:
            continue
        anchor = str(values.get("question") or "").strip()
        positive = str(values.get("top_chunk") or "").strip()
        if len(anchor) < 4 or len(positive) < 12:
            continue
        pairs.append(
            {
                "anchor": anchor,
                "positive": positive[:900],
                "source": "metrics",
                "gate_score": gate,
                "source_file": values.get("top_source"),
            }
        )
    return pairs


def collect_domain_pairs(
    *,
    chat_root: Optional[str] = None,
    audit_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    min_gate_score: float = 0.35,
) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    if chat_root:
        collected.extend(collect_from_chats(chat_root, min_gate_score=min_gate_score))
    if audit_path:
        collected.extend(collect_from_audit(audit_path, min_gate_score=min_gate_score))
    if metrics_path:
        collected.extend(collect_from_metrics(metrics_path, min_gate_score=min_gate_score))
    return dedupe_pairs(collected)


def collect_and_save(
    output_path: str,
    *,
    chat_root: Optional[str] = None,
    audit_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    min_gate_score: float = 0.35,
) -> Dict[str, Any]:
    pairs = collect_domain_pairs(
        chat_root=chat_root,
        audit_path=audit_path,
        metrics_path=metrics_path,
        min_gate_score=min_gate_score,
    )
    added = append_domain_pairs(pairs, output_path)
    total = len(load_domain_pairs(output_path))
    return {
        "collected": len(pairs),
        "added": added,
        "total": total,
        "output": output_path,
    }
