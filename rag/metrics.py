"""JSONL metrik kaydı ve özet dashboard."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import ENABLE_METRICS, METADATA_DIR, METRICS_PATH

_lock = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics_log_path(tenant_id: Optional[str] = None, base: Optional[str] = None) -> str:
    if base:
        return base
    if tenant_id:
        return os.path.join(METADATA_DIR, "tenants", tenant_id, "metrics.jsonl")
    return METRICS_PATH


def record_metric(
    kind: str,
    *,
    username: Optional[str] = None,
    tenant_id: Optional[str] = None,
    values: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Tek satır JSONL metrik yazar."""
    record = {
        "ts": _utcnow_iso(),
        "kind": kind,
        "username": username,
        "tenant_id": tenant_id,
        "values": values or {},
    }
    use = ENABLE_METRICS if enabled is None else enabled
    if not use:
        return record

    out_path = path or metrics_log_path(tenant_id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Opsiyonel Prometheus sink (hata yut — JSONL asıl kaynak)
    try:
        from rag.prometheus_sink import observe_metric

        observe_metric(kind, values=values, tenant_id=tenant_id)
    except Exception:
        pass
    return record


def read_metrics(
    *,
    path: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 5000,
    kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Son N metrik kaydını (eskiden yeniye) döndürür."""
    out_path = path or metrics_log_path(tenant_id)
    if not os.path.isfile(out_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind and row.get("kind") != kind:
                continue
            rows.append(row)
    return rows[-limit:]


def _avg(nums: List[float]) -> float:
    return sum(nums) / len(nums) if nums else 0.0


def summarize_metrics(
    rows: Optional[List[Dict[str, Any]]] = None,
    *,
    path: Optional[str] = None,
    limit: int = 5000,
) -> Dict[str, Any]:
    """Metrik özetini dashboard/CLI için üretir."""
    if rows is None:
        rows = read_metrics(path=path, limit=limit)

    by_kind: Dict[str, int] = {}
    gate_scores: List[float] = []
    latencies: List[float] = []
    n_sources: List[float] = []
    no_answer_count = 0
    query_count = 0
    ingest_chunks = 0
    ingest_count = 0
    rebuild_count = 0
    delete_count = 0
    push_revoked_count = 0
    push_prune_removed = 0
    judge_runs = 0
    judge_ok = 0
    judge_soft_fail = 0
    judge_accuracies: List[float] = []

    for row in rows:
        kind = str(row.get("kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        vals = row.get("values") or {}
        if not isinstance(vals, dict):
            vals = {}

        if kind == "query":
            query_count += 1
            if vals.get("no_answer"):
                no_answer_count += 1
            if vals.get("gate_score") is not None:
                gate_scores.append(float(vals["gate_score"]))
            if vals.get("latency_ms") is not None:
                latencies.append(float(vals["latency_ms"]))
            if vals.get("n_sources") is not None:
                n_sources.append(float(vals["n_sources"]))
        elif kind == "ingest":
            ingest_count += 1
            ingest_chunks += int(vals.get("chunks_added") or 0)
        elif kind == "rebuild":
            rebuild_count += 1
        elif kind == "delete":
            delete_count += 1
        elif kind == "push_token_revoked":
            push_revoked_count += 1
        elif kind == "push_token_prune":
            push_prune_removed += int(vals.get("removed") or 0)
        elif kind == "judge_run":
            judge_runs += 1
            if vals.get("ok"):
                judge_ok += 1
            if vals.get("soft_fail"):
                judge_soft_fail += 1
            if vals.get("accuracy") is not None:
                try:
                    judge_accuracies.append(float(vals["accuracy"]))
                except (TypeError, ValueError):
                    pass

    recent = list(reversed(rows[-10:]))
    recent_acc = judge_accuracies[-20:]

    return {
        "total_events": len(rows),
        "by_kind": by_kind,
        "query": {
            "count": query_count,
            "no_answer_count": no_answer_count,
            "no_answer_rate": (no_answer_count / query_count) if query_count else 0.0,
            "avg_gate_score": round(_avg(gate_scores), 4),
            "avg_latency_ms": round(_avg(latencies), 1),
            "avg_n_sources": round(_avg(n_sources), 2),
        },
        "ingest": {"count": ingest_count, "chunks_added": ingest_chunks},
        "rebuild": {"count": rebuild_count},
        "delete": {"count": delete_count},
        "push": {
            "token_revoked": push_revoked_count,
            "token_pruned": push_prune_removed,
        },
        "judge": {
            "runs": judge_runs,
            "ok": judge_ok,
            "failed": max(0, judge_runs - judge_ok),
            "soft_fail": judge_soft_fail,
            "soft_fail_rate": round(
                (judge_soft_fail / judge_runs) if judge_runs else 0.0, 4
            ),
            "avg_accuracy": round(_avg(judge_accuracies), 4),
            "recent_accuracies": [round(a, 4) for a in recent_acc],
        },
        "recent": recent,
    }
