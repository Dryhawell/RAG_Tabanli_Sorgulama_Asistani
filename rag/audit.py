"""JSONL audit log (login, ingest, query, admin işlemleri)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import AUDIT_LOG_PATH, ENABLE_AUDIT, METADATA_DIR

_lock = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit_log_path(tenant_id: Optional[str] = None, base: Optional[str] = None) -> str:
    if base:
        return base
    if tenant_id:
        return os.path.join(METADATA_DIR, "tenants", tenant_id, "audit.jsonl")
    return AUDIT_LOG_PATH


def write_audit(
    event: str,
    *,
    username: Optional[str] = None,
    tenant_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Tek satır JSONL yazar. enabled=None ise config ENABLE_AUDIT kullanılır."""
    record = {
        "ts": _utcnow_iso(),
        "event": event,
        "username": username,
        "tenant_id": tenant_id,
        "details": details or {},
    }
    use = ENABLE_AUDIT if enabled is None else enabled
    if not use:
        return record

    out_path = path or audit_log_path(tenant_id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record


def read_audit(
    *,
    path: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 100,
    event: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Son N audit kaydını (yeniden eskiye) döndürür."""
    out_path = path or audit_log_path(tenant_id)
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
            if event and row.get("event") != event:
                continue
            rows.append(row)
    return list(reversed(rows[-limit:]))
