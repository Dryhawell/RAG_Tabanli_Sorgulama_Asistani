"""Rich-text mark işlem geçmişi (audit JSONL)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import METADATA_DIR
from rag.collab_crdt import _safe_slug


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_audit_path(workspace_key: str, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_slug(workspace_key), "mark_audit.jsonl")


def log_mark_audit(
    workspace_key: str,
    action: str,
    *,
    mark: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "ts": _utcnow_iso(),
        "workspace_key": workspace_key,
        "action": action,
        "author": author,
        "mark": mark or {},
        "details": details or {},
    }
    path = mark_audit_path(workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        from rag.audit import write_audit

        write_audit(
            f"collab_mark_{action}",
            username=author,
            details={
                "workspace": workspace_key,
                "mark": mark,
                **(details or {}),
            },
        )
    except Exception:
        pass
    return record


def read_mark_audit(
    workspace_key: str,
    *,
    limit: int = 100,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = mark_audit_path(workspace_key, base=base)
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def comment_audit_path(workspace_key: str, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_slug(workspace_key), "comment_audit.jsonl")


def log_comment_audit(
    workspace_key: str,
    action: str,
    *,
    thread: Optional[Dict[str, Any]] = None,
    reply: Optional[Dict[str, Any]] = None,
    author: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "ts": _utcnow_iso(),
        "workspace_key": workspace_key,
        "action": action,
        "author": author,
        "thread": thread or {},
        "reply": reply or {},
        "details": details or {},
    }
    path = comment_audit_path(workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    try:
        from rag.audit import write_audit

        write_audit(
            f"collab_comment_{action}",
            username=author,
            details={
                "workspace": workspace_key,
                "thread": thread,
                "reply": reply,
                **(details or {}),
            },
        )
    except Exception:
        pass
    return record


def read_comment_audit(
    workspace_key: str,
    *,
    limit: int = 100,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = comment_audit_path(workspace_key, base=base)
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]
