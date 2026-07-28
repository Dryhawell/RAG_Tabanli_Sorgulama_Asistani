"""İşbirlikçi yorum @mention bildirimleri."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import METADATA_DIR
from rag.collab_crdt import _safe_slug

MENTION_RE = re.compile(r"@([a-zA-Z0-9_.-]{2,32})")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def notifications_path(workspace_key: str, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_slug(workspace_key), "notifications.jsonl")


def notify_center_path(base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", "_notify_center.jsonl")


def _append_center_line(event: Dict[str, Any], base: Optional[str] = None) -> None:
    path = notify_center_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def extract_mentions(text: str) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for m in MENTION_RE.findall(text):
        key = m.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(m.strip())
    return out


def append_notification(
    workspace_key: str,
    *,
    target_user: str,
    kind: str = "mention",
    from_user: Optional[str] = None,
    thread_id: Optional[str] = None,
    body_preview: str = "",
    base: Optional[str] = None,
) -> Dict[str, Any]:
    target = (target_user or "").strip()
    if not target:
        raise ValueError("Hedef kullanıcı boş")
    event = {
        "id": _new_id(),
        "workspace_key": workspace_key,
        "target_user": target,
        "kind": kind,
        "from_user": from_user,
        "thread_id": thread_id,
        "body_preview": (body_preview or "")[:240],
        "created_at": _utcnow_iso(),
        "read": False,
    }
    path = notifications_path(workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    _append_center_line(event, base=base)
    return event


def list_notifications(
    workspace_key: str,
    username: str,
    *,
    limit: int = 50,
    unread_only: bool = False,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = notifications_path(workspace_key, base=base)
    if not os.path.isfile(path):
        return []
    user_key = (username or "").strip().lower()
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("target_user") or "").strip().lower() != user_key:
                continue
            if unread_only and row.get("read"):
                continue
            rows.append(row)
    return rows[-limit:]


def mark_notifications_read(
    workspace_key: str,
    username: str,
    *,
    notification_ids: Optional[List[str]] = None,
    base: Optional[str] = None,
) -> int:
    path = notifications_path(workspace_key, base=base)
    if not os.path.isfile(path):
        return 0
    user_key = (username or "").strip().lower()
    want = set(notification_ids or [])
    all_rows: List[Dict[str, Any]] = []
    changed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("target_user") or "").strip().lower() != user_key:
                all_rows.append(row)
                continue
            if not want and not row.get("read"):
                row["read"] = True
                changed += 1
            elif row.get("id") in want and not row.get("read"):
                row["read"] = True
                changed += 1
            all_rows.append(row)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed


def list_notifications_global(
    username: str,
    *,
    limit: int = 50,
    unread_only: bool = False,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Tüm workspace'lerdeki bildirimleri merkez indeksinden listeler."""
    path = notify_center_path(base=base)
    if not os.path.isfile(path):
        return []
    user_key = (username or "").strip().lower()
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("target_user") or "").strip().lower() != user_key:
                continue
            if unread_only and row.get("read"):
                continue
            rows.append(row)
    rows.sort(key=lambda r: str(r.get("created_at") or ""))
    return rows[-limit:]


def mark_notifications_read_global(
    username: str,
    *,
    notification_ids: Optional[List[str]] = None,
    base: Optional[str] = None,
) -> int:
    """Merkez + workspace dosyalarında bildirimleri okundu işaretler."""
    path = notify_center_path(base=base)
    if not os.path.isfile(path):
        return 0
    user_key = (username or "").strip().lower()
    want = set(notification_ids or [])
    all_rows: List[Dict[str, Any]] = []
    changed = 0
    by_ws: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("target_user") or "").strip().lower() != user_key:
                all_rows.append(row)
                continue
            should_mark = (not want and not row.get("read")) or (
                row.get("id") in want and not row.get("read")
            )
            if should_mark:
                row["read"] = True
                changed += 1
                ws = str(row.get("workspace_key") or "")
                if ws and row.get("id"):
                    by_ws.setdefault(ws, []).append(str(row["id"]))
            all_rows.append(row)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for ws, ids in by_ws.items():
            mark_notifications_read(ws, username, notification_ids=ids, base=base)
    return changed


def notify_mentions(
    workspace_key: str,
    body: str,
    *,
    from_user: Optional[str] = None,
    thread_id: Optional[str] = None,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """@kullanıcı etiketlerini bildirim olarak kaydeder (gönderen hariç)."""
    events: List[Dict[str, Any]] = []
    sender = (from_user or "").strip().lower()
    preview = (body or "").strip()
    for mention in extract_mentions(body):
        if mention.lower() == sender:
            continue
        events.append(
            append_notification(
                workspace_key,
                target_user=mention,
                from_user=from_user,
                thread_id=thread_id,
                body_preview=preview,
                base=base,
            )
        )
    return events
