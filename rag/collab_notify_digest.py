"""Günlük / periyodik bildirim digest (e-posta özet)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from app.config import NOTIFY_DIGEST_HOURS, NOTIFY_FROM_EMAIL, NOTIFY_SMTP_HOST
except ImportError:
    NOTIFY_DIGEST_HOURS = 24
    NOTIFY_SMTP_HOST = ""
    NOTIFY_FROM_EMAIL = "noreply@localhost"

from rag.collab_notify import list_notifications_global


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def collect_digest_events(
    username: str,
    *,
    hours: Optional[int] = None,
    unread_only: bool = True,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    window = int(hours or NOTIFY_DIGEST_HOURS)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, window))
    rows = list_notifications_global(
        username,
        limit=500,
        unread_only=unread_only,
        base=base,
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        dt = _parse_iso(str(row.get("created_at") or ""))
        if dt is None or dt >= cutoff:
            out.append(row)
    return out


def build_digest_body(events: List[Dict[str, Any]], username: str) -> str:
    lines = [
        f"Bildirim özeti — {username}",
        f"Toplam: {len(events)}",
        "",
    ]
    for ev in events:
        lines.append(
            f"• [{ev.get('workspace_key') or '-'}] "
            f"{ev.get('from_user') or '-'}: {ev.get('body_preview') or ''}"
        )
    return "\n".join(lines)


def send_digest_email(
    username: str,
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    events = collect_digest_events(username, hours=hours, base=base)
    if not events:
        return {"sent": False, "count": 0, "reason": "empty"}
    if not NOTIFY_SMTP_HOST:
        return {"sent": False, "count": len(events), "reason": "no_smtp"}
    from rag.collab_notify_dispatch import dispatch_digest_email

    ok = dispatch_digest_email(username, events)
    return {
        "sent": ok,
        "count": len(events),
        "reason": None if ok else "send_failed",
    }
