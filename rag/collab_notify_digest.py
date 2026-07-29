"""Günlük / periyodik bildirim digest (e-posta özet)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from app.config import NOTIFY_DIGEST_HOURS, NOTIFY_FROM_EMAIL, NOTIFY_SMTP_HOST, NOTIFY_WEBHOOK_URL
except ImportError:
    NOTIFY_DIGEST_HOURS = 24
    NOTIFY_SMTP_HOST = ""
    NOTIFY_FROM_EMAIL = "noreply@localhost"
    NOTIFY_WEBHOOK_URL = ""

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


def list_digest_target_users(
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
) -> List[str]:
    """Özet gönderilecek kullanıcıları merkez indeksinden toplar."""
    try:
        from rag.collab_notify import notify_center_path
    except ImportError:
        return []
    path = notify_center_path(base=base)
    if not os.path.isfile(path):
        return []
    window = int(hours or NOTIFY_DIGEST_HOURS)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, window))
    users: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("read"):
                continue
            dt = _parse_iso(str(row.get("created_at") or ""))
            if dt is None or dt >= cutoff:
                target = str(row.get("target_user") or "").strip()
                if target:
                    users.add(target)
    return sorted(users)


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


def build_digest_slack_blocks(
    events: List[Dict[str, Any]],
    username: str,
) -> List[Dict[str, Any]]:
    """Slack Incoming Webhook Block Kit payload."""
    blocks: List[Dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Bildirim özeti — {username}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Toplam *{len(events)}* okunmamış bildirim",
            },
        },
        {"type": "divider"},
    ]
    for ev in events[:50]:
        workspace = ev.get("workspace_key") or "-"
        from_user = ev.get("from_user") or "-"
        preview = (ev.get("body_preview") or "").strip() or "—"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{from_user}* in `{workspace}`\n"
                        f">{preview}"
                    ),
                },
            }
        )
    if len(events) > 50:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_+{len(events) - 50} daha fazla bildirim_",
                    }
                ],
            }
        )
    return blocks


def is_slack_webhook_url(url: Optional[str]) -> bool:
    return "hooks.slack.com" in (url or "")


def is_discord_webhook_url(url: Optional[str]) -> bool:
    return "discord.com/api/webhooks" in (url or "")


def build_digest_discord_embed(
    events: List[Dict[str, Any]],
    username: str,
) -> List[Dict[str, Any]]:
    """Discord webhook embed payload."""
    fields: List[Dict[str, Any]] = []
    for ev in events[:25]:
        workspace = ev.get("workspace_key") or "-"
        from_user = ev.get("from_user") or "-"
        preview = (ev.get("body_preview") or "").strip() or "—"
        fields.append(
            {
                "name": f"{from_user} · {workspace}",
                "value": preview[:1024],
            }
        )
    description = f"Toplam **{len(events)}** okunmamış bildirim"
    if len(events) > 25:
        description += f" (+{len(events) - 25} daha)"
    return [
        {
            "title": f"Bildirim özeti — {username}",
            "description": description,
            "color": 5814783,
            "fields": fields,
        }
    ]


def send_digest_email(
    username: str,
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    events = collect_digest_events(username, hours=hours, base=base)
    if not events:
        return {"sent": False, "count": 0, "reason": "empty"}
    email_ok = False
    webhook_ok = False
    if NOTIFY_SMTP_HOST:
        from rag.collab_notify_dispatch import dispatch_digest_email

        email_ok = dispatch_digest_email(username, events)
    if NOTIFY_WEBHOOK_URL:
        from rag.collab_notify_dispatch import dispatch_digest_webhook

        webhook_ok = dispatch_digest_webhook(username, events)
    if not NOTIFY_SMTP_HOST and not NOTIFY_WEBHOOK_URL:
        return {"sent": False, "count": len(events), "reason": "no_channel"}
    sent = email_ok or webhook_ok
    reason = None if sent else "send_failed"
    return {
        "sent": sent,
        "count": len(events),
        "email": email_ok,
        "webhook": webhook_ok,
        "reason": reason,
    }


def send_digest_all(
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    users = list_digest_target_users(hours=hours, base=base)
    results: Dict[str, Any] = {}
    sent = 0
    for user in users:
        r = send_digest_email(user, hours=hours, base=base)
        results[user] = r
        if r.get("sent"):
            sent += 1
    return {"users": users, "sent": sent, "results": results}
