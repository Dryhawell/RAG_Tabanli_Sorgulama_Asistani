"""Günlük / periyodik bildirim digest (e-posta özet)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from app.config import (
        NOTIFY_DIGEST_GROUP_BY,
        NOTIFY_DIGEST_HOURS,
        NOTIFY_DIGEST_MENTIONS_ONLY,
        NOTIFY_FROM_EMAIL,
        NOTIFY_SMTP_HOST,
        NOTIFY_WEBHOOK_URL,
    )
except ImportError:
    NOTIFY_DIGEST_HOURS = 24
    NOTIFY_DIGEST_MENTIONS_ONLY = False
    NOTIFY_DIGEST_GROUP_BY = "thread"
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


def filter_mention_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Yalnızca @mention bildirimlerini döndürür."""
    return [
        ev
        for ev in events
        if str(ev.get("kind") or "mention").strip().lower() == "mention"
    ]


def _digest_group_key(ev: Dict[str, Any], group_by: str) -> str:
    if group_by == "workspace":
        return str(ev.get("workspace_key") or "-")
    thread_id = str(ev.get("thread_id") or "").strip()
    if thread_id:
        return f"thread:{thread_id}"
    workspace = str(ev.get("workspace_key") or "-")
    return f"workspace:{workspace}"


def group_digest_events(
    events: List[Dict[str, Any]],
    *,
    group_by: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Digest bildirimlerini thread veya workspace'e göre gruplar."""
    mode = (group_by or NOTIFY_DIGEST_GROUP_BY or "thread").strip().lower()
    if mode in {"none", "off", ""}:
        return {"all": list(events)}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events:
        key = _digest_group_key(ev, mode)
        grouped.setdefault(key, []).append(ev)
    return grouped


def prepare_digest_events(
    events: List[Dict[str, Any]],
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Mention filtreleme + gruplama meta ile hazırlar."""
    use_mentions = (
        mentions_only if mentions_only is not None else NOTIFY_DIGEST_MENTIONS_ONLY
    )
    filtered = filter_mention_events(events) if use_mentions else list(events)
    grouped = group_digest_events(filtered, group_by=group_by)
    return {
        "events": filtered,
        "grouped": grouped,
        "mentions_only": use_mentions,
        "group_by": group_by or NOTIFY_DIGEST_GROUP_BY,
        "total": len(filtered),
    }


def build_digest_body(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> str:
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    filtered = prep["events"]
    grouped = prep["grouped"]
    lines = [
        f"Bildirim özeti — {username}",
        f"Toplam: {prep['total']}",
    ]
    if prep["mentions_only"]:
        lines.append("(yalnızca @mention)")
    lines.append("")
    if len(grouped) == 1 and "all" in grouped:
        for ev in filtered:
            lines.append(
                f"• [{ev.get('workspace_key') or '-'}] "
                f"{ev.get('from_user') or '-'}: {ev.get('body_preview') or ''}"
            )
    else:
        for group_key, group_events in grouped.items():
            lines.append(f"=== {group_key} ({len(group_events)}) ===")
            for ev in group_events:
                lines.append(
                    f"  • {ev.get('from_user') or '-'}: {ev.get('body_preview') or ''}"
                )
            lines.append("")
    return "\n".join(lines).strip()


def build_digest_slack_blocks(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Slack Incoming Webhook Block Kit payload."""
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    filtered = prep["events"]
    grouped = prep["grouped"]
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
                "text": (
                    f"Toplam *{prep['total']}* bildirim"
                    + (" (yalnızca @mention)" if prep["mentions_only"] else "")
                ),
            },
        },
        {"type": "divider"},
    ]
    shown = 0
    for group_key, group_events in grouped.items():
        if group_key != "all":
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{group_key}* ({len(group_events)})",
                    },
                }
            )
        for ev in group_events:
            if shown >= 50:
                break
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
            shown += 1
        if shown >= 50:
            break
    if prep["total"] > 50:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_+{prep['total'] - 50} daha fazla bildirim_",
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
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discord webhook embed payload."""
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    filtered = prep["events"]
    fields: List[Dict[str, Any]] = []
    shown = 0
    for group_key, group_events in prep["grouped"].items():
        for ev in group_events:
            if shown >= 25:
                break
            workspace = ev.get("workspace_key") or "-"
            from_user = ev.get("from_user") or "-"
            preview = (ev.get("body_preview") or "").strip() or "—"
            prefix = f"{group_key} · " if group_key != "all" else ""
            fields.append(
                {
                    "name": f"{prefix}{from_user} · {workspace}",
                    "value": preview[:1024],
                }
            )
            shown += 1
        if shown >= 25:
            break
    description = f"Toplam **{prep['total']}** bildirim"
    if prep["mentions_only"]:
        description += " (yalnızca @mention)"
    if prep["total"] > 25:
        description += f" (+{prep['total'] - 25} daha)"
    return [
        {
            "title": f"Bildirim özeti — {username}",
            "description": description,
            "color": 5814783,
            "fields": fields,
        }
    ]


def is_teams_webhook_url(url: Optional[str]) -> bool:
    u = (url or "").lower()
    return "webhook.office.com" in u or "office.com/webhook" in u


def build_digest_teams_adaptive_card(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Microsoft Teams Adaptive Card v1.4 içeriği."""
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    body: List[Dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": f"Bildirim özeti — {username}",
            "weight": "Bolder",
            "size": "Large",
        },
        {
            "type": "TextBlock",
            "text": (
                f"Toplam {prep['total']} bildirim"
                + (" (yalnızca @mention)" if prep["mentions_only"] else "")
            ),
            "isSubtle": True,
            "wrap": True,
        },
    ]
    shown = 0
    for group_key, group_events in prep["grouped"].items():
        if group_key != "all":
            body.append(
                {
                    "type": "TextBlock",
                    "text": f"{group_key} ({len(group_events)})",
                    "weight": "Bolder",
                    "separator": True,
                }
            )
        facts: List[Dict[str, str]] = []
        for ev in group_events:
            if shown >= 25:
                break
            facts.append(
                {
                    "title": str(ev.get("from_user") or "-"),
                    "value": (
                        f"{ev.get('workspace_key') or '-'}: "
                        f"{(ev.get('body_preview') or '—')[:200]}"
                    ),
                }
            )
            shown += 1
        if facts:
            body.append({"type": "FactSet", "facts": facts})
        if shown >= 25:
            break
    if prep["total"] > 25:
        body.append(
            {
                "type": "TextBlock",
                "text": f"+{prep['total'] - 25} daha fazla bildirim",
                "isSubtle": True,
            }
        )
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
    }


def build_digest_teams_payload(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Teams workflow webhook payload (Adaptive Card attachment)."""
    card = build_digest_teams_adaptive_card(
        events,
        username,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card,
            }
        ],
    }


def send_digest_email(
    username: str,
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    events = collect_digest_events(username, hours=hours, base=base)
    if not events:
        return {"sent": False, "count": 0, "reason": "empty"}
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    filtered = prep["events"]
    if not filtered:
        return {"sent": False, "count": 0, "reason": "empty_after_filter"}
    email_ok = False
    webhook_ok = False
    if NOTIFY_SMTP_HOST:
        from rag.collab_notify_dispatch import dispatch_digest_email

        email_ok = dispatch_digest_email(
            username,
            events,
            mentions_only=mentions_only,
            group_by=group_by,
        )
    if NOTIFY_WEBHOOK_URL:
        from rag.collab_notify_dispatch import dispatch_digest_webhook

        webhook_ok = dispatch_digest_webhook(
            username,
            events,
            mentions_only=mentions_only,
            group_by=group_by,
        )
    if not NOTIFY_SMTP_HOST and not NOTIFY_WEBHOOK_URL:
        return {"sent": False, "count": len(filtered), "reason": "no_channel"}
    sent = email_ok or webhook_ok
    reason = None if sent else "send_failed"
    return {
        "sent": sent,
        "count": len(filtered),
        "email": email_ok,
        "webhook": webhook_ok,
        "reason": reason,
        "mentions_only": prep["mentions_only"],
        "group_by": prep["group_by"],
    }


def send_digest_all(
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    users = list_digest_target_users(hours=hours, base=base)
    results: Dict[str, Any] = {}
    sent = 0
    for user in users:
        r = send_digest_email(
            user,
            hours=hours,
            base=base,
            mentions_only=mentions_only,
            group_by=group_by,
        )
        results[user] = r
        if r.get("sent"):
            sent += 1
    return {"users": users, "sent": sent, "results": results}
