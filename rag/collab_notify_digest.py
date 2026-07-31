"""Günlük / periyodik bildirim digest (e-posta özet)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from app.config import (
        METADATA_DIR,
        NOTIFY_DIGEST_GROUP_BY,
        NOTIFY_DIGEST_HOURS,
        NOTIFY_DIGEST_HTML,
        NOTIFY_DIGEST_MENTIONS_ONLY,
        NOTIFY_DIGEST_MIN_PER_WORKSPACE,
        NOTIFY_DIGEST_QUIET_HOURS,
        NOTIFY_DIGEST_SLACK_THREAD_TS,
        NOTIFY_DIGEST_TEAMS_REPLY_ID,
        NOTIFY_DIGEST_THREAD_REPLY,
        NOTIFY_DIGEST_TIMEZONE,
        NOTIFY_FROM_EMAIL,
        NOTIFY_SLACK_BOT_TOKEN,
        NOTIFY_SLACK_CHANNEL,
        NOTIFY_SMTP_HOST,
        NOTIFY_TENANT_TIMEZONES,
        NOTIFY_WEBHOOK_URL,
    )
except ImportError:
    METADATA_DIR = "metadata"
    NOTIFY_DIGEST_HOURS = 24
    NOTIFY_DIGEST_MENTIONS_ONLY = False
    NOTIFY_DIGEST_GROUP_BY = "thread"
    NOTIFY_DIGEST_MIN_PER_WORKSPACE = 1
    NOTIFY_DIGEST_HTML = True
    NOTIFY_DIGEST_QUIET_HOURS = ""
    NOTIFY_DIGEST_TIMEZONE = "UTC"
    NOTIFY_TENANT_TIMEZONES = ""
    NOTIFY_DIGEST_THREAD_REPLY = True
    NOTIFY_DIGEST_SLACK_THREAD_TS = ""
    NOTIFY_DIGEST_TEAMS_REPLY_ID = ""
    NOTIFY_SLACK_BOT_TOKEN = ""
    NOTIFY_SLACK_CHANNEL = ""
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


def _parse_hhmm(value: str) -> Optional[int]:
    """HH:MM veya HH → dakika-of-day (0–1439)."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        if ":" in raw:
            hh, mm = raw.split(":", 1)
            h, m = int(hh), int(mm)
        else:
            h, m = int(raw), 0
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return h * 60 + m
    except Exception:
        return None


def parse_quiet_hours(spec: Optional[str] = None) -> Optional[tuple[int, int]]:
    """'22:00-07:00' → (start_min, end_min). Geçersiz/boş → None."""
    text = (spec if spec is not None else NOTIFY_DIGEST_QUIET_HOURS) or ""
    text = text.strip()
    if not text:
        return None
    if "-" not in text:
        return None
    left, right = text.split("-", 1)
    start = _parse_hhmm(left)
    end = _parse_hhmm(right)
    if start is None or end is None:
        return None
    return start, end


def parse_tenant_timezones(spec: Optional[str] = None) -> Dict[str, str]:
    """'default:Europe/Istanbul,acme:America/New_York' → dict."""
    text = (spec if spec is not None else NOTIFY_TENANT_TIMEZONES) or ""
    text = text.strip()
    out: Dict[str, str] = {}
    if not text:
        return out
    if text.startswith("{"):
        try:
            raw = json.loads(text)
            if isinstance(raw, dict):
                return {str(k): str(v) for k, v in raw.items() if v}
        except json.JSONDecodeError:
            pass
    for part in text.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        tid, tz = part.split(":", 1)
        tid, tz = tid.strip(), tz.strip()
        if tid and tz:
            out[tid] = tz
    return out


def resolve_digest_timezone(
    *,
    timezone_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    username: Optional[str] = None,
) -> str:
    """Öncelik: açık timezone > kullanıcı profili > tenant eşlemesi > global."""
    if timezone_name:
        return timezone_name.strip() or "UTC"
    if username:
        try:
            from rag.auth import get_user_timezone

            user_tz = get_user_timezone(username)
            if user_tz:
                return user_tz
        except Exception:
            pass
    if tenant_id:
        mapping = parse_tenant_timezones()
        if tenant_id in mapping:
            return mapping[tenant_id]
        if tenant_id.lower() in mapping:
            return mapping[tenant_id.lower()]
    return (NOTIFY_DIGEST_TIMEZONE or "UTC").strip() or "UTC"


def _zoneinfo(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def resolve_quiet_hours_spec(
    *,
    quiet_spec: Optional[str] = None,
    username: Optional[str] = None,
) -> Optional[str]:
    """Öncelik: açık spec > kullanıcı profili quiet_hours > global NOTIFY_DIGEST_QUIET_HOURS.

    Açık spec boş / off / none / disabled ise quiet hours kapalı (fallback yok).
    """
    if quiet_spec is not None:
        s = str(quiet_spec).strip()
        if not s or s.lower() in {"off", "none", "disabled"}:
            return None
        return s
    if username:
        try:
            from rag.auth import get_user_quiet_hours

            user_qh = get_user_quiet_hours(username)
            if user_qh:
                return user_qh
        except Exception:
            pass
    global_qh = (NOTIFY_DIGEST_QUIET_HOURS or "").strip()
    return global_qh or None


def is_quiet_hours(
    *,
    now: Optional[datetime] = None,
    quiet_spec: Optional[str] = None,
    timezone_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    username: Optional[str] = None,
) -> bool:
    """Yerel (kullanıcı/tenant) saati quiet hours aralığındaysa True."""
    resolved = resolve_quiet_hours_spec(quiet_spec=quiet_spec, username=username)
    bounds = parse_quiet_hours(resolved)
    if bounds is None:
        return False
    start, end = bounds
    tz_name = resolve_digest_timezone(
        timezone_name=timezone_name,
        tenant_id=tenant_id,
        username=username,
    )
    tz = _zoneinfo(tz_name)
    dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(tz)
    minute = local.hour * 60 + local.minute
    if start == end:
        return True
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


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


def apply_workspace_min_threshold(
    events: List[Dict[str, Any]],
    min_per_workspace: int,
) -> List[Dict[str, Any]]:
    """Workspace bazında min bildirim eşiğini uygular."""
    threshold = int(min_per_workspace)
    if threshold <= 1:
        return list(events)
    counts: Dict[str, int] = {}
    for ev in events:
        ws = str(ev.get("workspace_key") or "-")
        counts[ws] = counts.get(ws, 0) + 1
    allowed = {ws for ws, n in counts.items() if n >= threshold}
    return [
        ev
        for ev in events
        if str(ev.get("workspace_key") or "-") in allowed
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
    min_per_workspace: Optional[int] = None,
) -> Dict[str, Any]:
    """Mention filtreleme + gruplama meta ile hazırlar."""
    use_mentions = (
        mentions_only if mentions_only is not None else NOTIFY_DIGEST_MENTIONS_ONLY
    )
    filtered = filter_mention_events(events) if use_mentions else list(events)
    min_ws = (
        int(min_per_workspace)
        if min_per_workspace is not None
        else NOTIFY_DIGEST_MIN_PER_WORKSPACE
    )
    before_ws = len(filtered)
    filtered = apply_workspace_min_threshold(filtered, min_ws)
    grouped = group_digest_events(filtered, group_by=group_by)
    return {
        "events": filtered,
        "grouped": grouped,
        "mentions_only": use_mentions,
        "group_by": group_by or NOTIFY_DIGEST_GROUP_BY,
        "min_per_workspace": min_ws,
        "workspace_filtered": before_ws - len(filtered),
        "total": len(filtered),
    }


def build_digest_body(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
) -> str:
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
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


def build_digest_html(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
) -> str:
    """Digest özet e-postası HTML şablonu."""
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
    )
    filtered = prep["events"]
    grouped = prep["grouped"]
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:system-ui,sans-serif;color:#222;line-height:1.45}",
        "h1{font-size:1.2em;margin:0 0 8px}",
        ".meta{color:#666;font-size:0.9em;margin-bottom:16px}",
        ".group{margin:12px 0;padding:10px 12px;background:#f6f8fa;border-radius:6px}",
        ".group h2{font-size:0.95em;margin:0 0 8px;color:#444}",
        "ul{margin:0;padding-left:18px}",
        "li{margin:4px 0}",
        ".from{font-weight:600}",
        ".ws{color:#666;font-size:0.85em}",
        "</style></head><body>",
        f"<h1>Bildirim özeti — {username}</h1>",
        f"<div class='meta'>Toplam: {prep['total']}",
    ]
    if prep["mentions_only"]:
        parts.append(" · yalnızca @mention")
    if prep.get("min_per_workspace", 1) > 1:
        parts.append(f" · min/workspace={prep['min_per_workspace']}")
    parts.append("</div>")
    if len(grouped) == 1 and "all" in grouped:
        parts.append("<ul>")
        for ev in filtered:
            ws = ev.get("workspace_key") or "-"
            fr = ev.get("from_user") or "-"
            preview = ev.get("body_preview") or ""
            parts.append(
                f"<li><span class='from'>{fr}</span> "
                f"<span class='ws'>[{ws}]</span> {preview}</li>"
            )
        parts.append("</ul>")
    else:
        for group_key, group_events in grouped.items():
            parts.append(
                f"<div class='group'><h2>{group_key} ({len(group_events)})</h2><ul>"
            )
            for ev in group_events:
                fr = ev.get("from_user") or "-"
                preview = ev.get("body_preview") or ""
                parts.append(f"<li><span class='from'>{fr}</span>: {preview}</li>")
            parts.append("</ul></div>")
    parts.append("</body></html>")
    return "".join(parts)


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
    reply_to_id: Optional[str] = None,
    quiet_hours_summary: bool = False,
) -> Dict[str, Any]:
    """Teams workflow webhook payload (Adaptive Card attachment)."""
    if quiet_hours_summary:
        card = build_quiet_hours_teams_card(
            events,
            username,
            mentions_only=mentions_only,
            group_by=group_by,
        )
    else:
        card = build_digest_teams_adaptive_card(
            events,
            username,
            mentions_only=mentions_only,
            group_by=group_by,
        )
    payload: Dict[str, Any] = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card,
            }
        ],
    }
    rid = (reply_to_id or "").strip()
    if rid:
        payload["replyToId"] = rid
    return payload


def digest_thread_state_path(base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", "digest_threads.json")


def load_digest_thread_state(
    username: str,
    *,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    path = digest_thread_state_path(base=base)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    row = data.get((username or "").strip().lower())
    return dict(row) if isinstance(row, dict) else {}


def save_digest_thread_state(
    username: str,
    *,
    slack_thread_ts: Optional[str] = None,
    teams_reply_id: Optional[str] = None,
    channel: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    path = digest_thread_state_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    key = (username or "").strip().lower()
    row = dict(data.get(key) or {})
    if slack_thread_ts:
        row["slack_thread_ts"] = str(slack_thread_ts).strip()
    if teams_reply_id:
        row["teams_reply_id"] = str(teams_reply_id).strip()
    if channel:
        row["channel"] = str(channel).strip()
    row["updated_at"] = datetime.now(timezone.utc).isoformat()
    data[key] = row
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return row


def resolve_digest_thread_refs(
    username: str,
    *,
    base: Optional[str] = None,
    slack_thread_ts: Optional[str] = None,
    teams_reply_id: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Öncelik: açık argüman > env > kayıtlı state."""
    state = load_digest_thread_state(username, base=base)
    slack_ts = (
        (slack_thread_ts or "").strip()
        or NOTIFY_DIGEST_SLACK_THREAD_TS
        or str(state.get("slack_thread_ts") or "").strip()
        or None
    )
    teams_id = (
        (teams_reply_id or "").strip()
        or NOTIFY_DIGEST_TEAMS_REPLY_ID
        or str(state.get("teams_reply_id") or "").strip()
        or None
    )
    channel = (NOTIFY_SLACK_CHANNEL or str(state.get("channel") or "").strip() or None)
    return {
        "slack_thread_ts": slack_ts,
        "teams_reply_id": teams_id,
        "channel": channel,
    }


def build_quiet_hours_summary_text(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> str:
    """Quiet hours thread reply için kısa metin özeti."""
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    lines = [
        f"Quiet hours özeti — {username}",
        f"Toplam {prep['total']} bekleyen bildirim"
        + (" (yalnızca @mention)" if prep["mentions_only"] else ""),
        "Tam digest sessiz saat bitince gönderilecek.",
    ]
    shown = 0
    for group_key, group_events in prep["grouped"].items():
        if group_key != "all":
            lines.append(f"[{group_key}] {len(group_events)}")
        for ev in group_events[:5]:
            preview = (ev.get("body_preview") or "").strip() or "—"
            lines.append(
                f"- {ev.get('from_user') or '-'}: {preview[:120]}"
            )
            shown += 1
            if shown >= 8:
                break
        if shown >= 8:
            break
    if prep["total"] > shown:
        lines.append(f"+{prep['total'] - shown} daha")
    return "\n".join(lines)


def build_quiet_hours_slack_blocks(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Quiet hours özeti — {username}*\n"
                    f"Toplam *{prep['total']}* bekleyen bildirim. "
                    "Tam digest sessiz saat bitince gelecek."
                ),
            },
        }
    ]
    shown = 0
    for _gk, group_events in prep["grouped"].items():
        for ev in group_events:
            if shown >= 8:
                break
            preview = (ev.get("body_preview") or "").strip() or "—"
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"{ev.get('from_user') or '-'} · "
                                f"`{ev.get('workspace_key') or '-'}` · {preview[:140]}"
                            ),
                        }
                    ],
                }
            )
            shown += 1
        if shown >= 8:
            break
    return blocks


def build_quiet_hours_teams_card(
    events: List[Dict[str, Any]],
    username: str,
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Any]:
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
    )
    body: List[Dict[str, Any]] = [
        {
            "type": "TextBlock",
            "text": f"Quiet hours özeti — {username}",
            "weight": "Bolder",
            "size": "Medium",
        },
        {
            "type": "TextBlock",
            "text": (
                f"{prep['total']} bekleyen bildirim. "
                "Tam digest sessiz saat bitince gönderilecek."
            ),
            "wrap": True,
            "isSubtle": True,
        },
    ]
    facts: List[Dict[str, str]] = []
    for _gk, group_events in prep["grouped"].items():
        for ev in group_events:
            if len(facts) >= 8:
                break
            facts.append(
                {
                    "title": str(ev.get("from_user") or "-"),
                    "value": (ev.get("body_preview") or "—")[:160],
                }
            )
        if len(facts) >= 8:
            break
    if facts:
        body.append({"type": "FactSet", "facts": facts})
    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": body,
    }


def send_digest_email(
    username: str,
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    ignore_quiet_hours: bool = False,
    quiet_hours: Optional[str] = None,
    timezone_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    skip_quiet_tracking: bool = False,
) -> Dict[str, Any]:
    tz_resolved = resolve_digest_timezone(
        timezone_name=timezone_name,
        tenant_id=tenant_id,
        username=username,
    )
    resolved_qh = resolve_quiet_hours_spec(quiet_spec=quiet_hours, username=username)
    actually_quiet = is_quiet_hours(
        quiet_spec=quiet_hours,
        timezone_name=tz_resolved,
        tenant_id=tenant_id,
        username=username,
    )
    in_quiet = (not ignore_quiet_hours) and actually_quiet
    quiet_row: Dict[str, Any] = {}
    flushed_after_quiet = False
    if not skip_quiet_tracking:
        quiet_row = record_quiet_hours_observation(
            username,
            actually_quiet,
            base=base,
        )
        # Quiet bitmiş ve pending flush varsa tam digest'i zorla
        if (
            not ignore_quiet_hours
            and not actually_quiet
            and quiet_row.get("pending_flush")
        ):
            ignore_quiet_hours = True
            in_quiet = False
            flushed_after_quiet = True
    try:
        from rag.collab_notify_push import is_push_configured
    except ImportError:
        def is_push_configured() -> bool:  # type: ignore
            return False
    push_configured = is_push_configured()
    webhook_configured = bool((NOTIFY_WEBHOOK_URL or "").strip())
    thread_reply_ok = bool(NOTIFY_DIGEST_THREAD_REPLY) and webhook_configured
    # Quiet hours: push veya thread-reply kanalı yoksa erken çık.
    if in_quiet and not push_configured and not thread_reply_ok:
        return {
            "sent": False,
            "count": 0,
            "reason": "quiet_hours",
            "quiet_hours": resolved_qh,
            "timezone": tz_resolved,
            "tenant_id": tenant_id,
        }
    events = collect_digest_events(username, hours=hours, base=base)
    if not events:
        return {"sent": False, "count": 0, "reason": "empty"}
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
    )
    filtered = prep["events"]
    if not filtered:
        reason = (
            "below_workspace_threshold"
            if prep.get("workspace_filtered", 0) > 0
            else "empty_after_filter"
        )
        return {"sent": False, "count": 0, "reason": reason}
    email_ok = False
    webhook_ok = False
    push_ok = False
    thread_reply = False
    # E-posta: quiet hours'ta atlanır.
    # Webhook: normal digest veya quiet-hours thread reply.
    # Push: cihaz bazlı sessiz saat / geofence override ile filtrelenir.
    if not in_quiet:
        if NOTIFY_SMTP_HOST:
            from rag.collab_notify_dispatch import dispatch_digest_email

            email_ok = dispatch_digest_email(
                username,
                events,
                mentions_only=mentions_only,
                group_by=group_by,
                min_per_workspace=min_per_workspace,
            )
        if webhook_configured:
            from rag.collab_notify_dispatch import dispatch_digest_webhook

            webhook_ok = dispatch_digest_webhook(
                username,
                events,
                mentions_only=mentions_only,
                group_by=group_by,
                min_per_workspace=min_per_workspace,
                base=base,
                persist_thread=True,
            )
    elif thread_reply_ok:
        from rag.collab_notify_dispatch import dispatch_quiet_hours_thread_reply

        webhook_ok = dispatch_quiet_hours_thread_reply(
            username,
            events,
            mentions_only=mentions_only,
            group_by=group_by,
            min_per_workspace=min_per_workspace,
            base=base,
        )
        thread_reply = bool(webhook_ok)
    if push_configured:
        from rag.collab_notify_push import dispatch_digest_push

        push_ok = dispatch_digest_push(
            username,
            events,
            mentions_only=mentions_only,
            group_by=group_by,
            min_per_workspace=min_per_workspace,
            base=base,
        )
    if not NOTIFY_SMTP_HOST and not webhook_configured and not push_configured:
        return {"sent": False, "count": len(filtered), "reason": "no_channel"}
    sent = email_ok or webhook_ok or push_ok
    if not sent and in_quiet:
        reason = "quiet_hours"
    else:
        reason = None if sent else "send_failed"
    if flushed_after_quiet and not skip_quiet_tracking:
        mark_quiet_flush_done(
            username,
            base=base,
            result={
                "sent": sent,
                "reason": reason,
                "count": len(filtered),
            },
        )
    return {
        "sent": sent,
        "count": len(filtered),
        "email": email_ok,
        "webhook": webhook_ok,
        "push": push_ok,
        "thread_reply": thread_reply,
        "flushed_after_quiet": flushed_after_quiet,
        "reason": reason,
        "quiet_hours": resolved_qh,
        "mentions_only": prep["mentions_only"],
        "group_by": prep["group_by"],
        "min_per_workspace": prep.get("min_per_workspace"),
        "workspace_filtered": prep.get("workspace_filtered", 0),
        "timezone": tz_resolved,
        "tenant_id": tenant_id,
    }


def digest_quiet_flush_path(base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", "digest_quiet_flush.json")


def load_quiet_flush_state(*, base: Optional[str] = None) -> Dict[str, Any]:
    path = digest_quiet_flush_path(base=base)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_quiet_flush_state(
    state: Dict[str, Any],
    *,
    base: Optional[str] = None,
) -> str:
    path = digest_quiet_flush_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path


def record_quiet_hours_observation(
    username: str,
    in_quiet: bool,
    *,
    base: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Kullanıcının quiet/non-quiet geçişini state'e yazar."""
    key = (username or "").strip().lower()
    if not key:
        return {}
    state = load_quiet_flush_state(base=base)
    row = dict(state.get(key) or {})
    prev = row.get("in_quiet")
    current = bool(in_quiet)
    ts = (now or datetime.now(timezone.utc)).isoformat()
    row["in_quiet"] = current
    row["observed_at"] = ts
    # quiet → active geçişi: bir kez tam digest flush bekler
    if prev is True and not current:
        row["quiet_ended_at"] = ts
        row["pending_flush"] = True
    state[key] = row
    save_quiet_flush_state(state, base=base)
    return row


def mark_quiet_flush_done(
    username: str,
    *,
    base: Optional[str] = None,
    now: Optional[datetime] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key = (username or "").strip().lower()
    state = load_quiet_flush_state(base=base)
    row = dict(state.get(key) or {})
    row["pending_flush"] = False
    row["flushed_at"] = (now or datetime.now(timezone.utc)).isoformat()
    if result is not None:
        row["last_flush_result"] = {
            "sent": bool(result.get("sent")),
            "reason": result.get("reason"),
            "count": result.get("count"),
        }
    state[key] = row
    save_quiet_flush_state(state, base=base)
    return row


def maybe_flush_after_quiet_hours(
    username: str,
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    quiet_hours: Optional[str] = None,
    timezone_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    now: Optional[datetime] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Quiet hours bittikten sonra bir kez tam digest gönderir."""
    tz_resolved = resolve_digest_timezone(
        timezone_name=timezone_name,
        tenant_id=tenant_id,
        username=username,
    )
    in_quiet = is_quiet_hours(
        now=now,
        quiet_spec=quiet_hours,
        timezone_name=tz_resolved,
        tenant_id=tenant_id,
        username=username,
    )
    row = record_quiet_hours_observation(
        username,
        in_quiet,
        base=base,
        now=now,
    )
    if in_quiet:
        return {
            "flushed": False,
            "reason": "still_quiet",
            "state": row,
        }
    if not force and not row.get("pending_flush"):
        return {
            "flushed": False,
            "reason": "not_pending",
            "state": row,
        }

    result = send_digest_email(
        username,
        hours=hours,
        base=base,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
        ignore_quiet_hours=True,
        quiet_hours=quiet_hours,
        timezone_name=timezone_name,
        tenant_id=tenant_id,
        skip_quiet_tracking=True,
    )
    row = mark_quiet_flush_done(
        username,
        base=base,
        now=now,
        result=result,
    )
    return {
        "flushed": bool(result.get("sent")),
        "reason": result.get("reason") or ("sent" if result.get("sent") else "send_failed"),
        "digest": result,
        "state": row,
    }


def flush_quiet_hours_digests(
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    quiet_hours: Optional[str] = None,
    timezone_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    now: Optional[datetime] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Hedef kullanıcılar için quiet-hours sonrası flush dener."""
    users = list_digest_target_users(hours=hours, base=base)
    state = load_quiet_flush_state(base=base)
    extra = [
        u
        for u, row in state.items()
        if isinstance(row, dict) and row.get("pending_flush")
    ]
    seen = {u.lower() for u in users}
    for u in extra:
        if u.lower() not in seen:
            users.append(u)
            seen.add(u.lower())
    results: Dict[str, Any] = {}
    flushed = 0
    for user in users:
        r = maybe_flush_after_quiet_hours(
            user,
            hours=hours,
            base=base,
            mentions_only=mentions_only,
            group_by=group_by,
            min_per_workspace=min_per_workspace,
            quiet_hours=quiet_hours,
            timezone_name=timezone_name,
            tenant_id=tenant_id,
            now=now,
            force=force,
        )
        results[user] = r
        if r.get("flushed"):
            flushed += 1
    return {
        "users": users,
        "flushed": flushed,
        "results": results,
    }


def send_digest_all(
    *,
    hours: Optional[int] = None,
    base: Optional[str] = None,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    ignore_quiet_hours: bool = False,
    quiet_hours: Optional[str] = None,
    timezone_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Her kullanıcı için profil quiet_hours / timezone ile digest dener."""
    tz_resolved = resolve_digest_timezone(
        timezone_name=timezone_name,
        tenant_id=tenant_id,
    )
    users = list_digest_target_users(hours=hours, base=base)
    results: Dict[str, Any] = {}
    sent = 0
    skipped_quiet = 0
    for user in users:
        r = send_digest_email(
            user,
            hours=hours,
            base=base,
            mentions_only=mentions_only,
            group_by=group_by,
            min_per_workspace=min_per_workspace,
            ignore_quiet_hours=ignore_quiet_hours,
            quiet_hours=quiet_hours,
            timezone_name=timezone_name,
            tenant_id=tenant_id,
        )
        results[user] = r
        if r.get("sent"):
            sent += 1
        elif r.get("reason") == "quiet_hours":
            skipped_quiet += 1
    out: Dict[str, Any] = {
        "users": users,
        "sent": sent,
        "results": results,
        "timezone": tz_resolved,
        "tenant_id": tenant_id,
        "skipped_quiet_hours": skipped_quiet,
    }
    if not users:
        out["reason"] = "empty"
    elif sent == 0 and skipped_quiet == len(users):
        out["reason"] = "quiet_hours"
        out["quiet_hours"] = quiet_hours or NOTIFY_DIGEST_QUIET_HOURS
    return out
