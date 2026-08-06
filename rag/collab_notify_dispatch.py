"""Bildirim e-posta ve webhook (push) dağıtımı."""

from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

try:
    from app.config import (
        ENABLE_COLLAB_NOTIFY_DISPATCH,
        NOTIFY_FROM_EMAIL,
        NOTIFY_SMTP_HOST,
        NOTIFY_SMTP_PASSWORD,
        NOTIFY_SMTP_PORT,
        NOTIFY_SMTP_USER,
        NOTIFY_WEBHOOK_URL,
    )
except ImportError:
    ENABLE_COLLAB_NOTIFY_DISPATCH = True
    NOTIFY_SMTP_HOST = ""
    NOTIFY_SMTP_PORT = 587
    NOTIFY_SMTP_USER = ""
    NOTIFY_SMTP_PASSWORD = ""
    NOTIFY_FROM_EMAIL = "noreply@localhost"
    NOTIFY_WEBHOOK_URL = ""


def resolve_notify_email(target_user: str) -> Optional[str]:
    """users.json veya @ içeren kullanıcı adından e-posta çözümler."""
    target = (target_user or "").strip()
    if not target:
        return None
    if "@" in target:
        return target
    try:
        from rag.auth import ensure_users_file

        users = ensure_users_file()
        meta = users.get(target) or users.get(target.lower())
        if meta and meta.get("email"):
            return str(meta["email"]).strip()
    except Exception:
        pass
    return None


def dispatch_email(event: Dict[str, Any], *, to_email: Optional[str] = None) -> bool:
    """SMTP ile mention bildirimi gönderir (yapılandırılmamışsa False)."""
    if not NOTIFY_SMTP_HOST:
        return False
    to_addr = to_email or resolve_notify_email(str(event.get("target_user") or ""))
    if not to_addr:
        return False
    subject = f"[RAG Collab] {event.get('from_user') or 'Biri'} sizi etiketledi"
    body = (
        f"Workspace: {event.get('workspace_key') or '-'}\n"
        f"Gönderen: {event.get('from_user') or '-'}\n"
        f"Mesaj: {event.get('body_preview') or ''}\n"
        f"Thread: {event.get('thread_id') or '-'}\n"
    )
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = NOTIFY_FROM_EMAIL
    msg["To"] = to_addr
    try:
        with smtplib.SMTP(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=15) as smtp:
            if NOTIFY_SMTP_USER:
                smtp.starttls()
                smtp.login(NOTIFY_SMTP_USER, NOTIFY_SMTP_PASSWORD)
            smtp.sendmail(NOTIFY_FROM_EMAIL, [to_addr], msg.as_string())
        return True
    except Exception:
        return False


def dispatch_webhook(event: Dict[str, Any]) -> bool:
    """Webhook URL'ine JSON POST (Slack/Discord/generic push)."""
    url = (NOTIFY_WEBHOOK_URL or "").strip()
    if not url:
        return False
    try:
        import requests

        from rag.collab_notify_digest import is_slack_webhook_url, is_discord_webhook_url

        preview = event.get("body_preview") or ""
        text = (
            f"{event.get('from_user') or 'Biri'} mentioned "
            f"{event.get('target_user') or 'user'}: {preview}"
        )
        payload: Dict[str, Any] = {
            "type": "collab_mention",
            "notification": event,
            "text": text,
        }
        if is_slack_webhook_url(url):
            payload["blocks"] = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*@{event.get('target_user') or 'user'}* "
                            f"— {event.get('from_user') or 'Biri'}\n"
                            f">{preview or '—'}"
                        ),
                    },
                }
            ]
        elif is_discord_webhook_url(url):
            payload = {
                "embeds": [
                    {
                        "title": "Collab mention",
                        "description": preview or "—",
                        "color": 3447003,
                        "fields": [
                            {
                                "name": "From",
                                "value": str(event.get("from_user") or "-"),
                                "inline": True,
                            },
                            {
                                "name": "Target",
                                "value": str(event.get("target_user") or "-"),
                                "inline": True,
                            },
                            {
                                "name": "Workspace",
                                "value": str(event.get("workspace_key") or "-"),
                                "inline": True,
                            },
                        ],
                    }
                ]
            }
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code < 400
    except Exception:
        return False


def dispatch_notification(event: Dict[str, Any]) -> Dict[str, Any]:
    """E-posta ve webhook kanallarına bildirim gönderir."""
    if not ENABLE_COLLAB_NOTIFY_DISPATCH:
        return {"email": False, "webhook": False, "skipped": True}
    email_ok = dispatch_email(event)
    webhook_ok = dispatch_webhook(event)
    return {"email": email_ok, "webhook": webhook_ok, "skipped": False}


def dispatch_notifications(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dispatch_notification(ev) for ev in events or []]


def dispatch_digest_email(
    username: str,
    events: List[Dict[str, Any]],
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
) -> bool:
    """Okunmamış bildirimlerin günlük özet e-postası."""
    if not NOTIFY_SMTP_HOST or not events:
        return False
    to_addr = resolve_notify_email(username)
    if not to_addr:
        return False
    from rag.collab_notify_digest import (
        NOTIFY_DIGEST_HTML,
        build_digest_body,
        build_digest_html,
        prepare_digest_events,
    )

    body = build_digest_body(
        events,
        username,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
    )
    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
    )
    count = prep["total"]
    subject = f"[RAG Collab] Bildirim özeti ({count})"
    if NOTIFY_DIGEST_HTML:
        html = build_digest_html(
            events,
            username,
            mentions_only=mentions_only,
            group_by=group_by,
            min_per_workspace=min_per_workspace,
        )
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))
    else:
        msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = NOTIFY_FROM_EMAIL
    msg["To"] = to_addr
    try:
        with smtplib.SMTP(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=15) as smtp:
            if NOTIFY_SMTP_USER:
                smtp.starttls()
                smtp.login(NOTIFY_SMTP_USER, NOTIFY_SMTP_PASSWORD)
            smtp.sendmail(NOTIFY_FROM_EMAIL, [to_addr], msg.as_string())
        return True
    except Exception:
        return False


def dispatch_digest_webhook(
    username: str,
    events: List[Dict[str, Any]],
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    thread_ts: Optional[str] = None,
    reply_to_id: Optional[str] = None,
    quiet_hours_summary: bool = False,
    base: Optional[str] = None,
    persist_thread: bool = False,
) -> bool:
    """Digest özetini webhook'a gönderir (Teams/Slack/Discord/generic).

    Slack: `thread_ts` ile thread reply; Teams: `replyToId`.
    Quiet hours özeti: kısa summary payload.
    """
    url = (NOTIFY_WEBHOOK_URL or "").strip()
    if not url or not events:
        return False
    try:
        import requests

        from rag.collab_notify_digest import (
            build_digest_body,
            build_digest_discord_embed,
            build_digest_slack_blocks,
            build_digest_teams_payload,
            build_quiet_hours_slack_blocks,
            build_quiet_hours_summary_text,
            is_discord_webhook_url,
            is_slack_webhook_url,
            is_teams_webhook_url,
            prepare_digest_events,
            resolve_digest_thread_refs,
            save_digest_thread_state,
        )

        refs = resolve_digest_thread_refs(
            username,
            base=base,
            slack_thread_ts=thread_ts,
            teams_reply_id=reply_to_id,
        )
        slack_ts = refs.get("slack_thread_ts")
        teams_id = refs.get("teams_reply_id")

        prep = prepare_digest_events(
            events,
            mentions_only=mentions_only,
            group_by=group_by,
            min_per_workspace=min_per_workspace,
        )
        if quiet_hours_summary:
            text = build_quiet_hours_summary_text(
                events,
                username,
                mentions_only=mentions_only,
                group_by=group_by,
            )
        else:
            text = build_digest_body(
                events,
                username,
                mentions_only=mentions_only,
                group_by=group_by,
                min_per_workspace=min_per_workspace,
            )
        if is_teams_webhook_url(url):
            payload = build_digest_teams_payload(
                events,
                username,
                mentions_only=mentions_only,
                group_by=group_by,
                reply_to_id=teams_id if quiet_hours_summary else None,
                quiet_hours_summary=quiet_hours_summary,
            )
        elif is_discord_webhook_url(url):
            prefix = "Quiet hours özeti" if quiet_hours_summary else "Bildirim özeti"
            payload = {
                "content": f"{prefix} — {username} ({prep['total']})",
                "embeds": build_digest_discord_embed(
                    events,
                    username,
                    mentions_only=mentions_only,
                    group_by=group_by,
                ),
            }
            if quiet_hours_summary:
                payload["content"] = (
                    f"Quiet hours özeti — {username} ({prep['total']} bekleyen)"
                )
        else:
            payload: Dict[str, Any] = {
                "type": "collab_digest_quiet_hours" if quiet_hours_summary else "collab_digest",
                "username": username,
                "count": prep["total"],
                "text": text,
                "quiet_hours_summary": quiet_hours_summary,
            }
            if is_slack_webhook_url(url):
                if quiet_hours_summary:
                    payload["blocks"] = build_quiet_hours_slack_blocks(
                        events,
                        username,
                        mentions_only=mentions_only,
                        group_by=group_by,
                    )
                else:
                    payload["blocks"] = build_digest_slack_blocks(
                        events,
                        username,
                        mentions_only=mentions_only,
                        group_by=group_by,
                    )
                if quiet_hours_summary and slack_ts:
                    payload["thread_ts"] = slack_ts
                elif not quiet_hours_summary and slack_ts:
                    # Parent thread'e de yazılabilir (opsiyonel env)
                    payload["thread_ts"] = slack_ts
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code >= 400:
            return False
        # Bot/API yanıtından thread_ts yakala
        if persist_thread and not quiet_hours_summary:
            try:
                content = getattr(r, "content", None)
                body = r.json() if content else {}
            except Exception:
                body = {}
            if isinstance(body, dict):
                ts = str(body.get("ts") or body.get("message_ts") or "").strip()
                rid = str(body.get("id") or body.get("replyToId") or "").strip()
                if ts or rid:
                    save_digest_thread_state(
                        username,
                        slack_thread_ts=ts or None,
                        teams_reply_id=rid or None,
                        base=base,
                    )
        return True
    except Exception:
        return False


def dispatch_quiet_hours_thread_reply(
    username: str,
    events: List[Dict[str, Any]],
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    base: Optional[str] = None,
) -> bool:
    """Quiet hours sırasında Slack/Teams thread reply özeti gönderir."""
    from rag.collab_notify_digest import (
        NOTIFY_DIGEST_THREAD_REPLY,
        NOTIFY_SLACK_BOT_TOKEN,
        NOTIFY_SLACK_CHANNEL,
        resolve_digest_thread_refs,
    )

    if not NOTIFY_DIGEST_THREAD_REPLY:
        return False
    refs = resolve_digest_thread_refs(username, base=base)
    # Parent yoksa yine de quiet summary webhook'a gider (üst seviye mesaj).
    # Slack bot token varsa chat.postMessage ile thread reply dene.
    bot = (NOTIFY_SLACK_BOT_TOKEN or "").strip()
    channel = (NOTIFY_SLACK_CHANNEL or refs.get("channel") or "").strip()
    slack_ts = refs.get("slack_thread_ts")
    if bot and channel and slack_ts:
        try:
            import requests

            from rag.collab_notify_digest import (
                build_quiet_hours_slack_blocks,
                build_quiet_hours_summary_text,
            )

            text = build_quiet_hours_summary_text(
                events,
                username,
                mentions_only=mentions_only,
                group_by=group_by,
            )
            blocks = build_quiet_hours_slack_blocks(
                events,
                username,
                mentions_only=mentions_only,
                group_by=group_by,
            )
            r = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {bot}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                json={
                    "channel": channel,
                    "thread_ts": slack_ts,
                    "text": text,
                    "blocks": blocks,
                },
                timeout=10,
            )
            if r.status_code < 400:
                data = {}
                try:
                    data = r.json()
                except Exception:
                    pass
                if data.get("ok"):
                    return True
        except Exception:
            pass
    return dispatch_digest_webhook(
        username,
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
        quiet_hours_summary=True,
        base=base,
        persist_thread=False,
    )
