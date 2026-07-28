"""Bildirim e-posta ve webhook (push) dağıtımı."""

from __future__ import annotations

import json
import smtplib
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

        payload = {
            "type": "collab_mention",
            "notification": event,
            "text": (
                f"{event.get('from_user') or 'Biri'} mentioned "
                f"{event.get('target_user') or 'user'}: "
                f"{event.get('body_preview') or ''}"
            ),
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


def dispatch_digest_email(username: str, events: List[Dict[str, Any]]) -> bool:
    """Okunmamış bildirimlerin günlük özet e-postası."""
    if not NOTIFY_SMTP_HOST or not events:
        return False
    to_addr = resolve_notify_email(username)
    if not to_addr:
        return False
    from rag.collab_notify_digest import build_digest_body

    body = build_digest_body(events, username)
    subject = f"[RAG Collab] Bildirim özeti ({len(events)})"
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
