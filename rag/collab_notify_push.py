"""Mobil push bildirim PoC (FCM HTTP / generic webhook) + token TTL."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from app.config import (
        METADATA_DIR,
        NOTIFY_PUSH_API_KEY,
        NOTIFY_PUSH_PROVIDER,
        NOTIFY_PUSH_TOKEN_TTL_DAYS,
        NOTIFY_PUSH_URL,
    )
except ImportError:
    METADATA_DIR = "metadata"
    NOTIFY_PUSH_URL = ""
    NOTIFY_PUSH_API_KEY = ""
    NOTIFY_PUSH_PROVIDER = "generic"
    NOTIFY_PUSH_TOKEN_TTL_DAYS = 90


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


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


def device_tokens_path(base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", "push_tokens.jsonl")


def register_device_token(
    username: str,
    token: str,
    *,
    platform: str = "fcm",
    label: Optional[str] = None,
    device_name: Optional[str] = None,
    os_name: Optional[str] = None,
    os_version: Optional[str] = None,
    app_version: Optional[str] = None,
    ttl_days: Optional[int] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Kullanıcı cihaz token'ını kaydeder (TTL + cihaz meta)."""
    user = (username or "").strip()
    tok = (token or "").strip()
    if not user or not tok:
        raise ValueError("username ve token gerekli")
    days = int(ttl_days if ttl_days is not None else NOTIFY_PUSH_TOKEN_TTL_DAYS)
    now = _utcnow()
    expires = now + timedelta(days=max(1, days))
    record = {
        "username": user,
        "token": tok,
        "platform": (platform or "fcm").strip().lower(),
        "label": (label or "").strip() or None,
        "device_name": (device_name or "").strip() or None,
        "os_name": (os_name or "").strip() or None,
        "os_version": (os_version or "").strip() or None,
        "app_version": (app_version or "").strip() or None,
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "last_seen_at": now.isoformat(),
        "revoked": False,
    }
    path = device_tokens_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = _read_all_tokens(base=base)
    updated = False
    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user.lower()
            and str(row.get("token") or "") == tok
        ):
            # created_at koru
            created = row.get("created_at") or record["created_at"]
            row.update(record)
            row["created_at"] = created
            updated = True
        new_rows.append(row)
    if not updated:
        new_rows.append(record)
    _write_all_tokens(new_rows, base=base)
    return record


def touch_device_last_seen(
    username: str,
    token: str,
    *,
    base: Optional[str] = None,
) -> bool:
    """Başarılı push sonrası last_seen_at günceller."""
    user_key = (username or "").strip().lower()
    tok = (token or "").strip()
    rows = _read_all_tokens(base=base)
    changed = False
    now = _utcnow_iso()
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user_key
            and str(row.get("token") or "") == tok
            and not row.get("revoked")
        ):
            row["last_seen_at"] = now
            changed = True
    if changed:
        _write_all_tokens(rows, base=base)
    return changed


def _read_all_tokens(base: Optional[str] = None) -> List[Dict[str, Any]]:
    path = device_tokens_path(base=base)
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
    return rows


def _write_all_tokens(rows: List[Dict[str, Any]], base: Optional[str] = None) -> None:
    path = device_tokens_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_token_expired(row: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    if row.get("revoked"):
        return True
    exp = _parse_iso(str(row.get("expires_at") or ""))
    if exp is None:
        return False
    current = now or _utcnow()
    return exp <= current


def list_device_tokens(
    username: str,
    *,
    include_expired: bool = False,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_key = (username or "").strip().lower()
    rows = [
        row
        for row in _read_all_tokens(base=base)
        if str(row.get("username") or "").strip().lower() == user_key
    ]
    if include_expired:
        return rows
    return [row for row in rows if not is_token_expired(row)]


def revoke_device_token(
    username: str,
    token: str,
    *,
    base: Optional[str] = None,
) -> bool:
    user_key = (username or "").strip().lower()
    tok = (token or "").strip()
    rows = _read_all_tokens(base=base)
    changed = False
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user_key
            and str(row.get("token") or "") == tok
        ):
            row["revoked"] = True
            row["revoked_at"] = _utcnow_iso()
            changed = True
    if changed:
        _write_all_tokens(rows, base=base)
    return changed


def prune_expired_tokens(
    *,
    username: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Süresi dolmuş / revoked token'ları dosyadan temizler."""
    user_key = (username or "").strip().lower() if username else None
    rows = _read_all_tokens(base=base)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for row in rows:
        if user_key and str(row.get("username") or "").strip().lower() != user_key:
            kept.append(row)
            continue
        if is_token_expired(row):
            removed += 1
            continue
        kept.append(row)
    if removed:
        _write_all_tokens(kept, base=base)
    return {"removed": removed, "remaining": len(kept)}


def summarize_user_devices(
    username: str,
    *,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """UI için cihaz özeti (token kısaltılmış)."""
    out: List[Dict[str, Any]] = []
    for row in list_device_tokens(username, include_expired=True, base=base):
        tok = str(row.get("token") or "")
        preview = tok[:6] + "…" + tok[-4:] if len(tok) > 12 else tok
        out.append(
            {
                "token": tok,
                "token_preview": preview,
                "platform": row.get("platform") or "generic",
                "label": row.get("label"),
                "device_name": row.get("device_name"),
                "os_name": row.get("os_name"),
                "os_version": row.get("os_version"),
                "app_version": row.get("app_version"),
                "created_at": row.get("created_at"),
                "expires_at": row.get("expires_at"),
                "last_seen_at": row.get("last_seen_at"),
                "expired": is_token_expired(row),
                "revoked": bool(row.get("revoked")),
            }
        )
    return out


def build_fcm_payload(
    token: str,
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """FCM HTTP v1 tarzı message gövdesi (PoC)."""
    return {
        "message": {
            "token": token,
            "notification": {"title": title, "body": body[:200]},
            "data": {k: str(v) for k, v in (data or {}).items()},
        }
    }


def build_apns_payload(
    token: str,
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """APNs benzeri PoC payload (gateway üzerinden iletilir)."""
    return {
        "device_token": token,
        "aps": {
            "alert": {"title": title, "body": body[:200]},
            "sound": "default",
        },
        "data": data or {},
    }


def build_generic_push_payload(
    username: str,
    tokens: List[Dict[str, Any]],
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "type": "collab_push",
        "provider": NOTIFY_PUSH_PROVIDER,
        "username": username,
        "title": title,
        "body": body,
        "tokens": tokens,
        "data": data or {},
    }


def dispatch_push(
    username: str,
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
    base: Optional[str] = None,
) -> bool:
    """Kayıtlı (süresi dolmamış) cihaz token'larına push gönderir."""
    url = (NOTIFY_PUSH_URL or "").strip()
    if not url:
        return False
    tokens = list_device_tokens(username, include_expired=False, base=base)
    if not tokens:
        tokens = [{"username": username, "token": "", "platform": "generic"}]
    try:
        import requests

        provider = (NOTIFY_PUSH_PROVIDER or "generic").lower()
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if NOTIFY_PUSH_API_KEY:
            if provider == "fcm":
                headers["Authorization"] = f"Bearer {NOTIFY_PUSH_API_KEY}"
            else:
                headers["Authorization"] = f"key={NOTIFY_PUSH_API_KEY}"

        ok_any = False
        if provider == "fcm":
            for tok in tokens:
                t = str(tok.get("token") or "").strip()
                if not t:
                    continue
                payload = build_fcm_payload(t, title=title, body=body, data=data)
                r = requests.post(url, json=payload, headers=headers, timeout=10)
                if r.status_code < 400:
                    ok_any = True
                    touch_device_last_seen(username, t, base=base)
        elif provider == "apns":
            for tok in tokens:
                t = str(tok.get("token") or "").strip()
                if not t:
                    continue
                payload = build_apns_payload(t, title=title, body=body, data=data)
                r = requests.post(url, json=payload, headers=headers, timeout=10)
                if r.status_code < 400:
                    ok_any = True
                    touch_device_last_seen(username, t, base=base)
        else:
            payload = build_generic_push_payload(
                username,
                tokens,
                title=title,
                body=body,
                data=data,
            )
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            if r.status_code < 400:
                ok_any = True
                for tok in tokens:
                    t = str(tok.get("token") or "").strip()
                    if t:
                        touch_device_last_seen(username, t, base=base)
        return ok_any
    except Exception:
        return False


def dispatch_digest_push(
    username: str,
    events: List[Dict[str, Any]],
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    base: Optional[str] = None,
) -> bool:
    from rag.collab_notify_digest import prepare_digest_events

    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
    )
    count = prep["total"]
    if count <= 0:
        return False
    title = f"Bildirim özeti ({count})"
    body = f"{username}: {count} okunmamış collab bildirimi"
    return dispatch_push(
        username,
        title=title,
        body=body,
        data={"type": "collab_digest", "count": count},
        base=base,
    )
