"""Sohbet oturumu paylaşım linkleri (salt okunur token erişimi)."""

from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.config import METADATA_DIR, SHARE_LINK_DEFAULT_TTL_DAYS

_lock = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def shares_store_path(base: Optional[str] = None) -> str:
    return os.path.join(base or METADATA_DIR, "share_links.json")


def _normalize_chat_rel(chat_dir: str) -> str:
    """Proje köküne göre chat dizin yolunu sakla."""
    root = os.path.abspath(os.getcwd())
    path = os.path.abspath(chat_dir)
    if path.startswith(root + os.sep):
        return path[len(root) + 1 :].replace("\\", "/")
    return path.replace("\\", "/")


def _chat_dir_from_rel(chat_rel: str) -> str:
    raw = (chat_rel or "").replace("\\", "/")
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    root = os.path.abspath(os.getcwd())
    return os.path.join(root, raw.lstrip("/"))


def load_shares(base: Optional[str] = None) -> Dict[str, Any]:
    path = shares_store_path(base)
    if not os.path.isfile(path):
        return {"links": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"links": {}}
    data.setdefault("links", {})
    return data


def save_shares(data: Dict[str, Any], base: Optional[str] = None) -> str:
    path = shares_store_path(base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return path


@dataclass
class ShareLink:
    token: str
    chat_rel: str
    session_id: str
    owner: Optional[str] = None
    created_at: str = ""
    expires_at: str = ""
    revoked: bool = False

    @classmethod
    def from_dict(cls, token: str, data: dict) -> "ShareLink":
        return cls(
            token=token,
            chat_rel=str(data.get("chat_rel") or ""),
            session_id=str(data.get("session_id") or ""),
            owner=data.get("owner"),
            created_at=str(data.get("created_at") or ""),
            expires_at=str(data.get("expires_at") or ""),
            revoked=bool(data.get("revoked")),
        )


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def create_share_link(
    chat_dir: str,
    session_id: str,
    *,
    owner: Optional[str] = None,
    ttl_days: Optional[int] = None,
    base: Optional[str] = None,
) -> ShareLink:
    ttl = ttl_days if ttl_days is not None else SHARE_LINK_DEFAULT_TTL_DAYS
    token = secrets.token_urlsafe(24)
    now = _utcnow()
    expires = now + timedelta(days=max(1, int(ttl)))
    link = ShareLink(
        token=token,
        chat_rel=_normalize_chat_rel(chat_dir),
        session_id=session_id,
        owner=owner,
        created_at=now.isoformat(),
        expires_at=expires.isoformat(),
        revoked=False,
    )
    store = load_shares(base)
    store["links"][token] = {
        "chat_rel": link.chat_rel,
        "session_id": link.session_id,
        "owner": link.owner,
        "created_at": link.created_at,
        "expires_at": link.expires_at,
        "revoked": False,
    }
    save_shares(store, base)
    return link


def resolve_share_token(token: str, base: Optional[str] = None) -> Optional[ShareLink]:
    token = (token or "").strip()
    if not token:
        return None
    store = load_shares(base)
    raw = store.get("links", {}).get(token)
    if not raw:
        return None
    link = ShareLink.from_dict(token, raw)
    if link.revoked:
        return None
    exp = _parse_iso(link.expires_at)
    if exp is not None and _utcnow() > exp:
        return None
    if not link.chat_rel or not link.session_id:
        return None
    return link


def revoke_share_link(token: str, base: Optional[str] = None) -> bool:
    store = load_shares(base)
    raw = store.get("links", {}).get(token)
    if not raw:
        return False
    raw["revoked"] = True
    raw["revoked_at"] = _utcnow_iso()
    save_shares(store, base)
    return True


def list_links_for_session(
    chat_dir: str,
    session_id: str,
    *,
    base: Optional[str] = None,
    active_only: bool = True,
) -> List[ShareLink]:
    rel = _normalize_chat_rel(chat_dir)
    out: List[ShareLink] = []
    for token, raw in load_shares(base).get("links", {}).items():
        if raw.get("chat_rel") != rel or raw.get("session_id") != session_id:
            continue
        link = ShareLink.from_dict(token, raw)
        if active_only:
            if link.revoked:
                continue
            exp = _parse_iso(link.expires_at)
            if exp is not None and _utcnow() > exp:
                continue
        out.append(link)
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


def build_share_url(token: str, base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    return f"{base}?share={token}"


def load_shared_session(link: ShareLink) -> Optional[Dict[str, Any]]:
    from rag.chat_store import load_session

    chat_dir = _chat_dir_from_rel(link.chat_rel)
    return load_session(link.session_id, chat_dir=chat_dir)
