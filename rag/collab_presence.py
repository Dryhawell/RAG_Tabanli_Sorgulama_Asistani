"""Collab WebSocket presence / imleç / typing + disk kalıcılığı."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from app.config import COLLAB_PRESENCE_TTL_SEC, METADATA_DIR
except ImportError:
    METADATA_DIR = "metadata"
    COLLAB_PRESENCE_TTL_SEC = 90

_PRESENCE_COLORS = (
    "#4a90d9",
    "#e67e22",
    "#2ecc71",
    "#9b59b6",
    "#e74c3c",
    "#16a085",
    "#f39c12",
    "#8e44ad",
)

_client_meta: Dict[int, Dict[str, Any]] = {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def color_for_user(username: Optional[str]) -> str:
    name = (username or "anon").strip().lower()
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(_PRESENCE_COLORS)
    return _PRESENCE_COLORS[idx]


def _safe_ws_key(workspace_key: str) -> str:
    raw = (workspace_key or "shared").strip() or "shared"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)[:120]


def presence_store_path(workspace_key: str, *, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_ws_key(workspace_key), "presence.json")


def bind_client(websocket: Any, workspace_key: str, username: Optional[str]) -> Dict[str, Any]:
    now = _utcnow_iso()
    meta = {
        "workspace_key": workspace_key,
        "username": username or "anon",
        "cursor": 0,
        "selection_end": 0,
        "color": color_for_user(username),
        "typing": False,
        "updated_at": now,
    }
    _client_meta[id(websocket)] = meta
    return meta


def update_cursor(
    websocket: Any,
    *,
    cursor: int,
    selection_end: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    meta = _client_meta.get(id(websocket))
    if not meta:
        return None
    meta["cursor"] = max(0, int(cursor))
    if selection_end is not None:
        meta["selection_end"] = max(0, int(selection_end))
    meta["updated_at"] = _utcnow_iso()
    return dict(meta)


def update_typing(websocket: Any, typing: bool) -> Optional[Dict[str, Any]]:
    meta = _client_meta.get(id(websocket))
    if not meta:
        return None
    meta["typing"] = bool(typing)
    meta["updated_at"] = _utcnow_iso()
    return dict(meta)


def release_client(websocket: Any) -> Optional[Dict[str, Any]]:
    return _client_meta.pop(id(websocket), None)


def presence_entry(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": meta.get("username"),
        "cursor": int(meta.get("cursor") or 0),
        "selection_end": int(meta.get("selection_end") or 0),
        "color": meta.get("color"),
        "typing": bool(meta.get("typing")),
        "updated_at": meta.get("updated_at"),
        "stale": bool(meta.get("stale")),
    }


def presence_list(clients: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ws in clients:
        meta = _client_meta.get(id(ws))
        if meta:
            out.append(presence_entry(meta))
    out.sort(key=lambda x: (x.get("username") or ""))
    return out


def _parse_iso_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def is_presence_fresh(
    row: Dict[str, Any],
    *,
    now: Optional[float] = None,
    ttl_sec: Optional[int] = None,
) -> bool:
    ttl = int(ttl_sec if ttl_sec is not None else COLLAB_PRESENCE_TTL_SEC)
    if ttl <= 0:
        return True
    ts = _parse_iso_ts(str(row.get("updated_at") or ""))
    if ts is None:
        return False
    current = now if now is not None else time.time()
    return (current - ts) <= ttl


def load_presence_snapshot(
    workspace_key: str,
    *,
    base: Optional[str] = None,
    ttl_sec: Optional[int] = None,
    include_stale: bool = False,
) -> List[Dict[str, Any]]:
    path = presence_store_path(workspace_key, base=base)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    users = data.get("users") if isinstance(data, dict) else None
    if not isinstance(users, list):
        return []
    now = time.time()
    out: List[Dict[str, Any]] = []
    for row in users:
        if not isinstance(row, dict):
            continue
        fresh = is_presence_fresh(row, now=now, ttl_sec=ttl_sec)
        if not fresh and not include_stale:
            continue
        entry = presence_entry(row)
        entry["stale"] = not fresh
        out.append(entry)
    out.sort(key=lambda x: (x.get("username") or ""))
    return out


def save_presence_snapshot(
    workspace_key: str,
    users: List[Dict[str, Any]],
    *,
    base: Optional[str] = None,
) -> str:
    path = presence_store_path(workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "workspace_key": workspace_key,
        "updated_at": _utcnow_iso(),
        "users": [presence_entry(u) for u in users if u.get("username")],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def persist_room_presence(
    workspace_key: str,
    clients: set,
    *,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    users = presence_list(clients)
    save_presence_snapshot(workspace_key, users, base=base)
    return users


def merge_presence_for_join(
    workspace_key: str,
    clients: set,
    *,
    base: Optional[str] = None,
    ttl_sec: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Canlı istemciler + TTL içindeki disk snapshot (yeniden bağlanma PoC)."""
    live = {str(u.get("username") or "").lower(): u for u in presence_list(clients)}
    for row in load_presence_snapshot(
        workspace_key, base=base, ttl_sec=ttl_sec, include_stale=False
    ):
        key = str(row.get("username") or "").lower()
        if key and key not in live:
            live[key] = row
    out = list(live.values())
    out.sort(key=lambda x: (x.get("username") or ""))
    return out
