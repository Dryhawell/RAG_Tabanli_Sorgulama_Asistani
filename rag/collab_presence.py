"""Collab WebSocket presence / imleç yardımcıları."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

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


def color_for_user(username: Optional[str]) -> str:
    name = (username or "anon").strip().lower()
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(_PRESENCE_COLORS)
    return _PRESENCE_COLORS[idx]


def bind_client(websocket: Any, workspace_key: str, username: Optional[str]) -> Dict[str, Any]:
    meta = {
        "workspace_key": workspace_key,
        "username": username or "anon",
        "cursor": 0,
        "selection_end": 0,
        "color": color_for_user(username),
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
    return dict(meta)


def release_client(websocket: Any) -> Optional[Dict[str, Any]]:
    return _client_meta.pop(id(websocket), None)


def presence_entry(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": meta.get("username"),
        "cursor": int(meta.get("cursor") or 0),
        "selection_end": int(meta.get("selection_end") or 0),
        "color": meta.get("color"),
    }


def presence_list(clients: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ws in clients:
        meta = _client_meta.get(id(ws))
        if meta:
            out.append(presence_entry(meta))
    out.sort(key=lambda x: (x.get("username") or ""))
    return out
