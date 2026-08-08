"""Sohbet oturumlarını JSON olarak kalıcı saklar."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import CHAT_DIR


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_chat_dir(chat_dir: str = CHAT_DIR) -> str:
    os.makedirs(chat_dir, exist_ok=True)
    return chat_dir


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    return slug or "oturum"


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def session_path(session_id: str, chat_dir: str = CHAT_DIR) -> str:
    return os.path.join(ensure_chat_dir(chat_dir), f"{session_id}.json")


def create_session(title: str = "Yeni sohbet", chat_dir: str = CHAT_DIR) -> Dict[str, Any]:
    sid = new_session_id()
    data = {
        "id": sid,
        "title": title,
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "messages": [],
    }
    save_session(data, chat_dir=chat_dir)
    return data


def save_session(session: Dict[str, Any], chat_dir: str = CHAT_DIR) -> str:
    sid = session.get("id") or new_session_id()
    session["id"] = sid
    session["updated_at"] = _utcnow_iso()
    path = session_path(sid, chat_dir=chat_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return path


def load_session(session_id: str, chat_dir: str = CHAT_DIR) -> Optional[Dict[str, Any]]:
    path = session_path(session_id, chat_dir=chat_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions(chat_dir: str = CHAT_DIR) -> List[Dict[str, Any]]:
    ensure_chat_dir(chat_dir)
    items = []
    for name in os.listdir(chat_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(chat_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items.append(
                {
                    "id": data.get("id") or name[:-5],
                    "title": data.get("title") or name[:-5],
                    "updated_at": data.get("updated_at") or "",
                    "n_messages": len(data.get("messages") or []),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return items


def delete_session(session_id: str, chat_dir: str = CHAT_DIR) -> bool:
    path = session_path(session_id, chat_dir=chat_dir)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def append_messages(
    session: Dict[str, Any],
    messages: List[Dict[str, Any]],
    chat_dir: str = CHAT_DIR,
    auto_title: bool = True,
) -> Dict[str, Any]:
    session.setdefault("messages", []).extend(messages)
    if auto_title and session.get("title") in {None, "", "Yeni sohbet"}:
        for m in session["messages"]:
            if m.get("role") == "user" and m.get("content"):
                title = str(m["content"]).strip().replace("\n", " ")
                session["title"] = (title[:48] + "…") if len(title) > 48 else title
                break
    save_session(session, chat_dir=chat_dir)
    return session
