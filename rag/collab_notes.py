"""Çalışma alanına özel işbirlikçi paylaşımlı not (dosya tabanlı)."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import METADATA_DIR

_lock = threading.Lock()
_HISTORY_LIMIT = 30


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(key: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (key or "shared").strip().lower()).strip("-")
    return slug or "shared"


def note_path(workspace_key: str, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_slug(workspace_key), "shared_note.json")


@dataclass
class CollabNote:
    content: str = ""
    revision: int = 0
    updated_at: str = ""
    updated_by: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


def load_note(workspace_key: str, base: Optional[str] = None) -> CollabNote:
    path = note_path(workspace_key, base=base)
    if not os.path.isfile(path):
        return CollabNote()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return CollabNote()
    return CollabNote(
        content=str(data.get("content") or ""),
        revision=int(data.get("revision") or 0),
        updated_at=str(data.get("updated_at") or ""),
        updated_by=data.get("updated_by"),
        history=list(data.get("history") or []),
    )


def save_note(
    workspace_key: str,
    content: str,
    *,
    username: Optional[str] = None,
    expected_revision: Optional[int] = None,
    base: Optional[str] = None,
    force: bool = False,
) -> CollabNote:
    path = note_path(workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with _lock:
        current = load_note(workspace_key, base=base)
        if (
            expected_revision is not None
            and not force
            and current.revision != expected_revision
        ):
            raise ValueError(
                f"Çakışma: not başka biri tarafından güncellendi (rev {current.revision})."
            )

        preview = (content or "").strip().replace("\n", " ")
        if len(preview) > 96:
            preview = preview[:96] + "…"

        new_rev = current.revision + 1
        history = list(current.history or [])
        history.append(
            {
                "revision": new_rev,
                "ts": _utcnow_iso(),
                "user": username,
                "preview": preview,
            }
        )
        if len(history) > _HISTORY_LIMIT:
            history = history[-_HISTORY_LIMIT:]

        note = CollabNote(
            content=content or "",
            revision=new_rev,
            updated_at=_utcnow_iso(),
            updated_by=username,
            history=history,
        )
        payload = {
            "workspace_key": workspace_key,
            "content": note.content,
            "revision": note.revision,
            "updated_at": note.updated_at,
            "updated_by": note.updated_by,
            "history": note.history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return note
