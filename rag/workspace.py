"""Paylaşımlı veya kullanıcıya özel çalışma alanı yolları."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from app.config import (
    AUTH_SHARED_INDEX,
    CHAT_DIR,
    DATA_DIR,
    DOCSTORE_PATH,
    INDEX_PATH,
    INDEXES_DIR,
    METADATA_DIR,
)
from rag.auth import User, _safe_username, user_chat_dir


@dataclass(frozen=True)
class WorkspacePaths:
    data_dir: str
    indexes_dir: str
    index_path: str
    docstore_path: str
    source_meta_path: str
    chat_dir: str
    shared: bool
    username: Optional[str] = None

    @property
    def key(self) -> str:
        if self.shared or not self.username:
            return "shared"
        return f"user:{self.username}"


def resolve_workspace(
    user: Optional[User] = None,
    *,
    shared: Optional[bool] = None,
) -> WorkspacePaths:
    """Auth yoksa veya shared=True ise global yollar; aksi halde kullanıcı izolasyonu."""
    use_shared = AUTH_SHARED_INDEX if shared is None else shared

    if user is None or use_shared:
        return WorkspacePaths(
            data_dir=DATA_DIR,
            indexes_dir=INDEXES_DIR,
            index_path=INDEX_PATH,
            docstore_path=DOCSTORE_PATH,
            source_meta_path=os.path.join(METADATA_DIR, "sources.json"),
            chat_dir=user_chat_dir(user.username) if user else CHAT_DIR,
            shared=True,
            username=user.username if user else None,
        )

    safe = _safe_username(user.username) or "anon"
    data_dir = os.path.join(DATA_DIR, "users", safe)
    indexes_dir = os.path.join(INDEXES_DIR, "users", safe)
    meta_dir = os.path.join(METADATA_DIR, "users", safe)
    return WorkspacePaths(
        data_dir=data_dir,
        indexes_dir=indexes_dir,
        index_path=os.path.join(indexes_dir, "faiss.index"),
        docstore_path=os.path.join(meta_dir, "docstore.json"),
        source_meta_path=os.path.join(meta_dir, "sources.json"),
        chat_dir=user_chat_dir(user.username),
        shared=False,
        username=safe,
    )


def ensure_workspace_dirs(ws: WorkspacePaths) -> None:
    for d in {
        ws.data_dir,
        ws.indexes_dir,
        os.path.dirname(ws.docstore_path) or ".",
        os.path.dirname(ws.source_meta_path) or ".",
        ws.chat_dir,
    }:
        os.makedirs(d, exist_ok=True)
