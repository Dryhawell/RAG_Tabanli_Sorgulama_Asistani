"""Paylaşımlı / kullanıcıya özel / tenant çalışma alanı yolları."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from app.config import (
    AUTH_SHARED_INDEX,
    CHAT_DIR,
    DATA_DIR,
    DOCSTORE_PATH,
    ENABLE_TENANTS,
    INDEX_PATH,
    INDEXES_DIR,
    METADATA_DIR,
)
from rag.auth import User, _safe_tenant, _safe_username, user_chat_dir


@dataclass(frozen=True)
class WorkspacePaths:
    data_dir: str
    indexes_dir: str
    index_path: str
    docstore_path: str
    source_meta_path: str
    chat_dir: str
    audit_path: str
    metrics_path: str
    shared: bool
    username: Optional[str] = None
    tenant_id: Optional[str] = None

    @property
    def key(self) -> str:
        parts = []
        if self.tenant_id:
            parts.append(f"tenant:{self.tenant_id}")
        if self.shared or not self.username:
            parts.append("shared")
        else:
            parts.append(f"user:{self.username}")
        return "|".join(parts)


def resolve_workspace(
    user: Optional[User] = None,
    *,
    shared: Optional[bool] = None,
    enable_tenants: Optional[bool] = None,
) -> WorkspacePaths:
    """Auth yoksa veya shared=True ise ortak yollar; aksi halde kullanıcı izolasyonu.

    Tenant açıkken kökler `.../tenants/<tenant_id>/` altına alınır.
    """
    use_shared = AUTH_SHARED_INDEX if shared is None else shared
    use_tenants = ENABLE_TENANTS if enable_tenants is None else enable_tenants

    tenant = None
    if use_tenants and user is not None:
        tenant = _safe_tenant(user.tenant_id)

    def _root(base: str) -> str:
        if tenant:
            return os.path.join(base, "tenants", tenant)
        return base

    data_root = _root(DATA_DIR)
    indexes_root = _root(INDEXES_DIR)
    meta_root = _root(METADATA_DIR)
    chat_root = _root(CHAT_DIR)
    audit_path = os.path.join(meta_root, "audit.jsonl")
    metrics_path = os.path.join(meta_root, "metrics.jsonl")

    if user is None or use_shared:
        # Tenant yokken geriye dönük global yollar
        if tenant is None:
            return WorkspacePaths(
                data_dir=DATA_DIR,
                indexes_dir=INDEXES_DIR,
                index_path=INDEX_PATH,
                docstore_path=DOCSTORE_PATH,
                source_meta_path=os.path.join(METADATA_DIR, "sources.json"),
                chat_dir=user_chat_dir(user.username) if user else CHAT_DIR,
                audit_path=os.path.join(METADATA_DIR, "audit.jsonl"),
                metrics_path=os.path.join(METADATA_DIR, "metrics.jsonl"),
                shared=True,
                username=user.username if user else None,
                tenant_id=None,
            )
        return WorkspacePaths(
            data_dir=data_root,
            indexes_dir=indexes_root,
            index_path=os.path.join(indexes_root, "faiss.index"),
            docstore_path=os.path.join(meta_root, "docstore.json"),
            source_meta_path=os.path.join(meta_root, "sources.json"),
            chat_dir=user_chat_dir(user.username, base=chat_root) if user else chat_root,
            audit_path=audit_path,
            metrics_path=metrics_path,
            shared=True,
            username=user.username if user else None,
            tenant_id=tenant,
        )

    safe = _safe_username(user.username) or "anon"
    data_dir = os.path.join(data_root, "users", safe)
    indexes_dir = os.path.join(indexes_root, "users", safe)
    meta_dir = os.path.join(meta_root, "users", safe)
    return WorkspacePaths(
        data_dir=data_dir,
        indexes_dir=indexes_dir,
        index_path=os.path.join(indexes_dir, "faiss.index"),
        docstore_path=os.path.join(meta_dir, "docstore.json"),
        source_meta_path=os.path.join(meta_dir, "sources.json"),
        chat_dir=user_chat_dir(user.username, base=chat_root, tenant_id=None),
        audit_path=audit_path,
        metrics_path=metrics_path,
        shared=False,
        username=safe,
        tenant_id=tenant,
    )


def ensure_workspace_dirs(ws: WorkspacePaths) -> None:
    for d in {
        ws.data_dir,
        ws.indexes_dir,
        os.path.dirname(ws.docstore_path) or ".",
        os.path.dirname(ws.source_meta_path) or ".",
        os.path.dirname(ws.audit_path) or ".",
        os.path.dirname(ws.metrics_path) or ".",
        ws.chat_dir,
    }:
        os.makedirs(d, exist_ok=True)
