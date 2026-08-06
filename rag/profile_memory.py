"""Uzun vadeli kullanıcı profil belleği (vektör retrieval)."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.config import METADATA_DIR

_lock = threading.Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def profile_dir(username: str, base: Optional[str] = None) -> str:
    safe = re.sub(r"[^a-z0-9._-]+", "", (username or "anon").strip().lower()) or "anon"
    root = base or os.path.join(METADATA_DIR, "profiles")
    path = os.path.join(root, safe)
    os.makedirs(path, exist_ok=True)
    return path


def profile_store_path(username: str, base: Optional[str] = None) -> str:
    return os.path.join(profile_dir(username, base=base), "long_memory.json")


@dataclass
class MemoryItem:
    id: str
    text: str
    kind: str = "fact"  # fact | preference | note
    created_at: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "created_at": self.created_at or _utcnow_iso(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryItem":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            text=str(data.get("text") or ""),
            kind=str(data.get("kind") or "fact"),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass
class ProfileMemoryStore:
    username: str
    items: List[MemoryItem] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None  # (n, dim)
    path: str = ""

    @property
    def size(self) -> int:
        return len(self.items)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vecs / norms).astype(np.float32)


def load_profile_memory(
    username: str,
    *,
    base: Optional[str] = None,
) -> ProfileMemoryStore:
    path = profile_store_path(username, base=base)
    store = ProfileMemoryStore(username=username, path=path)
    if not os.path.isfile(path):
        return store
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items_raw = data.get("items") or []
    store.items = [MemoryItem.from_dict(x) for x in items_raw if isinstance(x, dict)]
    emb = data.get("embeddings")
    if emb is not None:
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == len(store.items):
            store.embeddings = arr
    return store


def save_profile_memory(store: ProfileMemoryStore) -> str:
    path = store.path or profile_store_path(store.username)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "username": store.username,
        "updated_at": _utcnow_iso(),
        "items": [i.to_dict() for i in store.items],
        "embeddings": store.embeddings.tolist() if store.embeddings is not None else None,
    }
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    store.path = path
    return path


def add_memory(
    store: ProfileMemoryStore,
    text: str,
    *,
    kind: str = "fact",
    embedder=None,
    dedupe: bool = True,
) -> MemoryItem:
    text = (text or "").strip()
    if not text:
        raise ValueError("Boş bellek metni")
    if dedupe:
        low = text.lower()
        for existing in store.items:
            if existing.text.lower() == low:
                return existing
    item = MemoryItem(
        id=uuid.uuid4().hex[:10],
        text=text,
        kind=kind,
        created_at=_utcnow_iso(),
    )
    store.items.append(item)
    if embedder is not None:
        vec = embedder.encode([text])
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        vec = _l2_normalize(vec)
        if store.embeddings is None or store.embeddings.size == 0:
            store.embeddings = vec
        else:
            # dim uyumsuzsa yeniden kur
            if store.embeddings.shape[1] != vec.shape[1] or store.embeddings.shape[0] != len(store.items) - 1:
                texts = [i.text for i in store.items]
                store.embeddings = _l2_normalize(embedder.encode(texts))
            else:
                store.embeddings = np.vstack([store.embeddings, vec])
    save_profile_memory(store)
    return item


def rebuild_embeddings(store: ProfileMemoryStore, embedder) -> ProfileMemoryStore:
    if not store.items:
        store.embeddings = None
        save_profile_memory(store)
        return store
    texts = [i.text for i in store.items]
    store.embeddings = _l2_normalize(embedder.encode(texts))
    save_profile_memory(store)
    return store


def search_memories(
    store: ProfileMemoryStore,
    query: str,
    embedder,
    *,
    top_k: int = 5,
    min_score: float = 0.25,
) -> List[MemoryItem]:
    if not store.items:
        return []
    if store.embeddings is None or store.embeddings.shape[0] != len(store.items):
        rebuild_embeddings(store, embedder)
    q = embedder.encode([query])
    q = _l2_normalize(q)
    scores = (store.embeddings @ q.T).reshape(-1)
    order = np.argsort(-scores)[:top_k]
    out: List[MemoryItem] = []
    for idx in order:
        sc = float(scores[int(idx)])
        if sc < min_score:
            continue
        item = store.items[int(idx)]
        out.append(
            MemoryItem(
                id=item.id,
                text=item.text,
                kind=item.kind,
                created_at=item.created_at,
                score=sc,
            )
        )
    return out


def memories_context_block(items: Sequence[MemoryItem]) -> str:
    if not items:
        return ""
    lines = ["Uzun vadeli kullanıcı belleği:"]
    for it in items:
        lines.append(f"- ({it.kind}) {it.text}")
    return "\n".join(lines)


def ingest_session_facts(
    store: ProfileMemoryStore,
    facts: Sequence[str],
    *,
    embedder=None,
    kind: str = "fact",
) -> int:
    """Kısa bellek olgularını uzun vadeli store'a taşır."""
    n = 0
    for fact in facts:
        fact = (fact or "").strip()
        if len(fact) < 8:
            continue
        before = store.size
        add_memory(store, fact, kind=kind, embedder=embedder, dedupe=True)
        if store.size > before:
            n += 1
    return n
