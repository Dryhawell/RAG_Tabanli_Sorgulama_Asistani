"""RGA-tarzı metin CRDT — eşzamanlı düzenlemeleri otomatik birleştirir."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import METADATA_DIR

_lock = threading.Lock()
ROOT_ID = "__ROOT__"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(key: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (key or "shared").strip().lower()).strip("-")
    return slug or "shared"


def crdt_path(workspace_key: str, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_slug(workspace_key), "crdt_note.json")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class CrdtNode:
    id: str
    after: str
    char: str
    deleted: bool = False
    lamport: int = 0
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "after": self.after,
            "char": self.char,
            "deleted": self.deleted,
            "lamport": self.lamport,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrdtNode":
        return cls(
            id=str(data.get("id") or _new_id()),
            after=str(data.get("after") or ROOT_ID),
            char=str(data.get("char") or ""),
            deleted=bool(data.get("deleted")),
            lamport=int(data.get("lamport") or 0),
            author=str(data.get("author") or ""),
        )


@dataclass
class CrdtDocument:
    workspace_key: str
    nodes: Dict[str, CrdtNode] = field(default_factory=dict)
    lamport: int = 0
    revision: int = 0
    updated_at: str = ""
    updated_by: Optional[str] = None
    path: str = ""

    def bump_lamport(self, incoming: int) -> int:
        self.lamport = max(self.lamport, incoming) + 1
        return self.lamport

    def children_ids(self, parent_id: str) -> List[str]:
        kids = [nid for nid, n in self.nodes.items() if n.after == parent_id]
        kids.sort(key=lambda nid: (
            self.nodes[nid].lamport,
            self.nodes[nid].author,
            nid,
        ))
        return kids

    def materialize(self) -> str:
        chars: List[str] = []

        def walk(parent_id: str) -> None:
            for cid in self.children_ids(parent_id):
                node = self.nodes[cid]
                if not node.deleted:
                    chars.append(node.char)
                walk(cid)

        walk(ROOT_ID)
        return "".join(chars)

    def ordered_visible_ids(self) -> List[str]:
        ids: List[str] = []

        def walk(parent_id: str) -> None:
            for cid in self.children_ids(parent_id):
                if not self.nodes[cid].deleted:
                    ids.append(cid)
                walk(cid)

        walk(ROOT_ID)
        return ids


def _text_to_nodes(text: str, author: str, lamport_start: int = 0) -> Tuple[Dict[str, CrdtNode], int]:
    nodes: Dict[str, CrdtNode] = {}
    lamport = lamport_start
    after = ROOT_ID
    for ch in text:
        lamport += 1
        nid = _new_id()
        nodes[nid] = CrdtNode(
            id=nid,
            after=after,
            char=ch,
            lamport=lamport,
            author=author,
        )
        after = nid
    return nodes, lamport


def load_crdt(workspace_key: str, base: Optional[str] = None) -> CrdtDocument:
    path = crdt_path(workspace_key, base=base)
    doc = CrdtDocument(workspace_key=workspace_key, path=path)
    if not os.path.isfile(path):
        return doc
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return doc
    doc.lamport = int(data.get("lamport") or 0)
    doc.revision = int(data.get("revision") or 0)
    doc.updated_at = str(data.get("updated_at") or "")
    doc.updated_by = data.get("updated_by")
    raw_nodes = data.get("nodes") or {}
    if isinstance(raw_nodes, dict):
        for nid, row in raw_nodes.items():
            if isinstance(row, dict):
                doc.nodes[str(nid)] = CrdtNode.from_dict(row)
    return doc


def save_crdt(doc: CrdtDocument, base: Optional[str] = None) -> str:
    path = doc.path or crdt_path(doc.workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc.updated_at = _utcnow_iso()
    payload = {
        "format": "crdt-rga",
        "workspace_key": doc.workspace_key,
        "lamport": doc.lamport,
        "revision": doc.revision,
        "updated_at": doc.updated_at,
        "updated_by": doc.updated_by,
        "content_cache": doc.materialize(),
        "nodes": {nid: n.to_dict() for nid, n in doc.nodes.items()},
    }
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    doc.path = path
    return path


def apply_op(doc: CrdtDocument, op: Dict[str, Any], author: str) -> None:
    """Tek CRDT op uygular (ins/del)."""
    kind = op.get("type") or op.get("op")
    lamport = int(op.get("lamport") or 0)
    doc.bump_lamport(lamport)

    if kind == "ins":
        nid = str(op.get("id") or _new_id())
        after = str(op.get("after") or ROOT_ID)
        char = str(op.get("char") or "")
        if not char:
            return
        if nid in doc.nodes:
            existing = doc.nodes[nid]
            existing.deleted = False
            existing.char = char
            existing.after = after
            existing.lamport = doc.lamport
            existing.author = author
        else:
            doc.nodes[nid] = CrdtNode(
                id=nid,
                after=after,
                char=char,
                lamport=doc.lamport,
                author=author,
            )
    elif kind == "del":
        target = str(op.get("target") or op.get("id") or "")
        if target in doc.nodes:
            doc.nodes[target].deleted = True
            doc.nodes[target].lamport = doc.lamport
            doc.nodes[target].author = author


def apply_ops(doc: CrdtDocument, ops: List[Dict[str, Any]], author: str) -> CrdtDocument:
    for op in ops:
        apply_op(doc, op, author)
    doc.revision += 1
    doc.updated_by = author
    save_crdt(doc)
    return doc


def _diff_to_ops(
    old_ids: List[str],
    old_chars: List[str],
    new_text: str,
    author: str,
    lamport_start: int,
) -> List[Dict[str, Any]]:
    """Basit karakter diff → insert/delete op listesi."""
    ops: List[Dict[str, Any]] = []
    lamport = lamport_start
    # Ortak prefix
    i = 0
    while i < len(old_chars) and i < len(new_text) and old_chars[i] == new_text[i]:
        i += 1
    # Ortak suffix
    j_old = len(old_chars) - 1
    j_new = len(new_text) - 1
    while j_old >= i and j_new >= i and old_chars[j_old] == new_text[j_new]:
        j_old -= 1
        j_new -= 1

    middle_old = old_ids[i : j_old + 1] if j_old >= i else []
    middle_new = new_text[i : j_new + 1] if j_new >= i else []

    for nid in middle_old:
        lamport += 1
        ops.append({"type": "del", "target": nid, "lamport": lamport})

    after_id = old_ids[i - 1] if i > 0 else ROOT_ID
    for ch in middle_new:
        lamport += 1
        nid = _new_id()
        ops.append(
            {
                "type": "ins",
                "id": nid,
                "after": after_id,
                "char": ch,
                "lamport": lamport,
            }
        )
        after_id = nid
    return ops


def apply_text_edit(
    workspace_key: str,
    new_text: str,
    *,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    """Tam metin düzenlemesini CRDT op'larına çevirip birleştirir (çakışma yok)."""
    doc = load_crdt(workspace_key, base=base)
    old_text = doc.materialize()
    if new_text == old_text:
        return doc

    if not doc.nodes and new_text:
        nodes, lamport = _text_to_nodes(new_text, author or "anon", doc.lamport)
        doc.nodes = nodes
        doc.lamport = lamport
        doc.revision += 1
        doc.updated_by = author
        save_crdt(doc, base=base)
        return doc

    old_ids = doc.ordered_visible_ids()
    old_chars = [doc.nodes[nid].char for nid in old_ids]
    ops = _diff_to_ops(old_ids, old_chars, new_text, author or "anon", doc.lamport)
    return apply_ops(doc, ops, author or "anon")


def merge_remote_ops(
    workspace_key: str,
    ops: List[Dict[str, Any]],
    *,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    doc = load_crdt(workspace_key, base=base)
    return apply_ops(doc, ops, author or "anon")
