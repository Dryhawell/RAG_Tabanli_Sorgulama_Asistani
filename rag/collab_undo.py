"""CRDT undo/redo yığını."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.collab_crdt import (
    CrdtDocument,
    _diff_to_ops,
    _text_to_nodes,
    apply_op,
    apply_ops,
    crdt_path,
    load_crdt,
    save_crdt,
)


def undo_path(workspace_key: str, base: Optional[str] = None) -> str:
    path = crdt_path(workspace_key, base=base)
    return path.replace("crdt_note.json", "undo_stack.json")


@dataclass
class UndoStack:
    workspace_key: str
    undo: List[Dict[str, Any]] = field(default_factory=list)  # {forward, inverse}
    redo: List[Dict[str, Any]] = field(default_factory=list)
    limit: int = 50
    path: str = ""

    def push_edit(self, forward: List[Dict[str, Any]], inverse: List[Dict[str, Any]]) -> None:
        if not forward:
            return
        self.undo.append({"forward": forward, "inverse": inverse})
        if len(self.undo) > self.limit:
            self.undo = self.undo[-self.limit :]
        self.redo.clear()


def load_undo_stack(workspace_key: str, base: Optional[str] = None) -> UndoStack:
    path = undo_path(workspace_key, base=base)
    stack = UndoStack(workspace_key=workspace_key, path=path)
    if not os.path.isfile(path):
        return stack
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stack.undo = list(data.get("undo") or [])
    stack.redo = list(data.get("redo") or [])
    stack.limit = int(data.get("limit") or 50)
    return stack


def save_undo_stack(stack: UndoStack) -> str:
    path = stack.path or undo_path(stack.workspace_key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "workspace_key": stack.workspace_key,
                "undo": stack.undo,
                "redo": stack.redo,
                "limit": stack.limit,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    stack.path = path
    return path


def invert_ops(ops: List[Dict[str, Any]], doc: CrdtDocument) -> List[Dict[str, Any]]:
    inverse: List[Dict[str, Any]] = []
    for op in reversed(ops):
        kind = op.get("type") or op.get("op")
        if kind == "ins":
            nid = str(op.get("id") or "")
            inverse.append(
                {"type": "del", "target": nid, "lamport": int(op.get("lamport") or 0) + 1}
            )
        elif kind == "del":
            target = str(op.get("target") or op.get("id") or "")
            node = doc.nodes.get(target)
            if node is None:
                continue
            inverse.append(
                {
                    "type": "ins",
                    "id": node.id,
                    "after": node.after,
                    "char": node.char,
                    "lamport": int(op.get("lamport") or 0) + 1,
                }
            )
    return inverse


def _apply_ops_keep_revision(doc: CrdtDocument, ops: List[Dict[str, Any]], author: str) -> None:
    for op in ops:
        apply_op(doc, op, author)
    doc.revision += 1
    doc.updated_by = author
    save_crdt(doc)


def apply_text_edit_with_undo(
    workspace_key: str,
    new_text: str,
    *,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    doc = load_crdt(workspace_key, base=base)
    old_text = doc.materialize()
    if new_text == old_text:
        return doc

    stack = load_undo_stack(workspace_key, base=base)
    user = author or "anon"

    if not doc.nodes and new_text:
        nodes, lamport = _text_to_nodes(new_text, user, doc.lamport)
        forward = [
            {
                "type": "ins",
                "id": n.id,
                "after": n.after,
                "char": n.char,
                "lamport": n.lamport,
            }
            for n in nodes.values()
        ]
        # sıralı zincir için materialize sırasını kullan
        ordered = []
        # nodes dict insertion order = creation order for py3.7+
        for n in nodes.values():
            ordered.append(
                {
                    "type": "ins",
                    "id": n.id,
                    "after": n.after,
                    "char": n.char,
                    "lamport": n.lamport,
                }
            )
        inverse = [
            {"type": "del", "target": n.id, "lamport": lamport + i + 1}
            for i, n in enumerate(nodes.values())
        ]
        stack.push_edit(ordered or forward, inverse)
        save_undo_stack(stack)
        doc.nodes = nodes
        doc.lamport = lamport
        doc.revision += 1
        doc.updated_by = user
        save_crdt(doc, base=base)
        return doc

    old_ids = doc.ordered_visible_ids()
    old_chars = [doc.nodes[nid].char for nid in old_ids]
    ops = _diff_to_ops(old_ids, old_chars, new_text, user, doc.lamport)
    inverse = invert_ops(ops, doc)
    stack.push_edit(ops, inverse)
    save_undo_stack(stack)
    return apply_ops(doc, ops, user)


def undo_edit(
    workspace_key: str,
    *,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    stack = load_undo_stack(workspace_key, base=base)
    if not stack.undo:
        return load_crdt(workspace_key, base=base)
    entry = stack.undo.pop()
    doc = load_crdt(workspace_key, base=base)
    _apply_ops_keep_revision(doc, list(entry.get("inverse") or []), author or "anon")
    stack.redo.append(entry)
    save_undo_stack(stack)
    return doc


def redo_edit(
    workspace_key: str,
    *,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    stack = load_undo_stack(workspace_key, base=base)
    if not stack.redo:
        return load_crdt(workspace_key, base=base)
    entry = stack.redo.pop()
    doc = load_crdt(workspace_key, base=base)
    _apply_ops_keep_revision(doc, list(entry.get("forward") or []), author or "anon")
    stack.undo.append(entry)
    save_undo_stack(stack)
    return doc
