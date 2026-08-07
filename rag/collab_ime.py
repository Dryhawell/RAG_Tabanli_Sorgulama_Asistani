"""CRDT paste / IME için metin aralığı işlemleri."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rag.collab_crdt import (
    ROOT_ID,
    CrdtDocument,
    _new_id,
    load_crdt,
)
from rag.collab_undo import apply_text_edit_with_undo


def _visible_ids(doc: CrdtDocument) -> List[str]:
    return doc.ordered_visible_ids()


def delete_range_ops(
    doc: CrdtDocument,
    start: int,
    end: int,
    *,
    lamport_start: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """[start, end) görünür karakter aralığını silen op'lar."""
    ids = _visible_ids(doc)
    start = max(0, min(start, len(ids)))
    end = max(start, min(end, len(ids)))
    ops: List[Dict[str, Any]] = []
    lamport = lamport_start
    for nid in ids[start:end]:
        lamport += 1
        ops.append({"type": "del", "target": nid, "lamport": lamport})
    return ops, lamport


def insert_text_ops(
    doc: CrdtDocument,
    index: int,
    text: str,
    *,
    lamport_start: int,
    author: str = "anon",
) -> Tuple[List[Dict[str, Any]], int]:
    """index konumuna metin ekleyen op zinciri (silme yokken doğru çalışır)."""
    ids = _visible_ids(doc)
    index = max(0, min(index, len(ids)))
    after = ids[index - 1] if index > 0 else ROOT_ID
    ops: List[Dict[str, Any]] = []
    lamport = lamport_start
    for ch in text:
        lamport += 1
        nid = _new_id()
        ops.append(
            {
                "type": "ins",
                "id": nid,
                "after": after,
                "char": ch,
                "lamport": lamport,
                "author": author,
            }
        )
        after = nid
    return ops, lamport


def replace_range_text(old: str, start: int, end: int, text: str) -> str:
    start = max(0, min(start, len(old)))
    end = max(start, min(end, len(old)))
    return old[:start] + text + old[end:]


def apply_paste(
    workspace_key: str,
    *,
    start: int,
    end: int,
    text: str,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    """Paste: seçili aralığı düz metinle değiştir (undo'lu full-doc diff).

    RGA'da aralık sil+ekle sibling sırasını bozabildiği için güvenli yol:
    materialize → string replace → apply_text_edit_with_undo.
    """
    doc = load_crdt(workspace_key, base=base)
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    new_text = replace_range_text(doc.materialize(), start, end, text)
    return apply_text_edit_with_undo(
        workspace_key,
        new_text,
        author=author,
        base=base,
    )


def apply_ime_commit(
    workspace_key: str,
    *,
    start: int,
    end: int,
    text: str,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CrdtDocument:
    """IME compositionend sonrası commit (paste ile aynı güvenli yol)."""
    return apply_paste(
        workspace_key,
        start=start,
        end=end,
        text=text,
        author=author,
        base=base,
    )
