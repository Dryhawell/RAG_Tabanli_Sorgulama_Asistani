"""CRDT rich-text işaretleri (bold/italic) ve yorum thread'leri."""

from __future__ import annotations

import json
import os
import re
import uuid
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import METADATA_DIR
from rag.collab_crdt import _safe_slug, load_crdt
from rag.collab_notify import notify_mentions
from rag.collab_richtext_audit import log_mark_audit


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


def richtext_path(workspace_key: str, base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", _safe_slug(workspace_key), "richtext.json")


ALLOWED_MARKS = {"bold", "italic", "code"}


def _map_old_index_to_new(
    idx: int,
    old: str,
    new: str,
    sm: Optional[SequenceMatcher] = None,
) -> int:
    if idx <= 0:
        return 0
    if idx >= len(old):
        return len(new)
    matcher = sm or SequenceMatcher(None, old, new)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" and i1 <= idx < i2:
            return j1 + (idx - i1)
        if tag == "delete" and i1 <= idx < i2:
            return j1
        if tag == "replace" and i1 <= idx < i2:
            offset = idx - i1
            span = max(1, j2 - j1)
            return j1 + min(offset, span - 1)
    return len(new)


def remap_span(start: int, end: int, old: str, new: str) -> tuple[int, int]:
    """Eski metin koordinatlarındaki [start, end) aralığını yeni metne taşır."""
    if not old and not new:
        return 0, 0
    sm = SequenceMatcher(None, old, new)
    new_start = _map_old_index_to_new(int(start), old, new, sm)
    if int(end) <= int(start):
        return new_start, new_start
    new_end = _map_old_index_to_new(int(end), old, new, sm)
    new_start = max(0, min(new_start, len(new)))
    new_end = max(new_start, min(new_end, len(new)))
    return new_start, new_end


def remap_richtext_after_text_change(
    workspace_key: str,
    old_text: str,
    new_text: str,
    *,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """CRDT düzenlemesi sonrası mark ve yorum aralıklarını yeniden eşler."""
    if old_text == new_text:
        return {"marks": 0, "comments": 0, "removed_marks": 0}
    store = load_richtext(workspace_key, base=base)
    removed_marks = 0
    new_marks: List[TextMark] = []
    for m in store.marks:
        ns, ne = remap_span(m.start, m.end, old_text, new_text)
        if ne <= ns:
            removed_marks += 1
            continue
        m.start, m.end = ns, ne
        new_marks.append(m)
    store.marks = new_marks
    store.marks = merge_overlapping_marks(store.marks)
    for c in store.comments:
        ns, ne = remap_span(c.start, c.end, old_text, new_text)
        c.start, c.end = ns, max(ns, ne)
    save_richtext(store, base=base)
    if removed_marks:
        log_mark_audit(
            workspace_key,
            "remap_prune",
            author=None,
            details={"removed_marks": removed_marks, "old_len": len(old_text), "new_len": len(new_text)},
            base=base,
        )
    return {
        "marks": len(store.marks),
        "comments": len(store.comments),
        "removed_marks": removed_marks,
    }


@dataclass
class TextMark:
    id: str
    mark: str  # bold | italic | code
    start: int
    end: int
    author: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mark": self.mark,
            "start": self.start,
            "end": self.end,
            "author": self.author,
            "created_at": self.created_at or _utcnow_iso(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextMark":
        return cls(
            id=str(data.get("id") or _new_id()),
            mark=str(data.get("mark") or "bold"),
            start=int(data.get("start") or 0),
            end=int(data.get("end") or 0),
            author=data.get("author"),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass
class CommentReply:
    id: str
    author: Optional[str]
    body: str
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author": self.author,
            "body": self.body,
            "created_at": self.created_at or _utcnow_iso(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommentReply":
        return cls(
            id=str(data.get("id") or _new_id()),
            author=data.get("author"),
            body=str(data.get("body") or ""),
            created_at=str(data.get("created_at") or ""),
        )


@dataclass
class CommentThread:
    id: str
    start: int
    end: int
    author: Optional[str]
    body: str
    resolved: bool = False
    created_at: str = ""
    replies: List[CommentReply] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "author": self.author,
            "body": self.body,
            "resolved": self.resolved,
            "created_at": self.created_at or _utcnow_iso(),
            "replies": [r.to_dict() for r in self.replies],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommentThread":
        replies = [
            CommentReply.from_dict(x)
            for x in (data.get("replies") or [])
            if isinstance(x, dict)
        ]
        return cls(
            id=str(data.get("id") or _new_id()),
            start=int(data.get("start") or 0),
            end=int(data.get("end") or 0),
            author=data.get("author"),
            body=str(data.get("body") or ""),
            resolved=bool(data.get("resolved")),
            created_at=str(data.get("created_at") or ""),
            replies=replies,
        )


@dataclass
class RichTextStore:
    workspace_key: str
    marks: List[TextMark] = field(default_factory=list)
    comments: List[CommentThread] = field(default_factory=list)
    path: str = ""


def load_richtext(workspace_key: str, base: Optional[str] = None) -> RichTextStore:
    path = richtext_path(workspace_key, base=base)
    store = RichTextStore(workspace_key=workspace_key, path=path)
    if not os.path.isfile(path):
        return store
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    store.marks = [
        TextMark.from_dict(x) for x in (data.get("marks") or []) if isinstance(x, dict)
    ]
    store.comments = [
        CommentThread.from_dict(x)
        for x in (data.get("comments") or [])
        if isinstance(x, dict)
    ]
    return store


def save_richtext(store: RichTextStore, base: Optional[str] = None) -> str:
    path = store.path or richtext_path(store.workspace_key, base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "workspace_key": store.workspace_key,
        "updated_at": _utcnow_iso(),
        "marks": [m.to_dict() for m in store.marks],
        "comments": [c.to_dict() for c in store.comments],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    store.path = path
    return path


def merge_overlapping_marks(marks: List[TextMark]) -> List[TextMark]:
    """Aynı türde çakışan veya bitişik mark aralıklarını birleştirir."""
    if not marks:
        return []
    by_type: Dict[str, List[TextMark]] = {}
    for m in marks:
        by_type.setdefault(m.mark, []).append(m)
    merged: List[TextMark] = []
    for mark_type, group in by_type.items():
        intervals = sorted(group, key=lambda x: (x.start, x.end))
        cur = intervals[0]
        for nxt in intervals[1:]:
            if nxt.start <= cur.end:
                cur = TextMark(
                    id=cur.id,
                    mark=mark_type,
                    start=min(cur.start, nxt.start),
                    end=max(cur.end, nxt.end),
                    author=cur.author or nxt.author,
                    created_at=cur.created_at or nxt.created_at,
                )
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    merged.sort(key=lambda x: (x.start, x.end, x.mark))
    return merged


def marks_in_range(
    store: RichTextStore,
    start: int,
    end: int,
) -> List[str]:
    """[start, end) aralığıyla kesişen mark türlerini döndürür."""
    start, end = int(start), int(end)
    if end <= start:
        return []
    found: set[str] = set()
    for m in store.marks:
        if m.start < end and m.end > start:
            found.add(m.mark)
    return sorted(found)


def summarize_mark_layers(
    workspace_key: str,
    *,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Metindeki katman bölgelerini (çoklu stil birleşimleri) özetler."""
    doc = load_crdt(workspace_key, base=base)
    text = doc.materialize()
    store = load_richtext(workspace_key, base=base)
    n = len(text)
    if n == 0:
        return []
    open_marks: List[set] = [set() for _ in range(n)]
    for m in store.marks:
        a = max(0, min(m.start, n))
        b = max(a, min(m.end, n))
        for i in range(a, b):
            open_marks[i].add(m.mark)
    regions: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        marks = open_marks[i]
        j = i + 1
        while j < n and open_marks[j] == marks:
            j += 1
        if marks:
            regions.append(
                {
                    "start": i,
                    "end": j,
                    "layers": sorted(marks),
                    "layer_key": "-".join(sorted(marks)),
                }
            )
        i = j
    return regions


def add_mark(
    workspace_key: str,
    *,
    mark: str,
    start: int,
    end: int,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> TextMark:
    mark = (mark or "").strip().lower()
    if mark not in ALLOWED_MARKS:
        raise ValueError(f"Geçersiz mark: {mark}")
    start, end = int(start), int(end)
    if end <= start:
        raise ValueError("Mark aralığı boş")
    store = load_richtext(workspace_key, base=base)
    item = TextMark(
        id=_new_id(),
        mark=mark,
        start=start,
        end=end,
        author=author,
        created_at=_utcnow_iso(),
    )
    store.marks.append(item)
    store.marks = merge_overlapping_marks(store.marks)
    save_richtext(store, base=base)
    log_mark_audit(
        workspace_key,
        "add",
        mark=item.to_dict(),
        author=author,
        base=base,
    )
    return item


def remove_mark(workspace_key: str, mark_id: str, base: Optional[str] = None) -> bool:
    store = load_richtext(workspace_key, base=base)
    before = len(store.marks)
    removed = None
    kept: List[TextMark] = []
    for m in store.marks:
        if m.id == mark_id:
            removed = m
        else:
            kept.append(m)
    store.marks = kept
    if len(store.marks) == before:
        return False
    save_richtext(store, base=base)
    if removed:
        log_mark_audit(
            workspace_key,
            "remove",
            mark=removed.to_dict(),
            author=removed.author,
            base=base,
        )
    return True


def add_comment(
    workspace_key: str,
    *,
    start: int,
    end: int,
    body: str,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CommentThread:
    body = (body or "").strip()
    if len(body) < 1:
        raise ValueError("Boş yorum")
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    store = load_richtext(workspace_key, base=base)
    thread = CommentThread(
        id=_new_id(),
        start=start,
        end=max(start, end),
        author=author,
        body=body,
        created_at=_utcnow_iso(),
    )
    store.comments.append(thread)
    save_richtext(store, base=base)
    return thread


def reply_comment(
    workspace_key: str,
    thread_id: str,
    *,
    body: str,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> CommentThread:
    body = (body or "").strip()
    if not body:
        raise ValueError("Boş yanıt")
    store = load_richtext(workspace_key, base=base)
    for thread in store.comments:
        if thread.id == thread_id:
            thread.replies.append(
                CommentReply(
                    id=_new_id(),
                    author=author,
                    body=body,
                    created_at=_utcnow_iso(),
                )
            )
            save_richtext(store, base=base)
            return thread
    raise KeyError(f"Thread yok: {thread_id}")


def resolve_comment(
    workspace_key: str,
    thread_id: str,
    *,
    resolved: bool = True,
    base: Optional[str] = None,
) -> CommentThread:
    store = load_richtext(workspace_key, base=base)
    for thread in store.comments:
        if thread.id == thread_id:
            thread.resolved = resolved
            save_richtext(store, base=base)
            return thread
    raise KeyError(f"Thread yok: {thread_id}")


def render_mention_html(text: str) -> str:
    """Yorum metninde @kullanıcı etiketlerini vurgular."""
    if not text:
        return ""
    parts: List[str] = []
    last = 0
    for match in re.finditer(r"@([a-zA-Z0-9_.-]{2,32})", text):
        a, b = match.span()
        chunk = (
            text[last:a]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        parts.append(chunk)
        user = match.group(1)
        parts.append(
            f'<span class="crdt-mention" style="color:#4a90d9;font-weight:600">@{user}</span>'
        )
        last = b
    tail = text[last:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts.append(tail)
    return "".join(parts)


def render_rich_html(
    workspace_key: str,
    *,
    base: Optional[str] = None,
) -> str:
    """Düz metni mark'larla HTML'e çevirir; yorum ankorlarını işaretler."""
    doc = load_crdt(workspace_key, base=base)
    text = doc.materialize()
    store = load_richtext(workspace_key, base=base)
    n = len(text)
    if n == 0:
        return ""

    # her karakter için açık etiket seti
    open_marks: List[set] = [set() for _ in range(n)]
    for m in store.marks:
        a = max(0, min(m.start, n))
        b = max(a, min(m.end, n))
        for i in range(a, b):
            open_marks[i].add(m.mark)

    comment_starts = {}
    for c in store.comments:
        if c.resolved:
            continue
        a = max(0, min(c.start, n))
        comment_starts.setdefault(a, []).append(c.id)

    parts: List[str] = []
    i = 0
    while i < n:
        if i in comment_starts:
            for cid in comment_starts[i]:
                parts.append(
                    f'<span class="crdt-comment" data-id="{cid}" title="yorum">&#9679;</span>'
                )
        marks = open_marks[i]
        j = i + 1
        while j < n and open_marks[j] == marks and j not in comment_starts:
            j += 1
        chunk = (
            text[i:j]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        if "code" in marks:
            chunk = f"<code>{chunk}</code>"
        if "bold" in marks:
            chunk = f"<strong>{chunk}</strong>"
        if "italic" in marks:
            chunk = f"<em>{chunk}</em>"
        if marks:
            layer_key = "-".join(sorted(marks))
            chunk = (
                f'<span class="crdt-layer crdt-layer-{layer_key}" '
                f'data-layers="{layer_key}">{chunk}</span>'
            )
        parts.append(chunk)
        i = j
    return "".join(parts)


def richtext_snapshot(workspace_key: str, base: Optional[str] = None) -> Dict[str, Any]:
    store = load_richtext(workspace_key, base=base)
    return {
        "marks": [m.to_dict() for m in store.marks],
        "comments": [c.to_dict() for c in store.comments],
        "html": render_rich_html(workspace_key, base=base),
        "layers": summarize_mark_layers(workspace_key, base=base),
    }


def apply_rich_ops(
    workspace_key: str,
    ops: List[Dict[str, Any]],
    *,
    author: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """WebSocket / UI için zengin metin işlemleri.

    Desteklenen op.type:
      add_mark, remove_mark, add_comment, reply_comment, resolve_comment
    """
    results: List[Dict[str, Any]] = []
    mention_events: List[Dict[str, Any]] = []
    for raw in ops or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("type") or raw.get("op") or "").strip().lower()
        try:
            if kind == "add_mark":
                item = add_mark(
                    workspace_key,
                    mark=str(raw.get("mark") or "bold"),
                    start=int(raw.get("start") or 0),
                    end=int(raw.get("end") or 0),
                    author=raw.get("author") or author,
                    base=base,
                )
                results.append({"type": kind, "ok": True, "item": item.to_dict()})
            elif kind == "remove_mark":
                ok = remove_mark(
                    workspace_key, str(raw.get("id") or raw.get("mark_id") or ""), base=base
                )
                results.append({"type": kind, "ok": ok})
            elif kind == "add_comment":
                thread = add_comment(
                    workspace_key,
                    start=int(raw.get("start") or 0),
                    end=int(raw.get("end") or 0),
                    body=str(raw.get("body") or ""),
                    author=raw.get("author") or author,
                    base=base,
                )
                mention_events.extend(
                    notify_mentions(
                        workspace_key,
                        thread.body,
                        from_user=thread.author,
                        thread_id=thread.id,
                        base=base,
                    )
                )
                results.append({"type": kind, "ok": True, "item": thread.to_dict()})
            elif kind == "reply_comment":
                thread = reply_comment(
                    workspace_key,
                    str(raw.get("thread_id") or raw.get("id") or ""),
                    body=str(raw.get("body") or ""),
                    author=raw.get("author") or author,
                    base=base,
                )
                mention_events.extend(
                    notify_mentions(
                        workspace_key,
                        str(raw.get("body") or ""),
                        from_user=raw.get("author") or author,
                        thread_id=thread.id,
                        base=base,
                    )
                )
                results.append({"type": kind, "ok": True, "item": thread.to_dict()})
            elif kind == "resolve_comment":
                thread = resolve_comment(
                    workspace_key,
                    str(raw.get("thread_id") or raw.get("id") or ""),
                    resolved=bool(raw.get("resolved", True)),
                    base=base,
                )
                results.append({"type": kind, "ok": True, "item": thread.to_dict()})
            else:
                results.append({"type": kind or "?", "ok": False, "error": "bilinmeyen op"})
        except Exception as exc:
            results.append({"type": kind, "ok": False, "error": str(exc)})
    snap = richtext_snapshot(workspace_key, base=base)
    snap["results"] = results
    snap["notifications"] = mention_events
    return snap
