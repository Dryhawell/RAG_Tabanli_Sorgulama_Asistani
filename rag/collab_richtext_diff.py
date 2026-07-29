"""Rich-text mark audit diff görselleştirme."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

ACTION_COLORS = {
    "add": "#2ecc71",
    "remove": "#e74c3c",
    "remap_prune": "#f39c12",
    "test": "#3498db",
}


def snippet_at(text: str, start: int, end: int, max_len: int = 48) -> str:
    if not text:
        return ""
    start = max(0, min(int(start), len(text)))
    end = max(start, min(int(end), len(text)))
    chunk = text[start:end]
    if len(chunk) > max_len:
        return chunk[:max_len] + "…"
    return chunk


def diff_mark_snapshots(
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    before_map = {str(m.get("id")): m for m in before if m.get("id")}
    after_map = {str(m.get("id")): m for m in after if m.get("id")}
    added = [after_map[k] for k in after_map if k not in before_map]
    removed = [before_map[k] for k in before_map if k not in after_map]
    changed: List[Dict[str, Any]] = []
    for key in before_map:
        if key in after_map and before_map[key] != after_map[key]:
            changed.append({"before": before_map[key], "after": after_map[key]})
    return {"added": added, "removed": removed, "changed": changed}


def format_mark_diff_line(entry: Dict[str, Any], text: str) -> str:
    action = str(entry.get("action") or "?")
    mark = entry.get("mark") or {}
    kind = str(mark.get("mark") or "")
    start = mark.get("start", "")
    end = mark.get("end", "")
    snippet = snippet_at(text, int(mark.get("start") or 0), int(mark.get("end") or 0))
    author = entry.get("author") or "-"
    return f"{action} · {kind} [{start}:{end}] «{snippet}» · {author}"


def html_mark_diff_entry(entry: Dict[str, Any], text: str) -> str:
    action = str(entry.get("action") or "?")
    mark = entry.get("mark") or {}
    color = ACTION_COLORS.get(action, "#95a5a6")
    start = int(mark.get("start") or 0)
    end = int(mark.get("end") or start)
    kind = html.escape(str(mark.get("mark") or ""))
    snippet = html.escape(snippet_at(text, start, end))
    author = html.escape(str(entry.get("author") or "-"))
    label = f"{action} · {kind} [{start}:{end}]"
    return (
        f'<div style="margin:4px 0;padding:6px 8px;border-left:3px solid {color};'
        f'background:{color}12;">'
        f'<span style="color:{color};font-weight:600;">{html.escape(label)}</span> '
        f'<span style="background:{color}33;padding:2px 6px;border-radius:3px;">'
        f"{snippet or '—'}</span> "
        f'<span style="color:#666;font-size:0.85em;">{author}</span>'
        f"</div>"
    )


def rebuild_mark_timeline(
    audit_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Audit kayıtlarını replay ederek her adım için before/after diff üretir."""
    marks: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    for entry in audit_rows:
        before = [dict(m) for m in marks]
        action = str(entry.get("action") or "")
        mark = entry.get("mark") or {}
        if action == "add" and mark.get("id"):
            marks.append(dict(mark))
        elif action == "remove" and mark.get("id"):
            marks = [m for m in marks if m.get("id") != mark.get("id")]
        after = [dict(m) for m in marks]
        timeline.append(
            {
                "entry": entry,
                "before": before,
                "after": after,
                "diff": diff_mark_snapshots(before, after),
            }
        )
    return timeline


def summarize_mark_audit_diffs(
    audit_rows: List[Dict[str, Any]],
    text: str,
    *,
    limit: int = 15,
) -> List[Dict[str, Any]]:
    rows = audit_rows[-limit:]
    timeline = rebuild_mark_timeline(rows)
    out: List[Dict[str, Any]] = []
    for item in timeline:
        entry = item["entry"]
        diff = item["diff"]
        out.append(
            {
                "ts": entry.get("ts"),
                "action": entry.get("action"),
                "line": format_mark_diff_line(entry, text),
                "html": html_mark_diff_entry(entry, text),
                "added": len(diff.get("added") or []),
                "removed": len(diff.get("removed") or []),
                "changed": len(diff.get("changed") or []),
            }
        )
    return out
