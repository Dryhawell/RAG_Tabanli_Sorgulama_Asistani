"""Sohbet oturumunu JSON / Markdown olarak dışa aktarma."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def export_session_json(session: Dict[str, Any], *, indent: int = 2) -> str:
    payload = {
        "id": session.get("id"),
        "title": session.get("title"),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "messages": session.get("messages") or [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def export_session_markdown(session: Dict[str, Any]) -> str:
    title = session.get("title") or "Sohbet"
    lines: List[str] = [
        f"# {title}",
        "",
        f"- id: `{session.get('id')}`",
        f"- created_at: {session.get('created_at') or '-'}",
        f"- updated_at: {session.get('updated_at') or '-'}",
        "",
        "---",
        "",
    ]
    for msg in session.get("messages") or []:
        role = msg.get("role") or "unknown"
        content = (msg.get("content") or "").strip()
        heading = "Kullanıcı" if role == "user" else ("Asistan" if role == "assistant" else role)
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(content or "_(boş)_")
        lines.append("")
        sources = msg.get("sources") or []
        if sources:
            lines.append("### Kaynaklar")
            lines.append("")
            for src in sources:
                label = src.get("label") or src.get("source_file") or "?"
                score = src.get("score")
                if score is not None:
                    lines.append(f"- {label} (skor: {float(score):.3f})")
                else:
                    lines.append(f"- {label}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
