"""Yanıt metninde kaynak chunk örtüşmelerini vurgula."""

from __future__ import annotations

import html
import re
from typing import Dict, List, Optional, Sequence, Tuple


def _tokenize(text: str) -> List[Tuple[str, int, int]]:
    """(token, start, end) — harf/rakam dizileri."""
    out = []
    for m in re.finditer(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{3,}", text or ""):
        out.append((m.group(0).lower(), m.start(), m.end()))
    return out


def find_overlap_spans(
    answer: str,
    chunk_text: str,
    *,
    min_tokens: int = 4,
) -> List[Tuple[int, int]]:
    """answer içinde chunk ile örtüşen en uzun ortak token zincirlerinin span'leri."""
    a_toks = _tokenize(answer)
    c_toks = _tokenize(chunk_text)
    if len(a_toks) < min_tokens or len(c_toks) < min_tokens:
        return []

    c_set_seqs = set()
    c_words = [t[0] for t in c_toks]
    for i in range(len(c_words) - min_tokens + 1):
        c_set_seqs.add(tuple(c_words[i : i + min_tokens]))

    spans: List[Tuple[int, int]] = []
    a_words = [t[0] for t in a_toks]
    i = 0
    while i <= len(a_words) - min_tokens:
        matched = False
        # mümkün olan en uzun eşleşme
        max_n = min(12, len(a_words) - i)
        for n in range(max_n, min_tokens - 1, -1):
            seq = tuple(a_words[i : i + n])
            # chunk içinde bu n-gram var mı?
            found = seq[:min_tokens] in c_set_seqs or any(
                c_words[j : j + n] == list(seq)
                for j in range(len(c_words) - n + 1)
            )
            if found:
                start = a_toks[i][1]
                end = a_toks[i + n - 1][2]
                spans.append((start, end))
                i += n
                matched = True
                break
        if not matched:
            i += 1
    return _merge_spans(spans)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def highlight_answer_html(
    answer: str,
    sources: Sequence[dict],
    *,
    min_tokens: int = 4,
) -> str:
    """Kaynak örtüşmelerini <mark> ile HTML'e çevirir; data-anchor ile bağlar."""
    if not answer:
        return ""
    # her kaynak için span topla → (start, end, anchor, label)
    tagged: List[Tuple[int, int, str, str]] = []
    for src in sources:
        text = src.get("text") or ""
        anchor = src.get("anchor") or ""
        label = src.get("label") or src.get("source_file") or ""
        for s, e in find_overlap_spans(answer, text, min_tokens=min_tokens):
            tagged.append((s, e, anchor, label))
    tagged = sorted(tagged, key=lambda x: (x[0], -(x[1] - x[0])))
    # örtüşen span'lerde ilkini tut
    chosen: List[Tuple[int, int, str, str]] = []
    occupied_until = -1
    for s, e, anchor, label in tagged:
        if s < occupied_until:
            continue
        chosen.append((s, e, anchor, label))
        occupied_until = e

    if not chosen:
        return html.escape(answer).replace("\n", "<br/>")

    parts = []
    cursor = 0
    for s, e, anchor, label in chosen:
        if cursor < s:
            parts.append(html.escape(answer[cursor:s]))
        frag = html.escape(answer[s:e])
        title = html.escape(label)
        if anchor:
            parts.append(
                f'<mark title="{title}" style="background:#fff3b0;padding:0 2px;">'
                f'<a href="#{html.escape(anchor)}" style="color:inherit;text-decoration:underline;">{frag}</a>'
                f"</mark>"
            )
        else:
            parts.append(
                f'<mark title="{title}" style="background:#fff3b0;padding:0 2px;">{frag}</mark>'
            )
        cursor = e
    if cursor < len(answer):
        parts.append(html.escape(answer[cursor:]))
    return "".join(parts).replace("\n", "<br/>")


def highlight_answer_markdown(
    answer: str,
    sources: Sequence[dict],
    *,
    min_tokens: int = 4,
) -> str:
    """Markdown **kalın** ile basit vurgu (stream/log için)."""
    if not answer:
        return ""
    spans: List[Tuple[int, int]] = []
    for src in sources:
        spans.extend(find_overlap_spans(answer, src.get("text") or "", min_tokens=min_tokens))
    spans = _merge_spans(spans)
    if not spans:
        return answer
    parts = []
    cursor = 0
    for s, e in spans:
        parts.append(answer[cursor:s])
        parts.append("**" + answer[s:e] + "**")
        cursor = e
    parts.append(answer[cursor:])
    return "".join(parts)
