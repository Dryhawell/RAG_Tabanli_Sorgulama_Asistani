"""Rich-text mark remap testleri."""

from rag.collab_crdt import apply_text_edit
from rag.collab_richtext import (
    add_mark,
    load_richtext,
    remap_richtext_after_text_change,
    remap_span,
)


def test_remap_span_insert():
    old = "hello world"
    new = "hello big world"
    start, end = remap_span(6, 11, old, new)
    assert start == 10
    assert end == 15


def test_remap_after_edit(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    ws = "remap-ws"
    apply_text_edit(ws, "Merhaba kalin dunya", author="a")
    add_mark(ws, mark="bold", start=8, end=13, author="a")
    apply_text_edit(ws, "Merhaba kalindunya", author="a")
    store = load_richtext(ws)
    assert len(store.marks) == 1
    m = store.marks[0]
    assert m.start >= 8
    assert m.end > m.start


def test_remap_manual(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    ws = "manual"
    apply_text_edit(ws, "abcdef", author="a", base=str(tmp_path))
    add_mark(ws, mark="italic", start=1, end=4, author="a", base=str(tmp_path))
    stats = remap_richtext_after_text_change(ws, "abcdef", "abXdef", base=str(tmp_path))
    assert stats["marks"] == 1
    store = load_richtext(ws, base=str(tmp_path))
    assert store.marks[0].start == 1
    assert store.marks[0].end == 4
