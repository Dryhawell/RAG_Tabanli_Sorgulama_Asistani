"""CRDT rich-text marks ve yorum thread testleri."""

from rag.collab_crdt import apply_text_edit
from rag.collab_richtext import (
    add_comment,
    add_mark,
    apply_rich_ops,
    load_richtext,
    render_rich_html,
    reply_comment,
    resolve_comment,
)


def test_marks_and_comments(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    ws = "rich-ws"
    apply_text_edit(ws, "Merhaba kalin dunya", author="alice")
    m = add_mark(ws, mark="bold", start=8, end=13, author="alice")
    assert m.mark == "bold"
    add_mark(ws, mark="italic", start=14, end=19, author="bob")
    thread = add_comment(ws, start=0, end=7, body="Selamlama?", author="bob")
    reply_comment(ws, thread.id, body="Evet", author="alice")
    html = render_rich_html(ws)
    assert "<strong>" in html
    assert "<em>" in html
    assert "crdt-comment" in html
    store = load_richtext(ws)
    assert len(store.marks) == 2
    assert len(store.comments) == 1
    assert len(store.comments[0].replies) == 1
    resolve_comment(ws, thread.id, resolved=True)
    html2 = render_rich_html(ws)
    assert "crdt-comment" not in html2


def test_apply_rich_ops(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    ws = "ops-ws"
    apply_text_edit(ws, "abc def", author="a")
    snap = apply_rich_ops(
        ws,
        [
            {"type": "add_mark", "mark": "code", "start": 0, "end": 3},
            {"type": "add_comment", "start": 4, "end": 7, "body": "nedir?"},
        ],
        author="a",
    )
    assert snap["results"][0]["ok"] is True
    assert snap["results"][1]["ok"] is True
    assert "<code>" in snap["html"]
    tid = snap["results"][1]["item"]["id"]
    snap2 = apply_rich_ops(
        ws,
        [{"type": "reply_comment", "thread_id": tid, "body": "cevap"}],
        author="b",
    )
    assert snap2["results"][0]["ok"] is True
