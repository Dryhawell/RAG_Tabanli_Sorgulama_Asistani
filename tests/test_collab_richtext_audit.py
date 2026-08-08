"""Mark audit testleri."""

from rag.collab_richtext import add_mark, remove_mark
from rag.collab_richtext_audit import log_mark_audit, read_mark_audit


def test_mark_audit_log(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_richtext_audit.METADATA_DIR", str(tmp_path))
    ws = "audit-ws"
    m = add_mark(ws, mark="bold", start=0, end=3, author="alice")
    rows = read_mark_audit(ws)
    assert len(rows) >= 1
    assert rows[-1]["action"] == "add"
    remove_mark(ws, m.id)
    rows2 = read_mark_audit(ws)
    assert any(r["action"] == "remove" for r in rows2)


def test_log_mark_audit_direct(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_richtext_audit.METADATA_DIR", str(tmp_path))
    log_mark_audit("ws", "test", mark={"mark": "italic"}, author="bob")
    rows = read_mark_audit("ws")
    assert len(rows) == 1
    assert rows[0]["action"] == "test"


def test_comment_audit_log(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_richtext_audit.METADATA_DIR", str(tmp_path))
    from rag.collab_richtext_audit import log_comment_audit, read_comment_audit

    log_comment_audit(
        "ws-c",
        "add",
        thread={"id": "t1", "body": "hello", "start": 0, "end": 3},
        author="alice",
    )
    rows = read_comment_audit("ws-c")
    assert len(rows) == 1
    assert rows[0]["action"] == "add"
