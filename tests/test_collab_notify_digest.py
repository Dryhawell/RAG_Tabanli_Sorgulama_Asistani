"""Bildirim digest testleri."""

from datetime import datetime, timezone

from rag.collab_notify_digest import build_digest_body, collect_digest_events


def test_collect_digest_events(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify.METADATA_DIR", str(tmp_path))
    from rag.collab_notify import notify_mentions

    notify_mentions("ws-d", "Hi @alice", from_user="bob", thread_id="t1")
    rows = collect_digest_events("alice", hours=24, base=str(tmp_path))
    assert len(rows) >= 1


def test_build_digest_body():
    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "test mesaj",
        }
    ]
    body = build_digest_body(events, "alice")
    assert "alice" in body
    assert "bob" in body
    assert "test mesaj" in body
