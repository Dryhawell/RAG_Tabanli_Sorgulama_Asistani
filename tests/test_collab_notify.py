"""@mention bildirim testleri."""

from rag.collab_notify import (
    extract_mentions,
    list_notifications,
    mark_notifications_read,
    notify_mentions,
)


def test_extract_mentions():
    assert extract_mentions("Merhaba @alice ve @bob") == ["alice", "bob"]
    assert extract_mentions("@alice @alice") == ["alice"]
    assert extract_mentions("düz metin") == []


def test_notify_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify.METADATA_DIR", str(tmp_path))
    ws = "notify-ws"
    events = notify_mentions(
        ws,
        "Lütfen @alice bak",
        from_user="bob",
        thread_id="th1",
    )
    assert len(events) == 1
    assert events[0]["target_user"] == "alice"
    rows = list_notifications(ws, "alice", unread_only=True)
    assert len(rows) == 1
    assert mark_notifications_read(ws, "alice") == 1
    assert list_notifications(ws, "alice", unread_only=True) == []
