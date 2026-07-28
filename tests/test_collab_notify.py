"""@mention bildirim testleri."""

from rag.collab_notify import (
    extract_mentions,
    list_notifications,
    list_notifications_global,
    mark_notifications_read,
    mark_notifications_read_global,
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


def test_notify_center_global(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify.METADATA_DIR", str(tmp_path))
    notify_mentions("ws-a", "Hey @alice", from_user="bob", thread_id="t1")
    notify_mentions("ws-b", "Ping @alice", from_user="carol", thread_id="t2")
    rows = list_notifications_global("alice", unread_only=True)
    assert len(rows) == 2
    workspaces = {r.get("workspace_key") for r in rows}
    assert workspaces == {"ws-a", "ws-b"}
    assert mark_notifications_read_global("alice") == 2
    assert list_notifications_global("alice", unread_only=True) == []
