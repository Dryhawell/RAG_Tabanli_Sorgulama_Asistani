"""Collab presence persistence + typing testleri."""

from rag.collab_presence import (
    bind_client,
    color_for_user,
    load_presence_snapshot,
    merge_presence_for_join,
    persist_room_presence,
    presence_entry,
    presence_list,
    release_client,
    update_cursor,
    update_typing,
)


class _FakeWs:
    pass


def test_bind_update_release_presence():
    ws = _FakeWs()
    meta = bind_client(ws, "room-1", "alice")
    assert meta["username"] == "alice"
    assert meta["color"] == color_for_user("alice")
    assert meta.get("typing") is False

    updated = update_cursor(ws, cursor=12, selection_end=18)
    assert updated is not None
    assert updated["cursor"] == 12
    assert updated["selection_end"] == 18

    typed = update_typing(ws, True)
    assert typed is not None
    assert typed["typing"] is True

    clients = {ws}
    plist = presence_list(clients)
    assert len(plist) == 1
    assert plist[0]["username"] == "alice"
    assert plist[0]["cursor"] == 12
    assert plist[0]["typing"] is True

    released = release_client(ws)
    assert released["username"] == "alice"
    assert presence_list(clients) == []

    entry = presence_entry(meta)
    assert entry["color"] == meta["color"]
    assert "typing" in entry


def test_presence_snapshot_persist_and_merge(tmp_path):
    ws = _FakeWs()
    bind_client(ws, "room-a", "bob")
    update_cursor(ws, cursor=5)
    update_typing(ws, True)
    clients = {ws}
    persist_room_presence("room-a", clients, base=str(tmp_path))
    snap = load_presence_snapshot("room-a", base=str(tmp_path))
    assert len(snap) == 1
    assert snap[0]["username"] == "bob"
    assert snap[0]["typing"] is True

    # Canlı istemci yok; TTL içi disk satırı merge edilir
    release_client(ws)
    merged = merge_presence_for_join("room-a", set(), base=str(tmp_path))
    assert any(u["username"] == "bob" for u in merged)
