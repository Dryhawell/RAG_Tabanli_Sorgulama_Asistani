"""Collab presence testleri."""

from rag.collab_presence import (
    bind_client,
    color_for_user,
    presence_entry,
    presence_list,
    release_client,
    update_cursor,
)


class _FakeWs:
    pass


def test_bind_update_release_presence():
    ws = _FakeWs()
    meta = bind_client(ws, "room-1", "alice")
    assert meta["username"] == "alice"
    assert meta["color"] == color_for_user("alice")

    updated = update_cursor(ws, cursor=12, selection_end=18)
    assert updated is not None
    assert updated["cursor"] == 12
    assert updated["selection_end"] == 18

    clients = {ws}
    plist = presence_list(clients)
    assert len(plist) == 1
    assert plist[0]["username"] == "alice"
    assert plist[0]["cursor"] == 12

    released = release_client(ws)
    assert released["username"] == "alice"
    assert presence_list(clients) == []

    entry = presence_entry(meta)
    assert entry["color"] == meta["color"]
