"""Bildirim digest testleri."""

from rag.collab_notify_digest import (
    build_digest_body,
    collect_digest_events,
    list_digest_target_users,
)


def test_collect_digest_events(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify.METADATA_DIR", str(tmp_path))
    from rag.collab_notify import notify_mentions

    notify_mentions("ws-d", "Hi @alice", from_user="bob", thread_id="t1")
    rows = collect_digest_events("alice", hours=24, base=str(tmp_path))
    assert len(rows) >= 1


def test_list_digest_target_users(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify.METADATA_DIR", str(tmp_path))
    from rag.collab_notify import notify_mentions

    notify_mentions("ws-x", "Hi @bob", from_user="alice", thread_id="t1")
    users = list_digest_target_users(hours=24, base=str(tmp_path))
    assert "bob" in users


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


def test_build_digest_slack_blocks():
    from rag.collab_notify_digest import build_digest_slack_blocks

    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "hello @alice",
        }
    ]
    blocks = build_digest_slack_blocks(events, "alice")
    assert blocks[0]["type"] == "header"
    assert any(b.get("type") == "section" for b in blocks)
    section = next(b for b in blocks if b.get("type") == "section" and "bob" in b["text"]["text"])
    assert "hello" in section["text"]["text"]


def test_build_digest_discord_embed():
    from rag.collab_notify_digest import build_digest_discord_embed

    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "ping",
        }
    ]
    embeds = build_digest_discord_embed(events, "alice")
    assert len(embeds) == 1
    assert "alice" in embeds[0]["title"]
    assert embeds[0]["fields"][0]["name"].startswith("bob")
