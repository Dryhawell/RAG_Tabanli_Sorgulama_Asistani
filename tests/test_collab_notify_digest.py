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
            "kind": "mention",
            "thread_id": "t1",
        }
    ]
    embeds = build_digest_discord_embed(events, "alice")
    assert len(embeds) == 1
    assert "alice" in embeds[0]["title"]
    assert embeds[0]["fields"][0]["name"].find("bob") >= 0


def test_filter_and_group_digest_events():
    from rag.collab_notify_digest import (
        filter_mention_events,
        group_digest_events,
        prepare_digest_events,
    )

    events = [
        {
            "kind": "mention",
            "thread_id": "t1",
            "workspace_key": "ws",
            "from_user": "a",
            "body_preview": "hi",
        },
        {
            "kind": "system",
            "thread_id": "t2",
            "workspace_key": "ws",
            "from_user": "b",
            "body_preview": "sys",
        },
        {
            "kind": "mention",
            "thread_id": "t1",
            "workspace_key": "ws",
            "from_user": "c",
            "body_preview": "again",
        },
    ]
    mentions = filter_mention_events(events)
    assert len(mentions) == 2
    grouped = group_digest_events(mentions, group_by="thread")
    assert "thread:t1" in grouped
    assert len(grouped["thread:t1"]) == 2
    prep = prepare_digest_events(events, mentions_only=True, group_by="thread")
    assert prep["total"] == 2
    assert "thread:t1" in prep["grouped"]


def test_build_digest_teams_adaptive_card():
    from rag.collab_notify_digest import build_digest_teams_payload

    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "teams ping",
            "kind": "mention",
            "thread_id": "th1",
        }
    ]
    payload = build_digest_teams_payload(events, "alice")
    assert payload["type"] == "message"
    assert payload["attachments"][0]["contentType"] == (
        "application/vnd.microsoft.card.adaptive"
    )
    card = payload["attachments"][0]["content"]
    assert card["type"] == "AdaptiveCard"
    assert any(
        b.get("type") == "FactSet" for b in card.get("body") or []
    )
