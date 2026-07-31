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


def test_apply_workspace_min_threshold():
    from rag.collab_notify_digest import (
        apply_workspace_min_threshold,
        prepare_digest_events,
    )

    events = [
        {"workspace_key": "ws1", "kind": "mention", "from_user": "a"},
        {"workspace_key": "ws2", "kind": "mention", "from_user": "b"},
        {"workspace_key": "ws2", "kind": "mention", "from_user": "c"},
    ]
    out = apply_workspace_min_threshold(events, 2)
    assert len(out) == 2
    assert all(ev.get("workspace_key") == "ws2" for ev in out)
    prep = prepare_digest_events(events, min_per_workspace=2)
    assert prep["total"] == 2
    assert prep["workspace_filtered"] == 1


def test_build_digest_html():
    from rag.collab_notify_digest import build_digest_html

    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "hello",
            "kind": "mention",
        }
    ]
    html = build_digest_html(events, "alice")
    assert "<html>" in html
    assert "alice" in html
    assert "bob" in html
    assert "hello" in html


def test_quiet_hours_overnight():
    from datetime import datetime, timezone

    from rag.collab_notify_digest import is_quiet_hours, parse_quiet_hours

    assert parse_quiet_hours("22:00-07:00") == (22 * 60, 7 * 60)
    night = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    morning = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    assert is_quiet_hours(now=night, quiet_spec="22:00-07:00") is True
    assert is_quiet_hours(now=morning, quiet_spec="22:00-07:00") is False
    assert is_quiet_hours(now=night, quiet_spec="") is False


def test_quiet_hours_tenant_timezone(monkeypatch):
    from datetime import datetime, timezone

    from rag.collab_notify_digest import (
        is_quiet_hours,
        parse_tenant_timezones,
        resolve_digest_timezone,
    )

    monkeypatch.setattr(
        "rag.collab_notify_digest.NOTIFY_TENANT_TIMEZONES",
        "default:Europe/Istanbul,acme:America/New_York",
    )
    mapping = parse_tenant_timezones()
    assert mapping["default"] == "Europe/Istanbul"
    assert resolve_digest_timezone(tenant_id="default") == "Europe/Istanbul"
    # 21:30 UTC = 00:30 Istanbul (UTC+3) → quiet if 22:00-07:00 local
    utc_evening = datetime(2026, 1, 1, 21, 30, tzinfo=timezone.utc)
    assert (
        is_quiet_hours(
            now=utc_evening,
            quiet_spec="22:00-07:00",
            timezone_name="Europe/Istanbul",
        )
        is True
    )
    assert (
        is_quiet_hours(
            now=utc_evening,
            quiet_spec="22:00-07:00",
            timezone_name="UTC",
        )
        is False
    )


def test_quiet_hours_user_profile_timezone(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from rag.auth import update_user_timezone
    from rag.collab_notify_digest import is_quiet_hours, resolve_digest_timezone

    users_path = str(tmp_path / "users.json")
    with open(users_path, "w", encoding="utf-8") as f:
        f.write('{"alice": {"password_hash": "x$y", "role": "user", "tenant_id": "default"}}')
    monkeypatch.setattr("rag.auth.USERS_PATH", users_path)
    update_user_timezone("alice", "Europe/Istanbul", path=users_path)
    assert resolve_digest_timezone(username="alice") == "Europe/Istanbul"
    utc_evening = datetime(2026, 1, 1, 21, 30, tzinfo=timezone.utc)
    assert (
        is_quiet_hours(
            now=utc_evening,
            quiet_spec="22:00-07:00",
            username="alice",
        )
        is True
    )


def test_send_digest_skips_quiet_hours(monkeypatch):
    from rag.collab_notify_digest import send_digest_email

    monkeypatch.setattr(
        "rag.collab_notify_digest.is_quiet_hours",
        lambda **kwargs: True,
    )
    result = send_digest_email("alice", quiet_hours="22:00-07:00")
    assert result["sent"] is False
    assert result["reason"] == "quiet_hours"


def test_quiet_hours_user_profile_spec(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from rag.auth import update_user_quiet_hours
    from rag.collab_notify_digest import is_quiet_hours, resolve_quiet_hours_spec

    users_path = str(tmp_path / "users.json")
    with open(users_path, "w", encoding="utf-8") as f:
        f.write('{"alice": {"password_hash": "x$y", "role": "user", "tenant_id": "default"}}')
    monkeypatch.setattr("rag.auth.USERS_PATH", users_path)
    monkeypatch.setattr(
        "rag.collab_notify_digest.NOTIFY_DIGEST_QUIET_HOURS",
        "00:00-01:00",
    )
    update_user_quiet_hours("alice", "22:00-07:00", path=users_path)
    assert resolve_quiet_hours_spec(username="alice") == "22:00-07:00"
    night = datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc)
    assert is_quiet_hours(now=night, username="alice", timezone_name="UTC") is True
    # açık off quiet hours'ı kapatır
    assert resolve_quiet_hours_spec(quiet_spec="off", username="alice") is None
    assert is_quiet_hours(now=night, quiet_spec="off", username="alice") is False

