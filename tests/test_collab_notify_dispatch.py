"""Bildirim dispatch testleri."""

from rag.collab_notify_dispatch import (
    dispatch_notification,
    dispatch_webhook,
    resolve_notify_email,
)


def test_resolve_notify_email_from_username():
    email = resolve_notify_email("alice@corp.com")
    assert email == "alice@corp.com"


def test_dispatch_webhook_mock(monkeypatch):
    monkeypatch.setattr("rag.collab_notify_dispatch.NOTIFY_WEBHOOK_URL", "http://hook.test/notify")
    calls = []

    class FakeResp:
        status_code = 200

    def fake_post(url, json=None, timeout=10):
        calls.append({"url": url, "json": json})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    event = {
        "target_user": "alice",
        "from_user": "bob",
        "body_preview": "test",
        "workspace_key": "ws1",
    }
    ok = dispatch_webhook(event)
    assert ok is True
    assert len(calls) == 1
    assert calls[0]["json"]["type"] == "collab_mention"


def test_dispatch_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr("rag.collab_notify_dispatch.ENABLE_COLLAB_NOTIFY_DISPATCH", False)
    result = dispatch_notification({"target_user": "a"})
    assert result.get("skipped") is True


def test_dispatch_digest_webhook_slack_blocks(monkeypatch):
    from rag.collab_notify_dispatch import dispatch_digest_webhook

    monkeypatch.setattr(
        "rag.collab_notify_dispatch.NOTIFY_WEBHOOK_URL",
        "https://hooks.slack.com/services/T/B/X",
    )
    calls = []

    class FakeResp:
        status_code = 200

    def fake_post(url, json=None, timeout=10):
        calls.append({"url": url, "json": json})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "ping",
        }
    ]
    ok = dispatch_digest_webhook("alice", events)
    assert ok is True
    assert calls[0]["json"]["type"] == "collab_digest"
    assert "blocks" in calls[0]["json"]
    assert calls[0]["json"]["blocks"][0]["type"] == "header"
