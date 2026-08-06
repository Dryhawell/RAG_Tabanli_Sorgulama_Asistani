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


def test_dispatch_digest_webhook_discord_embed(monkeypatch):
    from rag.collab_notify_dispatch import dispatch_digest_webhook

    monkeypatch.setattr(
        "rag.collab_notify_dispatch.NOTIFY_WEBHOOK_URL",
        "https://discord.com/api/webhooks/123/abc",
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
    assert "embeds" in calls[0]["json"]
    assert calls[0]["json"]["embeds"][0]["title"].startswith("Bildirim")


def test_dispatch_digest_webhook_teams_adaptive(monkeypatch):
    from rag.collab_notify_dispatch import dispatch_digest_webhook

    monkeypatch.setattr(
        "rag.collab_notify_dispatch.NOTIFY_WEBHOOK_URL",
        "https://outlook.office.com/webhook/abc/IncomingWebhook/xyz",
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
            "kind": "mention",
        }
    ]
    ok = dispatch_digest_webhook("alice", events)
    assert ok is True
    assert calls[0]["json"]["type"] == "message"
    assert calls[0]["json"]["attachments"][0]["content"]["type"] == "AdaptiveCard"


def test_dispatch_digest_email_html(monkeypatch):
    from rag.collab_notify_dispatch import dispatch_digest_email

    monkeypatch.setattr("rag.collab_notify_dispatch.NOTIFY_SMTP_HOST", "smtp.test")
    monkeypatch.setattr("rag.collab_notify_dispatch.resolve_notify_email", lambda u: "a@test.com")
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=15):
            pass

        def starttls(self):
            return None

        def login(self, user, password):
            return None

        def sendmail(self, from_addr, to_addrs, msg):
            sent.append(msg)
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr("rag.collab_notify_digest.NOTIFY_DIGEST_HTML", True)
    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "ping",
            "kind": "mention",
        }
    ]
    ok = dispatch_digest_email("alice@test.com", events)
    assert ok is True
    assert len(sent) == 1
    assert "multipart/alternative" in sent[0]
    assert "text/html" in sent[0]


def test_dispatch_quiet_hours_thread_reply_slack(monkeypatch, tmp_path):
    from rag.collab_notify_digest import save_digest_thread_state
    from rag.collab_notify_dispatch import dispatch_quiet_hours_thread_reply

    monkeypatch.setattr(
        "rag.collab_notify_dispatch.NOTIFY_WEBHOOK_URL",
        "https://hooks.slack.com/services/T/B/X",
    )
    monkeypatch.setattr("rag.collab_notify_digest.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_digest.NOTIFY_DIGEST_THREAD_REPLY", True)
    monkeypatch.setattr("rag.collab_notify_digest.NOTIFY_DIGEST_SLACK_THREAD_TS", "")
    save_digest_thread_state("alice", slack_thread_ts="1234.5678", base=str(tmp_path))
    calls = []

    class FakeResp:
        status_code = 200
        content = b""

    def fake_post(url, json=None, timeout=10, headers=None):
        calls.append({"url": url, "json": json})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    events = [
        {
            "workspace_key": "ws",
            "from_user": "bob",
            "body_preview": "gece ping",
            "kind": "mention",
        }
    ]
    ok = dispatch_quiet_hours_thread_reply("alice", events, base=str(tmp_path))
    assert ok is True
    assert calls[0]["json"]["type"] == "collab_digest_quiet_hours"
    assert calls[0]["json"]["thread_ts"] == "1234.5678"
    assert calls[0]["json"]["quiet_hours_summary"] is True


def test_dispatch_quiet_hours_teams_reply_to(monkeypatch, tmp_path):
    from rag.collab_notify_dispatch import dispatch_digest_webhook

    monkeypatch.setattr(
        "rag.collab_notify_dispatch.NOTIFY_WEBHOOK_URL",
        "https://outlook.office.com/webhook/abc/IncomingWebhook/xyz",
    )
    monkeypatch.setattr("rag.collab_notify_digest.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        "rag.collab_notify_digest.NOTIFY_DIGEST_TEAMS_REPLY_ID",
        "msg-parent-1",
    )
    calls = []

    class FakeResp:
        status_code = 200
        content = b""

    def fake_post(url, json=None, timeout=10, headers=None):
        calls.append(json)
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    events = [{"workspace_key": "ws", "from_user": "bob", "body_preview": "x"}]
    ok = dispatch_digest_webhook(
        "alice",
        events,
        quiet_hours_summary=True,
        base=str(tmp_path),
    )
    assert ok is True
    assert calls[0]["replyToId"] == "msg-parent-1"
    assert "Quiet hours" in calls[0]["attachments"][0]["content"]["body"][0]["text"]
