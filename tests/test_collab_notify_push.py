"""Mobil push PoC testleri."""

from datetime import datetime, timedelta, timezone

from rag.collab_notify_push import (
    build_fcm_payload,
    dispatch_push,
    is_token_expired,
    list_device_tokens,
    prune_expired_tokens,
    register_device_token,
    revoke_device_token,
    summarize_user_devices,
)


def test_register_and_list_device_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    register_device_token("alice", "tok-1", platform="fcm", label="phone")
    register_device_token("alice", "tok-1", platform="fcm")  # refresh
    rows = list_device_tokens("alice")
    assert len(rows) == 1
    assert rows[0]["token"] == "tok-1"
    assert rows[0].get("expires_at")
    assert rows[0].get("last_seen_at")


def test_device_meta_and_last_seen(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "http://push.test/send")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "generic")
    register_device_token(
        "dave",
        "tok-meta",
        device_name="Pixel 8",
        os_name="Android",
        os_version="14",
        app_version="1.2.3",
    )
    devices = summarize_user_devices("dave")
    assert devices[0]["device_name"] == "Pixel 8"
    assert devices[0]["os_name"] == "Android"

    class FakeResp:
        status_code = 200

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    assert dispatch_push("dave", title="x", body="y") is True
    devices2 = summarize_user_devices("dave")
    assert devices2[0].get("last_seen_at")


def test_build_fcm_payload():
    payload = build_fcm_payload("abc", title="T", body="B", data={"k": 1})
    assert payload["message"]["token"] == "abc"
    assert payload["message"]["notification"]["title"] == "T"
    assert payload["message"]["data"]["k"] == "1"


def test_dispatch_push_generic(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "http://push.test/send")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "generic")
    register_device_token("bob", "device-x", platform="generic")
    calls = []

    class FakeResp:
        status_code = 200

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    ok = dispatch_push("bob", title="Hi", body="There")
    assert ok is True
    assert calls[0]["json"]["type"] == "collab_push"
    assert calls[0]["json"]["username"] == "bob"


def test_token_ttl_revoke_and_prune(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    rec = register_device_token("carol", "tok-old", ttl_days=1)
    # force expire
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    rows = list_device_tokens("carol", include_expired=True)
    assert rows
    rows[0]["expires_at"] = past
    from rag.collab_notify_push import _write_all_tokens

    _write_all_tokens(rows)
    assert is_token_expired(rows[0]) is True
    assert list_device_tokens("carol") == []
    result = prune_expired_tokens(username="carol")
    assert result["removed"] == 1
    register_device_token("carol", "tok-new")
    assert revoke_device_token("carol", "tok-new") is True
    devices = summarize_user_devices("carol")
    assert any(d.get("revoked") for d in devices)
