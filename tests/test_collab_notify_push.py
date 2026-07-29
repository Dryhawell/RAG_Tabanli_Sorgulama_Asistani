"""Mobil push PoC testleri."""

from rag.collab_notify_push import (
    build_fcm_payload,
    dispatch_push,
    list_device_tokens,
    register_device_token,
)


def test_register_and_list_device_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    register_device_token("alice", "tok-1", platform="fcm")
    register_device_token("alice", "tok-1", platform="fcm")  # duplicate
    rows = list_device_tokens("alice")
    assert len(rows) == 1
    assert rows[0]["token"] == "tok-1"


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
