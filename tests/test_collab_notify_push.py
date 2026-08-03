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


def test_device_quiet_hours_and_geofence(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from rag.collab_notify_push import (
        device_inside_geofence,
        dispatch_push,
        haversine_m,
        is_device_in_quiet_hours,
        register_device_token,
        update_device_location,
    )

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "http://push.test/send")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "generic")
    monkeypatch.setattr(
        "rag.collab_notify_digest.NOTIFY_DIGEST_QUIET_HOURS",
        "22:00-07:00",
    )

    assert haversine_m(41.0, 29.0, 41.0, 29.0) < 1.0

    register_device_token(
        "eve",
        "tok-qh",
        quiet_hours="off",
        geofence_lat=41.0,
        geofence_lon=29.0,
        geofence_radius_m=500,
        last_lat=41.0,
        last_lon=29.0,
    )
    night = datetime(2026, 1, 1, 23, 30, tzinfo=timezone.utc)
    # cihaz off → quiet değil
    rows = list_device_tokens("eve")
    assert is_device_in_quiet_hours(rows[0], "eve", now=night, timezone_name="UTC") is False

    register_device_token(
        "eve",
        "tok-geo",
        quiet_hours="22:00-07:00",
        geofence_lat=41.0,
        geofence_lon=29.0,
        geofence_radius_m=200,
        last_lat=41.0,
        last_lon=29.0,
    )
    rows2 = [r for r in list_device_tokens("eve") if r["token"] == "tok-geo"]
    assert device_inside_geofence(rows2[0]) is True
    assert is_device_in_quiet_hours(rows2[0], "eve", now=night, timezone_name="UTC") is True
    # geofence dışı → quiet hours yok (seyahat)
    assert update_device_location("eve", "tok-geo", lat=42.0, lon=30.0) is True
    rows3 = [r for r in list_device_tokens("eve") if r["token"] == "tok-geo"]
    assert device_inside_geofence(rows3[0]) is False
    assert is_device_in_quiet_hours(rows3[0], "eve", now=night, timezone_name="UTC") is False

    calls = []

    class FakeResp:
        status_code = 200

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append(json)
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    # tok-geo quiet değil (dışarıda), tok-qh off → ikisi de gidebilir
    assert dispatch_push("eve", title="t", body="b") is True
    assert calls
    tok_list = [t["token"] for t in calls[0]["tokens"]]
    assert "tok-qh" in tok_list
    assert "tok-geo" in tok_list


def test_fcm_endpoint_and_auth_header(tmp_path, monkeypatch):
    from rag.collab_notify_push import (
        load_fcm_service_account_info,
        resolve_fcm_auth_header,
        resolve_fcm_endpoint,
    )

    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_FCM_PROJECT_ID", "demo-proj")
    assert resolve_fcm_endpoint().endswith("/projects/demo-proj/messages:send")

    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "https://gateway/push")
    assert resolve_fcm_endpoint() == "https://gateway/push"

    sa = tmp_path / "sa.json"
    sa.write_text('{"type": "service_account", "project_id": "x"}', encoding="utf-8")
    info = load_fcm_service_account_info(str(sa))
    assert info["type"] == "service_account"

    monkeypatch.setattr(
        "rag.collab_notify_push.get_fcm_access_token",
        lambda **kwargs: "oauth-token",
    )
    assert resolve_fcm_auth_header() == "Bearer oauth-token"

    monkeypatch.setattr(
        "rag.collab_notify_push.get_fcm_access_token",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_API_KEY", "static-key")
    assert resolve_fcm_auth_header() == "Bearer static-key"


def test_dispatch_push_fcm_uses_oauth_header(tmp_path, monkeypatch):
    from rag.collab_notify_push import dispatch_push, register_device_token

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "fcm")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_FCM_PROJECT_ID", "p1")
    monkeypatch.setattr(
        "rag.collab_notify_push.resolve_fcm_auth_header",
        lambda **kwargs: "Bearer tok-oauth",
    )
    register_device_token("alice", "device-fcm", platform="fcm")
    calls = []

    class FakeResp:
        status_code = 200

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    assert dispatch_push("alice", title="T", body="B") is True
    assert "fcm.googleapis.com/v1/projects/p1/messages:send" in calls[0]["url"]
    assert calls[0]["headers"]["Authorization"] == "Bearer tok-oauth"
    assert calls[0]["json"]["message"]["token"] == "device-fcm"




def test_apns_http2_body_and_endpoint(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    from rag.collab_notify_push import (
        build_apns_http2_body,
        get_apns_provider_token,
        load_apns_p8,
        resolve_apns_endpoint,
    )

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    p8 = tmp_path / "AuthKey.p8"
    p8.write_text(pem, encoding="utf-8")

    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_APNS_KEY_ID", "KEY1234567")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_APNS_TOPIC", "com.example.app")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_APNS_P8_PATH", str(p8))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_APNS_USE_SANDBOX", True)
    monkeypatch.setattr("rag.collab_notify_push._APNS_TOKEN_CACHE", {"token": None, "expires_at": 0.0})

    assert "BEGIN PRIVATE KEY" in (load_apns_p8() or "")
    token = get_apns_provider_token()
    assert token and isinstance(token, str)
    url = resolve_apns_endpoint("abcd1234")
    assert url.startswith("https://api.sandbox.push.apple.com/3/device/")
    body = build_apns_http2_body(title="T", body="B", data={"k": "v"})
    assert body["aps"]["alert"]["title"] == "T"
    assert body["k"] == "v"


def test_dispatch_apns_native_httpx(tmp_path, monkeypatch):
    from rag.collab_notify_push import dispatch_push, register_device_token

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "apns")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "")
    monkeypatch.setattr("rag.collab_notify_push.apns_native_configured", lambda: True)
    monkeypatch.setattr(
        "rag.collab_notify_push.get_apns_provider_token",
        lambda **kwargs: "jwt-token",
    )
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_APNS_TOPIC", "com.example.app")
    monkeypatch.setattr(
        "rag.collab_notify_push.resolve_apns_endpoint",
        lambda tok, **kwargs: f"https://api.push.apple.com/3/device/{tok}",
    )
    register_device_token("alice", "devicetok", platform="apns")
    calls = []

    class FakeResp:
        status_code = 200

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return FakeResp()

    monkeypatch.setattr("httpx.Client", FakeClient)
    assert dispatch_push("alice", title="Hi", body="There") is True
    assert calls
    assert "authorization" in calls[0]["headers"]
    assert calls[0]["headers"]["apns-topic"] == "com.example.app"
    assert "aps" in calls[0]["json"]


def test_fcm_invalid_token_auto_revoke(tmp_path, monkeypatch):
    from rag.collab_notify_push import (
        dispatch_push,
        is_fcm_invalid_token_response,
        list_device_tokens,
        parse_fcm_error_code,
        register_device_token,
        summarize_user_devices,
    )

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.content = b"x"

        def json(self):
            return self._payload

    bad = FakeResp(
        404,
        {
            "error": {
                "code": 404,
                "status": "NOT_FOUND",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.firebase.fcm.v1.FcmError",
                        "errorCode": "UNREGISTERED",
                    }
                ],
            }
        },
    )
    assert parse_fcm_error_code(bad) == "UNREGISTERED"
    assert is_fcm_invalid_token_response(bad) is True

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "fcm")
    monkeypatch.setattr(
        "rag.collab_notify_push.resolve_fcm_endpoint",
        lambda **k: "https://fcm.googleapis.com/v1/projects/p/messages:send",
    )
    monkeypatch.setattr(
        "rag.collab_notify_push.resolve_fcm_auth_header",
        lambda **k: "Bearer x",
    )
    register_device_token("alice", "dead-token", platform="fcm")
    monkeypatch.setattr("requests.post", lambda *a, **k: bad)
    assert dispatch_push("alice", title="t", body="b") is False
    devices = summarize_user_devices("alice")
    assert devices and devices[0]["revoked"] is True
    assert list_device_tokens("alice") == []


def test_revoke_records_metric_and_prune_breakdown(tmp_path, monkeypatch):
    from rag.collab_notify_push import (
        prune_expired_tokens,
        register_device_token,
        revoke_device_token,
        summarize_user_devices,
    )

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    metrics = []

    def fake_record(kind, **kwargs):
        metrics.append({"kind": kind, **kwargs})
        return {}

    monkeypatch.setattr("rag.metrics.record_metric", fake_record)
    register_device_token("alice", "tok-bad", platform="apns")
    assert revoke_device_token(
        "alice",
        "tok-bad",
        reason="BadDeviceToken",
        provider="apns",
    )
    assert any(m["kind"] == "push_token_revoked" for m in metrics)
    devices = summarize_user_devices("alice")
    assert devices[0].get("revoked") is True

    result = prune_expired_tokens(username="alice")
    assert result["removed"] == 1
    assert result["revoked_pruned"] == 1
    assert any(m["kind"] == "push_token_prune" for m in metrics)


def test_webpush_parse_and_dispatch(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from rag.collab_notify_push import (
        dispatch_push,
        is_push_configured,
        parse_webpush_subscription,
        register_device_token,
        send_webpush,
        vapid_configured,
    )

    assert parse_webpush_subscription("not-json") is None
    sub = {
        "endpoint": "https://push.example.com/sub/abc",
        "keys": {"p256dh": "p256dh-key", "auth": "auth-key"},
    }
    parsed = parse_webpush_subscription(json.dumps(sub))
    assert parsed is not None
    assert parsed["endpoint"].startswith("https://")

    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_VAPID_PUBLIC", "")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_VAPID_PRIVATE", "")
    assert vapid_configured() is False

    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_VAPID_PUBLIC", "pub")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_VAPID_PRIVATE", "priv")
    monkeypatch.setattr(
        "rag.collab_notify_push.NOTIFY_PUSH_VAPID_SUBJECT",
        "mailto:test@example.com",
    )
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "webpush")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_URL", "")
    assert is_push_configured() is True

    calls = []

    class FakeResp:
        status_code = 201

    def fake_webpush(**kwargs):
        calls.append(kwargs)
        return FakeResp()

    fake_mod = types.ModuleType("pywebpush")
    fake_mod.webpush = fake_webpush
    monkeypatch.setitem(sys.modules, "pywebpush", fake_mod)

    result = send_webpush(parsed, title="T", body="B", data={"k": 1})
    assert result["ok"] is True
    assert result["provider"] == "webpush"
    assert calls and "subscription_info" in calls[0]

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    register_device_token("alice", json.dumps(sub), platform="webpush")
    assert dispatch_push("alice", title="Hi", body="There") is True


def test_webpush_gone_revokes_token(tmp_path, monkeypatch):
    import json
    import sys
    import types

    from rag.collab_notify_push import (
        dispatch_push,
        register_device_token,
        summarize_user_devices,
    )

    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_PROVIDER", "webpush")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_VAPID_PUBLIC", "pub")
    monkeypatch.setattr("rag.collab_notify_push.NOTIFY_PUSH_VAPID_PRIVATE", "priv")
    sub = {
        "endpoint": "https://push.example.com/sub/gone",
        "keys": {"p256dh": "p", "auth": "a"},
    }
    register_device_token("bob", json.dumps(sub), platform="webpush")

    def boom(**kwargs):
        raise RuntimeError("410 Gone")

    fake_mod = types.ModuleType("pywebpush")
    fake_mod.webpush = boom
    monkeypatch.setitem(sys.modules, "pywebpush", fake_mod)

    assert dispatch_push("bob", title="t", body="b") is False
    devices = summarize_user_devices("bob")
    assert devices and devices[0].get("revoked") is True


def test_vapid_application_server_key_and_sw(tmp_path, monkeypatch):
    import os

    from rag.collab_notify_push import (
        load_service_worker_js,
        service_worker_js_path,
        vapid_application_server_key,
        vapid_public_key,
    )

    monkeypatch.setattr(
        "rag.collab_notify_push.NOTIFY_PUSH_VAPID_PUBLIC",
        "BNcRdQ+/abc+def/ghi=",
    )
    assert vapid_public_key().startswith("BNc")
    key = vapid_application_server_key()
    assert key is not None
    assert "+" not in key and "/" not in key
    assert os.path.isfile(service_worker_js_path()) or load_service_worker_js() == ""
    sw = load_service_worker_js()
    if sw:
        assert "addEventListener" in sw and "push" in sw


def test_generate_vapid_keys():
    from rag.collab_notify_push import (
        format_vapid_env,
        generate_vapid_keys,
        vapid_application_server_key,
    )

    keys = generate_vapid_keys(subject="mailto:ops@example.com")
    assert keys["public"]
    assert "BEGIN PRIVATE KEY" in keys["private"]
    assert keys["subject"] == "mailto:ops@example.com"
    assert vapid_application_server_key(keys["public"]) is not None
    env = format_vapid_env(keys)
    assert "RAG_NOTIFY_PUSH_VAPID_PUBLIC=" in env
    assert "RAG_NOTIFY_PUSH_VAPID_PRIVATE=" in env

