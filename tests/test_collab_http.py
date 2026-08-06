"""Collab HTTP (Web Push SW / register) testleri."""

import json
import threading
from http.client import HTTPConnection

from rag.collab_http import (
    handle_judge_ack,
    handle_judge_alert_state,
    handle_sw_js,
    handle_webpush_register,
)
from rag.collab_notify_push import list_device_tokens, summarize_user_devices


def test_handle_sw_js():
    code, headers, body = handle_sw_js()
    assert code == 200
    assert "javascript" in headers["Content-Type"]
    assert b"push" in body


def test_handle_judge_ack(tmp_path, monkeypatch):
    from rag.judge_alert import save_judge_alert_state

    state = str(tmp_path / "judge_state.json")
    save_judge_alert_state(
        {
            "soft_fail": True,
            "source": "report",
            "acknowledged": False,
            "last_action": "alert",
        },
        state,
    )
    monkeypatch.setenv("RAG_JUDGE_ACK_TOKEN", "secret-ack")
    monkeypatch.setenv("RAG_JUDGE_ALERT_STATE", state)
    monkeypatch.delenv("RAG_JUDGE_SLACK_WEBHOOK", raising=False)
    monkeypatch.delenv("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RAG_JUDGE_OPSGENIE_API_KEY", raising=False)

    code, _, body = handle_judge_ack(b'{"actor":"alice"}')
    assert code == 401

    code, _, body = handle_judge_ack(
        json.dumps({"actor": "alice", "token": "secret-ack", "note": "looking"}).encode()
    )
    assert code == 200
    data = json.loads(body.decode("utf-8"))
    assert data["ok"] is True
    assert data["state"]["acknowledged_by"] == "alice"

    code, _, body = handle_judge_alert_state()
    assert code == 200
    st = json.loads(body.decode("utf-8"))["state"]
    assert st["soft_fail"] is True
    assert st["acknowledged"] is True


def test_handle_judge_ack_form():
    from rag.collab_http import handle_judge_ack_form

    code, headers, body = handle_judge_ack_form()
    assert code == 200
    assert "text/html" in headers["Content-Type"]
    assert b"/judge/ack" in body
    assert b"Acknowledge" in body


def test_handle_judge_slack_interactive(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import time
    from urllib.parse import quote

    from rag.collab_http import handle_judge_slack_interactive
    from rag.judge_alert import save_judge_alert_state

    state = str(tmp_path / "state.json")
    save_judge_alert_state(
        {"soft_fail": True, "source": "report", "acknowledged": False},
        state,
    )
    secret = "slack-secret"
    monkeypatch.setenv("RAG_JUDGE_SLACK_SIGNING_SECRET", secret)
    monkeypatch.setenv("RAG_JUDGE_ALERT_STATE", state)
    payload = {
        "type": "block_actions",
        "user": {"username": "ops"},
        "actions": [{"action_id": "judge_ack_interactive", "value": "ack"}],
    }
    form = f"payload={quote(json.dumps(payload))}".encode("utf-8")
    ts = str(int(time.time()))
    base = f"v0:{ts}:{form.decode()}"
    sig = "v0=" + hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    code, _, body = handle_judge_slack_interactive(
        form,
        headers={
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": sig,
        },
    )
    assert code == 200
    data = json.loads(body.decode("utf-8"))
    assert "acknowledged" in data.get("text", "").lower() or data.get("text")


def test_handle_webpush_register(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    sub = {
        "endpoint": "https://push.example.com/x",
        "keys": {"p256dh": "p", "auth": "a"},
    }
    code, headers, body = handle_webpush_register(
        json.dumps({"username": "alice", "subscription": sub}).encode("utf-8")
    )
    assert code == 200
    data = json.loads(body.decode("utf-8"))
    assert data["ok"] is True
    tokens = list_device_tokens("alice")
    assert len(tokens) == 1
    assert tokens[0]["platform"] == "webpush"


def test_collab_http_server_get_sw(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notify_push.METADATA_DIR", str(tmp_path))
    from http.server import ThreadingHTTPServer

    from rag.collab_http import CollabHTTPHandler

    class H(CollabHTTPHandler):
        public_base = "http://127.0.0.1:0"

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        conn.request("GET", "/sw.js")
        resp = conn.getresponse()
        assert resp.status == 200
        body = resp.read()
        assert b"addEventListener" in body
        conn.close()

        sub = {
            "endpoint": "https://push.example.com/y",
            "keys": {"p256dh": "pp", "auth": "aa"},
        }
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        payload = json.dumps({"username": "carol", "subscription": sub})
        conn.request(
            "POST",
            "/webpush/register",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["ok"] is True
        conn.close()
        devices = summarize_user_devices("carol")
        assert devices and devices[0]["platform"] == "webpush"
    finally:
        server.shutdown()
        server.server_close()
