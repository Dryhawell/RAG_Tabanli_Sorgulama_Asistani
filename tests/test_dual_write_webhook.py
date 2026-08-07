from __future__ import annotations

import json
from pathlib import Path

from rag.dual_write_webhook import (
    extract_firing_dual_write_alerts,
    handle_dual_write_alertmanager_webhook,
    plan_actions_for_alerts,
)


def test_extract_and_plan_burn_alert() -> None:
    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "RagDualWriteLagShadowBurn",
                    "service": "rag-ingest",
                    "secondary_backend": "qdrant",
                    "severity": "critical",
                },
                "annotations": {"summary": "burn"},
            },
            {
                "status": "resolved",
                "labels": {
                    "alertname": "RagDualWriteLagHigh",
                    "service": "rag-ingest",
                },
            },
        ],
    }
    firing = extract_firing_dual_write_alerts(payload)
    assert len(firing) == 1
    assert firing[0]["alertname"] == "RagDualWriteLagShadowBurn"
    actions = plan_actions_for_alerts(firing)
    assert actions == ["catch_up", "shadow_compare"]


def test_webhook_dry_run_and_cooldown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "3600")
    state = tmp_path / "state.json"
    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "RagDualWriteLagHigh",
                    "service": "rag-ingest",
                },
            }
        ],
    }
    first = handle_dual_write_alertmanager_webhook(
        payload, dry_run=True, state_path=str(state)
    )
    assert first["ok"] is True
    assert first["skipped"] is False
    assert first["planned_actions"] == ["catch_up"]
    assert first["results"][0]["dry_run"] is True

    # Persist a recent trigger then expect cooldown
    state.write_text(
        json.dumps({"last_trigger_ts": __import__("time").time()}),
        encoding="utf-8",
    )
    second = handle_dual_write_alertmanager_webhook(
        payload, dry_run=True, state_path=str(state)
    )
    assert second["skipped"] is True
    assert second["reason"] == "cooldown"

    forced = handle_dual_write_alertmanager_webhook(
        payload, dry_run=True, force=True, state_path=str(state)
    )
    assert forced["skipped"] is False


def test_webhook_circuit_breaker_and_rate_limit_headers(
    tmp_path: Path, monkeypatch
) -> None:
    import json
    import time

    from rag.dual_write_webhook import (
        handle_alertmanager_webhook_http,
        handle_dual_write_alertmanager_webhook,
        record_webhook_circuit_result,
        webhook_rate_limit_headers,
    )

    monkeypatch.delenv("RAG_ALERTMANAGER_WEBHOOK_SIGNING_SECRET", raising=False)
    monkeypatch.delenv("RAG_ALERTMANAGER_WEBHOOK_TOKEN", raising=False)
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "0")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_CB_FAILURES", "2")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_CB_OPEN_SEC", "3600")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_STRICT_RL", "1")

    hdrs = webhook_rate_limit_headers(
        limit=900, remaining=0, reset_at=time.time() + 30, retry_after=30
    )
    assert hdrs["RateLimit-Remaining"] == "0"
    assert hdrs["Retry-After"] == "30"

    state = tmp_path / "cb.json"
    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "RagDualWriteLagHigh",
                    "service": "rag-ingest",
                },
            }
        ],
    }

    from unittest.mock import patch

    with patch(
        "rag.dual_write_webhook.run_dual_write_catch_up",
        return_value={"ok": False, "action": "catch_up", "error": "boom"},
    ):
        monkeypatch.setenv("RAG_DUAL_WRITE_GH_DISPATCH", "0")
        first = handle_dual_write_alertmanager_webhook(
            payload, force=True, state_path=str(state)
        )
        second = handle_dual_write_alertmanager_webhook(
            payload, force=True, state_path=str(state)
        )
    assert first["ok"] is False
    assert second["ok"] is False

    blocked = handle_dual_write_alertmanager_webhook(
        payload, force=False, state_path=str(state)
    )
    assert blocked["skipped"] is True
    assert blocked["reason"] == "circuit_open"

    code, headers, body = handle_alertmanager_webhook_http(
        json.dumps(payload).encode(),
        headers={},
        state_path=str(state),
    )
    assert code == 429
    assert "RateLimit-Limit" in headers
    assert headers.get("Retry-After")
    assert json.loads(body.decode())["reason"] == "circuit_open"

    # Recovery after success
    st = json.loads(state.read_text(encoding="utf-8"))
    record_webhook_circuit_result(st, ok=True)
    assert st["consecutive_failures"] == 0
    assert not st.get("circuit_open_until")


def test_http_adapter_cooldown_rate_limit_headers(
    tmp_path: Path, monkeypatch
) -> None:
    import json
    import time

    from rag.dual_write_webhook import handle_alertmanager_webhook_http

    monkeypatch.delenv("RAG_ALERTMANAGER_WEBHOOK_SIGNING_SECRET", raising=False)
    monkeypatch.delenv("RAG_ALERTMANAGER_WEBHOOK_TOKEN", raising=False)
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "3600")
    monkeypatch.delenv("RAG_DUAL_WRITE_WEBHOOK_STRICT_RL", raising=False)
    state = tmp_path / "rl.json"
    state.write_text(json.dumps({"last_trigger_ts": time.time()}), encoding="utf-8")
    payload = b'{"status":"firing","alerts":[{"status":"firing","labels":{"alertname":"RagDualWriteLagHigh","service":"rag-ingest"}}]}'
    code, headers, body = handle_alertmanager_webhook_http(
        payload, headers={}, state_path=str(state)
    )
    assert code == 200
    assert "RateLimit-Remaining" in headers
    assert headers["RateLimit-Remaining"] == "0"
    assert "Retry-After" in headers
    assert json.loads(body.decode())["reason"] == "cooldown"


def test_http_adapter_unauthorized(monkeypatch) -> None:
    from rag.dual_write_webhook import handle_alertmanager_webhook_http

    monkeypatch.delenv("RAG_ALERTMANAGER_WEBHOOK_SIGNING_SECRET", raising=False)
    monkeypatch.setenv("RAG_ALERTMANAGER_WEBHOOK_TOKEN", "secret")
    code, _hdrs, body = handle_alertmanager_webhook_http(
        b'{"alerts":[]}', headers={}
    )
    assert code == 401
    assert json.loads(body.decode())["error"] == "unauthorized"

    code2, _, body2 = handle_alertmanager_webhook_http(
        b'{"status":"firing","alerts":[]}',
        headers={"X-Webhook-Token": "secret"},
    )
    assert code2 == 200
    assert json.loads(body2.decode())["skipped"] is True


def test_webhook_hmac_and_replay(tmp_path: Path, monkeypatch) -> None:
    import time

    from rag.dual_write_webhook import (
        build_webhook_signature,
        check_webhook_replay,
        handle_alertmanager_webhook_http,
        verify_webhook_signature,
    )

    secret = "hmac-test-secret"
    monkeypatch.setenv("RAG_ALERTMANAGER_WEBHOOK_SIGNING_SECRET", secret)
    monkeypatch.delenv("RAG_ALERTMANAGER_WEBHOOK_TOKEN", raising=False)
    body = b'{"status":"firing","alerts":[]}'
    ts = str(int(time.time()))
    sig = build_webhook_signature(body, timestamp=ts, secret=secret)
    assert verify_webhook_signature(
        body, timestamp=ts, signature=sig, signing_secret=secret
    )
    assert not verify_webhook_signature(
        body, timestamp=ts, signature="v0=deadbeef", signing_secret=secret
    )

    state = tmp_path / "wh.json"
    code, _, raw = handle_alertmanager_webhook_http(
        body,
        headers={
            "X-Webhook-Timestamp": ts,
            "X-Webhook-Signature": sig,
            "X-Webhook-Nonce": "n-1",
        },
        state_path=str(state),
    )
    assert code == 200
    assert json.loads(raw.decode()).get("auth") == "hmac"

    # Replay same nonce → 409
    code2, _, raw2 = handle_alertmanager_webhook_http(
        body,
        headers={
            "X-Webhook-Timestamp": ts,
            "X-Webhook-Signature": sig,
            "X-Webhook-Nonce": "n-1",
        },
        state_path=str(state),
    )
    assert code2 == 409
    assert json.loads(raw2.decode())["error"] == "replay"

    # Missing nonce
    code3, _, raw3 = handle_alertmanager_webhook_http(
        body,
        headers={
            "X-Webhook-Timestamp": ts,
            "X-Webhook-Signature": sig,
        },
        state_path=str(state),
    )
    assert code3 == 401
    assert json.loads(raw3.decode())["error"] == "nonce_missing"

    replay = check_webhook_replay(
        nonce="n-2",
        timestamp=ts,
        state_path=str(state),
        persist=True,
    )
    assert replay["ok"] is True
    assert check_webhook_replay(
        nonce="n-2", timestamp=ts, state_path=str(state), persist=False
    )["error"] == "replay"


def test_github_dispatch_fallback_on_not_dual_write(monkeypatch, tmp_path: Path) -> None:
    from unittest.mock import patch

    from rag.dual_write_webhook import (
        handle_dual_write_alertmanager_webhook,
        maybe_github_dispatch_fallback,
        trigger_github_workflow_dispatch,
    )

    monkeypatch.setenv("RAG_DUAL_WRITE_GH_DISPATCH", "1")
    monkeypatch.setenv("GH_PAT", "ghp_test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "0")

    calls = []

    def fake_dispatch(workflow, **kwargs):
        calls.append({"workflow": workflow, **kwargs})
        return {"ok": True, "method": "rest", "workflow": workflow}

    row = {"ok": False, "action": "catch_up", "error": "not_dual_write"}
    with patch(
        "rag.dual_write_webhook.trigger_github_workflow_dispatch",
        side_effect=fake_dispatch,
    ):
        out = maybe_github_dispatch_fallback(row)
    assert out["github_fallback"]["ok"] is True
    assert calls[0]["workflow"] == "dual-write-catch-up.yml"

    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "RagDualWriteLagHigh",
                    "service": "rag-ingest",
                },
            }
        ],
    }
    with patch(
        "rag.dual_write_webhook.run_dual_write_catch_up",
        return_value={"ok": False, "action": "catch_up", "error": "not_dual_write"},
    ):
        with patch(
            "rag.dual_write_webhook.trigger_github_workflow_dispatch",
            side_effect=fake_dispatch,
        ):
            report = handle_dual_write_alertmanager_webhook(
                payload, force=True, state_path=str(tmp_path / "st.json")
            )
    assert report["ok"] is True  # fallback ok
    assert report["results"][0]["github_fallback"]["ok"] is True


def test_trigger_github_workflow_dispatch_rest(monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    from rag.dual_write_webhook import trigger_github_workflow_dispatch

    monkeypatch.setenv("GH_PAT", "ghp_x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return 204

        status = 204

        def read(self):
            return b""

    with patch("subprocess.run", side_effect=FileNotFoundError("no gh")):
        with patch("urllib.request.urlopen", return_value=_Resp()) as opener:
            result = trigger_github_workflow_dispatch(
                "dual-write-shadow-compare.yml",
                inputs={"auto_catch_up_on_fail": "true"},
                ref="main",
            )
    assert result["ok"] is True
    assert result["method"] == "rest"
    req = opener.call_args[0][0]
    assert req.full_url.endswith(
        "/repos/acme/rag/actions/workflows/dual-write-shadow-compare.yml/dispatches"
    )
