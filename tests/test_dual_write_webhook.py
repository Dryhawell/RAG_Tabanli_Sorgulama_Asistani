from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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


def test_dual_write_webhook_dlq_and_metrics(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import patch

    from rag.dual_write_webhook import (
        dual_write_dlq_depth,
        handle_dual_write_alertmanager_webhook,
        read_dual_write_dlq,
        replay_dual_write_dlq,
    )

    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "0")
    monkeypatch.setenv("RAG_DUAL_WRITE_GH_DISPATCH", "0")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_DLQ", str(tmp_path / "dlq.jsonl"))
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_CB_FAILURES", "99")
    monkeypatch.setattr("rag.metrics.ENABLE_METRICS", True)
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("RAG_METRICS_PATH", str(metrics_path))

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
        return_value={"ok": False, "action": "catch_up", "error": "boom"},
    ):
        with patch(
            "rag.dual_write_webhook.emit_dual_write_webhook_metric"
        ) as emit:
            report = handle_dual_write_alertmanager_webhook(
                payload, force=True, state_path=str(tmp_path / "st.json")
            )
    assert report["ok"] is False
    assert report["dlq"]["ok"] is True
    assert dual_write_dlq_depth() == 1
    rows = read_dual_write_dlq()
    assert rows[0]["reason"] == "action_failed"
    assert emit.called

    with patch(
        "rag.dual_write_webhook.run_dual_write_catch_up",
        return_value={"ok": True, "action": "catch_up"},
    ):
        replayed = replay_dual_write_dlq(
            limit=5, force=True, state_path=str(tmp_path / "st2.json")
        )
    assert replayed["ok"] is True
    assert replayed["replayed"] == 1
    assert dual_write_dlq_depth() == 0


def test_prune_dual_write_dlq_by_age(tmp_path: Path, monkeypatch) -> None:
    import json
    from datetime import datetime, timedelta, timezone

    from rag.dual_write_webhook import prune_dual_write_dlq, read_dual_write_dlq

    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_DLQ", str(tmp_path / "dlq.jsonl"))
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=10)).isoformat()
    young = (now - timedelta(hours=1)).isoformat()
    path = tmp_path / "dlq.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": old, "payload_digest": "old", "payload": {}}) + "\n")
        f.write(json.dumps({"ts": young, "payload_digest": "young", "payload": {}}) + "\n")

    dry = prune_dual_write_dlq(days=7, dry_run=True)
    assert dry["removed"] == 1
    assert dry["after"] == 1
    assert len(read_dual_write_dlq()) == 2

    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_SLACK_WEBHOOK", "https://hooks.slack.test/dlq")
    with patch("rag.judge_alert.post_slack", return_value=True) as post:
        report = prune_dual_write_dlq(days=7, dry_run=False, notify=True)
    assert report["removed"] == 1
    assert report["after"] == 1
    assert read_dual_write_dlq()[0]["payload_digest"] == "young"
    assert report["notify"]["posted"] is True
    assert post.called


def test_dlq_quarantine_and_replay_budget(tmp_path: Path, monkeypatch) -> None:
    import json

    from rag.dual_write_webhook import (
        dual_write_dlq_depth,
        read_dual_write_dlq,
        read_dual_write_dlq_quarantine,
        replay_dual_write_dlq,
    )

    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_DLQ", str(tmp_path / "dlq.jsonl"))
    monkeypatch.setenv(
        "RAG_DUAL_WRITE_WEBHOOK_DLQ_QUARANTINE", str(tmp_path / "quarantine.jsonl")
    )
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_QUARANTINE_AFTER", "2")
    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_REPLAY_MAX_PER_RUN", "1")
    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_REPLAY_RUN_ID", "run-a")
    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_SLACK_WEBHOOK", "https://hooks.slack.test/q"
    )
    monkeypatch.setenv("RAG_DUAL_WRITE_GH_DISPATCH", "0")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "0")
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_CB_FAILURES", "99")

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
    entry = {
        "ts": "2026-01-01T00:00:00+00:00",
        "payload_digest": "abc",
        "payload": payload,
        "attempts": 1,
        "reason": "failed",
        "alerts": ["RagDualWriteLagHigh"],
    }
    (tmp_path / "dlq.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8"
    )

    with patch(
        "rag.dual_write_webhook.handle_dual_write_alertmanager_webhook",
        return_value={"ok": False, "reason": "boom"},
    ):
        with patch("rag.judge_alert.post_slack", return_value=True) as post:
            first = replay_dual_write_dlq(limit=5, dry_run=False, force=True)

    assert first["quarantined"] == 1
    assert dual_write_dlq_depth() == 0
    assert len(read_dual_write_dlq_quarantine()) == 1
    assert post.called

    # Same run id: budget exhausted (1/1 used)
    entry2 = {**entry, "payload_digest": "def", "attempts": 0}
    (tmp_path / "dlq.jsonl").write_text(json.dumps(entry2) + "\n", encoding="utf-8")
    with patch(
        "rag.dual_write_webhook.handle_dual_write_alertmanager_webhook",
        return_value={"ok": True},
    ):
        budgeted = replay_dual_write_dlq(limit=5, dry_run=False, force=True)
    assert budgeted.get("skipped") is True
    assert budgeted.get("reason") == "replay_budget_exhausted"
    assert dual_write_dlq_depth() == 1

    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_REPLAY_RUN_ID", "run-b")
    with patch(
        "rag.dual_write_webhook.handle_dual_write_alertmanager_webhook",
        return_value={"ok": True},
    ):
        next_run = replay_dual_write_dlq(limit=5, dry_run=False, force=True)
    assert next_run.get("skipped") is not True
    assert next_run["replayed"] == 1
    assert dual_write_dlq_depth() == 0
    assert len(read_dual_write_dlq()) == 0


def test_prune_dual_write_dlq_quarantine_aging(tmp_path: Path, monkeypatch) -> None:
    import json
    from datetime import datetime, timedelta, timezone

    from rag.dual_write_webhook import (
        prune_dual_write_dlq_quarantine,
        read_dual_write_dlq_quarantine,
    )

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_WEBHOOK_DLQ_QUARANTINE", str(tmp_path / "quarantine.jsonl")
    )
    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_SLACK_WEBHOOK", "https://hooks.slack.test/q"
    )
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=10)).isoformat()
    young = (now - timedelta(hours=1)).isoformat()
    path = tmp_path / "quarantine.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": old,
                    "quarantined_at": old,
                    "payload_digest": "old-q",
                    "payload": {},
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "ts": young,
                    "quarantined_at": young,
                    "payload_digest": "young-q",
                    "payload": {},
                }
            )
            + "\n"
        )

    with patch("rag.judge_alert.post_slack", return_value=True) as post:
        report = prune_dual_write_dlq_quarantine(days=7, dry_run=False, notify=True)
    assert report["removed"] == 1
    assert report["after"] == 1
    assert read_dual_write_dlq_quarantine()[0]["payload_digest"] == "young-q"
    assert report["notify"]["posted"] is True
    assert post.called
    assert "quarantine" in (post.call_args[0][1]["text"] or "").lower()


def test_quarantine_slack_block_kit_status(tmp_path: Path, monkeypatch) -> None:
    from rag.dual_write_webhook import (
        build_dual_write_dlq_quarantine_slack_payload,
        maybe_alert_dual_write_dlq_quarantine,
    )

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_WEBHOOK_DLQ_QUARANTINE", str(tmp_path / "quarantine.jsonl")
    )
    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_SLACK_WEBHOOK", "https://hooks.slack.test/q"
    )
    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_QUARANTINE_BLOCK_KIT", "1")
    entry = {
        "payload_digest": "abc123",
        "attempts": 3,
        "quarantine_reason": "replay_exhausted",
    }
    payload = build_dual_write_dlq_quarantine_slack_payload(
        mode="enqueue", entry=entry, path=str(tmp_path / "quarantine.jsonl"), depth=1
    )
    assert isinstance(payload.get("blocks"), list)
    assert payload["blocks"][0]["type"] == "header"
    assert "Dual-write DLQ quarantine" in payload["text"]

    aging = build_dual_write_dlq_quarantine_slack_payload(
        mode="aging",
        path=str(tmp_path / "quarantine.jsonl"),
        depth=2,
        reasons=["depth `2` ≥ `1`"],
        oldest_age_hours=50.0,
    )
    assert aging["blocks"][0]["type"] == "header"
    assert "aging" in aging["text"].lower()
    assert "50.0h" in aging["blocks"][1]["text"]["text"]

    with patch("rag.judge_alert.post_slack", return_value=True) as post:
        out = maybe_alert_dual_write_dlq_quarantine(entry=entry, depth=1)
    assert out.get("ok") is True
    assert out.get("block_kit") is True
    assert out.get("posted") is True
    assert post.call_args[0][1]["blocks"][0]["type"] == "header"

    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_QUARANTINE_BLOCK_KIT", "0")
    plain = build_dual_write_dlq_quarantine_slack_payload(
        mode="enqueue", entry=entry, depth=1
    )
    assert "blocks" not in plain


def test_replay_dual_write_dlq_quarantine(tmp_path: Path, monkeypatch) -> None:
    import json

    from rag.dual_write_webhook import (
        dual_write_dlq_depth,
        dual_write_dlq_quarantine_depth,
        read_dual_write_dlq,
        read_dual_write_dlq_quarantine,
        replay_dual_write_dlq_quarantine,
    )

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_WEBHOOK_DLQ_QUARANTINE", str(tmp_path / "quarantine.jsonl")
    )
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_DLQ", str(tmp_path / "dlq.jsonl"))
    monkeypatch.setenv("RAG_DUAL_WRITE_WEBHOOK_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("RAG_DUAL_WRITE_DLQ_REPLAY_RUN_ID", "q-run")
    monkeypatch.delenv("RAG_DUAL_WRITE_DLQ_REPLAY_MAX_PER_RUN", raising=False)
    monkeypatch.delenv("RAG_DUAL_WRITE_DLQ_REPLAY_MAX_PER_HOUR", raising=False)

    payload = {
        "status": "firing",
        "alerts": [
            {
                "status": "firing",
                "labels": {"alertname": "RagDualWriteLagHigh", "service": "rag-ingest"},
            }
        ],
    }
    entry = {
        "ts": "2026-01-01T00:00:00+00:00",
        "payload_digest": "q1",
        "payload": payload,
        "attempts": 3,
        "quarantine_reason": "replay_exhausted",
    }
    (tmp_path / "quarantine.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with patch(
        "rag.dual_write_webhook.handle_dual_write_alertmanager_webhook",
        return_value={"ok": True},
    ):
        ok = replay_dual_write_dlq_quarantine(limit=5, dry_run=False, force=True)
    assert ok["replayed"] == 1
    assert ok["remaining"] == 0
    assert dual_write_dlq_quarantine_depth() == 0

    (tmp_path / "quarantine.jsonl").write_text(
        json.dumps({**entry, "payload_digest": "q2"}) + "\n", encoding="utf-8"
    )
    with patch(
        "rag.dual_write_webhook.handle_dual_write_alertmanager_webhook",
        return_value={"ok": False, "reason": "still_bad"},
    ):
        rq = replay_dual_write_dlq_quarantine(
            limit=5, dry_run=False, force=True, requeue=True
        )
    assert rq["requeued"] == 1
    assert dual_write_dlq_quarantine_depth() == 0
    assert dual_write_dlq_depth() == 1
    assert read_dual_write_dlq()[0]["payload_digest"] == "q2"
    assert len(read_dual_write_dlq_quarantine()) == 0


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
