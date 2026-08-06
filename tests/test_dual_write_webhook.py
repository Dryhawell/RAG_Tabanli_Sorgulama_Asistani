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


def test_http_adapter_unauthorized(monkeypatch) -> None:
    from rag.dual_write_webhook import handle_alertmanager_webhook_http

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
