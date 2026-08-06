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
