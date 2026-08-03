"""Grafana / workflow artifact smoke tests."""

from pathlib import Path


def test_grafana_judge_soft_fail_alert_yaml():
    path = Path("grafana/alerting/rag_judge_soft_fail.yaml")
    assert path.is_file()
    blob = path.read_text(encoding="utf-8")
    assert "apiVersion: 1" in blob
    assert "rag_judge_soft_fail_total" in blob
    assert "rag_judge_runs_total" in blob
    assert "0.3" in blob
    assert "rag-judge" in blob


def test_digest_alert_workflow_yaml():
    path = Path(".github/workflows/digest-alert.yml")
    text = path.read_text(encoding="utf-8")
    assert "digest-alert-check" in text
    assert "digest-alert-all" in text
    assert "RAG_DIGEST_ALERT_WEBHOOK_URL" in text
    assert "schedule:" in text


def test_vapid_rotate_workflow_yaml():
    path = Path(".github/workflows/vapid-rotate.yml")
    text = path.read_text(encoding="utf-8")
    assert "rotate-vapid" in text
    assert "update-github-secrets" in text
    assert "workflow_dispatch" in text
    assert "vapid_keys.json" in text
    assert "GH_PAT" in text
