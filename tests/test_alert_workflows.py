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


def test_grafana_judge_dashboard_has_llm_and_exemplars():
    path = Path("grafana/dashboards/rag_judge.json")
    text = path.read_text(encoding="utf-8")
    assert "rag_llm_tokens_total" in text
    assert "exemplar" in text
    assert "Tempo" in text or "exemplars" in text.lower()


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
    assert "github_environment" in text
    assert "vapid-github-environment" in text


def test_ci_judge_alert_state_persistence_yaml():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "judge_alert_state.json" in text
    assert "RAG_JUDGE_ALERT_STATE" in text
    assert "judge-alert-state-" in text
    assert "actions/cache@v4" in text
