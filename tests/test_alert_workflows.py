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
    assert "LLM cost" in text
    assert '"uid": "prometheus"' in text


def test_grafana_tempo_datasource_provisioning():
    path = Path("grafana/provisioning/datasources/datasources.yml")
    text = path.read_text(encoding="utf-8")
    assert "uid: prometheus" in text
    assert "uid: tempo" in text
    assert "exemplarTraceIdDestinations" in text
    assert "trace_id" in text
    assert Path("grafana/tempo.yaml").is_file()
    assert Path("grafana/prometheus.yml").is_file()
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert 'profiles: ["obs"]' in compose
    assert "tempo:" in compose
    assert "grafana:" in compose


def test_alertmanager_and_llm_cost_rules():
    rules = Path("grafana/rules/rag_llm_cost.yml").read_text(encoding="utf-8")
    assert "rag:llm_cost_usd_per_hour" in rules
    assert "rag:llm_cost_usd_1h" in rules
    assert "RagLlmCostHigh" in rules
    prom = Path("grafana/prometheus.yml").read_text(encoding="utf-8")
    assert "rule_files:" in prom
    assert "alertmanager:9093" in prom
    am = Path("grafana/alertmanager.yml").read_text(encoding="utf-8")
    assert "judge-webhook" in am
    assert "rag-webhook" in am
    tmpl = Path("grafana/alertmanager.yml.template").read_text(encoding="utf-8")
    assert "${RAG_ALERTMANAGER_SLACK_WEBHOOK}" in tmpl
    assert "${RAG_ALERTMANAGER_WEBHOOK_URL}" in tmpl
    assert "judge-slack" in tmpl
    assert Path("scripts/render_alertmanager_config.sh").is_file()
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "alertmanager:" in compose
    assert "alertmanager.yml.template" in compose
    assert "render_alertmanager_config.sh" in compose
    assert "RAG_ALERTMANAGER_SLACK_WEBHOOK" in compose
    dash = Path("grafana/dashboards/rag_judge.json").read_text(encoding="utf-8")
    assert "rag:llm_cost_usd_1h" in dash


def test_render_alertmanager_config_script(tmp_path, monkeypatch):
    import os
    import subprocess

    out = tmp_path / "am.yml"
    env = os.environ.copy()
    env["RAG_ALERTMANAGER_SLACK_WEBHOOK"] = "https://hooks.slack.test/T/B/xxx"
    env["RAG_ALERTMANAGER_WEBHOOK_URL"] = "http://example.test/hook"
    env["ALERTMANAGER_OUTPUT"] = str(out)
    rc = subprocess.run(
        ["sh", "scripts/render_alertmanager_config.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0, rc.stderr
    text = out.read_text(encoding="utf-8")
    assert "https://hooks.slack.test/T/B/xxx" in text
    assert "http://example.test/hook" in text
    assert "${RAG_" not in text


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
