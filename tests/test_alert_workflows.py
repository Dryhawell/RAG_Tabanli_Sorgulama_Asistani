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


def test_rotate_alertmanager_slack_webhook(tmp_path, monkeypatch):
    from rag.alertmanager_ops import rotate_alertmanager_slack_webhook

    out = tmp_path / "am.yml"
    env_path = tmp_path / "am.env"
    reloads = []

    def fake_reload(*, url=None, timeout=5.0):
        reloads.append(url)
        return {"ok": True, "status": 200, "url": url or "http://127.0.0.1:9093/-/reload"}

    monkeypatch.setattr("rag.alertmanager_ops.reload_alertmanager", fake_reload)
    report = rotate_alertmanager_slack_webhook(
        "https://hooks.slack.test/new/wh",
        write_env=str(env_path),
        output=str(out),
        reload=True,
        reload_url="http://am.test/-/reload",
    )
    assert report["ok"] is True
    assert "https://hooks.slack.test/new/wh" in env_path.read_text(encoding="utf-8")
    assert "https://hooks.slack.test/new/wh" in out.read_text(encoding="utf-8")
    assert reloads == ["http://am.test/-/reload"]


def test_cli_alertmanager_parser():
    from rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "alertmanager",
            "--rotate-slack-webhook",
            "https://hooks.slack.test/x",
            "--no-reload",
            "--update-github-secrets",
            "--secrets-dry-run",
        ]
    )
    assert args.command == "alertmanager"
    assert args.rotate_slack_webhook.startswith("https://")
    assert args.no_reload is True
    assert args.update_github_secrets is True
    assert args.secrets_dry_run is True
    silence_args = parser.parse_args(
        [
            "alertmanager",
            "--silence",
            "--matcher",
            "service=rag-judge",
            "--matcher",
            "alertname=~Rag.*",
            "--duration",
            "2h",
            "--comment",
            "maint",
        ]
    )
    assert silence_args.silence is True
    assert silence_args.matcher == ["service=rag-judge", "alertname=~Rag.*"]
    assert silence_args.duration == "2h"


def test_create_silence_posts_api(monkeypatch):
    from rag.alertmanager_ops import create_silence, parse_duration_sec, parse_silence_matcher

    assert parse_duration_sec("2h") == 7200.0
    assert parse_silence_matcher("service=rag-judge")["isRegex"] is False
    assert parse_silence_matcher("alertname=~Rag.*")["isRegex"] is True

    captured = {}

    class FakeResp:
        status = 200

        def read(self):
            return b'{"silenceID":"abc-123"}'

        def getcode(self):
            return 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=5.0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data
        return FakeResp()

    monkeypatch.setattr("rag.alertmanager_ops.request.urlopen", fake_urlopen)
    out = create_silence(
        matchers=[parse_silence_matcher("service=rag-judge")],
        duration_sec=3600,
        comment="test",
        base_url="http://am.test:9093",
    )
    assert out["ok"] is True
    assert out["silenceID"] == "abc-123"
    assert captured["url"].endswith("/api/v2/silences")
    assert captured["method"] == "POST"


def test_generate_inhibit_rules_from_grafana_labels(tmp_path):
    from rag.alertmanager_ops import (
        extract_grafana_alert_labels,
        generate_inhibit_rules,
        render_inhibit_rules_yaml,
        write_inhibit_rules,
    )

    labels = extract_grafana_alert_labels(
        ["grafana/alerting/rag_judge_soft_fail.yaml", "grafana/rules/rag_llm_cost.yml"]
    )
    assert labels
    assert any(
        (item.get("labels") or {}).get("service") == "rag-judge" for item in labels
    )
    rules = generate_inhibit_rules(labels)
    assert rules
    yaml_text = render_inhibit_rules_yaml(rules)
    assert "inhibit_rules:" in yaml_text
    assert "source_matchers:" in yaml_text
    out = tmp_path / "inhibit.yml"
    report = write_inhibit_rules(
        paths=[
            "grafana/alerting/rag_judge_soft_fail.yaml",
            "grafana/rules/rag_llm_cost.yml",
        ],
        output=str(out),
    )
    assert report["ok"] is True
    assert out.is_file()
    assert "severity = critical" in out.read_text(encoding="utf-8")


def test_cli_alertmanager_generate_inhibit_parser():
    from rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["alertmanager", "--generate-inhibit", "--equal", "alertname,service"]
    )
    assert args.generate_inhibit is True
    assert args.equal == "alertname,service"


def test_dual_write_catch_up_workflow_yaml():
    path = Path(".github/workflows/dual-write-catch-up.yml")
    text = path.read_text(encoding="utf-8")
    assert "migrate-vector --catch-up" in text
    assert "schedule:" in text
    assert "RAG_VECTOR_DUAL_WRITE" in text
    assert "auto_cutover" in text
    assert "--auto-cutover" in text


def test_alertmanager_slack_rotate_workflow_yaml():
    path = Path(".github/workflows/alertmanager-slack-rotate.yml")
    text = path.read_text(encoding="utf-8")
    assert "rotate-slack-webhook" in text
    assert "update-github-secrets" in text
    assert "GH_PAT" in text
    assert "workflow_dispatch" in text


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
