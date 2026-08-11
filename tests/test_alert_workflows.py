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
    assert "${RAG_ALERTMANAGER_INGEST_WEBHOOK_URL}" in tmpl
    assert "${RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL}" in tmpl
    assert "ingest-webhook" in tmpl
    assert "ingest-dlq-quarantine-webhook" in tmpl
    assert "RagDualWriteDlqQuarantineDepthHigh" in tmpl
    assert "service = rag-ingest" in tmpl
    assert "judge-slack" in tmpl
    assert Path("scripts/render_alertmanager_config.sh").is_file()
    render_sh = Path("scripts/render_alertmanager_config.sh").read_text(encoding="utf-8")
    assert "RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL" in render_sh
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "alertmanager:" in compose
    assert "alertmanager.yml.template" in compose
    assert "render_alertmanager_config.sh" in compose
    assert "RAG_ALERTMANAGER_SLACK_WEBHOOK" in compose
    dash = Path("grafana/dashboards/rag_judge.json").read_text(encoding="utf-8")
    assert "rag:llm_cost_usd_1h" in dash
    am = Path("grafana/alertmanager.yml").read_text(encoding="utf-8")
    assert "ingest-webhook" in am
    assert "ingest-dlq-quarantine-webhook" in am
    assert "rag-ingest" in am
    assert "dual-write-catch-up" in am
    assert "dual-write-dlq-quarantine" in am
    assert "RagDualWriteDlqQuarantineDepthHigh" in am


def test_render_alertmanager_config_script(tmp_path, monkeypatch):
    import os
    import subprocess

    out = tmp_path / "am.yml"
    env = os.environ.copy()
    env["RAG_ALERTMANAGER_SLACK_WEBHOOK"] = "https://hooks.slack.test/T/B/xxx"
    env["RAG_ALERTMANAGER_WEBHOOK_URL"] = "http://example.test/hook"
    env["RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL"] = (
        "http://example.test/hooks/dual-write-dlq-quarantine"
    )
    env["ALERTMANAGER_OUTPUT"] = str(out)
    # no inhibit file → no merge noise
    env["ALERTMANAGER_INHIBIT"] = str(tmp_path / "missing_inhibit.yml")
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
    assert "dual-write-dlq-quarantine" in text
    assert "ingest-dlq-quarantine-webhook" in text
    assert "silence-burn-slack" in text
    assert "RagInhibitEqualCanarySilenceBurn" in text
    assert "${RAG_" not in text


def test_render_merges_generated_inhibit_rules(tmp_path):
    import os
    import subprocess

    from rag.alertmanager_ops import merge_inhibit_rules_into_config, write_inhibit_rules

    out = tmp_path / "am.yml"
    inhibit = tmp_path / "inhibit.yml"
    write_inhibit_rules(
        paths=[
            "grafana/alerting/rag_judge_soft_fail.yaml",
            "grafana/rules/rag_llm_cost.yml",
        ],
        output=str(inhibit),
    )
    env = os.environ.copy()
    env["RAG_ALERTMANAGER_SLACK_WEBHOOK"] = "https://hooks.slack.test/T/B/xxx"
    env["RAG_ALERTMANAGER_WEBHOOK_URL"] = "http://example.test/hook"
    env["ALERTMANAGER_OUTPUT"] = str(out)
    env["ALERTMANAGER_INHIBIT"] = str(inhibit)
    rc = subprocess.run(
        ["sh", "scripts/render_alertmanager_config.sh"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rc.returncode == 0, rc.stderr
    assert "inhibit_merged:" in (rc.stdout or "")
    text = out.read_text(encoding="utf-8")
    assert text.count("source_matchers:") >= 2
    # idempotent python merge
    report = merge_inhibit_rules_into_config(str(out), inhibit_path=str(inhibit))
    assert report["ok"] is True
    assert report.get("added", 0) == 0 or report.get("merged") is False


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


def test_cli_alertmanager_inhibit_arg_parser():
    from rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "alertmanager",
            "--render",
            "--inhibit",
            "metadata/inhibit.yml",
            "--output",
            "metadata/am.yml",
        ]
    )
    assert args.render is True
    assert args.inhibit == "metadata/inhibit.yml"
    assert args.output == "metadata/am.yml"


def test_render_inhibit_then_check_config(tmp_path, monkeypatch):
    """CI pipeline: generate-inhibit → render --inhibit → check-config."""
    import json
    from unittest.mock import patch

    from rag.alertmanager_ops import check_alertmanager_config, write_inhibit_rules
    from rag.cli import main

    inhibit = tmp_path / "inhibit.yml"
    out = tmp_path / "am.yml"
    report = write_inhibit_rules(output=str(inhibit))
    assert report["ok"] is True
    assert inhibit.is_file()

    code = main(
        [
            "alertmanager",
            "--render",
            "--output",
            str(out),
            "--inhibit",
            str(inhibit),
            "--slack-webhook",
            "https://hooks.slack.test/T/B/x",
            "--webhook-url",
            "http://127.0.0.1/hook",
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "route:" in text
    assert "receivers:" in text
    if "source_matchers:" in inhibit.read_text(encoding="utf-8"):
        assert "inhibit_rules:" in text

    with patch("shutil.which", return_value=None):
        checked = check_alertmanager_config(str(out))
    assert checked["ok"] is True
    assert checked["method"] == "structural"



def test_dual_write_catch_up_workflow_yaml():
    path = Path(".github/workflows/dual-write-catch-up.yml")
    text = path.read_text(encoding="utf-8")
    assert "migrate-vector --catch-up" in text
    assert "schedule:" in text
    assert "RAG_VECTOR_DUAL_WRITE" in text
    assert "auto_cutover" in text
    assert "--auto-cutover" in text


def test_dual_write_shadow_compare_workflow_yaml():
    path = Path(".github/workflows/dual-write-shadow-compare.yml")
    text = path.read_text(encoding="utf-8")
    assert "migrate-vector" in text
    assert "--shadow-compare" in text
    assert "--auto-catch-up-on-fail" in text
    assert "auto_catch_up_on_fail" in text
    assert "RAG_DUAL_WRITE_SHADOW_MIN_OVERLAP" in text
    assert "schedule:" in text
    assert "workflow_dispatch" in text
    assert "dual_write_shadow_compare.json" in text


def test_dual_write_dlq_replay_workflow_yaml():
    path = Path(".github/workflows/dual-write-dlq-replay.yml")
    text = path.read_text(encoding="utf-8")
    assert "dual-write-dlq" in text
    assert "--replay" in text
    assert "15 */6 * * *" in text
    assert "dual_write_dlq_replay.json" in text
    assert "--prune" in text
    assert "dual_write_dlq_prune.json" in text
    assert "--quarantine" in text
    assert "dual_write_dlq_quarantine.json" in text
    assert "--replay-quarantine" in text
    assert "dual_write_dlq_quarantine_replay.json" in text
    assert "--prune-quarantine" in text
    assert "dual_write_dlq_quarantine_prune.json" in text
    assert "RAG_DUAL_WRITE_DLQ_QUARANTINE_RETENTION_DAYS" in text
    assert "RAG_DUAL_WRITE_DLQ_QUARANTINE_AUTO_REPLAY" in text
    assert "RAG_DUAL_WRITE_DLQ_QUARANTINE_AUTO_REQUEUE" in text
    assert 'github.event_name }}" = "schedule"' in text or "schedule" in text
    assert "RAG_DUAL_WRITE_DLQ_REPLAY_MAX_PER_RUN" in text
    assert "RAG_DUAL_WRITE_DLQ_QUARANTINE_AFTER" in text
    assert "workflow_dispatch" in text


def test_ci_inhibit_equal_opsgenie_canary_env():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY" in text
    assert "INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON" in text
    assert "INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS" in text
    assert "INHIBIT_EQUAL_CANARY_CLOSE_ON_GREEN" in text
    assert "INHIBIT_EQUAL_CANARY_AUTO_SILENCE" in text
    assert "INHIBIT_EQUAL_CANARY_AUTO_SILENCE_DURATION" in text
    assert "INHIBIT_EQUAL_CANARY_AUTO_SILENCE_MATCHERS" in text
    assert "INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY" in text
    assert "INHIBIT_EQUAL_CANARY_PD_SEVERITY" in text
    assert "INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON" in text
    assert "INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL" in text
    assert (
        "INHIBIT_EQUAL_CANARY_PD_SEVERITY: "
        "${{ vars.INHIBIT_EQUAL_CANARY_PD_SEVERITY || 'error' }}"
    ) in text
    assert (
        "INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON: "
        "${{ vars.INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON || '' }}"
    ) in text
    assert (
        "INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL: "
        "${{ vars.INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL || '' }}"
    ) in text


def test_readme_sonraki_adaylar_after_mute_revoke_canvas_rotate_digest_og_folder():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "## Sonraki adaylar" in text
    assert "presence multi-worker (Redis)" in text
    assert "VAPID OIDC" in text
    sonraki = text.split("## Sonraki adaylar")[1].split("##")[0]
    assert "mute export revoke canvas fan-out audit trail" in sonraki
    assert "sidecar rotate notify digest thread" in sonraki
    assert "Alertmanager folder receiver policy" in sonraki
    # Completed this round — should not remain as next candidates
    assert "mute export revoke Slack canvas refresh on revoke" not in sonraki
    assert "sidecar rotate notify digest\n" not in sonraki
    assert "Grafana folder alert routing" not in sonraki
    assert "webhook-signing-sidecar" in text
    assert "--export-mute-snapshots" in text
    assert "--upload-mute-snapshots" in text
    assert "--prune-mute-exports" in text
    assert "--sign-mute-export" in text
    assert "--revoke-mute-export" in text
    assert "--sweep-mute-export-urls" in text
    assert "/judge/mute-export-revoke" in text
    assert "judge_ack_digest_revoke_mute_export" in text
    assert "judge_mute_export_revoke_modal" in text
    assert "mute_export_sign" in text
    assert "mute_export_revoke" in text
    assert "mute_export_revokes" in text
    assert "mute_export_sweep" in text
    assert "mute_export_signed_urls.json" in text
    assert "/judge/mute-snapshots" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SIGNING_SECRET" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_RETENTION_DAYS" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SWEEP" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_REVOKE_CONFIRM" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_REVOKE_THREAD" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_REVOKE_CANVAS" in text
    assert "tenant|latest" in text
    assert "RAG_JUDGE_ACK_RATE_LIMIT_SEC" in text
    assert "judge_ack_digest_export_mute_snapshots" in text
    assert "judge-ack-digest.yml" in text
    assert "webhook-signing-sidecar-rotate.yml" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_UPLOAD_CHANNEL" in text
    assert "rag_webhook_signing_sidecar_forward_total" in text
    assert "rag_webhook_signing_sidecar_cert_expiry_days" in text
    assert "RagWebhookSigningSidecarCertExpirySoon" in text
    assert "--check-certs" in text
    assert "--rotate-certs" in text
    assert "--rotate-certs-if-expiring" in text
    assert "--notify" in text
    assert "--dry-run" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PAGERDUTY_ROUTING_KEY" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY_BY_ERROR" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_DRY_RUN_GATE" in text
    assert "would_rotate" in text
    assert "notify_pd" in text
    assert "runbook:silence-burn" in text
    assert "INHIBIT_EQUAL_CANARY_OPSGENIE_TAGS" in text
    assert "opsgenie_alert_deep_link" in text
    assert "INHIBIT_EQUAL_CANARY_OPSGENIE_ALERT_URL" in text
    assert "opsgenie_url" in text
    assert "INHIBIT_EQUAL_CANARY_CLOSE_ACK_SYNC" in text
    assert "opsgenie_ack_regions" in text
    assert "green recover close" in text
    assert "RagWebhookSigningSidecarForwardFail" in text
    assert "rag_webhook_signing_sidecar.yml" in text
    assert "RagInhibitEqualCanarySilenceBurn" in text
    assert "rag:inhibit_equal_canary_silence_fail_ratio" in text
    assert "/ops/silence-burn" in text
    assert "/ops/amtool" in text
    assert "INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL" in text
    assert "Opsgenie" in text and "runbook_url" in text
    assert "RAG Silence Burn" in text
    assert "grafana_folder=rag-silence-burn" in text
    assert "silence-burn-webhook" in text
    assert "rag-silence-burn-slack" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT" in text
    assert "INHIBIT_EQUAL_CANARY_SILENCE_EXPIRY_WEBHOOK" in text
    assert "--canary-silence-expiry" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_MTLS" in text
    assert "rag_judge_ack_digest_msgref_reconcile_total" in text
    assert "INHIBIT_EQUAL_CANARY_AUTO_SILENCE" in text
    assert "rag_inhibit_equal_canary_silence_total" in text
    assert "RAG_JUDGE_ACK_DIGEST_HISTORY_RECONCILE" in text
    assert "RagInhibitEqualCanaryResolveFail" in text
    assert "RAG_DUAL_WRITE_DLQ_QUARANTINE_WEBHOOK_SIGNING_SECRET" in text
    assert "RAG_ALERTMANAGER_WEBHOOK_REQUIRE_AUTH=0" in text
    assert "Auth varsayılan zorunlu" in text


def test_webhook_signing_sidecar_rotate_workflow_yaml():
    path = Path(".github/workflows/webhook-signing-sidecar-rotate.yml")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "--rotate-certs-if-expiring" in text
    assert "--rotate-certs" in text
    assert "--check-certs" in text
    assert "--notify" in text
    assert "--dry-run" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PAGERDUTY_ROUTING_KEY" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_NOTIFY" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY_BY_ERROR" in text
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_DRY_RUN_GATE" in text
    assert "DRY_RUN_GATE_FAIL" in text
    assert "would_rotate" in text or "would rotate" in text
    assert "sidecar_cert_rotate_dry_run.json" in text
    assert "webhook-signing-sidecar" in text
    assert "metadata/sidecar-tls" in text
    assert "sidecar_cert_rotate.json" in text
    assert "schedule:" in text
    assert "workflow_dispatch" in text
    assert "0 4 * * *" in text


def test_ci_inhibit_equal_slack_thread_env():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN" in text
    assert "INHIBIT_EQUAL_CANARY_SLACK_CHANNEL" in text
    assert "INHIBIT_EQUAL_CANARY_SLACK_THREAD_REPLY" in text
    assert "INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS" in text
    assert "metadata/inhibit_equal_canary_slack.json" in text


def test_judge_ack_digest_mute_prune_workflow_yaml():
    text = Path(".github/workflows/judge-ack-digest.yml").read_text(encoding="utf-8")
    assert "--prune-mutes" in text
    assert "--reconcile-messages" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_TTL_DAYS" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_PRUNE" in text
    assert "RAG_JUDGE_ACK_DIGEST_MESSAGES_TTL_DAYS" in text
    assert "RAG_JUDGE_ACK_DIGEST_MESSAGES_PRUNE" in text
    assert "RAG_JUDGE_ACK_DIGEST_HISTORY_RECONCILE" in text
    assert "RAG_JUDGE_SLACK_BOT_TOKEN" in text
    assert "judge_ack_digest_messages.json" in text
    assert "--export-mute-snapshots" in text
    assert "--upload-mute-snapshots" in text
    assert "--prune-mute-exports" in text
    assert "--sign-mute-export" in text
    assert "--sweep-mute-export-urls" in text
    assert "judge_ack_digest_mute_snapshots.csv" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_UPLOAD_CHANNEL" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_RETENTION_DAYS" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SIGNING_SECRET" in text
    assert "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SWEEP" in text
    assert "metadata/mute_exports/" in text
    assert "mute_export_signed_urls.json" in text
    assert "Export mute snapshots" in text


def test_canary_silence_expiry_workflow_yaml():
    path = Path(".github/workflows/inhibit-equal-canary-silence-expiry.yml")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "--canary-silence-expiry" in text
    assert "INHIBIT_EQUAL_CANARY_SILENCE_EXPIRY_WEBHOOK" in text
    assert "inhibit_equal_canary_silence.json" in text
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT" in compose
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY" in compose
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CA" in compose


def test_judge_ack_digest_workflow_yaml():
    path = Path(".github/workflows/judge-ack-digest.yml")
    text = path.read_text(encoding="utf-8")
    assert "judge-ack-digest" in text
    assert "--fan-out" in text
    assert "RAG_JUDGE_SLACK_WEBHOOK" in text
    assert "RAG_JUDGE_ACK_DIGEST_WEBHOOKS_JSON" in text
    assert "RAG_JUDGE_ACK_DIGEST_QUIET_HOURS" in text
    assert "RAG_JUDGE_ACK_DIGEST_QUIET_HOURS_JSON" in text
    assert "RAG_JUDGE_ACK_AUDIT_RETENTION_DAYS" in text
    assert "RAG_JUDGE_ACK_AUDIT_PRUNE" in text
    assert "judge_ack_digest_last.json" in text
    assert "schedule:" in text
    assert "0 9 * * 1" in text
    assert "workflow_dispatch" in text


def test_ci_alertmanager_check_config_step():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "--check-config" in text
    assert "--generate-inhibit" in text
    assert "--inhibit" in text
    assert "--tune-equal" in text
    assert "ci_inhibit_equal_pr_comment.py" in text
    assert "ci_inhibit_equal_apply.py" in text
    assert "INHIBIT_EQUAL_APPLY_PR_COMMENT_POST" in text
    assert "needs.test.result" in text or "needs: [test]" in text
    assert "inhibit-equal-apply" in text
    assert "alertmanager-inhibit-" in text
    assert "alertmanager --render" in text
    assert "Alertmanager check-config" in text
    assert "inhibit_equal_tune.json" in text
    assert "inhibit_equal_apply_pr_comment.md" in text


def test_dual_write_shadow_alert_artifacts():
    rules = Path("grafana/rules/rag_dual_write.yml").read_text(encoding="utf-8")
    assert "RagDualWriteShadowOverlapLow" in rules
    assert "RagDualWriteLagHigh" in rules
    assert "RagDualWriteLagShadowBurn" in rules
    assert "RagDualWriteDlqDepthHigh" in rules
    assert "RagDualWriteDlqQuarantineDepthHigh" in rules
    assert "rag_dual_write_webhook_dlq_depth" in rules
    assert "rag_dual_write_webhook_dlq_quarantine_depth" in rules
    assert "rag:dual_write_dlq_quarantine_depth:avg1h" in rules
    assert "rag_vector_dual_write_shadow_overlap" in rules
    assert "rag:dual_write_shadow_overlap:avg1h" in rules
    assert "rag:dual_write_lag:avg1h" in rules
    assert "rag:dual_write_shadow_burn:1h" in rules
    alerting = Path("grafana/alerting/rag_dual_write_shadow.yaml").read_text(
        encoding="utf-8"
    )
    assert "rag-dual-write-shadow-overlap-low" in alerting
    assert "rag-dual-write-lag-shadow-burn" in alerting
    assert "rag-dual-write-dlq-quarantine-depth" in alerting
    assert "rag_dual_write_webhook_dlq_quarantine_depth" in alerting
    assert "Dashboard: rag-judge" in alerting
    assert "--quarantine --notify" in alerting
    assert "--replay-quarantine" in alerting
    assert "0.95" in alerting
    rules = Path("grafana/rules/rag_dual_write.yml").read_text(encoding="utf-8")
    assert "dashboard_uid: rag-judge" in rules
    assert "Prune: python -m rag.cli dual-write-dlq --prune-quarantine" in rules
    dash = Path("grafana/dashboards/rag_judge.json").read_text(encoding="utf-8")
    assert "rag_vector_dual_write_shadow_overlap" in dash
    assert "rag:dual_write_shadow_overlap:avg1h" in dash
    assert "rag:dual_write_lag:avg1h" in dash
    assert "rag:dual_write_shadow_burn:1h" in dash
    assert "DLQ quarantine depth" in dash
    assert "rag_dual_write_webhook_dlq_quarantine_depth >= 1" in dash


def test_inhibit_equal_canary_resolve_alert_artifacts():
    rules = Path("grafana/rules/rag_inhibit_equal.yml").read_text(encoding="utf-8")
    assert "RagInhibitEqualCanaryResolveFail" in rules
    assert "RagInhibitEqualCanaryResolveFailBurst" in rules
    assert "rag_inhibit_equal_canary_resolve_total" in rules
    assert 'result="fail"' in rules
    assert "dashboard_uid: rag-judge" in rules
    assert "RagInhibitEqualCanarySilenceBurn" in rules
    assert "rag:inhibit_equal_canary_silence_fail_ratio:1h" in rules
    assert "rag:inhibit_equal_canary_silence_fail_ratio:6h" in rules
    assert "rag_inhibit_equal_canary_silence_total" in rules
    assert 'panel_id: "25"' in rules
    assert "runbook_url: /ops/silence-burn" in rules
    assert "opsgenie_url:" in rules
    assert "rag-judge-soft-fail/inhibit-equal-canary" in rules
    assert "--list-silences" in rules
    assert "--check-config" in rules
    assert "INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL" in rules
    assert "Opsgenie" in rules or "details.runbook_url" in rules
    assert "grafana_folder: rag-silence-burn" in rules
    assert "RAG Silence Burn" in rules
    alerting = Path("grafana/alerting/rag_inhibit_equal_canary.yaml").read_text(
        encoding="utf-8"
    )
    assert "rag-inhibit-equal-canary-resolve-fail" in alerting
    assert "rag-inhibit-equal-canary-resolve-fail-burst" in alerting
    assert "rag-inhibit-equal-canary-silence-burn" in alerting
    assert "rag_inhibit_equal_canary_resolve_total" in alerting
    assert "Inhibit equal canary resolve" in alerting
    assert "silence burn-rate" in alerting
    assert "runbook_url: /ops/silence-burn" in alerting
    assert "opsgenie_url:" in alerting
    assert "/ops/amtool" in alerting
    assert "INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL" in alerting
    assert "Opsgenie" in alerting or "details.runbook_url" in alerting
    assert "folder: RAG Silence Burn" in alerting
    assert "name: rag-inhibit-equal-silence-burn" in alerting
    assert "grafana_folder: rag-silence-burn" in alerting
    contact = Path("grafana/provisioning/alerting/rag_judge_contact.yaml").read_text(
        encoding="utf-8"
    )
    assert "rag-silence-burn-slack" in contact
    assert "RAG Silence Burn" in contact
    assert '["grafana_folder", "=", "RAG Silence Burn"]' in contact
    assert "RagInhibitEqualCanarySilenceBurn" in contact
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "webhook-signing-sidecar" in compose
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM" in compose
    am = Path("grafana/alertmanager.yml").read_text(encoding="utf-8")
    assert "webhook-signing-sidecar" in am
    assert "silence-burn-webhook" in am
    assert "RagInhibitEqualCanarySilenceBurn" in am
    assert "grafana_folder = rag-silence-burn" in am
    tmpl = Path("grafana/alertmanager.yml.template").read_text(encoding="utf-8")
    assert "signing sidecar" in tmpl
    assert "silence-burn-slack" in tmpl
    assert "RagInhibitEqualCanarySilenceBurn" in tmpl
    dash = Path("grafana/dashboards/rag_judge.json").read_text(encoding="utf-8")
    assert "Silence burn" in dash
    assert "opsgenie.com/alert/list" in dash
    assert "rag:inhibit_equal_canary_silence_fail_ratio:1h > 0.2" in dash


def test_webhook_signing_sidecar_grafana_alert_artifacts():
    rules = Path("grafana/rules/rag_webhook_signing_sidecar.yml").read_text(
        encoding="utf-8"
    )
    assert "RagWebhookSigningSidecarForwardFail" in rules
    assert "RagWebhookSigningSidecarForwardFailBurst" in rules
    assert "RagWebhookSigningSidecarCertExpirySoon" in rules
    assert "RagWebhookSigningSidecarCertExpired" in rules
    assert "rag_webhook_signing_sidecar_forward_total" in rules
    assert "rag_webhook_signing_sidecar_cert_expiry_days" in rules
    assert 'result!~"ok|dry_run"' in rules
    assert 'panel_id: "24"' in rules
    assert 'panel_id: "26"' in rules
    assert "--check-certs" in rules
    assert "--rotate-certs" in rules
    assert "rag-ingest" in rules
    alerting = Path("grafana/alerting/rag_webhook_signing_sidecar.yaml").read_text(
        encoding="utf-8"
    )
    assert "rag-webhook-signing-sidecar-forward-fail" in alerting
    assert "rag-webhook-signing-sidecar-forward-fail-burst" in alerting
    assert "rag-webhook-signing-sidecar-cert-expiry-soon" in alerting
    assert "rag-webhook-signing-sidecar-cert-expired" in alerting
    assert "Signing sidecar forward" in alerting
    assert "cert expiry" in alerting
    assert "rag-ingest" in alerting


def test_suggest_equal_labels_and_from_live(tmp_path, monkeypatch):
    from unittest.mock import patch

    from rag.alertmanager_ops import (
        suggest_equal_labels,
        tune_inhibit_equal_from_live,
        write_inhibit_rules,
    )

    equal = suggest_equal_labels(
        [
            {"labels": {"alertname": "A", "service": "rag-ingest", "secondary_backend": "qdrant"}},
            {"labels": {"alertname": "B", "service": "rag-ingest", "secondary_backend": "qdrant"}},
            {"labels": {"alertname": "C", "service": "rag-judge", "tenant": "acme"}},
        ],
        min_count=2,
    )
    assert "alertname" in equal
    assert "service" in equal
    assert "secondary_backend" in equal

    fake_alerts = {
        "ok": True,
        "alerts": [
            {
                "labels": {
                    "alertname": "RagDualWriteLagShadowBurn",
                    "service": "rag-ingest",
                    "secondary_backend": "qdrant",
                    "severity": "critical",
                }
            },
            {
                "labels": {
                    "alertname": "RagDualWriteLagHigh",
                    "service": "rag-ingest",
                    "secondary_backend": "qdrant",
                    "severity": "warning",
                }
            },
        ],
    }
    with patch("rag.alertmanager_ops.list_alerts", return_value=fake_alerts):
        tune = tune_inhibit_equal_from_live(include_static=False, min_count=2)
        assert tune["ok"] is True
        assert "secondary_backend" in tune["equal"]
        out = tmp_path / "inhibit.yml"
        report = write_inhibit_rules(
            output=str(out),
            from_live=True,
            paths=["grafana/rules/rag_dual_write.yml"],
        )
    assert report["ok"] is True
    assert report["from_live"] is True
    assert "secondary_backend" in report["equal"] or "service" in report["equal"]
    text = out.read_text(encoding="utf-8")
    assert "equal:" in text


def test_cli_alertmanager_from_live_parser():
    from rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["alertmanager", "--generate-inhibit", "--from-live", "--tune-equal"]
    )
    assert args.from_live is True
    assert args.tune_equal is True




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
