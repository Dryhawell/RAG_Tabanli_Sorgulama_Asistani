from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rag.alertmanager_ops import (
    apply_inhibit_equal_with_gate,
    canonicalize_inhibit_yaml,
    diff_inhibit_rules,
    render_inhibit_rules_yaml,
    write_inhibit_rules,
)
from scripts.ci_inhibit_equal_apply import resolve_dry_run


def test_diff_inhibit_unchanged(tmp_path: Path) -> None:
    text = render_inhibit_rules_yaml(
        [
            {
                "source_matchers": ["severity = critical"],
                "target_matchers": ["severity = warning"],
                "equal": ["alertname", "service"],
            }
        ]
    )
    cur = tmp_path / "cur.yml"
    cur.write_text(text, encoding="utf-8")
    diff = diff_inhibit_rules(str(cur), text)
    assert diff["ok"] is True
    assert diff["changed"] is False


def test_diff_inhibit_changed(tmp_path: Path) -> None:
    old = render_inhibit_rules_yaml(
        [
            {
                "source_matchers": ["severity = critical"],
                "target_matchers": ["severity = warning"],
                "equal": ["alertname", "service"],
            }
        ]
    )
    new = render_inhibit_rules_yaml(
        [
            {
                "source_matchers": ["severity = critical"],
                "target_matchers": ["severity = warning"],
                "equal": ["alertname", "service", "team"],
            }
        ]
    )
    cur = tmp_path / "cur.yml"
    cur.write_text(old, encoding="utf-8")
    diff = diff_inhibit_rules(str(cur), new)
    assert diff["changed"] is True
    assert "team" in diff["unified_diff"]


def test_canonicalize_strips_comments() -> None:
    raw = "# greeting-line\ninhibit_rules:\n  - equal: [\"a\"]\n\n"
    out = canonicalize_inhibit_yaml(raw)
    assert "greeting-line" not in out
    assert "inhibit_rules:" in out
    assert out.startswith("inhibit_rules:")


def test_apply_inhibit_equal_dry_run_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "inhibit.yml"
    # seed different content so changed=True
    out.write_text("inhibit_rules: []\n", encoding="utf-8")
    with patch(
        "rag.alertmanager_ops.check_alertmanager_config",
        return_value={"ok": True, "method": "structural"},
    ):
        with patch(
            "rag.alertmanager_ops.render_alertmanager_config",
            return_value={"ok": True, "output": str(tmp_path / "am.yml")},
        ):
            report = apply_inhibit_equal_with_gate(
                output=str(out),
                from_live=False,
                dry_run=True,
            )
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["applied"] is False
    # file unchanged because dry_run
    assert out.read_text(encoding="utf-8") == "inhibit_rules: []\n"


def test_apply_inhibit_equal_writes_when_gate_ok(tmp_path: Path) -> None:
    out = tmp_path / "inhibit.yml"
    out.write_text("inhibit_rules: []\n", encoding="utf-8")
    with patch(
        "rag.alertmanager_ops.check_alertmanager_config",
        return_value={"ok": True, "method": "amtool"},
    ):
        with patch(
            "rag.alertmanager_ops.render_alertmanager_config",
            return_value={"ok": True, "output": str(tmp_path / "am.yml")},
        ):
            report = apply_inhibit_equal_with_gate(
                output=str(out),
                from_live=False,
                dry_run=False,
            )
    assert report["ok"] is True
    assert report["applied"] is True
    assert report.get("rolled_back") is False
    assert "source_matchers:" in out.read_text(encoding="utf-8")
    assert Path(report["backup"]).is_file()


def test_apply_inhibit_equal_rollback_on_amtool_regression(tmp_path: Path) -> None:
    out = tmp_path / "inhibit.yml"
    previous = "inhibit_rules: []\n"
    out.write_text(previous, encoding="utf-8")
    checks = iter(
        [
            {"ok": True, "method": "amtool"},
            {"ok": False, "method": "amtool", "error": "bad config"},
        ]
    )
    with patch(
        "rag.alertmanager_ops.check_alertmanager_config",
        side_effect=lambda *a, **k: next(checks),
    ):
        with patch(
            "rag.alertmanager_ops.render_alertmanager_config",
            return_value={"ok": True, "output": str(tmp_path / "am.yml")},
        ):
            report = apply_inhibit_equal_with_gate(
                output=str(out),
                from_live=False,
                dry_run=False,
                rollback_on_regression=True,
                backup_path=str(tmp_path / "bak.yml"),
            )
    assert report["ok"] is False
    assert report["applied"] is False
    assert report["rolled_back"] is True
    assert report["reason"] == "amtool_regression"
    assert out.read_text(encoding="utf-8") == previous
    assert (tmp_path / "bak.yml").read_text(encoding="utf-8") == previous


def test_resolve_dry_run_auto(monkeypatch) -> None:
    monkeypatch.setenv("INHIBIT_EQUAL_APPLY", "auto")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    assert resolve_dry_run() is True
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    assert resolve_dry_run() is False
    monkeypatch.setenv("INHIBIT_EQUAL_APPLY", "dry-run")
    assert resolve_dry_run() is True


def test_ci_workflow_has_apply_job() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "inhibit-equal-apply" in text
    assert "ci_inhibit_equal_apply.py" in text
    assert "INHIBIT_EQUAL_APPLY" in text
    assert "INHIBIT_EQUAL_APPLY_PR_COMMENT_POST" in text
    assert "needs: [test]" in text
    assert "needs.test.result" in text
    assert "INHIBIT_EQUAL_APPLY_REQUIRE_GREEN" in text
    assert "INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK" in text
    assert "INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY" in text
    assert "inhibit-equal-apply-bot" in Path("scripts/ci_inhibit_equal_apply.py").read_text(
        encoding="utf-8"
    )
    assert "dual-write-dlq" in Path("rag/cli.py").read_text(encoding="utf-8")
    assert "--diff-inhibit" in Path("rag/cli.py").read_text(encoding="utf-8") or (
        "--apply-equal" in Path("rag/cli.py").read_text(encoding="utf-8")
    )


def test_resolve_apply_allowed(monkeypatch) -> None:
    from scripts.ci_inhibit_equal_apply import resolve_apply_allowed

    monkeypatch.delenv("INHIBIT_EQUAL_APPLY_REQUIRE_GREEN", raising=False)
    assert resolve_apply_allowed()["ok"] is True
    monkeypatch.setenv("INHIBIT_EQUAL_APPLY_REQUIRE_GREEN", "1")
    monkeypatch.setenv("CI_TEST_RESULT", "success")
    assert resolve_apply_allowed()["ok"] is True
    monkeypatch.setenv("CI_TEST_RESULT", "failure")
    denied = resolve_apply_allowed()
    assert denied["ok"] is False
    assert denied["reason"] == "ci_not_green"


def test_build_inhibit_equal_apply_comment() -> None:
    from scripts.ci_inhibit_equal_apply import (
        INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER,
        build_inhibit_equal_apply_comment,
        write_apply_preview_artifacts,
    )

    report = {
        "ok": True,
        "dry_run": True,
        "reason": "dry_run_changed",
        "output": "grafana/inhibit_rules.generated.yml",
        "generated": {"equal": ["alertname", "service"]},
        "diff": {
            "changed": True,
            "unified_diff": "--- a\n+++ b\n+equal: [team]\n",
        },
    }
    md = build_inhibit_equal_apply_comment(report)
    assert INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER in md
    assert "```diff" in md
    assert "equal: [team]" in md
    assert "dry-run preview" in md.lower()

    unchanged = build_inhibit_equal_apply_comment(
        {
            "ok": True,
            "dry_run": True,
            "reason": "unchanged",
            "generated": {"equal": ["alertname"]},
            "diff": {"changed": False, "unified_diff": ""},
        }
    )
    assert "No inhibit diff" in unchanged


def test_resolve_inhibit_equal_canary_pd_severity(monkeypatch) -> None:
    import json

    from scripts.ci_inhibit_equal_apply import resolve_inhibit_equal_canary_pd_severity

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY", raising=False)
    monkeypatch.delenv("RAG_INHIBIT_EQUAL_CANARY_PD_SEVERITY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON", raising=False)
    monkeypatch.delenv("RAG_INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON", raising=False)
    monkeypatch.delenv("CI_TEST_RESULT", raising=False)

    assert (
        resolve_inhibit_equal_canary_pd_severity({"reason": "amtool_regression"})
        == "critical"
    )
    assert resolve_inhibit_equal_canary_pd_severity({"reason": "unknown"}) == "error"
    monkeypatch.setenv("CI_TEST_RESULT", "failure")
    assert resolve_inhibit_equal_canary_pd_severity({"reason": "unknown"}) == "critical"
    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON",
        json.dumps({"backup_missing": "warning"}),
    )
    assert (
        resolve_inhibit_equal_canary_pd_severity({"reason": "backup_missing"})
        == "warning"
    )
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY", "info")
    assert (
        resolve_inhibit_equal_canary_pd_severity({"reason": "amtool_regression"})
        == "info"
    )


def test_notify_inhibit_equal_rollback_canary(monkeypatch) -> None:
    from scripts.ci_inhibit_equal_apply import notify_inhibit_equal_rollback_canary

    report = {
        "rolled_back": True,
        "reason": "amtool_regression",
        "backup": "metadata/bak.yml",
        "generated": {"equal": ["alertname"]},
        "diff": {"unified_diff": "+x\n"},
        "post_check": {"ok": False, "method": "amtool"},
    }
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS", raising=False)
    monkeypatch.delenv("RAG_INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("RAG_INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("RAG_JUDGE_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_CHANNEL", raising=False)
    monkeypatch.delenv("CI_TEST_RESULT", raising=False)
    skipped = notify_inhibit_equal_rollback_canary(report)
    assert skipped["skipped"] is True

    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", "https://hooks.slack.test/x")
    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL", "https://ops.example.test/ops/silence-burn"
    )
    with patch("rag.judge_alert.post_slack", return_value=True) as post:
        out = notify_inhibit_equal_rollback_canary(report)
    assert out["ok"] is True
    assert out["posted"] is True
    assert out["slack"] is True
    assert post.called
    payload = post.call_args[0][1]
    assert "rolled back" in payload["text"].lower() or "rollback" in payload["text"].lower()
    assert payload.get("runbook_url")
    assert "/ops/silence-burn" in str(payload.get("runbook_url"))
    assert any(
        (b.get("type") == "actions")
        for b in (payload.get("blocks") or [])
    )

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", raising=False)
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", "pd-key")
    with patch("rag.judge_alert.post_pagerduty", return_value=True) as pd:
        with patch("rag.judge_alert.post_slack", return_value=False):
            pd_out = notify_inhibit_equal_rollback_canary(report)
    assert pd_out["pagerduty"] is True
    assert pd_out["posted"] is True
    assert pd_out.get("pagerduty_severity") == "critical"
    assert pd.called
    assert pd.call_args.kwargs.get("source") == "inhibit-equal-canary"
    assert pd.call_args.kwargs.get("routing_key") == "pd-key"
    assert pd.call_args.kwargs.get("severity") == "critical"
    assert pd.call_args.kwargs.get("runbook_url")
    assert "silence-burn" in str(pd.call_args.kwargs.get("runbook_url"))
    assert (pd.call_args.kwargs.get("report") or {}).get("runbook_url")

    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY", "warning")
    with patch("rag.judge_alert.post_pagerduty", return_value=True) as pd_warn:
        with patch("rag.judge_alert.post_slack", return_value=False):
            tuned = notify_inhibit_equal_rollback_canary(report)
    assert tuned.get("pagerduty_severity") == "warning"
    assert pd_warn.call_args.kwargs.get("severity") == "warning"
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PD_SEVERITY", raising=False)

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", "og-key")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS", "us")
    with patch("rag.judge_alert.post_opsgenie", return_value=True) as og:
        with patch("rag.judge_alert.post_slack", return_value=False):
            with patch("rag.judge_alert.post_pagerduty", return_value=False):
                og_out = notify_inhibit_equal_rollback_canary(report)
    assert og_out["opsgenie"] is True
    assert og_out["posted"] is True
    assert og.called
    assert og.call_args.kwargs.get("source") == "inhibit-equal-canary"
    assert og.call_args.kwargs.get("api_key") == "og-key"
    assert og.call_args.kwargs.get("region") == "us"
    assert og.call_args.kwargs.get("runbook_url")
    assert "silence-burn" in str(og.call_args.kwargs.get("runbook_url"))

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", raising=False)
    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON",
        '{"us":"k-us","eu":"k-eu"}',
    )
    with patch("rag.judge_alert.post_opsgenie", return_value=True) as og_multi:
        with patch("rag.judge_alert.post_slack", return_value=False):
            with patch("rag.judge_alert.post_pagerduty", return_value=False):
                multi = notify_inhibit_equal_rollback_canary(report)
    assert multi["opsgenie"] is True
    assert multi["opsgenie_regions"]["us"] is True
    assert multi["opsgenie_regions"]["eu"] is True
    assert og_multi.call_count == 2
    regions = {c.kwargs.get("region") for c in og_multi.call_args_list}
    assert regions == {"us", "eu"}


def test_opsgenie_api_base_regions() -> None:
    from rag.judge_alert import opsgenie_api_base

    assert opsgenie_api_base(region="us").endswith("api.opsgenie.com")
    assert "eu.api.opsgenie.com" in opsgenie_api_base(region="eu")
    assert opsgenie_api_base(base_url="https://example.test/og") == "https://example.test/og"


def test_post_opsgenie_silence_burn_runbook_tags(monkeypatch) -> None:
    from rag.judge_alert import post_opsgenie

    calls: list = []

    class FakeResp:
        status_code = 202

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_OPSGENIE_TAGS", "canary,silence")
    ok = post_opsgenie(
        report={"summary": {"ok": False, "accuracy": 0.1, "failed": 1, "total": 2}},
        source="inhibit-equal-canary",
        api_key="og-key",
        region="eu",
        runbook_url="https://ops.example.test/ops/silence-burn#silence-burn-rate--multi-region",
    )
    assert ok is True
    assert calls
    body = calls[0]["json"]
    tags = [str(t) for t in (body.get("tags") or [])]
    assert "silence-burn" in tags
    assert "runbook" in tags
    assert "runbook:silence-burn" in tags
    assert "region:eu" in tags
    assert "canary" in tags
    assert "silence" in tags
    assert body.get("details", {}).get("runbook_url")
    assert "silence-burn" in str(body["details"]["runbook_url"])
    assert body.get("details", {}).get("opsgenie_url")
    assert "opsgenie.com/alert/list" in str(body["details"]["opsgenie_url"])
    assert "inhibit-equal-canary" in str(body["details"]["opsgenie_url"])
    # EU region → eu.app host
    assert "eu.app.opsgenie.com" in str(body["details"]["opsgenie_url"])


def test_opsgenie_alert_deep_link_regions(monkeypatch) -> None:
    from rag.judge_alert import opsgenie_alert_deep_link

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_ALERT_URL", raising=False)
    monkeypatch.delenv("RAG_OPSGENIE_ALERT_URL", raising=False)
    us = opsgenie_alert_deep_link(source="inhibit-equal-canary", region="us")
    assert us.startswith("https://app.opsgenie.com/alert/list?")
    eu = opsgenie_alert_deep_link(source="inhibit-equal-canary", region="eu")
    assert eu.startswith("https://eu.app.opsgenie.com/alert/list?")
    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_OPSGENIE_ALERT_URL",
        "https://ops.example/alert/custom",
    )
    assert opsgenie_alert_deep_link() == "https://ops.example/alert/custom"


def test_notify_inhibit_equal_close_on_green(monkeypatch) -> None:
    from scripts.ci_inhibit_equal_apply import (
        build_inhibit_equal_resolve_canary_payload,
        notify_inhibit_equal_close_on_green,
    )

    report = {
        "applied": True,
        "rolled_back": False,
        "generated": {"equal": ["alertname"]},
        "git": {"ok": True},
    }
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("RAG_JUDGE_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", raising=False)
    skipped = notify_inhibit_equal_close_on_green(report)
    assert skipped["skipped"] is True

    payload = build_inhibit_equal_resolve_canary_payload(report)
    assert "resolved" in payload["text"].lower()
    assert "green" in payload["text"].lower()
    assert payload.get("opsgenie_url")
    assert "opsgenie.com/alert/list" in str(payload.get("opsgenie_url"))
    assert "Opsgenie:" in payload["text"] or "opsgenie" in payload["text"].lower()

    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", "https://hooks.slack.test/r")
    with patch("rag.judge_alert.post_slack", return_value=True) as slack:
        slack_out = notify_inhibit_equal_close_on_green(report)
    assert slack_out["closed"] is True
    assert slack_out["slack"] is True
    assert slack_out.get("opsgenie_url")
    assert slack.called
    assert "resolved" in slack.call_args[0][1]["text"].lower()
    assert "opsgenie.com" in slack.call_args[0][1]["text"].lower()

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", raising=False)
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", "og-key")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS", "us,eu")
    with patch("rag.judge_alert.post_opsgenie_close", return_value=True) as close:
        out = notify_inhibit_equal_close_on_green(report)
    assert out["closed"] is True
    assert out["opsgenie"] is True
    assert out.get("opsgenie_deep_link")
    assert out["opsgenie_regions"]["us"] is True
    assert out["opsgenie_regions"]["eu"] is True
    assert close.call_count == 2
    assert all(
        c.kwargs.get("source") == "inhibit-equal-canary" for c in close.call_args_list
    )
    assert all(
        "opsgenie.com" in str(c.kwargs.get("note") or "").lower()
        for c in close.call_args_list
    )

    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", raising=False)
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", "pd-key")
    with patch("rag.judge_alert.post_pagerduty", return_value=True) as pd:
        with patch("rag.judge_alert.post_opsgenie_close", return_value=False):
            pd_out = notify_inhibit_equal_close_on_green(report)
    assert pd_out["pagerduty"] is True
    assert pd.call_args.kwargs.get("event_action") == "resolve"
    assert pd.call_args.kwargs.get("source") == "inhibit-equal-canary"


def test_inhibit_equal_canary_resolve_prometheus_metric(monkeypatch) -> None:
    from scripts.ci_inhibit_equal_apply import notify_inhibit_equal_close_on_green

    seen: list[dict] = []

    def _fake_record(kind, values=None, **kwargs):
        seen.append({"kind": kind, "values": values or {}})

    monkeypatch.setattr("rag.metrics.record_metric", _fake_record)
    # Patch where emit imports from
    monkeypatch.setattr(
        "scripts.ci_inhibit_equal_apply.emit_inhibit_equal_canary_resolve_metric",
        lambda **kw: seen.append({"kind": "inhibit_equal_canary_resolve", "values": kw}),
    )

    skipped = notify_inhibit_equal_close_on_green(
        {"applied": False, "rolled_back": False}
    )
    assert skipped["skipped"] is True
    assert any(
        s["values"].get("result") == "skipped" for s in seen if "result" in s["values"]
    )

    seen.clear()
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", raising=False)
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", "xoxb")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_CHANNEL", "C1")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", "111.1")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_THREAD_REPLY", "1")
    with patch(
        "rag.judge_alert.post_slack_thread_message",
        return_value={"ok": True, "ts": "111.2"},
    ):
        closed = notify_inhibit_equal_close_on_green(
            {
                "applied": True,
                "rolled_back": False,
                "generated": {"equal": ["alertname"]},
                "git": {"ok": True},
            }
        )
    assert closed["slack_thread_reply"] is True
    assert any(
        s["values"].get("result") == "ok" and s["values"].get("via") == "thread"
        for s in seen
    )


def test_inhibit_equal_canary_auto_silence(monkeypatch, tmp_path: Path) -> None:
    from scripts.ci_inhibit_equal_apply import (
        auto_silence_inhibit_equal_canary_resolve,
        inhibit_equal_canary_auto_silence_enabled,
        load_inhibit_equal_canary_silence_state,
    )

    seen: list[dict] = []
    monkeypatch.setattr(
        "scripts.ci_inhibit_equal_apply.emit_inhibit_equal_canary_silence_metric",
        lambda **kw: seen.append(kw),
    )
    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_SILENCE_STATE", str(tmp_path / "silence.json")
    )

    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_AUTO_SILENCE", "0")
    assert inhibit_equal_canary_auto_silence_enabled() is False
    disabled = auto_silence_inhibit_equal_canary_resolve(
        {"applied": True, "rolled_back": False}
    )
    assert disabled["skipped"] is True
    assert disabled["reason"] == "disabled"

    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_AUTO_SILENCE", "1")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_AUTO_SILENCE_DURATION", "30m")
    with patch(
        "rag.alertmanager_ops.create_silence",
        return_value={
            "ok": True,
            "silenceID": "sil-1",
            "request": {
                "startsAt": "2026-01-01T00:00:00Z",
                "endsAt": "2026-01-01T00:30:00Z",
            },
        },
    ) as create:
        out = auto_silence_inhibit_equal_canary_resolve(
            {"applied": True, "rolled_back": False}
        )
    assert out["ok"] is True
    assert out["silenceID"] == "sil-1"
    assert out["duration_sec"] == 1800.0
    assert create.called
    matchers = create.call_args.kwargs.get("matchers") or []
    names = {m.get("name") for m in matchers}
    assert "alertname" in names
    assert "service" in names
    assert any(s.get("result") == "ok" for s in seen)
    state = load_inhibit_equal_canary_silence_state()
    assert state.get("silenceID") == "sil-1"
    assert state.get("endsAt") == "2026-01-01T00:30:00Z"

    skipped = auto_silence_inhibit_equal_canary_resolve(
        {"applied": False, "rolled_back": False}
    )
    assert skipped["reason"] == "not_applied_green"


def test_inhibit_equal_canary_silence_expiry_webhook(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts.ci_inhibit_equal_apply import (
        check_inhibit_equal_canary_silence_expiry,
        save_inhibit_equal_canary_silence_state,
    )

    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_SILENCE_STATE", str(tmp_path / "silence.json")
    )
    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_SILENCE_EXPIRY_WEBHOOK", "https://hooks.example/x"
    )
    seen: list[dict] = []
    monkeypatch.setattr(
        "scripts.ci_inhibit_equal_apply.emit_inhibit_equal_canary_silence_metric",
        lambda **kw: seen.append(kw),
    )

    empty = check_inhibit_equal_canary_silence_expiry()
    assert empty["skipped"] is True
    assert empty["reason"] == "no_silence_state"

    save_inhibit_equal_canary_silence_state(
        silence_id="sil-exp",
        ends_at="2020-01-01T00:00:00Z",
        starts_at="2019-12-31T22:00:00Z",
        matchers=[{"name": "alertname", "value": "X", "isRegex": False, "isEqual": True}],
        expiry_notified_at="",
    )
    with patch(
        "rag.alertmanager_ops.list_silences",
        return_value={"ok": True, "silences": []},
    ):
        with patch("rag.judge_alert.post_slack", return_value=True) as post:
            out = check_inhibit_equal_canary_silence_expiry()
    assert out["expired"] is True
    assert out["notified"] is True
    assert out["metric_result"] == "webhook_ok"
    assert post.called
    assert any(s.get("result") == "webhook_ok" for s in seen)

    # Second call should skip (already notified)
    seen.clear()
    again = check_inhibit_equal_canary_silence_expiry()
    assert again["skipped"] is True
    assert again["reason"] == "already_notified"


def test_inhibit_equal_canary_slack_state_artifact_ensured(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.ci_inhibit_equal_apply import (
        inhibit_equal_canary_slack_state_path,
        load_inhibit_equal_canary_slack_state,
        save_inhibit_equal_canary_slack_state,
    )

    monkeypatch.setenv(
        "INHIBIT_EQUAL_CANARY_SLACK_STATE", str(tmp_path / "inhibit_equal_canary_slack.json")
    )
    path = Path(inhibit_equal_canary_slack_state_path())
    assert not path.is_file()
    # Simulate main() ensure-write
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    assert path.is_file()
    assert load_inhibit_equal_canary_slack_state() == {}
    save_inhibit_equal_canary_slack_state(
        thread_ts="1.2", channel="C9", reason="amtool_regression"
    )
    loaded = load_inhibit_equal_canary_slack_state()
    assert loaded.get("thread_ts") == "1.2"
    assert loaded.get("channel") == "C9"
    assert "updated_at" in loaded


def test_inhibit_equal_canary_slack_thread_reply_on_resolve(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.ci_inhibit_equal_apply import (
        notify_inhibit_equal_close_on_green,
        notify_inhibit_equal_rollback_canary,
        save_inhibit_equal_canary_slack_state,
    )

    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_STATE", str(tmp_path / "canary_slack.json"))
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", "xoxb-canary")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_CHANNEL", "C-canary")
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_THREAD_REPLY", "1")
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", raising=False)
    monkeypatch.delenv("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", raising=False)

    rollback = {
        "rolled_back": True,
        "reason": "amtool_regression",
        "backup": "metadata/bak.yml",
        "generated": {"equal": ["alertname"]},
        "diff": {"unified_diff": "+x\n"},
        "post_check": {"ok": False, "method": "amtool"},
    }
    with patch(
        "rag.judge_alert.post_slack_thread_message",
        return_value={"ok": True, "ts": "111.1", "channel": "C-canary"},
    ) as bot_post:
        rolled = notify_inhibit_equal_rollback_canary(rollback)
    assert rolled["posted"] is True
    assert rolled["slack"] is True
    assert rolled["slack_via"] == "bot"
    assert rolled["slack_thread_ts"] == "111.1"
    assert bot_post.called
    assert bot_post.call_args.kwargs.get("thread_ts") is None

    green = {
        "applied": True,
        "rolled_back": False,
        "generated": {"equal": ["alertname"]},
        "git": {"ok": True},
    }
    with patch(
        "rag.judge_alert.post_slack_thread_message",
        return_value={"ok": True, "ts": "111.2", "thread_ts": "111.1"},
    ) as reply:
        with patch("rag.judge_alert.post_slack", return_value=False) as webhook:
            closed = notify_inhibit_equal_close_on_green(green)
    assert closed["closed"] is True
    assert closed["slack_thread_reply"] is True
    assert closed["slack_via"] == "thread"
    assert closed["slack_thread_ts"] == "111.1"
    assert reply.called
    assert reply.call_args.kwargs.get("thread_ts") == "111.1"
    assert webhook.called is False

    # Explicit env thread_ts wins over state file
    save_inhibit_equal_canary_slack_state(
        thread_ts="222.2", channel="C-canary", base=str(tmp_path)
    )
    monkeypatch.setenv("INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", "333.3")
    with patch(
        "rag.judge_alert.post_slack_thread_message",
        return_value={"ok": True, "ts": "333.4"},
    ) as reply2:
        again = notify_inhibit_equal_close_on_green(green)
    assert again["slack_thread_ts"] == "333.3"
    assert reply2.call_args.kwargs.get("thread_ts") == "333.3"
