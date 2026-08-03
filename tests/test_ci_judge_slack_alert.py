"""Judge soft-fail multi-channel alert tests."""

import json

import scripts.ci_judge_slack_alert as alert_script
from rag.judge_alert import (
    detect_soft_fail,
    dispatch_judge_alerts,
    soft_fail_from_metrics,
    soft_fail_from_report,
)


def test_soft_fail_from_report_and_metrics(tmp_path):
    report = {"summary": {"ok": False, "accuracy": 0.5, "passed": 1, "failed": 1, "total": 2}}
    assert soft_fail_from_report(report, soft_fail_env=True) is True
    assert soft_fail_from_report(report, soft_fail_env=False) is False
    assert soft_fail_from_report({"summary": {"ok": True}}, soft_fail_env=True) is False

    metrics = tmp_path / "m.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "kind": "judge_run",
                "values": {"soft_fail": True, "ok": False, "accuracy": 0.4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert soft_fail_from_metrics(str(metrics)) is True
    assert soft_fail_from_metrics(str(tmp_path / "missing.jsonl")) is False


def test_dispatch_pagerduty_and_opsgenie(monkeypatch):
    calls = []

    class FakeResp:
        status_code = 202

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    report = {"summary": {"ok": False, "accuracy": 0.2, "failed": 3, "total": 5}}
    result = dispatch_judge_alerts(
        report,
        source="report",
        slack_webhook="",
        pagerduty_routing_key="pd-key",
        opsgenie_api_key="og-key",
    )
    assert result["alerted"] is True
    assert result["channels"]["pagerduty"] is True
    assert result["channels"]["opsgenie"] is True
    urls = [c["url"] for c in calls]
    assert any("pagerduty.com" in u for u in urls)
    assert any("opsgenie.com" in u for u in urls)


def test_ci_judge_slack_alert_posts(tmp_path, monkeypatch):
    report = tmp_path / "judge.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "ok": False,
                    "accuracy": 0.5,
                    "passed": 4,
                    "failed": 4,
                    "total": 8,
                    "mode": "llm",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_JUDGE_OUTPUT", str(report))
    monkeypatch.setenv("RAG_JUDGE_METRICS_PATH", str(tmp_path / "none.jsonl"))
    monkeypatch.setenv("RAG_JUDGE_SOFT_FAIL", "1")
    monkeypatch.setenv("RAG_JUDGE_SLACK_WEBHOOK", "https://hooks.slack.test/xxx")
    monkeypatch.delenv("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RAG_JUDGE_OPSGENIE_API_KEY", raising=False)
    calls = []

    def fake_post(url, json=None, headers=None, timeout=10):
        calls.append({"url": url, "json": json})

        class R:
            status_code = 200

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    assert alert_script.main() == 0
    assert calls and "soft-fail" in calls[0]["json"]["text"]


def test_ci_judge_slack_alert_noop_when_ok(tmp_path, monkeypatch):
    report = tmp_path / "judge.json"
    report.write_text(
        json.dumps({"summary": {"ok": True, "accuracy": 1.0}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_JUDGE_OUTPUT", str(report))
    monkeypatch.setenv("RAG_JUDGE_METRICS_PATH", str(tmp_path / "none.jsonl"))
    monkeypatch.setenv("RAG_JUDGE_SOFT_FAIL", "1")
    monkeypatch.setenv("RAG_JUDGE_SLACK_WEBHOOK", "https://hooks.slack.test/xxx")
    calls = []
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: calls.append(1),
    )
    assert alert_script.main() == 0
    assert not calls


def test_detect_soft_fail_helper(tmp_path):
    report = tmp_path / "j.json"
    report.write_text(json.dumps({"summary": {"ok": False}}), encoding="utf-8")
    fired, source, _ = detect_soft_fail(
        report_path=str(report),
        metrics_path=str(tmp_path / "m.jsonl"),
        soft_fail_env=True,
    )
    assert fired and source == "report"
