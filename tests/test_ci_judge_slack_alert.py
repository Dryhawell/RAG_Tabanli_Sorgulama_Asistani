"""Judge soft-fail multi-channel alert tests."""

import json

import scripts.ci_judge_slack_alert as alert_script
from rag.judge_alert import (
    detect_and_dispatch,
    detect_soft_fail,
    dispatch_judge_alerts,
    dispatch_judge_resolve,
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
    monkeypatch.setenv("RAG_JUDGE_ALERT_STATE", str(tmp_path / "state.json"))
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
    monkeypatch.setenv("RAG_JUDGE_ALERT_STATE", str(tmp_path / "state.json"))
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


def test_dispatch_judge_resolve_channels(monkeypatch):
    calls = []

    class FakeResp:
        status_code = 202

    def fake_post(url, json=None, headers=None, params=None, timeout=10):
        calls.append({"url": url, "json": json, "params": params})
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    report = {"summary": {"ok": True, "accuracy": 1.0}}
    result = dispatch_judge_resolve(
        report,
        source="report",
        slack_webhook="https://hooks.slack.test/resolve",
        pagerduty_routing_key="pd-key",
        opsgenie_api_key="og-key",
    )
    assert result["resolved"] is True
    assert result["channels"]["slack"] is True
    assert result["channels"]["pagerduty"] is True
    assert result["channels"]["opsgenie"] is True
    pd = next(c for c in calls if "pagerduty.com" in c["url"])
    assert pd["json"]["event_action"] == "resolve"
    assert any("/close" in c["url"] for c in calls)
    slack = next(c for c in calls if "hooks.slack.test" in c["url"])
    assert "resolved" in slack["json"]["text"].lower()


def test_detect_and_dispatch_auto_resolve(tmp_path, monkeypatch):
    state = str(tmp_path / "state.json")
    fail_report = tmp_path / "fail.json"
    ok_report = tmp_path / "ok.json"
    fail_report.write_text(
        json.dumps({"summary": {"ok": False, "accuracy": 0.4}}),
        encoding="utf-8",
    )
    ok_report.write_text(
        json.dumps({"summary": {"ok": True, "accuracy": 1.0}}),
        encoding="utf-8",
    )
    metrics = str(tmp_path / "none.jsonl")
    monkeypatch.setenv("RAG_JUDGE_SLACK_WEBHOOK", "https://hooks.slack.test/xxx")
    monkeypatch.delenv("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RAG_JUDGE_OPSGENIE_API_KEY", raising=False)
    calls = []

    def fake_post(url, json=None, headers=None, params=None, timeout=10):
        calls.append({"url": url, "json": json})

        class R:
            status_code = 200

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    alert = detect_and_dispatch(
        report_path=str(fail_report),
        metrics_path=metrics,
        soft_fail_env=True,
        state_path=state,
    )
    assert alert["action"] == "alert"
    assert alert["soft_fail"] is True
    with open(state, encoding="utf-8") as f:
        assert json.load(f)["soft_fail"] is True

    calls.clear()
    resolved = detect_and_dispatch(
        report_path=str(ok_report),
        metrics_path=metrics,
        soft_fail_env=True,
        state_path=state,
    )
    assert resolved["action"] == "resolve"
    assert resolved["soft_fail"] is False
    assert calls and "resolved" in calls[0]["json"]["text"].lower()
    with open(state, encoding="utf-8") as f:
        st = json.load(f)
    assert st["soft_fail"] is False
    assert st["last_action"] == "resolve"


def test_acknowledge_holds_realert(tmp_path, monkeypatch):
    from rag.judge_alert import acknowledge_judge_alert, detect_and_dispatch

    state = str(tmp_path / "state.json")
    fail_report = tmp_path / "fail.json"
    fail_report.write_text(
        json.dumps({"summary": {"ok": False, "accuracy": 0.3}}),
        encoding="utf-8",
    )
    metrics = str(tmp_path / "none.jsonl")
    monkeypatch.setenv("RAG_JUDGE_SLACK_WEBHOOK", "https://hooks.slack.test/xxx")
    monkeypatch.delenv("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.delenv("RAG_JUDGE_OPSGENIE_API_KEY", raising=False)
    calls = []

    def fake_post(url, json=None, headers=None, params=None, timeout=10):
        calls.append({"url": url, "json": json})

        class R:
            status_code = 200

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    first = detect_and_dispatch(
        report_path=str(fail_report),
        metrics_path=metrics,
        soft_fail_env=True,
        state_path=state,
    )
    assert first["action"] == "alert"
    calls.clear()
    ack = acknowledge_judge_alert(
        actor="oncall",
        note="investigating",
        state_path=state,
        notify=True,
    )
    assert ack["ok"] is True
    assert ack["state"]["acknowledged"] is True
    assert calls and "ACK" in calls[0]["json"]["text"]
    calls.clear()
    held = detect_and_dispatch(
        report_path=str(fail_report),
        metrics_path=metrics,
        soft_fail_env=True,
        state_path=state,
    )
    assert held["action"] == "ack_hold"
    assert held["alerted"] is False
    assert not calls

