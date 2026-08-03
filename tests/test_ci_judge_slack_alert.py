"""Judge soft-fail Slack alert script tests."""

import json

import scripts.ci_judge_slack_alert as alert


def test_soft_fail_from_report_and_metrics(tmp_path):
    report = {"summary": {"ok": False, "accuracy": 0.5, "passed": 1, "failed": 1, "total": 2}}
    assert alert.soft_fail_from_report(report, soft_fail_env=True) is True
    assert alert.soft_fail_from_report(report, soft_fail_env=False) is False
    assert alert.soft_fail_from_report({"summary": {"ok": True}}, soft_fail_env=True) is False

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
    assert alert.soft_fail_from_metrics(str(metrics)) is True
    assert alert.soft_fail_from_metrics(str(tmp_path / "missing.jsonl")) is False


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
    calls = []

    def fake_post(url, json=None, timeout=10):
        calls.append({"url": url, "json": json})

        class R:
            status_code = 200

        return R()

    monkeypatch.setattr("requests.post", fake_post)
    assert alert.main() == 0
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
    assert alert.main() == 0
    assert not calls
