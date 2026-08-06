from __future__ import annotations

from pathlib import Path

from rag.judge_alert import (
    append_judge_ack_audit,
    dispatch_judge_ack_digest,
    summarize_judge_ack_audit,
)
from rag.prometheus_sink import observe_metric, prometheus_available, render_prometheus


def test_summarize_judge_ack_audit_by_event(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    append_judge_ack_audit("unack", actor="bob", path=path)
    append_judge_ack_audit("ack", actor="alice", path=path)

    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    assert summary["ok"] is True
    assert summary["total"] == 3
    assert summary["by_event"] == {"ack": 2, "unack": 1}
    assert summary["actor_count"] == 2
    assert set(summary["actors"]) == {"alice", "bob"}
    assert "ack" in summary["last_by_event"]


def test_dispatch_judge_ack_digest_dry_run(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    result = dispatch_judge_ack_digest(summary, dry_run=True)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert "Total events" in result["payload"]["text"]


def test_dispatch_judge_ack_digest_skips_without_webhook(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_JUDGE_SLACK_WEBHOOK", raising=False)
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    result = dispatch_judge_ack_digest(summary, dry_run=False)
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["configured"] is False


def test_judge_ack_audit_prometheus_metric(monkeypatch) -> None:
    if not prometheus_available():
        return
    monkeypatch.setattr("rag.prometheus_sink.ENABLE_PROMETHEUS", True)
    observe_metric(
        "judge_ack_audit",
        values={"event": "ack", "source": "slack"},
        enabled=True,
    )
    body = render_prometheus().decode("utf-8")
    assert "rag_judge_ack_audit_total" in body
    assert 'event="ack"' in body


def test_append_forwards_ack_metric(tmp_path: Path, monkeypatch) -> None:
    if not prometheus_available():
        return
    monkeypatch.setattr("rag.prometheus_sink.ENABLE_PROMETHEUS", True)
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("resolve", actor="ops", path=path)
    body = render_prometheus().decode("utf-8")
    assert "rag_judge_ack_audit_total" in body
    assert 'event="resolve"' in body
