from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rag.judge_alert import (
    append_judge_ack_audit,
    dispatch_judge_ack_digest,
    dispatch_judge_ack_digest_fanout,
    parse_judge_ack_digest_webhooks,
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


def test_summarize_filters_by_tenant(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="a", path=path, extra={"tenant_id": "acme"})
    append_judge_ack_audit("ack", actor="b", path=path, extra={"tenant_id": "beta"})
    append_judge_ack_audit("unack", actor="a", path=path, extra={"tenant_id": "acme"})

    all_sum = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    assert all_sum["by_tenant"] == {"acme": 2, "beta": 1}

    acme = summarize_judge_ack_audit(path=path, since_hours=24 * 365, tenant_id="acme")
    assert acme["total"] == 2
    assert acme["tenant_id"] == "acme"
    assert acme["by_event"] == {"ack": 1, "unack": 1}


def test_dispatch_judge_ack_digest_dry_run(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    result = dispatch_judge_ack_digest(summary, dry_run=True)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert "Total events" in result["payload"]["text"]
    assert result["block_kit"] is True
    assert isinstance(result["payload"].get("blocks"), list)
    assert result["payload"]["blocks"][0]["type"] == "header"


def test_dispatch_no_block_kit(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    result = dispatch_judge_ack_digest(summary, dry_run=True, block_kit=False)
    assert "blocks" not in result["payload"]
    assert result["block_kit"] is False


def test_tenant_quiet_hours_override_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_JUDGE_ACK_DIGEST_WEBHOOKS_JSON", raising=False)
    monkeypatch.delenv("RAG_JUDGE_ACK_DIGEST_QUIET_HOURS_JSON", raising=False)
    (tmp_path / "judge_ack_digest_webhooks.json").write_text(
        '{"acme": "https://hooks.slack.test/acme", "beta": "https://hooks.slack.test/beta"}',
        encoding="utf-8",
    )
    (tmp_path / "judge_ack_digest_quiet_hours.json").write_text(
        '{"acme": "22:00-07:00", "beta": "off"}',
        encoding="utf-8",
    )
    from rag.judge_alert import (
        parse_judge_ack_digest_quiet_hours,
        resolve_judge_ack_digest_quiet_hours,
    )

    qmap = parse_judge_ack_digest_quiet_hours(base=str(tmp_path))
    assert qmap["acme"] == "22:00-07:00"
    assert resolve_judge_ack_digest_quiet_hours("acme", base=str(tmp_path)) == "22:00-07:00"
    assert resolve_judge_ack_digest_quiet_hours("beta", base=str(tmp_path)) == "off"

    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", path=path, extra={"tenant_id": "acme"})

    def fake_quiet(*, quiet_spec=None, tenant_id=None, **kwargs):
        return quiet_spec == "22:00-07:00"

    with patch("rag.collab_notify_digest.is_quiet_hours", side_effect=fake_quiet):
        report = dispatch_judge_ack_digest_fanout(
            path=path,
            since_hours=24 * 365,
            dry_run=True,
            base=str(tmp_path),
        )
    by_tid = {r["tenant_id"]: r for r in report["results"]}
    assert by_tid["acme"]["reason"] == "quiet_hours"
    assert by_tid["beta"].get("reason") != "quiet_hours"
    assert by_tid["beta"]["dry_run"] is True
    assert report["quiet_overrides"] == 2


def test_dispatch_skips_quiet_hours(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    with patch("rag.collab_notify_digest.is_quiet_hours", return_value=True):
        result = dispatch_judge_ack_digest(
            summary,
            webhook="https://hooks.slack.test/T/B/x",
            quiet_hours="22:00-07:00",
        )
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "quiet_hours"


def test_dispatch_force_ignores_quiet_hours(tmp_path: Path) -> None:
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    with patch("rag.collab_notify_digest.is_quiet_hours", return_value=True):
        result = dispatch_judge_ack_digest(
            summary,
            dry_run=True,
            webhook="https://hooks.slack.test/T/B/x",
            quiet_hours="22:00-07:00",
            ignore_quiet_hours=True,
        )
    assert result.get("reason") != "quiet_hours"
    assert result["dry_run"] is True
    assert result["configured"] is True


def test_parse_and_fanout_webhooks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_JUDGE_ACK_DIGEST_WEBHOOKS_JSON", raising=False)
    mapping_path = tmp_path / "judge_ack_digest_webhooks.json"
    mapping_path.write_text(
        '{"acme": "https://hooks.slack.test/acme", "beta": "https://hooks.slack.test/beta"}',
        encoding="utf-8",
    )
    mapping = parse_judge_ack_digest_webhooks(base=str(tmp_path))
    assert set(mapping) == {"acme", "beta"}

    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", path=path, extra={"tenant_id": "acme"})
    report = dispatch_judge_ack_digest_fanout(
        path=path,
        since_hours=24 * 365,
        dry_run=True,
        ignore_quiet_hours=True,
        base=str(tmp_path),
    )
    assert report["ok"] is True
    assert report["fanout"] is True
    assert report["tenant_count"] == 2
    tenants = {r["tenant_id"] for r in report["results"]}
    assert tenants == {"acme", "beta"}


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
