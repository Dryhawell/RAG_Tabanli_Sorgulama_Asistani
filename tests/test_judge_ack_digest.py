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


def test_dispatch_judge_ack_digest_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_JUDGE_SLACK_INTERACTIVE", "1")
    monkeypatch.setenv("RAG_JUDGE_ACK_PUBLIC_URL", "http://example.test/judge/ack-form")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_DIFF", "1")
    monkeypatch.setenv(
        "RAG_JUDGE_ACK_DIGEST_SNAPSHOT", str(tmp_path / "digest_last.json")
    )
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    result = dispatch_judge_ack_digest(summary, dry_run=True, base=str(tmp_path))
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert "Total events" in result["payload"]["text"]
    assert "vs last digest" in result["payload"]["text"]
    assert result["diff"] is not None
    assert result["block_kit"] is True
    assert isinstance(result["payload"].get("blocks"), list)
    assert result["payload"]["blocks"][0]["type"] == "header"
    actions = [
        b for b in result["payload"]["blocks"] if b.get("type") == "actions"
    ]
    assert actions
    ids = {e.get("action_id") for e in actions[0].get("elements") or []}
    assert "judge_ack_digest_reexport" in ids
    assert "judge_ack_digest_heatmap_zoom" in ids
    assert "judge_ack_digest_open" in ids
    assert "judge_ack_interactive" in ids
    # CSV dropped when form URL present so zoom fits the 5-button cap
    assert "judge_ack_digest_reexport_csv" not in ids or len(actions[0]["elements"]) <= 5


def test_digest_diff_snapshot_delta(tmp_path: Path, monkeypatch) -> None:
    from rag.judge_alert import (
        diff_judge_ack_digest,
        save_judge_ack_digest_snapshot,
        build_judge_ack_digest_slack_blocks,
    )

    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_DIFF", "1")
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    prev = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    save_judge_ack_digest_snapshot(prev, base=str(tmp_path))
    append_judge_ack_audit("unack", actor="bob", path=path)
    append_judge_ack_audit("ack", actor="carol", path=path)
    curr = summarize_judge_ack_audit(path=path, since_hours=24 * 365)
    from rag.judge_alert import load_judge_ack_digest_snapshot

    loaded = load_judge_ack_digest_snapshot(base=str(tmp_path))
    diff = diff_judge_ack_digest(loaded, curr)
    assert diff["has_previous"] is True
    assert diff["delta_total"] == 2
    assert "unack" in diff["new_events"] or diff["by_event_delta"].get("unack", 0) > 0
    assert "bob" in diff["actors_added"] or "carol" in diff["actors_added"]
    blocks = build_judge_ack_digest_slack_blocks(curr, diff=diff)
    texts = " ".join(
        str((b.get("text") or {}).get("text") or "") for b in blocks if b.get("type") == "section"
    )
    assert "vs last digest" in texts
    assert "Δ total" in texts or "delta" in texts.lower() or "+" in texts

    result = dispatch_judge_ack_digest(
        curr, dry_run=True, base=str(tmp_path), include_diff=True
    )
    assert result["diff"]["delta_total"] == 2
    assert "vs last digest" in result["payload"]["text"]


def test_ack_heatmap_and_tenant_canvas_urls(tmp_path: Path, monkeypatch) -> None:
    from rag.judge_alert import (
        build_ack_heatmap,
        format_ack_heatmap_mrkdwn,
        judge_ack_canvas_urls,
    )

    monkeypatch.setenv("RAG_JUDGE_ACK_PUBLIC_URL", "http://example.test/judge/ack-form")
    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT", str(tmp_path / "ack.jsonl"))
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="a", path=path, extra={"tenant_id": "acme"})
    append_judge_ack_audit("unack", actor="b", path=path, extra={"tenant_id": "acme"})
    append_judge_ack_audit("ack", actor="c", path=path, extra={"tenant_id": "beta"})
    heat = build_ack_heatmap(path=path, since_hours=24 * 365, tenant_id="acme", bucket="day")
    assert heat["ok"] is True
    assert heat["total"] == 2
    assert "ack" in heat["events"]
    md = format_ack_heatmap_mrkdwn(heat)
    assert "Ack heatmap" in md
    urls = judge_ack_canvas_urls(tenant_id="acme", hours=168)
    assert "tenant=acme" in urls["form"]
    assert "tenant=acme" in urls["export"]
    assert "format=csv" in urls["export_csv"]

    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365, tenant_id="acme")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_HEATMAP", "1")
    result = dispatch_judge_ack_digest(summary, dry_run=True, tenant_id="acme")
    assert result["canvas"]["form"]
    assert "tenant=acme" in result["canvas"]["form"]
    assert "Ack heatmap" in result["payload"]["text"]
    assert "Canvas:" in result["payload"]["text"] or "tenant=acme" in result["payload"]["text"]

    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_HEATMAP_BUCKET", "hour")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_HEATMAP_MAX_BUCKETS", "24")
    hour = build_ack_heatmap(
        path=path, since_hours=24 * 365, tenant_id="acme", bucket="hour", max_buckets=24
    )
    assert hour["bucket"] == "hour"
    assert hour["total"] == 2


def test_tenant_mute_skips_digest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_MUTE", "acme,beta")
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="a", path=path, extra={"tenant_id": "acme"})
    summary = summarize_judge_ack_audit(path=path, since_hours=24 * 365, tenant_id="acme")
    result = dispatch_judge_ack_digest(summary, dry_run=True, tenant_id="acme", base=str(tmp_path))
    assert result["skipped"] is True
    assert result["reason"] == "muted"

    (tmp_path / "judge_ack_digest_webhooks.json").write_text(
        '{"acme": "https://hooks.slack.test/acme", "gamma": "https://hooks.slack.test/gamma"}',
        encoding="utf-8",
    )
    fan = dispatch_judge_ack_digest_fanout(
        path=path, since_hours=24 * 365, dry_run=True, base=str(tmp_path)
    )
    by_tid = {r.get("tenant_id"): r for r in fan["results"]}
    assert by_tid["acme"]["reason"] == "muted"
    assert by_tid["gamma"].get("reason") != "muted"
    assert "acme" in fan["muted"]


def test_heatmap_zoom_interactive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT", str(tmp_path / "ack.jsonl"))
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_PREFS", str(tmp_path / "prefs.json"))
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_BASE", str(tmp_path))
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    from rag.judge_alert import handle_slack_interactive_ack, resolve_heatmap_bucket

    with patch("rag.judge_alert.slack_api", return_value={"ok": True}):
        result = handle_slack_interactive_ack(
            {
                "type": "block_actions",
                "actions": [
                    {"action_id": "judge_ack_digest_heatmap_zoom", "value": "hour"}
                ],
                "user": {"id": "U1", "username": "ops"},
                "channel": {"id": "C1"},
            }
        )
    assert result["ok"] is True
    assert result["mode"] == "heatmap_zoom"
    assert result["bucket"] == "hour"
    assert "Ack heatmap" in result["text"]
    assert result["ephemeral"]["ok"] is True
    assert result["pref"]["ok"] is True
    assert resolve_heatmap_bucket(base=str(tmp_path)) == "hour"

    with patch("rag.judge_alert.slack_api", return_value={"ok": True}):
        tenant = handle_slack_interactive_ack(
            {
                "type": "block_actions",
                "actions": [
                    {
                        "action_id": "judge_ack_digest_heatmap_zoom",
                        "value": "acme|day",
                    }
                ],
                "user": {"id": "U1", "username": "ops"},
                "channel": {"id": "C1"},
            }
        )
    assert tenant["bucket"] == "day"
    assert tenant["tenant_id"] == "acme"
    assert resolve_heatmap_bucket("acme", base=str(tmp_path)) == "day"


def test_mute_ui_interactive_and_block_kit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_JUDGE_SLACK_INTERACTIVE", "1")
    monkeypatch.setenv("RAG_JUDGE_ACK_PUBLIC_URL", "http://example.test/judge/ack-form")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_BASE", str(tmp_path))
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_PREFS", str(tmp_path / "prefs.json"))
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-test")
    from rag.judge_alert import (
        build_judge_ack_digest_slack_blocks,
        handle_slack_interactive_ack,
        is_judge_ack_digest_muted,
        set_heatmap_bucket_pref,
    )

    set_heatmap_bucket_pref("hour", tenant_id="acme", base=str(tmp_path))
    summary = {
        "ok": True,
        "since_hours": 168,
        "total": 1,
        "actor_count": 1,
        "by_event": {"ack": 1},
        "actors": ["a"],
        "tenant_id": "acme",
    }
    blocks = build_judge_ack_digest_slack_blocks(
        summary, tenant_id="acme", base=str(tmp_path)
    )
    actions = [b for b in blocks if b.get("type") == "actions"]
    assert actions
    els = actions[0]["elements"]
    assert len(els) <= 5
    ids = {e.get("action_id") for e in els}
    assert "judge_ack_digest_mute" in ids
    assert "judge_ack_digest_heatmap_zoom" in ids
    zoom = next(e for e in els if e["action_id"] == "judge_ack_digest_heatmap_zoom")
    assert zoom["value"] == "acme|day"
    assert "Heatmap day" in zoom["text"]["text"]

    with patch("rag.judge_alert.slack_api", return_value={"ok": True}):
        muted = handle_slack_interactive_ack(
            {
                "type": "block_actions",
                "actions": [
                    {"action_id": "judge_ack_digest_mute", "value": "acme"}
                ],
                "user": {"id": "U1", "username": "ops"},
                "channel": {"id": "C1"},
            }
        )
    assert muted["ok"] is True
    assert muted["mode"] == "digest_mute"
    assert muted["muted"] is True
    assert is_judge_ack_digest_muted("acme", base=str(tmp_path))

    with patch("rag.judge_alert.slack_api", return_value={"ok": True}):
        unmuted = handle_slack_interactive_ack(
            {
                "type": "block_actions",
                "actions": [
                    {"action_id": "judge_ack_digest_unmute", "value": "acme"}
                ],
                "user": {"id": "U1", "username": "ops"},
                "channel": {"id": "C1"},
            }
        )
    assert unmuted["muted"] is False
    assert not is_judge_ack_digest_muted("acme", base=str(tmp_path))


def test_dual_write_dlq_cli_parser() -> None:
    from rag.cli import build_parser

    args = build_parser().parse_args(
        ["dual-write-dlq", "--replay", "--limit", "3", "--dry-run"]
    )
    assert args.command == "dual-write-dlq"
    assert args.replay is True
    assert args.limit == 3
    assert args.dry_run is True
    prune_args = build_parser().parse_args(
        ["dual-write-dlq", "--prune", "--days", "14", "--notify", "--dry-run"]
    )
    assert prune_args.prune is True
    assert prune_args.days == 14.0
    assert prune_args.notify is True
    q_args = build_parser().parse_args(["dual-write-dlq", "--quarantine", "--limit", "10"])
    assert q_args.quarantine is True
    assert q_args.limit == 10
    nb = build_parser().parse_args(["dual-write-dlq", "--replay", "--no-budget"])
    assert nb.no_budget is True
    rq = build_parser().parse_args(
        ["dual-write-dlq", "--replay-quarantine", "--requeue", "--limit", "2"]
    )
    assert rq.replay_quarantine is True
    assert rq.requeue is True
    assert rq.limit == 2
    pm = build_parser().parse_args(["judge-ack-digest", "--prune-mutes", "--dry-run"])
    assert pm.prune_mutes is True


def test_mute_ttl_retention_and_prune(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone

    from rag.judge_alert import (
        is_judge_ack_digest_muted,
        maybe_prune_judge_ack_digest_mutes,
        prune_judge_ack_digest_mutes,
        set_judge_ack_digest_mute,
        set_heatmap_bucket_pref,
        prune_judge_ack_digest_prefs,
    )

    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_MUTE_TTL_DAYS", "7")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_PREFS", str(tmp_path / "prefs.json"))
    saved = set_judge_ack_digest_mute("acme", muted=True, base=str(tmp_path), ttl_days=7)
    assert saved["ok"] is True
    assert saved["expires_at"]
    assert is_judge_ack_digest_muted("acme", base=str(tmp_path))

    # Force expired entry
    import json

    path = tmp_path / "judge_ack_digest_mutes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["acme"]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not is_judge_ack_digest_muted("acme", base=str(tmp_path))
    pruned = prune_judge_ack_digest_mutes(base=str(tmp_path), dry_run=False)
    assert pruned["removed"] == 1
    assert "acme" in pruned["removed_tenants"]

    set_heatmap_bucket_pref("hour", tenant_id="old", base=str(tmp_path))
    prefs_path = tmp_path / "prefs.json"
    prefs = json.loads(prefs_path.read_text(encoding="utf-8"))
    prefs["tenants"]["old"]["updated_at"] = (
        datetime.now(timezone.utc) - timedelta(days=120)
    ).isoformat()
    prefs_path.write_text(json.dumps(prefs), encoding="utf-8")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_PREFS_RETENTION_DAYS", "30")
    monkeypatch.setenv("RAG_JUDGE_ACK_DIGEST_MUTE_PRUNE", "1")
    report = maybe_prune_judge_ack_digest_mutes(base=str(tmp_path), dry_run=False)
    assert report["ok"] is True
    assert report["prefs"]["removed"] == 1


def test_digest_reexport_interactive_action(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT", str(tmp_path / "ack.jsonl"))
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-test")
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    from rag.judge_alert import handle_slack_interactive_ack

    uploaded = {}
    thread_calls = []

    def fake_upload(**kwargs):
        uploaded.update(kwargs)
        return {
            "ok": True,
            "file_id": "F1",
            "permalink": "https://slack.test/file",
            "filename": kwargs.get("filename"),
        }

    def fake_thread(**kwargs):
        thread_calls.append(kwargs)
        return {"ok": True, "ts": "333.444", "thread_ts": kwargs.get("thread_ts")}

    with patch("rag.judge_alert.slack_files_upload", side_effect=fake_upload):
        with patch(
            "rag.judge_alert.post_judge_digest_reexport_thread_reply",
            side_effect=fake_thread,
        ):
            result = handle_slack_interactive_ack(
                {
                    "type": "block_actions",
                    "actions": [
                        {"action_id": "judge_ack_digest_reexport", "value": "jsonl"}
                    ],
                    "user": {"username": "ops", "id": "U1"},
                    "channel": {"id": "C123"},
                    "message": {"ts": "111.222"},
                }
            )
    assert result["ok"] is True
    assert result["mode"] == "digest_reexport"
    assert result["format"] == "jsonl"
    assert result["summary"]["total"] >= 1
    assert "Total events" in result["text"]
    assert result["upload"]["ok"] is True
    assert uploaded.get("channels") == "C123"
    assert uploaded.get("thread_ts") == "111.222"
    assert "judge_ack_audit.jsonl" in (uploaded.get("filename") or "")
    assert "Attached `judge_ack_audit.jsonl`" in result["text"]
    assert result["thread_reply"]["ok"] is True
    assert thread_calls
    assert thread_calls[0].get("thread_ts") == "111.222"
    assert thread_calls[0].get("fmt") == "jsonl"


def test_digest_reexport_csv_with_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT", str(tmp_path / "ack.jsonl"))
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-test")
    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="alice", path=path)
    from rag.judge_alert import handle_slack_interactive_ack

    progress_calls = []

    def fake_api(method, **kwargs):
        progress_calls.append({"method": method, **kwargs})
        return {"ok": True, "message_ts": "1.2"}

    def fake_upload(**kwargs):
        return {
            "ok": True,
            "file_id": "F2",
            "permalink": "https://slack.test/csv",
            "filename": kwargs.get("filename"),
        }

    with patch("rag.judge_alert.slack_api", side_effect=fake_api):
        with patch("rag.judge_alert.slack_files_upload", side_effect=fake_upload):
            with patch(
                "rag.judge_alert.post_judge_digest_reexport_thread_reply",
                return_value={"ok": True, "ts": "9.9"},
            ):
                result = handle_slack_interactive_ack(
                    {
                        "type": "block_actions",
                        "actions": [
                            {
                                "action_id": "judge_ack_digest_reexport_csv",
                                "value": "csv",
                            }
                        ],
                        "user": {"id": "U9", "username": "ops"},
                        "channel": {"id": "C9"},
                        "message": {"ts": "55.66"},
                    }
                )
    assert result["ok"] is True
    assert result["format"] == "csv"
    assert result["export"]["format"] == "csv"
    assert result["progress"]["ok"] is True
    assert progress_calls and progress_calls[0]["method"] == "chat.postEphemeral"
    assert "csv" in (progress_calls[0]["json_body"]["text"] or "")
    assert "Attached `judge_ack_audit.csv`" in result["text"]
    assert result["thread_reply"]["ok"] is True


def test_maybe_purge_judge_ack_audit_from_env(tmp_path: Path, monkeypatch) -> None:
    from rag.judge_alert import append_judge_ack_audit, maybe_purge_judge_ack_audit

    path = str(tmp_path / "ack.jsonl")
    append_judge_ack_audit("ack", actor="a", path=path)
    monkeypatch.delenv("RAG_JUDGE_ACK_AUDIT_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("RAG_JUDGE_ACK_AUDIT_KEEP", raising=False)
    monkeypatch.delenv("RAG_JUDGE_ACK_AUDIT_PRUNE", raising=False)
    skipped = maybe_purge_judge_ack_audit(path=path)
    assert skipped.get("skipped") is True

    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT_KEEP", "1")
    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT_PRUNE", "1")
    append_judge_ack_audit("ack", actor="b", path=path)
    append_judge_ack_audit("ack", actor="c", path=path)
    purged = maybe_purge_judge_ack_audit(path=path)
    assert purged["ok"] is True
    assert purged["after"] == 1
    assert purged["removed"] >= 1

    monkeypatch.setenv("RAG_JUDGE_ACK_AUDIT_PRUNE", "0")
    disabled = maybe_purge_judge_ack_audit(path=path)
    assert disabled.get("reason") == "prune_disabled"


def test_slack_files_upload_requires_token() -> None:
    from rag.judge_alert import slack_files_upload

    result = slack_files_upload(content="x", channels="C1", bot_token="")
    assert result["ok"] is False
    assert result["error"] == "bot_token_missing"



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
