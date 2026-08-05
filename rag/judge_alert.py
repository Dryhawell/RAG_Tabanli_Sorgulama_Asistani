"""Judge soft-fail çok kanallı alert (Slack / PagerDuty / Opsgenie) + auto-resolve."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def judge_alert_state_path(base: Optional[str] = None) -> str:
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    return os.path.join(root, "judge_alert_state.json")


def load_judge_alert_state(path: Optional[str] = None) -> Dict[str, Any]:
    # Explicit path → sadece dosya (test/izolasyon); aksi halde remote öncelikli
    if path is None:
        remote = _load_remote_judge_state()
        if remote is not None:
            return remote
    p = path or judge_alert_state_path()
    if not os.path.isfile(p):
        return {"soft_fail": False, "source": None, "updated_at": None}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"soft_fail": False}
    except Exception:
        return {"soft_fail": False, "source": None, "updated_at": None}


def save_judge_alert_state(state: Dict[str, Any], path: Optional[str] = None) -> str:
    p = path or judge_alert_state_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    _save_remote_judge_state(state)
    return p


def judge_alert_state_url() -> str:
    return (
        os.environ.get("RAG_JUDGE_ALERT_STATE_URL", "").strip()
        or os.environ.get("RAG_JUDGE_STATE_URL", "").strip()
    )


def _load_remote_judge_state() -> Optional[Dict[str, Any]]:
    url = judge_alert_state_url()
    if not url:
        return None
    try:
        import requests

        r = requests.get(url, timeout=10)
        if r.status_code >= 400:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_remote_judge_state(state: Dict[str, Any]) -> bool:
    url = judge_alert_state_url()
    if not url:
        return False
    try:
        import requests

        r = requests.put(url, json=state, timeout=10)
        if r.status_code >= 400:
            r = requests.post(url, json=state, timeout=10)
        return r.status_code < 400
    except Exception:
        return False


def judge_ack_public_url() -> str:
    """Slack deep-link hedefi (ack form)."""
    explicit = (
        os.environ.get("RAG_JUDGE_ACK_PUBLIC_URL", "").strip()
        or os.environ.get("RAG_JUDGE_ACK_URL", "").strip()
    )
    if explicit:
        return explicit.rstrip("/")
    try:
        from app.config import COLLAB_HTTP_PORT, COLLAB_WS_PUBLIC_HOST

        host = (COLLAB_WS_PUBLIC_HOST or "localhost").strip() or "localhost"
        port = int(COLLAB_HTTP_PORT or 8766)
        return f"http://{host}:{port}/judge/ack-form"
    except Exception:
        base = os.environ.get("RAG_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if base:
            return f"{base}/judge/ack-form"
    return ""


def load_judge_report(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def soft_fail_from_report(
    report: Dict[str, Any],
    *,
    soft_fail_env: bool,
) -> bool:
    if not soft_fail_env:
        return False
    summary = report.get("summary") or {}
    return not bool(summary.get("ok"))


def soft_fail_from_metrics(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    last: Optional[Dict[str, Any]] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "judge_run":
                    last = row
    except Exception:
        return False
    if not last:
        return False
    vals = last.get("values") or {}
    return bool(vals.get("soft_fail"))


def detect_soft_fail(
    *,
    report_path: str,
    metrics_path: str,
    soft_fail_env: bool,
) -> Tuple[bool, str, Dict[str, Any]]:
    report = load_judge_report(report_path)
    if not report:
        alt = os.environ.get("RAG_JUDGE_OUTPUT_LLM", "metadata/judge_report_llm.json")
        report = load_judge_report(alt)
    from_report = soft_fail_from_report(report, soft_fail_env=soft_fail_env)
    from_metrics = soft_fail_from_metrics(metrics_path)
    if from_report:
        return True, "report", report
    if from_metrics:
        return True, "metrics", report or {"summary": {}}
    return False, "", report or {}


def _summary_text(report: Dict[str, Any], *, source: str) -> str:
    summary = report.get("summary") or {}
    mode = summary.get("mode") or report.get("mode") or "?"
    return (
        f"RAG judge soft-fail ({source}): mode={mode} "
        f"accuracy={summary.get('accuracy')} passed={summary.get('passed')} "
        f"failed={summary.get('failed')} total={summary.get('total')}"
    )


def _resolve_text(report: Dict[str, Any], *, source: str) -> str:
    summary = report.get("summary") or {}
    return (
        f"RAG judge soft-fail RESOLVED ({source}): "
        f"accuracy={summary.get('accuracy')} ok={summary.get('ok')}"
    )


def build_slack_payload(report: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    text = _summary_text(report, source=source)
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail*\n{text}"},
        }
    ]
    ack_url = judge_ack_public_url()
    if ack_url:
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Acknowledge"},
                        "url": ack_url,
                        "action_id": "judge_ack_open",
                    }
                ],
            }
        )
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Ack form: <{ack_url}|open>"},
                ],
            }
        )
    return {"text": text, "blocks": blocks}


def build_slack_resolve_payload(report: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    text = _resolve_text(report, source=source)
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail resolved*\n{text}"},
        }
    ]
    return {"text": text, "blocks": blocks}


def post_slack(webhook: str, payload: Dict[str, Any]) -> bool:
    url = (webhook or "").strip()
    if not url:
        return False
    try:
        import requests

        r = requests.post(url, json=payload, timeout=10)
        return r.status_code < 400
    except Exception:
        return False


def post_pagerduty(
    *,
    routing_key: str,
    report: Dict[str, Any],
    source: str,
    severity: str = "warning",
    event_action: str = "trigger",
) -> bool:
    key = (routing_key or "").strip()
    if not key:
        return False
    action = event_action if event_action in {"trigger", "acknowledge", "resolve"} else "trigger"
    text = (
        _resolve_text(report, source=source)
        if action == "resolve"
        else _summary_text(report, source=source)
    )
    summary = report.get("summary") or {}
    sev = severity if severity in {"info", "warning", "error", "critical"} else "warning"
    body: Dict[str, Any] = {
        "routing_key": key,
        "event_action": action,
        "dedup_key": f"rag-judge-soft-fail/{source}",
    }
    if action == "trigger":
        body["payload"] = {
            "summary": text[:1024],
            "severity": "rag-judge",
            "source": "rag-ci",
            "severity": sev,
            "component": "judge",
            "group": "ci",
            "class": "soft_fail",
            "custom_details": {
                "accuracy": summary.get("accuracy"),
                "passed": summary.get("passed"),
                "failed": summary.get("failed"),
                "total": summary.get("total"),
                "source": source,
                "mode": summary.get("mode") or report.get("mode"),
            },
        }
    try:
        import requests

        r = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=body,
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def post_opsgenie(
    *,
    api_key: str,
    report: Dict[str, Any],
    source: str,
    priority: str = "P3",
) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    text = _summary_text(report, source=source)
    summary = report.get("summary") or {}
    pri = priority if priority in {"P1", "P2", "P3", "P4", "P5"} else "P3"
    body = {
        "message": text[:130],
        "alias": f"rag-judge-soft-fail/{source}",
        "description": text,
        "priority": pri,
        "tags": ["rag", "judge", "soft-fail", source],
        "details": {
            "accuracy": str(summary.get("accuracy")),
            "passed": str(summary.get("passed")),
            "failed": str(summary.get("failed")),
            "total": str(summary.get("total")),
            "mode": str(summary.get("mode") or report.get("mode") or ""),
        },
        "entity": "rag-judge",
        "source": "rag-ci",
    }
    try:
        import requests

        r = requests.post(
            "https://api.opsgenie.com/v2/alerts",
            headers={
                "Authorization": f"GenieKey {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def post_opsgenie_close(*, api_key: str, source: str) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    alias = f"rag-judge-soft-fail/{source}"
    try:
        import requests
        from urllib.parse import quote

        r = requests.post(
            f"https://api.opsgenie.com/v2/alerts/{quote(alias, safe='')}/close",
            params={"identifierType": "alias"},
            headers={
                "Authorization": f"GenieKey {key}",
                "Content-Type": "application/json",
            },
            json={"note": "RAG judge soft-fail resolved in CI"},
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def _channel_creds(
    *,
    slack_webhook: Optional[str],
    pagerduty_routing_key: Optional[str],
    opsgenie_api_key: Optional[str],
) -> Tuple[str, str, str]:
    slack_url = (
        slack_webhook
        if slack_webhook is not None
        else (
            os.environ.get("RAG_JUDGE_SLACK_WEBHOOK", "").strip()
            or os.environ.get("RAG_DIGEST_ALERT_WEBHOOK_URL", "").strip()
            or os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        )
    )
    pd_key = (
        pagerduty_routing_key
        if pagerduty_routing_key is not None
        else os.environ.get("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", "").strip()
    )
    og_key = (
        opsgenie_api_key
        if opsgenie_api_key is not None
        else os.environ.get("RAG_JUDGE_OPSGENIE_API_KEY", "").strip()
    )
    return slack_url, pd_key, og_key


def dispatch_judge_alerts(
    report: Dict[str, Any],
    *,
    source: str,
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    slack_url, pd_key, og_key = _channel_creds(
        slack_webhook=slack_webhook,
        pagerduty_routing_key=pagerduty_routing_key,
        opsgenie_api_key=opsgenie_api_key,
    )
    channels: Dict[str, Any] = {}
    any_ok = False
    if slack_url:
        ok = post_slack(slack_url, build_slack_payload(report, source=source))
        channels["slack"] = ok
        any_ok = any_ok or ok
    if pd_key:
        ok = post_pagerduty(routing_key=pd_key, report=report, source=source)
        channels["pagerduty"] = ok
        any_ok = any_ok or ok
    if og_key:
        ok = post_opsgenie(api_key=og_key, report=report, source=source)
        channels["opsgenie"] = ok
        any_ok = any_ok or ok
    return {
        "alerted": any_ok,
        "channels": channels,
        "source": source,
        "soft_fail": True,
        "configured": bool(slack_url or pd_key or og_key),
    }


def dispatch_judge_resolve(
    report: Dict[str, Any],
    *,
    source: str,
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    slack_url, pd_key, og_key = _channel_creds(
        slack_webhook=slack_webhook,
        pagerduty_routing_key=pagerduty_routing_key,
        opsgenie_api_key=opsgenie_api_key,
    )
    channels: Dict[str, Any] = {}
    any_ok = False
    if slack_url:
        ok = post_slack(slack_url, build_slack_resolve_payload(report, source=source))
        channels["slack"] = ok
        any_ok = any_ok or ok
    if pd_key:
        ok = post_pagerduty(
            routing_key=pd_key,
            report=report,
            source=source,
            event_action="resolve",
        )
        channels["pagerduty"] = ok
        any_ok = any_ok or ok
    if og_key:
        ok = post_opsgenie_close(api_key=og_key, source=source)
        channels["opsgenie"] = ok
        any_ok = any_ok or ok
    return {
        "resolved": any_ok,
        "channels": channels,
        "source": source,
        "soft_fail": False,
        "configured": bool(slack_url or pd_key or og_key),
    }


def build_slack_ack_payload(
    report: Dict[str, Any],
    *,
    source: str,
    actor: str,
    note: str = "",
) -> Dict[str, Any]:
    text = (
        f"RAG judge soft-fail ACK ({source}): by={actor}"
        + (f" note={note}" if note else "")
    )
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail acknowledged*\n{text}"},
        }
    ]
    return {"text": text, "blocks": blocks}


def post_opsgenie_ack(*, api_key: str, source: str, note: str = "") -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    alias = f"rag-judge-soft-fail/{source}"
    body: Dict[str, Any] = {"user": "rag-ci"}
    if note:
        body["note"] = note[:15000]
    try:
        import requests
        from urllib.parse import quote

        r = requests.post(
            f"https://api.opsgenie.com/v2/alerts/{quote(alias, safe='')}/acknowledge",
            params={"identifierType": "alias"},
            headers={
                "Authorization": f"GenieKey {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def dispatch_judge_ack(
    report: Dict[str, Any],
    *,
    source: str,
    actor: str,
    note: str = "",
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    slack_url, pd_key, og_key = _channel_creds(
        slack_webhook=slack_webhook,
        pagerduty_routing_key=pagerduty_routing_key,
        opsgenie_api_key=opsgenie_api_key,
    )
    channels: Dict[str, Any] = {}
    any_ok = False
    if slack_url:
        ok = post_slack(
            slack_url,
            build_slack_ack_payload(report, source=source, actor=actor, note=note),
        )
        channels["slack"] = ok
        any_ok = any_ok or ok
    if pd_key:
        ok = post_pagerduty(
            routing_key=pd_key,
            report=report,
            source=source,
            event_action="acknowledge",
        )
        channels["pagerduty"] = ok
        any_ok = any_ok or ok
    if og_key:
        ok = post_opsgenie_ack(api_key=og_key, source=source, note=note or f"acked by {actor}")
        channels["opsgenie"] = ok
        any_ok = any_ok or ok
    return {
        "acked": any_ok,
        "channels": channels,
        "source": source,
        "soft_fail": True,
        "configured": bool(slack_url or pd_key or og_key),
        "actor": actor,
        "note": note,
    }


def acknowledge_judge_alert(
    *,
    actor: str,
    note: str = "",
    state_path: Optional[str] = None,
    notify: bool = True,
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manuel soft-fail acknowledge; state'e yazar, opsiyonel kanal bildirimi."""
    who = (actor or "").strip()
    if not who:
        return {"ok": False, "error": "actor_required"}
    state = load_judge_alert_state(state_path)
    if not state.get("soft_fail"):
        return {
            "ok": False,
            "error": "no_active_soft_fail",
            "state": state,
        }
    if state.get("acknowledged"):
        return {
            "ok": True,
            "already": True,
            "state": state,
            "notify": {"acked": False, "reason": "already_acknowledged"},
        }
    source = str(state.get("source") or "report")
    new_state = {
        **state,
        "soft_fail": True,
        "source": source,
        "acknowledged": True,
        "acknowledged_at": _utcnow_iso(),
        "acknowledged_by": who,
        "ack_note": (note or "")[:500],
        "updated_at": _utcnow_iso(),
        "last_action": "ack",
    }
    save_judge_alert_state(new_state, state_path)
    notify_result: Dict[str, Any] = {"acked": False, "skipped": True}
    if notify:
        notify_result = dispatch_judge_ack(
            report or {"summary": {"ok": False}},
            source=source,
            actor=who,
            note=note or "",
        )
    return {"ok": True, "already": False, "state": new_state, "notify": notify_result}


def detect_and_dispatch(
    *,
    report_path: str,
    metrics_path: str,
    soft_fail_env: bool,
    state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    soft_fail → alert + state; ack'lı soft-fail → yeniden alert yok;
    önceki soft_fail + şimdi ok → resolve + state temizle.
    """
    fired, source, report = detect_soft_fail(
        report_path=report_path,
        metrics_path=metrics_path,
        soft_fail_env=soft_fail_env,
    )
    state = load_judge_alert_state(state_path)
    prev = bool(state.get("soft_fail"))
    prev_source = str(state.get("source") or "report")
    acked = bool(state.get("acknowledged"))

    if fired:
        if prev and acked:
            return {
                "action": "ack_hold",
                "soft_fail": True,
                "alerted": False,
                "reason": "acknowledged",
                "acknowledged_by": state.get("acknowledged_by"),
                "acknowledged_at": state.get("acknowledged_at"),
                "source": source or prev_source,
            }
        result = dispatch_judge_alerts(report or {"summary": {}}, source=source or "unknown")
        save_judge_alert_state(
            {
                "soft_fail": True,
                "source": source or "unknown",
                "updated_at": _utcnow_iso(),
                "last_action": "alert",
                "acknowledged": False,
                "acknowledged_at": None,
                "acknowledged_by": None,
                "ack_note": None,
            },
            state_path,
        )
        return {
            "action": "alert",
            "soft_fail": True,
            **result,
        }

    if prev:
        result = dispatch_judge_resolve(
            report or {"summary": {"ok": True}},
            source=prev_source,
        )
        save_judge_alert_state(
            {
                "soft_fail": False,
                "source": None,
                "updated_at": _utcnow_iso(),
                "last_action": "resolve",
                "acknowledged": False,
                "acknowledged_at": None,
                "acknowledged_by": None,
                "ack_note": None,
            },
            state_path,
        )
        return {
            "action": "resolve",
            "soft_fail": False,
            **result,
        }

    return {
        "action": "noop",
        "alerted": False,
        "resolved": False,
        "reason": "no_soft_fail",
        "soft_fail": False,
    }
