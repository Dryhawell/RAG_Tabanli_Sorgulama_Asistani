"""Judge soft-fail çok kanallı alert (Slack / PagerDuty / Opsgenie)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


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


def build_slack_payload(report: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    text = _summary_text(report, source=source)
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail*\n{text}"},
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
) -> bool:
    key = (routing_key or "").strip()
    if not key:
        return False
    text = _summary_text(report, source=source)
    summary = report.get("summary") or {}
    sev = severity if severity in {"info", "warning", "error", "critical"} else "warning"
    body = {
        "routing_key": key,
        "event_action": "trigger",
        "dedup_key": f"rag-judge-soft-fail/{source}",
        "payload": {
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


def dispatch_judge_alerts(
    report: Dict[str, Any],
    *,
    source: str,
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
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
