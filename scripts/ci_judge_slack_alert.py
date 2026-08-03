#!/usr/bin/env python3
"""CI: Judge soft-fail olduğunda Slack webhook bildirimi."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


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


def build_slack_payload(report: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    summary = report.get("summary") or {}
    accuracy = summary.get("accuracy")
    mode = summary.get("mode") or report.get("mode") or "?"
    text = (
        f"RAG judge soft-fail ({source}): mode={mode} "
        f"accuracy={accuracy} passed={summary.get('passed')} "
        f"failed={summary.get('failed')} total={summary.get('total')}"
    )
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
    except Exception as exc:  # noqa: BLE001
        print(f"slack post failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    report_path = os.environ.get("RAG_JUDGE_OUTPUT", "metadata/judge_report.json")
    metrics_path = os.environ.get("RAG_JUDGE_METRICS_PATH", "metadata/metrics.jsonl")
    soft_env = os.environ.get("RAG_JUDGE_SOFT_FAIL", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    webhook = (
        os.environ.get("RAG_JUDGE_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_DIGEST_ALERT_WEBHOOK_URL", "").strip()
        or os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    )

    report = load_judge_report(report_path)
    # LLM job may write judge_report_llm.json
    if not report:
        alt = os.environ.get("RAG_JUDGE_OUTPUT_LLM", "metadata/judge_report_llm.json")
        report = load_judge_report(alt)

    from_report = soft_fail_from_report(report, soft_fail_env=soft_env)
    from_metrics = soft_fail_from_metrics(metrics_path)
    if not (from_report or from_metrics):
        print(json.dumps({"alerted": False, "reason": "no_soft_fail"}, ensure_ascii=False))
        return 0

    if not webhook:
        print(
            json.dumps(
                {"alerted": False, "reason": "no_webhook", "soft_fail": True},
                ensure_ascii=False,
            )
        )
        # soft-fail detected but webhook missing — do not fail CI
        return 0

    source = "report" if from_report else "metrics"
    payload = build_slack_payload(report or {"summary": {}}, source=source)
    ok = post_slack(webhook, payload)
    print(
        json.dumps(
            {"alerted": ok, "source": source, "soft_fail": True},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
