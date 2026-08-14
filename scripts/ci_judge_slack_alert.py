#!/usr/bin/env python3
"""CI: Judge soft-fail → Slack / PagerDuty / Opsgenie (+ auto-resolve)."""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from rag.judge_alert import detect_and_dispatch

    report_path = os.environ.get("RAG_JUDGE_OUTPUT", "metadata/judge_report.json")
    metrics_path = os.environ.get("RAG_JUDGE_METRICS_PATH", "metadata/metrics.jsonl")
    state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "metadata/judge_alert_state.json")
    soft_env = os.environ.get("RAG_JUDGE_SOFT_FAIL", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }

    result = detect_and_dispatch(
        report_path=report_path,
        metrics_path=metrics_path,
        soft_fail_env=soft_env,
        state_path=state_path,
    )
    action = result.get("action")
    if action == "noop":
        print(json.dumps({"alerted": False, "reason": "no_soft_fail"}, ensure_ascii=False))
        return 0
    if action == "alert" and not result.get("configured"):
        print(
            json.dumps(
                {"alerted": False, "reason": "no_channel", "soft_fail": True, "action": "alert"},
                ensure_ascii=False,
            )
        )
        return 0
    if action == "resolve" and not result.get("configured"):
        print(
            json.dumps(
                {
                    "resolved": False,
                    "reason": "no_channel",
                    "soft_fail": False,
                    "action": "resolve",
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
