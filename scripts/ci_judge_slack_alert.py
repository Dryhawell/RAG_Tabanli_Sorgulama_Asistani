#!/usr/bin/env python3
"""CI: Judge soft-fail → Slack / PagerDuty / Opsgenie."""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from rag.judge_alert import detect_soft_fail, dispatch_judge_alerts

    report_path = os.environ.get("RAG_JUDGE_OUTPUT", "metadata/judge_report.json")
    metrics_path = os.environ.get("RAG_JUDGE_METRICS_PATH", "metadata/metrics.jsonl")
    soft_env = os.environ.get("RAG_JUDGE_SOFT_FAIL", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }

    fired, source, report = detect_soft_fail(
        report_path=report_path,
        metrics_path=metrics_path,
        soft_fail_env=soft_env,
    )
    if not fired:
        print(json.dumps({"alerted": False, "reason": "no_soft_fail"}, ensure_ascii=False))
        return 0

    result = dispatch_judge_alerts(report or {"summary": {}}, source=source or "unknown")
    if not result.get("configured"):
        print(
            json.dumps(
                {
                    "alerted": False,
                    "reason": "no_channel",
                    "soft_fail": True,
                },
                ensure_ascii=False,
            )
        )
        return 0
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
