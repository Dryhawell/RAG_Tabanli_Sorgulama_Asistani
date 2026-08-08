#!/usr/bin/env python3
"""CI: inhibit equal-tune dry-run artifact + PR comment upsert."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Allow `python scripts/...` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INHIBIT_EQUAL_PR_COMMENT_MARKER = "<!-- inhibit-equal-tune-bot -->"


def build_inhibit_equal_comment(tune: Dict[str, Any]) -> str:
    equal = tune.get("equal") or []
    lines = [
        INHIBIT_EQUAL_PR_COMMENT_MARKER,
        "### Alertmanager inhibit equal tune (dry-run)",
        "",
        f"- **source**: `{tune.get('source')}`",
        f"- **equal**: `{', '.join(equal) if isinstance(equal, list) else equal}`",
        f"- **live_ok**: `{tune.get('live_ok')}`",
        f"- **live_alerts**: `{tune.get('live_alerts', 0)}`",
        f"- **static_label_sets**: `{tune.get('static_label_sets', 0)}`",
        f"- **label_sets**: `{tune.get('label_sets', tune.get('static_label_sets', 0))}`",
        "",
        "CI artifact: `metadata/inhibit_equal_tune.json` · "
        "`metadata/inhibit_rules.ci.yml`",
        "",
        "Apply locally:",
        "```bash",
        "python -m rag.cli alertmanager --generate-inhibit --from-live "
        f"--equal {','.join(equal) if equal else 'alertname,service'}",
        "```",
    ]
    if tune.get("live_error"):
        lines.insert(
            6, f"- **live_error**: `{tune.get('live_error')}` (static fallback used)"
        )
    return "\n".join(lines) + "\n"


def write_artifacts(tune: Dict[str, Any], *, base: str = "metadata") -> Dict[str, str]:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    tune_path = root / "inhibit_equal_tune.json"
    comment_path = root / "inhibit_equal_pr_comment.md"
    tune_path.write_text(
        json.dumps(tune, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    comment = build_inhibit_equal_comment(tune)
    comment_path.write_text(comment, encoding="utf-8")
    return {"tune": str(tune_path), "comment": str(comment_path)}


def _resolve_pr_number() -> Optional[str]:
    from scripts.ci_lora_pr_status import _resolve_pr_number as _lora_pr

    return _lora_pr()


def post_pr_comment(tune: Dict[str, Any], *, comment_path: str) -> Dict[str, Any]:
    from scripts.ci_lora_pr_status import upsert_pr_comment

    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    event = (os.environ.get("GITHUB_EVENT_NAME") or "").strip()
    pr_number = _resolve_pr_number()
    if not token or event not in {"pull_request", "pull_request_target"} or not pr_number:
        return {"posted": False, "reason": "not_pr_or_missing"}
    if os.environ.get("INHIBIT_EQUAL_PR_COMMENT_POST", "1").lower() in {
        "0",
        "false",
        "no",
    }:
        return {"posted": False, "reason": "disabled"}
    body = ""
    if os.path.isfile(comment_path):
        body = Path(comment_path).read_text(encoding="utf-8")
    if not body.strip():
        body = build_inhibit_equal_comment(tune)
    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if not repo:
        return {"posted": False, "reason": "missing_repo"}
    result = upsert_pr_comment(
        repo=repo,
        pr_number=pr_number,
        body=body,
        marker=INHIBIT_EQUAL_PR_COMMENT_MARKER,
        token=token,
    )
    result["pr_number"] = pr_number
    return result


def main() -> int:
    from rag.alertmanager_ops import tune_inhibit_equal_from_live

    paths = None
    raw_paths = os.environ.get("INHIBIT_EQUAL_ALERTING_PATHS", "").strip()
    if raw_paths:
        paths = [p.strip() for p in raw_paths.split(",") if p.strip()]
    tune = tune_inhibit_equal_from_live(
        paths=paths,
        include_static=True,
        min_count=1,
    )
    written = write_artifacts(tune)
    comment = post_pr_comment(tune, comment_path=written["comment"])
    out = {"tune": tune, "written": written, "pr_comment": comment}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0 if tune.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
