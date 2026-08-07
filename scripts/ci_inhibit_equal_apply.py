#!/usr/bin/env python3
"""CI: inhibit equal apply on main (diff + amtool/structural gate + optional commit).

PR'larda dry-run diff'i `<!-- inhibit-equal-apply-bot -->` marker'ı ile
yorum olarak upsert edilir (apply yok — sadece preview).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER = "<!-- inhibit-equal-apply-bot -->"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_dry_run() -> bool:
    """auto: dry-run unless refs/heads/main; explicit 1/0 overrides."""
    mode = os.environ.get("INHIBIT_EQUAL_APPLY", "auto").strip().lower() or "auto"
    if mode in {"1", "true", "yes", "on", "apply"}:
        return False
    if mode in {"0", "false", "no", "off", "dry-run", "diff"}:
        return True
    ref = (os.environ.get("GITHUB_REF", "") or "").strip()
    return ref != "refs/heads/main"


def resolve_apply_allowed() -> Dict[str, Any]:
    """Green-CI gate: require CI_TEST_RESULT=success when REQUIRE_GREEN=1."""
    require = os.environ.get("INHIBIT_EQUAL_APPLY_REQUIRE_GREEN", "0").strip().lower()
    if require in {"0", "false", "no", "off", ""}:
        return {"ok": True, "skipped": True, "reason": "not_required"}
    result = (os.environ.get("CI_TEST_RESULT") or "success").strip().lower()
    if result == "success":
        return {"ok": True, "ci_test_result": result}
    return {
        "ok": False,
        "ci_test_result": result,
        "reason": "ci_not_green",
    }


def maybe_commit_and_push(path: str, *, message: str) -> Dict[str, Any]:
    """Force-add gitignored inhibit file and push when enabled."""
    if os.environ.get("INHIBIT_EQUAL_GIT_COMMIT", "1").lower() in {
        "0",
        "false",
        "no",
    }:
        return {"ok": True, "skipped": True, "reason": "commit_disabled"}

    _git("config", "user.email", "github-actions[bot]@users.noreply.github.com")
    _git("config", "user.name", "github-actions[bot]")
    add = _git("add", "-f", path)
    if add.returncode != 0:
        return {
            "ok": False,
            "error": "git_add_failed",
            "stderr": (add.stderr or "")[:500],
        }
    status = _git("status", "--porcelain", "--", path)
    if not (status.stdout or "").strip():
        return {"ok": True, "skipped": True, "reason": "nothing_to_commit"}

    commit = _git("commit", "-m", message)
    if commit.returncode != 0:
        return {
            "ok": False,
            "error": "git_commit_failed",
            "stderr": (commit.stderr or "")[:500],
        }
    push = _git("push")
    return {
        "ok": push.returncode == 0,
        "committed": True,
        "pushed": push.returncode == 0,
        "stderr": (push.stderr or "")[:500] if push.returncode != 0 else None,
    }


def build_inhibit_equal_apply_comment(report: Dict[str, Any]) -> str:
    """Dry-run apply diff preview for PR comments (diff only)."""
    equal = (report.get("generated") or {}).get("equal") or []
    if not isinstance(equal, list):
        equal = [str(equal)]
    diff = report.get("diff") or {}
    changed = bool(diff.get("changed"))
    unified = str(diff.get("unified_diff") or "").strip()
    reason = str(report.get("reason") or "")
    lines = [
        INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER,
        "### Alertmanager inhibit equal apply (dry-run preview)",
        "",
        f"- **ok**: `{report.get('ok')}`",
        f"- **dry_run**: `{report.get('dry_run')}`",
        f"- **changed**: `{changed}`",
        f"- **reason**: `{reason}`",
        f"- **equal**: `{', '.join(equal) if equal else '—'}`",
        f"- **output**: `{report.get('output')}`",
        "",
        "Artifacts: `metadata/inhibit_equal_apply.json` · "
        "`metadata/inhibit_equal.diff`",
        "",
    ]
    if not changed:
        lines.append("_No inhibit diff vs current file._")
    else:
        # Cap diff body so PR comments stay readable
        max_chars = int(os.environ.get("INHIBIT_EQUAL_APPLY_DIFF_MAX_CHARS", "12000") or 12000)
        body = unified
        truncated = False
        if max_chars > 0 and len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n… (truncated)"
            truncated = True
        lines.append("```diff")
        lines.append(body or "(empty diff)")
        lines.append("```")
        if truncated:
            lines.append("")
            lines.append("_Diff truncated for comment size; see CI artifact._")
    return "\n".join(lines) + "\n"


def write_apply_preview_artifacts(
    report: Dict[str, Any], *, base: str = "metadata"
) -> Dict[str, str]:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    comment_path = root / "inhibit_equal_apply_pr_comment.md"
    comment = build_inhibit_equal_apply_comment(report)
    comment_path.write_text(comment, encoding="utf-8")
    return {"comment": str(comment_path)}


def _resolve_pr_number() -> Optional[str]:
    from scripts.ci_lora_pr_status import _resolve_pr_number as _lora_pr

    return _lora_pr()


def post_apply_pr_preview_comment(
    report: Dict[str, Any], *, comment_path: Optional[str] = None
) -> Dict[str, Any]:
    """Upsert dry-run apply diff comment on PR (never applies)."""
    from scripts.ci_lora_pr_status import upsert_pr_comment

    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    event = (os.environ.get("GITHUB_EVENT_NAME") or "").strip()
    pr_number = _resolve_pr_number()
    enabled = os.environ.get("INHIBIT_EQUAL_APPLY_PR_COMMENT_POST", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not enabled:
        return {"posted": False, "reason": "disabled"}
    if not token or event not in {"pull_request", "pull_request_target"} or not pr_number:
        return {"posted": False, "reason": "not_pr_or_missing"}
    # Preview is for dry-run only
    if not report.get("dry_run"):
        return {"posted": False, "reason": "not_dry_run"}
    body = ""
    if comment_path and os.path.isfile(comment_path):
        body = Path(comment_path).read_text(encoding="utf-8")
    if not body.strip():
        body = build_inhibit_equal_apply_comment(report)
    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if not repo:
        return {"posted": False, "reason": "missing_repo"}
    result = upsert_pr_comment(
        repo=repo,
        pr_number=pr_number,
        body=body,
        marker=INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER,
        token=token,
    )
    result["pr_number"] = pr_number
    return result


def build_inhibit_equal_rollback_canary_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Slack payload for amtool regression rollback canary."""
    equal = (report.get("generated") or {}).get("equal") or []
    if not isinstance(equal, list):
        equal = [str(equal)]
    diff = str(((report.get("diff") or {}).get("unified_diff") or "")).strip()
    excerpt = diff[:1200] + ("…" if len(diff) > 1200 else "")
    post = report.get("post_check") or {}
    text = (
        "Inhibit equal apply *rolled back* (amtool regression)\n"
        f"equal=`{', '.join(equal) or '—'}` · backup=`{report.get('backup')}`"
    )
    blocks: list = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Inhibit equal apply rollback",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*reason*: `{report.get('reason')}`\n"
                    f"*equal*: `{', '.join(equal) or '—'}`\n"
                    f"*backup*: `{report.get('backup')}`\n"
                    f"*post_check*: `{post.get('method')}` ok=`{post.get('ok')}`"
                ),
            },
        },
    ]
    if excerpt:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```diff\n{excerpt}\n```"},
            }
        )
    return {"text": text, "blocks": blocks}


def notify_inhibit_equal_rollback_canary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Post canary Slack (+ optional PagerDuty) when apply rolled back."""
    if not report.get("rolled_back"):
        return {"ok": True, "skipped": True, "reason": "not_rolled_back"}
    if os.environ.get("INHIBIT_EQUAL_CANARY_NOTIFY", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return {"ok": True, "skipped": True, "reason": "disabled"}

    webhook = (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", "").strip()
    )
    pd_key = (
        os.environ.get("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", "").strip()
    )
    if not webhook and not pd_key:
        return {"ok": True, "skipped": True, "reason": "webhook_missing"}

    from rag.judge_alert import post_pagerduty, post_slack

    payload = build_inhibit_equal_rollback_canary_payload(report)
    slack_ok = False
    if webhook:
        slack_ok = bool(post_slack(webhook, payload))

    pd_ok = False
    if pd_key:
        equal = (report.get("generated") or {}).get("equal") or []
        pd_report = {
            "summary": {
                "ok": False,
                "mode": "inhibit_equal_rollback",
                "accuracy": 0.0,
                "passed": 0,
                "failed": 1,
                "total": 1,
                "equal": equal,
                "backup": report.get("backup"),
                "reason": report.get("reason"),
            },
            "mode": "inhibit_equal_rollback",
        }
        pd_ok = bool(
            post_pagerduty(
                routing_key=pd_key,
                report=pd_report,
                source="inhibit-equal-canary",
                severity=os.environ.get("INHIBIT_EQUAL_CANARY_PD_SEVERITY", "error")
                or "error",
            )
        )

    posted = slack_ok or pd_ok
    return {
        "ok": posted if (webhook or pd_key) else True,
        "skipped": False,
        "posted": posted,
        "slack": slack_ok if webhook else None,
        "pagerduty": pd_ok if pd_key else None,
    }


def main() -> int:
    from rag.alertmanager_ops import apply_inhibit_equal_with_gate

    green = resolve_apply_allowed()
    if not green.get("ok"):
        out = {
            "ok": False,
            "skipped": True,
            "reason": green.get("reason") or "ci_not_green",
            "green_gate": green,
            "dry_run": True,
            "applied": False,
        }
        meta = ROOT / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "inhibit_equal_apply.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return 1

    dry = resolve_dry_run()
    require_amtool = os.environ.get("INHIBIT_EQUAL_REQUIRE_AMTOOL", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    out_path = os.environ.get(
        "ALERTMANAGER_INHIBIT",
        str(ROOT / "grafana" / "inhibit_rules.generated.yml"),
    )
    report = apply_inhibit_equal_with_gate(
        output=out_path,
        from_live=True,
        require_amtool=require_amtool,
        dry_run=dry,
    )
    report["green_gate"] = green
    if report.get("applied") and not dry and not report.get("rolled_back"):
        equal = (report.get("generated") or {}).get("equal") or []
        msg = (
            "chore(alertmanager): auto-apply inhibit equal labels "
            f"[{', '.join(equal)}]"
        )
        report["git"] = maybe_commit_and_push(out_path, message=msg)
    elif report.get("rolled_back"):
        report["git"] = {
            "ok": True,
            "skipped": True,
            "reason": "rolled_back_amtool_regression",
        }
        report["canary_notify"] = notify_inhibit_equal_rollback_canary(report)

    meta = ROOT / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "inhibit_equal_apply.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    diff_text = (report.get("diff") or {}).get("unified_diff") or ""
    (meta / "inhibit_equal.diff").write_text(diff_text, encoding="utf-8")

    preview = write_apply_preview_artifacts(report, base=str(meta))
    report["pr_preview"] = post_apply_pr_preview_comment(
        report, comment_path=preview.get("comment")
    )
    report["pr_preview_artifacts"] = preview

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not report.get("ok"):
        return 1
    git_result = report.get("git") or {}
    if git_result and not git_result.get("ok") and not git_result.get("skipped"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
