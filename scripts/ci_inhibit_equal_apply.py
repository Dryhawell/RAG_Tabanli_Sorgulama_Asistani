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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def resolve_opsgenie_canary_targets() -> List[Dict[str, str]]:
    """Multi-region Opsgenie targets for inhibit-equal canary.

    Precedence:
    1. ``INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON`` → ``{"us":"k1","eu":"k2"}``
    2. Single key + ``INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS`` CSV (same key fan-out)
    3. Single ``INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY`` (+ optional region)
    """
    targets: List[Dict[str, str]] = []
    raw_json = (
        os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEYS_JSON", "").strip()
    )
    if raw_json:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for region, key in data.items():
                k = str(key or "").strip()
                if k:
                    targets.append({"region": str(region).strip().lower() or "us", "api_key": k})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    k = str(item.get("api_key") or item.get("key") or "").strip()
                    if k:
                        targets.append(
                            {
                                "region": str(item.get("region") or "us").strip().lower(),
                                "api_key": k,
                            }
                        )
    if targets:
        return targets

    og_key = (
        os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_OPSGENIE_API_KEY", "").strip()
    )
    if not og_key:
        return []
    regions_raw = (
        os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_OPSGENIE_REGIONS", "").strip()
        or os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_REGION", "").strip()
        or os.environ.get("RAG_OPSGENIE_REGION", "").strip()
        or "us"
    )
    regions = [p.strip().lower() for p in regions_raw.replace(";", ",").split(",") if p.strip()]
    if not regions:
        regions = ["us"]
    return [{"region": r, "api_key": og_key} for r in regions]


def inhibit_equal_canary_slack_state_path(base: Optional[str] = None) -> str:
    env = (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_STATE", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_STATE", "").strip()
    )
    if env:
        return env
    root = base or str(ROOT / "metadata")
    return str(Path(root) / "inhibit_equal_canary_slack.json")


def load_inhibit_equal_canary_slack_state(
    *, base: Optional[str] = None
) -> Dict[str, Any]:
    path = inhibit_equal_canary_slack_state_path(base=base)
    if not Path(path).is_file():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_inhibit_equal_canary_slack_state(
    *,
    thread_ts: Optional[str] = None,
    channel: Optional[str] = None,
    reason: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(inhibit_equal_canary_slack_state_path(base=base))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = load_inhibit_equal_canary_slack_state(base=base)
    if thread_ts:
        row["thread_ts"] = str(thread_ts).strip()
    if channel:
        row["channel"] = str(channel).strip()
    if reason:
        row["reason"] = str(reason).strip()
    row["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def resolve_inhibit_equal_canary_slack_bot_token() -> str:
    return (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_BOT_TOKEN", "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )


def resolve_inhibit_equal_canary_slack_channel() -> str:
    return (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_CHANNEL", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_CHANNEL", "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "").strip()
    )


def inhibit_equal_canary_slack_thread_reply_enabled() -> bool:
    raw = (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_THREAD_REPLY", "1").strip().lower()
        or "1"
    )
    return raw not in {"0", "false", "no", "off"}


def resolve_inhibit_equal_canary_pd_severity(
    report: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve PagerDuty severity for inhibit-equal canary from CI/env/reason.

    Precedence:
    1. ``INHIBIT_EQUAL_CANARY_PD_SEVERITY`` (explicit)
    2. ``INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON`` JSON map keyed by report.reason
    3. Reason defaults (``amtool_regression`` → critical)
    4. ``CI_TEST_RESULT`` bump (failure/cancelled → critical)
    5. fallback ``error``
    """
    allowed = {"info", "warning", "error", "critical"}
    explicit = (
        os.environ.get("INHIBIT_EQUAL_CANARY_PD_SEVERITY", "").strip().lower()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_PD_SEVERITY", "").strip().lower()
    )
    if explicit in allowed:
        return explicit

    reason = str((report or {}).get("reason") or "").strip()
    raw_map = (
        os.environ.get("INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_PD_SEVERITY_BY_REASON", "").strip()
    )
    if raw_map:
        try:
            data = json.loads(raw_map)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and reason:
            mapped = str(data.get(reason) or "").strip().lower()
            if mapped in allowed:
                return mapped

    defaults = {
        "amtool_regression": "critical",
        "amtool_failed": "critical",
        "post_check_failed": "error",
    }
    if reason in defaults:
        return defaults[reason]

    ci = (os.environ.get("CI_TEST_RESULT", "") or "").strip().lower()
    if ci in {"failure", "cancelled", "timed_out"}:
        return "critical"
    return "error"


def notify_inhibit_equal_rollback_canary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Post canary Slack (+ optional PagerDuty / multi-region Opsgenie) on rollback."""
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
    og_targets = resolve_opsgenie_canary_targets()
    bot_token_early = resolve_inhibit_equal_canary_slack_bot_token()
    channel_early = resolve_inhibit_equal_canary_slack_channel()
    if (
        not webhook
        and not pd_key
        and not og_targets
        and not (bot_token_early and channel_early)
    ):
        return {"ok": True, "skipped": True, "reason": "webhook_missing"}

    from rag.judge_alert import (
        post_opsgenie,
        post_pagerduty,
        post_slack,
        post_slack_thread_message,
    )

    payload = build_inhibit_equal_rollback_canary_payload(report)
    bot_token = resolve_inhibit_equal_canary_slack_bot_token()
    channel = resolve_inhibit_equal_canary_slack_channel()
    slack_ok = False
    slack_thread_ts: Optional[str] = None
    slack_via: Optional[str] = None
    if bot_token and channel:
        bot_post = post_slack_thread_message(
            text=str(payload.get("text") or "Inhibit equal apply rolled back"),
            channel_id=channel,
            thread_ts=None,
            bot_token=bot_token,
        )
        slack_ok = bool(bot_post.get("ok"))
        if slack_ok:
            slack_via = "bot"
            slack_thread_ts = str(bot_post.get("ts") or "").strip() or None
            if slack_thread_ts:
                save_inhibit_equal_canary_slack_state(
                    thread_ts=slack_thread_ts,
                    channel=str(bot_post.get("channel") or channel),
                    reason=str(report.get("reason") or "") or None,
                )
    if (not slack_ok) and webhook:
        slack_ok = bool(post_slack(webhook, payload))
        if slack_ok:
            slack_via = "webhook"

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

    pd_severity = resolve_inhibit_equal_canary_pd_severity(report)
    pd_ok = False
    if pd_key:
        pd_ok = bool(
            post_pagerduty(
                routing_key=pd_key,
                report=pd_report,
                source="inhibit-equal-canary",
                severity=pd_severity,
            )
        )

    og_by_region: Dict[str, bool] = {}
    priority = (
        os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_PRIORITY", "P2") or "P2"
    )
    for t in og_targets:
        region = t.get("region") or "us"
        ok = bool(
            post_opsgenie(
                api_key=t["api_key"],
                report=pd_report,
                source="inhibit-equal-canary",
                priority=priority,
                region=region,
            )
        )
        og_by_region[region] = ok
    og_ok = any(og_by_region.values()) if og_by_region else False

    has_slack = bool(webhook or (bot_token and channel))
    posted = slack_ok or pd_ok or og_ok
    return {
        "ok": posted if (has_slack or pd_key or og_targets) else True,
        "skipped": False,
        "posted": posted,
        "slack": slack_ok if has_slack else None,
        "slack_via": slack_via,
        "slack_thread_ts": slack_thread_ts,
        "pagerduty": pd_ok if pd_key else None,
        "pagerduty_severity": pd_severity if pd_key else None,
        "opsgenie": og_ok if og_targets else None,
        "opsgenie_regions": og_by_region or None,
    }


def build_inhibit_equal_resolve_canary_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Slack payload when green apply succeeds (canary resolve)."""
    equal = (report.get("generated") or {}).get("equal") or []
    if not isinstance(equal, list):
        equal = [str(equal)]
    text = (
        "Inhibit equal apply *resolved* (green CI · canary close)\n"
        f"equal=`{', '.join(equal) or '—'}` · applied=`{report.get('applied')}`"
    )
    blocks: list = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Inhibit equal apply resolved",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*status*: green apply succeeded\n"
                    f"*equal*: `{', '.join(equal) or '—'}`\n"
                    f"*backup*: `{report.get('backup') or '—'}`\n"
                    f"*git*: `{(report.get('git') or {}).get('ok')}`"
                ),
            },
        },
    ]
    return {"text": text, "blocks": blocks}


def notify_inhibit_equal_close_on_green(report: Dict[str, Any]) -> Dict[str, Any]:
    """Close Opsgenie (+ PD resolve + Slack resolve) after successful green apply."""
    if not report.get("applied") or report.get("rolled_back"):
        return {"ok": True, "skipped": True, "reason": "not_applied_green"}
    if os.environ.get("INHIBIT_EQUAL_CANARY_NOTIFY", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return {"ok": True, "skipped": True, "reason": "disabled"}
    if os.environ.get("INHIBIT_EQUAL_CANARY_CLOSE_ON_GREEN", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return {"ok": True, "skipped": True, "reason": "close_disabled"}

    webhook = (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_WEBHOOK", "").strip()
    )
    pd_key = (
        os.environ.get("INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_PAGERDUTY_ROUTING_KEY", "").strip()
    )
    og_targets = resolve_opsgenie_canary_targets()
    state_early = load_inhibit_equal_canary_slack_state()
    thread_early = (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", "").strip()
        or str(state_early.get("thread_ts") or "").strip()
    )
    bot_early = resolve_inhibit_equal_canary_slack_bot_token()
    if (
        not webhook
        and not pd_key
        and not og_targets
        and not (bot_early and thread_early)
    ):
        return {"ok": True, "skipped": True, "reason": "targets_missing"}

    from rag.judge_alert import (
        post_opsgenie_close,
        post_pagerduty,
        post_slack,
        post_slack_thread_message,
    )

    equal = (report.get("generated") or {}).get("equal") or []
    pd_report = {
        "summary": {
            "ok": True,
            "mode": "inhibit_equal_apply",
            "accuracy": 1.0,
            "passed": 1,
            "failed": 0,
            "total": 1,
            "equal": equal,
        },
        "mode": "inhibit_equal_apply",
    }

    resolve_payload = build_inhibit_equal_resolve_canary_payload(report)
    bot_token = resolve_inhibit_equal_canary_slack_bot_token()
    state = load_inhibit_equal_canary_slack_state()
    thread_ts = (
        os.environ.get("INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_SLACK_THREAD_TS", "").strip()
        or str(state.get("thread_ts") or "").strip()
    )
    channel = (
        resolve_inhibit_equal_canary_slack_channel()
        or str(state.get("channel") or "").strip()
    )
    slack_ok = False
    slack_thread_reply = False
    slack_via: Optional[str] = None
    if (
        inhibit_equal_canary_slack_thread_reply_enabled()
        and bot_token
        and thread_ts
    ):
        reply = post_slack_thread_message(
            text=str(resolve_payload.get("text") or "Inhibit equal apply resolved"),
            channel_id=channel or None,
            thread_ts=thread_ts,
            bot_token=bot_token,
        )
        slack_ok = bool(reply.get("ok"))
        if slack_ok:
            slack_thread_reply = True
            slack_via = "thread"
    if (not slack_ok) and webhook:
        slack_ok = bool(post_slack(webhook, resolve_payload))
        if slack_ok:
            slack_via = "webhook"

    pd_ok = False
    if pd_key:
        pd_ok = bool(
            post_pagerduty(
                routing_key=pd_key,
                report=pd_report,
                source="inhibit-equal-canary",
                event_action="resolve",
            )
        )

    og_by_region: Dict[str, bool] = {}
    for t in og_targets:
        region = t.get("region") or "us"
        ok = bool(
            post_opsgenie_close(
                api_key=t["api_key"],
                source="inhibit-equal-canary",
                region=region,
            )
        )
        og_by_region[region] = ok
    og_ok = any(og_by_region.values()) if og_by_region else False
    has_slack = bool(webhook or slack_thread_reply or (bot_token and thread_ts))
    closed = slack_ok or pd_ok or og_ok
    return {
        "ok": closed if (has_slack or pd_key or og_targets) else True,
        "skipped": False,
        "closed": closed,
        "slack": slack_ok if has_slack else None,
        "slack_via": slack_via,
        "slack_thread_reply": slack_thread_reply,
        "slack_thread_ts": thread_ts or None,
        "pagerduty": pd_ok if pd_key else None,
        "opsgenie": og_ok if og_targets else None,
        "opsgenie_regions": og_by_region or None,
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
        canary_state_path = Path(inhibit_equal_canary_slack_state_path(base=str(meta)))
        if not canary_state_path.is_file():
            canary_state_path.write_text("{}\n", encoding="utf-8")
        out["canary_slack_state_path"] = str(canary_state_path)
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
        report["canary_resolve"] = notify_inhibit_equal_close_on_green(report)
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

    # Always emit canary Slack thread-state artifact (empty {} when unused).
    canary_state_path = Path(inhibit_equal_canary_slack_state_path(base=str(meta)))
    if not canary_state_path.is_file():
        canary_state_path.parent.mkdir(parents=True, exist_ok=True)
        canary_state_path.write_text("{}\n", encoding="utf-8")
    report["canary_slack_state_path"] = str(canary_state_path)
    report["canary_slack_state"] = load_inhibit_equal_canary_slack_state(base=str(meta))

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
