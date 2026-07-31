#!/usr/bin/env python3
"""CI: LoRA rollback önerisinden PR status check + yorum artifact üretir."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _load_suggestion() -> dict:
    path = os.environ.get("LORA_ROLLBACK_JSON", "metadata/lora_rollback.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "suggestion" in data:
                return data.get("suggestion") or {}
            if "should_rollback" in data:
                return data
            if "rollback_suggestion" in data:
                return data.get("rollback_suggestion") or {}
    # Placeholder suggestion when no eval artifact
    from rag.lora_dp_eval import suggest_lora_rollback

    report = {
        "gate_ok": False,
        "lora_ok": False,
        "delta_ok": False,
        "delta_accuracy": -0.1,
        "regressions": [{"case_id": "ci-placeholder"}],
        "lora_dir": os.environ.get("LORA_DIR", "models/lora-dp-embed"),
    }
    return suggest_lora_rollback(report)


def _post_commit_status(status: dict) -> dict:
    """GITHUB_TOKEN varsa commit status oluşturur (opsiyonel)."""
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if not token or not sha or not repo:
        return {"posted": False, "reason": "missing_token_or_sha"}
    if os.environ.get("LORA_PR_STATUS_POST", "1").lower() in {"0", "false", "no"}:
        return {"posted": False, "reason": "disabled"}
    payload = {
        "state": status.get("state") or "success",
        "description": status.get("description") or "",
        "context": status.get("context") or "lora/rollback",
    }
    target = os.environ.get("LORA_PR_STATUS_TARGET_URL", "").strip()
    if target:
        payload["target_url"] = target
    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/statuses/{sha}",
                "-X",
                "POST",
                "--input",
                "-",
            ],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "GH_TOKEN": token},
        )
        return {
            "posted": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:500],
            "stderr": (proc.stderr or "")[:500],
        }
    except FileNotFoundError:
        return {"posted": False, "reason": "gh_not_found"}
    except Exception as exc:
        return {"posted": False, "reason": str(exc)}


def _post_pr_comment(status: dict) -> dict:
    """PR olayındaysa yorum dosyasını gh ile gönderir (opsiyonel)."""
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    event = (os.environ.get("GITHUB_EVENT_NAME") or "").strip()
    pr_number = (os.environ.get("LORA_PR_NUMBER") or "").strip()
    if not pr_number:
        # pull_request event payload
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        if event_path and os.path.isfile(event_path):
            try:
                with open(event_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                pr_number = str((ev.get("pull_request") or {}).get("number") or "")
            except Exception:
                pr_number = ""
    if not token or event not in {"pull_request", "pull_request_target"} or not pr_number:
        return {"posted": False, "reason": "not_pr_or_missing"}
    if os.environ.get("LORA_PR_COMMENT_POST", "1").lower() in {"0", "false", "no"}:
        return {"posted": False, "reason": "disabled"}
    body = status.get("comment_markdown") or ""
    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/{pr_number}/comments",
                "-X",
                "POST",
                "-f",
                f"body={body}",
            ],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "GH_TOKEN": token},
        )
        return {
            "posted": proc.returncode == 0,
            "returncode": proc.returncode,
            "pr_number": pr_number,
        }
    except FileNotFoundError:
        return {"posted": False, "reason": "gh_not_found"}
    except Exception as exc:
        return {"posted": False, "reason": str(exc)}


def main() -> int:
    from rag.lora_dp_eval import build_lora_pr_status, write_lora_pr_status

    suggestion = _load_suggestion()
    status = build_lora_pr_status(suggestion)
    written = write_lora_pr_status(status)
    post_status = _post_commit_status(status)
    post_comment = _post_pr_comment(status)
    out = {
        "status": {
            "state": status.get("state"),
            "should_rollback": status.get("should_rollback"),
            "reason": status.get("reason"),
        },
        "written": written,
        "commit_status": post_status,
        "pr_comment": post_comment,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    # Check failure: rollback önerisi varsa exit 1 (status check kırmızı)
    fail_on_rollback = os.environ.get("LORA_PR_STATUS_FAIL_ON_ROLLBACK", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if fail_on_rollback and status.get("should_rollback"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
