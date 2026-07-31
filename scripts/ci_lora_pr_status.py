#!/usr/bin/env python3
"""CI: LoRA rollback önerisinden PR status check + yorum artifact üretir."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional


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


def _gh_api(
    args: List[str],
    *,
    payload: Optional[dict] = None,
    token: str,
) -> Dict[str, Any]:
    cmd = ["gh", "api", *args]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "GH_TOKEN": token},
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except FileNotFoundError:
        return {"ok": False, "reason": "gh_not_found", "stdout": "", "stderr": ""}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "stdout": "", "stderr": ""}


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
    result = _gh_api(
        [f"repos/{repo}/statuses/{sha}", "-X", "POST", "--input", "-"],
        payload=payload,
        token=token,
    )
    return {
        "posted": bool(result.get("ok")),
        "returncode": result.get("returncode"),
        "stdout": (result.get("stdout") or "")[:500],
        "stderr": (result.get("stderr") or "")[:500],
        "reason": result.get("reason"),
    }


def _resolve_pr_number() -> str:
    pr_number = (os.environ.get("LORA_PR_NUMBER") or "").strip()
    if pr_number:
        return pr_number
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if event_path and os.path.isfile(event_path):
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            return str((ev.get("pull_request") or {}).get("number") or "")
        except Exception:
            return ""
    return ""


def find_bot_comment_id(
    comments: List[Dict[str, Any]],
    marker: str,
) -> Optional[int]:
    """Marker içeren mevcut bot yorumunun id'sini bulur."""
    for row in comments or []:
        body = str(row.get("body") or "")
        if marker and marker in body:
            try:
                return int(row.get("id"))
            except (TypeError, ValueError):
                continue
    return None


def upsert_pr_comment(
    *,
    repo: str,
    pr_number: str,
    body: str,
    marker: str,
    token: str,
    gh_api: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Marker ile mevcut yorumu PATCH eder; yoksa yeni POST."""
    api = gh_api or _gh_api
    listed = api(
        [f"repos/{repo}/issues/{pr_number}/comments", "--paginate"],
        token=token,
    )
    if not listed.get("ok") and listed.get("reason") == "gh_not_found":
        return {"posted": False, "updated": False, "reason": "gh_not_found"}
    existing_id = None
    if listed.get("ok"):
        try:
            comments = json.loads(listed.get("stdout") or "[]")
            if isinstance(comments, dict):
                comments = [comments]
            existing_id = find_bot_comment_id(comments or [], marker)
        except Exception:
            existing_id = None
    if existing_id is not None:
        patched = api(
            [
                f"repos/{repo}/issues/comments/{existing_id}",
                "-X",
                "PATCH",
                "--input",
                "-",
            ],
            payload={"body": body},
            token=token,
        )
        return {
            "posted": bool(patched.get("ok")),
            "updated": True,
            "comment_id": existing_id,
            "returncode": patched.get("returncode"),
            "stderr": (patched.get("stderr") or "")[:500],
            "reason": patched.get("reason"),
        }
    created = api(
        [
            f"repos/{repo}/issues/{pr_number}/comments",
            "-X",
            "POST",
            "--input",
            "-",
        ],
        payload={"body": body},
        token=token,
    )
    comment_id = None
    if created.get("ok"):
        try:
            comment_id = json.loads(created.get("stdout") or "{}").get("id")
        except Exception:
            comment_id = None
    return {
        "posted": bool(created.get("ok")),
        "updated": False,
        "comment_id": comment_id,
        "returncode": created.get("returncode"),
        "stderr": (created.get("stderr") or "")[:500],
        "reason": created.get("reason"),
    }


def _post_pr_comment(status: dict) -> dict:
    """PR olayındaysa bot yorumunu upsert eder (opsiyonel)."""
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    event = (os.environ.get("GITHUB_EVENT_NAME") or "").strip()
    pr_number = _resolve_pr_number()
    if not token or event not in {"pull_request", "pull_request_target"} or not pr_number:
        return {"posted": False, "reason": "not_pr_or_missing"}
    if os.environ.get("LORA_PR_COMMENT_POST", "1").lower() in {"0", "false", "no"}:
        return {"posted": False, "reason": "disabled"}
    body = status.get("comment_markdown") or ""
    marker = status.get("comment_marker") or "<!-- lora-rollback-bot -->"
    # Dosyadan oku (CI artifact ile hizalı)
    comment_path = os.environ.get("LORA_PR_COMMENT_PATH", "metadata/lora_pr_comment.md")
    if os.path.isfile(comment_path):
        try:
            with open(comment_path, "r", encoding="utf-8") as f:
                file_body = f.read()
            if file_body.strip():
                body = file_body
        except Exception:
            pass
    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if not repo:
        return {"posted": False, "reason": "missing_repo"}
    result = upsert_pr_comment(
        repo=repo,
        pr_number=pr_number,
        body=body,
        marker=marker,
        token=token,
    )
    result["pr_number"] = pr_number
    return result


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
