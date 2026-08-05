"""GitHub Actions secret güncelleme (Alertmanager Slack webhook) — GH_PAT gerekir."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from scripts.update_github_vapid_secrets import (
    _gh_secret_set,
    parse_repo_slug,
    put_actions_secret,
    put_environment_secret,
)


def update_alertmanager_github_secrets(
    *,
    slack_webhook: str,
    webhook_url: Optional[str] = None,
    dry_run: bool = False,
    token: Optional[str] = None,
    repo: Optional[str] = None,
    environment: Optional[str] = None,
) -> Dict[str, Any]:
    """RAG_ALERTMANAGER_SLACK_WEBHOOK (+ opsiyonel WEBHOOK_URL) Actions secret."""
    tok = (
        (token or "").strip()
        or os.environ.get("GH_PAT", "").strip()
        or os.environ.get("ALERTMANAGER_GH_PAT", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    slug = (repo or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    env_name = (
        (
            environment
            if environment is not None
            else os.environ.get("ALERTMANAGER_GITHUB_ENVIRONMENT", "")
        )
        or ""
    ).strip()
    result: Dict[str, Any] = {
        "dry_run": dry_run,
        "repo": slug or None,
        "environment": env_name or None,
        "updated": [],
        "failed": [],
        "skipped": [],
        "method": None,
    }
    wh = (slack_webhook or "").strip()
    if not wh:
        result["failed"].append("slack_webhook_empty")
        return result
    if not tok:
        result["failed"].append("no_token")
        return result
    if not slug:
        result["failed"].append("no_repo")
        return result

    pairs = [("RAG_ALERTMANAGER_SLACK_WEBHOOK", wh)]
    extra = (webhook_url or "").strip()
    if extra:
        pairs.append(("RAG_ALERTMANAGER_WEBHOOK_URL", extra))

    if dry_run:
        result["updated"] = [n for n, _ in pairs]
        result["method"] = "dry_run"
        return result

    all_ok = True
    for name, value in pairs:
        if _gh_secret_set(
            name,
            value,
            repo=slug,
            env_token=tok,
            environment=env_name or None,
        ):
            result["updated"].append(name)
            result["method"] = "gh"
        else:
            all_ok = False
            break
    if all_ok and result["updated"]:
        return result

    result["updated"] = []
    result["method"] = "api"
    try:
        owner, name = parse_repo_slug(slug)
    except ValueError:
        result["failed"].append("bad_repo")
        return result
    for secret_name, value in pairs:
        if env_name:
            ok = put_environment_secret(
                token=tok,
                owner=owner,
                repo=name,
                environment=env_name,
                name=secret_name,
                value=value,
            )
        else:
            ok = put_actions_secret(
                token=tok,
                owner=owner,
                repo=name,
                name=secret_name,
                value=value,
            )
        if ok:
            result["updated"].append(secret_name)
        else:
            result["failed"].append(secret_name)
    return result


def main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Update GitHub Alertmanager Actions secrets")
    p.add_argument(
        "--slack-webhook",
        default=os.environ.get("RAG_ALERTMANAGER_SLACK_WEBHOOK", ""),
    )
    p.add_argument(
        "--webhook-url",
        default=os.environ.get("RAG_ALERTMANAGER_WEBHOOK_URL", ""),
    )
    p.add_argument("--repo", default=None)
    p.add_argument(
        "--environment",
        "--github-environment",
        dest="environment",
        default=os.environ.get("ALERTMANAGER_GITHUB_ENVIRONMENT", ""),
        help="GitHub Environment adı",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--from-env-file", default=None, help="alertmanager.slack.env yolu")
    args = p.parse_args(argv)

    slack = args.slack_webhook
    webhook_url = args.webhook_url or None
    if args.from_env_file:
        with open(args.from_env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == "RAG_ALERTMANAGER_SLACK_WEBHOOK":
                    slack = v
                elif k == "RAG_ALERTMANAGER_WEBHOOK_URL":
                    webhook_url = v
    result = update_alertmanager_github_secrets(
        slack_webhook=slack,
        webhook_url=webhook_url,
        dry_run=args.dry_run,
        repo=args.repo,
        environment=args.environment or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
