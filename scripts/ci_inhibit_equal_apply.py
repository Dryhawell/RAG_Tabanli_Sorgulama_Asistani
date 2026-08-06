#!/usr/bin/env python3
"""CI: inhibit equal apply on main (diff + amtool/structural gate + optional commit)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def main() -> int:
    from rag.alertmanager_ops import apply_inhibit_equal_with_gate

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
    if report.get("applied") and not dry:
        equal = (report.get("generated") or {}).get("equal") or []
        msg = (
            "chore(alertmanager): auto-apply inhibit equal labels "
            f"[{', '.join(equal)}]"
        )
        report["git"] = maybe_commit_and_push(out_path, message=msg)

    meta = ROOT / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "inhibit_equal_apply.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    diff_text = (report.get("diff") or {}).get("unified_diff") or ""
    (meta / "inhibit_equal.diff").write_text(diff_text, encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not report.get("ok"):
        return 1
    git_result = report.get("git") or {}
    if git_result and not git_result.get("ok") and not git_result.get("skipped"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
