"""Alertmanager webhook → dual-write catch-up / shadow-compare auto-trigger."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


CATCHUP_ALERTNAMES: Set[str] = {
    "RagDualWriteLagShadowBurn",
    "RagDualWriteLagHigh",
    "RagDualWriteShadowOverlapLow",
}

# alertname → actions
_ALERT_ACTIONS = {
    "RagDualWriteLagShadowBurn": ("catch_up", "shadow_compare"),
    "RagDualWriteLagHigh": ("catch_up",),
    "RagDualWriteShadowOverlapLow": ("shadow_compare",),
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dual_write_webhook_state_path(base: Optional[str] = None) -> str:
    env = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_STATE", "").strip()
    if env:
        return env
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    return os.path.join(root, "dual_write_webhook_state.json")


def load_dual_write_webhook_state(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or dual_write_webhook_state_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_dual_write_webhook_state(
    state: Dict[str, Any], *, path: Optional[str] = None
) -> None:
    p = path or dual_write_webhook_state_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def webhook_cooldown_sec() -> float:
    raw = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "900").strip()
    try:
        return max(0.0, float(raw or 900))
    except Exception:
        return 900.0


def extract_firing_dual_write_alerts(
    payload: Dict[str, Any],
    *,
    alertnames: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Alertmanager webhook body → firing dual-write alerts."""
    wanted = {a for a in (alertnames or CATCHUP_ALERTNAMES)}
    out: List[Dict[str, Any]] = []
    # AM may wrap as {status, alerts:[...]} or send a single alert
    alerts = payload.get("alerts")
    if alerts is None and isinstance(payload.get("labels"), dict):
        alerts = [payload]
    if not isinstance(alerts, list):
        return out
    for item in alerts:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or payload.get("status") or "").lower()
        if status and status not in {"firing", "active"}:
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        name = str(labels.get("alertname") or "").strip()
        if name not in wanted:
            continue
        svc = str(labels.get("service") or "").strip()
        if svc and svc not in {"rag-ingest", "rag"}:
            # still allow if alertname matches known set
            pass
        out.append(
            {
                "alertname": name,
                "labels": dict(labels),
                "annotations": item.get("annotations")
                if isinstance(item.get("annotations"), dict)
                else {},
                "status": status or "firing",
            }
        )
    return out


def plan_actions_for_alerts(alerts: Sequence[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    seen = set()
    for alert in alerts:
        name = str(alert.get("alertname") or "")
        for act in _ALERT_ACTIONS.get(name, ()):
            if act not in seen:
                seen.add(act)
                actions.append(act)
    return actions


def run_dual_write_catch_up(*, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"ok": True, "dry_run": True, "action": "catch_up"}
    try:
        from rag.cli import DOCSTORE_PATH, INDEX_PATH
        from rag.store import (
            DualWriteIndex,
            create_index,
            dual_write_backend,
            dual_write_catch_up,
            load_index,
        )

        index = load_index(INDEX_PATH, DOCSTORE_PATH)
        if not isinstance(index, DualWriteIndex):
            secondary = dual_write_backend() or "qdrant"
            try:
                sec = create_index(
                    dim=getattr(index, "dim", 384),
                    embedding_model=getattr(index, "embedding_model", None),
                    backend=secondary,
                    dual_write="",
                )
                index = DualWriteIndex(index, sec, secondary_backend=secondary)
            except Exception as exc:
                return {
                    "ok": False,
                    "action": "catch_up",
                    "error": "not_dual_write",
                    "detail": str(exc),
                }
        report = dual_write_catch_up(index)
        report["action"] = "catch_up"
        return report
    except Exception as exc:
        return {
            "ok": False,
            "action": "catch_up",
            "error": type(exc).__name__,
            "detail": str(exc),
        }


def run_dual_write_shadow_compare(*, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"ok": True, "dry_run": True, "action": "shadow_compare"}
    try:
        import subprocess
        import sys

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag.cli",
                "migrate-vector",
                "--shadow-compare",
                "--auto-catch-up-on-fail",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=float(os.environ.get("RAG_DUAL_WRITE_WEBHOOK_TIMEOUT_SEC", "120") or 120),
        )
        parsed: Any = None
        try:
            parsed = json.loads(proc.stdout or "")
        except Exception:
            parsed = None
        err = None
        if isinstance(parsed, dict) and parsed.get("error") == "not_dual_write":
            err = "not_dual_write"
        return {
            "ok": proc.returncode == 0,
            "action": "shadow_compare",
            "returncode": proc.returncode,
            "report": parsed,
            "error": err,
            "stdout": (proc.stdout or "")[:2000],
            "stderr": (proc.stderr or "")[:500],
        }
    except Exception as exc:
        return {
            "ok": False,
            "action": "shadow_compare",
            "error": type(exc).__name__,
            "detail": str(exc),
        }


_WORKFLOW_BY_ACTION = {
    "catch_up": "dual-write-catch-up.yml",
    "shadow_compare": "dual-write-shadow-compare.yml",
}


def github_dispatch_enabled() -> bool:
    raw = os.environ.get("RAG_DUAL_WRITE_GH_DISPATCH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def trigger_github_workflow_dispatch(
    workflow: str,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    ref: Optional[str] = None,
    token: Optional[str] = None,
    repo: Optional[str] = None,
) -> Dict[str, Any]:
    """GitHub Actions workflow_dispatch (GH_PAT / GITHUB_TOKEN)."""
    import subprocess
    from urllib import error, request

    tok = (
        (token or "").strip()
        or os.environ.get("GH_PAT", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    if not tok:
        return {"ok": False, "error": "token_missing"}
    slug = (repo or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if "/" not in slug:
        return {"ok": False, "error": "repo_missing"}
    branch = (
        (ref or "").strip()
        or os.environ.get("RAG_DUAL_WRITE_GH_REF", "").strip()
        or os.environ.get("GITHUB_REF_NAME", "").strip()
        or "main"
    )
    wf = (workflow or "").strip()
    if not wf:
        return {"ok": False, "error": "workflow_missing"}
    payload_inputs = inputs or {}

    # Prefer gh CLI
    env = os.environ.copy()
    env["GH_TOKEN"] = tok
    env["GITHUB_TOKEN"] = tok
    cmd = [
        "gh",
        "workflow",
        "run",
        wf,
        "--repo",
        slug,
        "--ref",
        branch,
    ]
    for k, v in payload_inputs.items():
        cmd.extend(["-f", f"{k}={v}"])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        if proc.returncode == 0:
            return {
                "ok": True,
                "method": "gh",
                "workflow": wf,
                "ref": branch,
                "repo": slug,
                "inputs": payload_inputs,
            }
        gh_err = (proc.stderr or proc.stdout or "").strip()[:500]
    except FileNotFoundError:
        gh_err = "gh_not_found"
    except Exception as exc:
        gh_err = f"{type(exc).__name__}:{exc}"

    # REST fallback
    url = (
        f"https://api.github.com/repos/{slug}/actions/workflows/"
        f"{wf}/dispatches"
    )
    body = json.dumps({"ref": branch, "inputs": payload_inputs}).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return {
                "ok": 200 <= int(code) < 300,
                "method": "rest",
                "status": int(code),
                "workflow": wf,
                "ref": branch,
                "repo": slug,
                "inputs": payload_inputs,
                "gh_error": gh_err,
            }
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = str(exc)
        return {
            "ok": False,
            "method": "rest",
            "status": int(exc.code),
            "error": detail or str(exc),
            "workflow": wf,
            "gh_error": gh_err,
        }
    except Exception as exc:
        return {
            "ok": False,
            "method": "rest",
            "error": type(exc).__name__,
            "detail": str(exc),
            "gh_error": gh_err,
            "workflow": wf,
        }


def maybe_github_dispatch_fallback(result: Dict[str, Any]) -> Dict[str, Any]:
    """In-process fail (özellikle not_dual_write) → workflow_dispatch."""
    if not github_dispatch_enabled():
        result["github_fallback"] = {"ok": False, "skipped": True, "reason": "disabled"}
        return result
    if result.get("dry_run"):
        result["github_fallback"] = {"ok": True, "skipped": True, "reason": "dry_run"}
        return result
    should = (not result.get("ok")) and (
        result.get("error") == "not_dual_write"
        or os.environ.get("RAG_DUAL_WRITE_GH_DISPATCH_ON_ANY_FAIL", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not should:
        result["github_fallback"] = {"ok": False, "skipped": True, "reason": "not_needed"}
        return result
    action = str(result.get("action") or "")
    workflow = _WORKFLOW_BY_ACTION.get(action)
    if not workflow:
        result["github_fallback"] = {
            "ok": False,
            "skipped": True,
            "reason": "unknown_action",
        }
        return result
    inputs: Dict[str, Any]
    if action == "catch_up":
        inputs = {
            "force_cutover_check": "true",
            "auto_cutover": "false",
        }
    else:
        inputs = {
            "sample": "16",
            "top_k": "6",
            "min_overlap": os.environ.get("RAG_DUAL_WRITE_SHADOW_MIN_OVERLAP", "1.0"),
            "auto_catch_up_on_fail": "true",
        }
    dispatched = trigger_github_workflow_dispatch(workflow, inputs=inputs)
    result["github_fallback"] = dispatched
    return result


def handle_dual_write_alertmanager_webhook(
    payload: Dict[str, Any],
    *,
    dry_run: bool = False,
    force: bool = False,
    state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Firing dual-write alert → catch-up / shadow-compare (debounce'lu)."""
    alerts = extract_firing_dual_write_alerts(payload)
    if not alerts:
        return {
            "ok": True,
            "skipped": True,
            "reason": "no_matching_firing_alerts",
            "alerts": [],
        }

    actions = plan_actions_for_alerts(alerts)
    st_path = state_path or dual_write_webhook_state_path()
    state = load_dual_write_webhook_state(st_path)
    now = time.time()
    cooldown = webhook_cooldown_sec()
    last = float(state.get("last_trigger_ts") or 0)
    if not force and cooldown > 0 and last and (now - last) < cooldown:
        return {
            "ok": True,
            "skipped": True,
            "reason": "cooldown",
            "retry_after_sec": max(0, int(cooldown - (now - last))),
            "alerts": [a.get("alertname") for a in alerts],
            "planned_actions": actions,
        }

    results: List[Dict[str, Any]] = []
    for act in actions:
        if act == "catch_up":
            row = run_dual_write_catch_up(dry_run=dry_run)
        elif act == "shadow_compare":
            row = run_dual_write_shadow_compare(dry_run=dry_run)
        else:
            continue
        results.append(maybe_github_dispatch_fallback(row))

    ok = all(r.get("ok") or (r.get("github_fallback") or {}).get("ok") for r in results) if results else True
    if not dry_run:
        state.update(
            {
                "last_trigger_ts": now,
                "last_trigger_at": _utcnow_iso(),
                "last_alerts": [a.get("alertname") for a in alerts],
                "last_actions": actions,
                "last_ok": ok,
            }
        )
        try:
            save_dual_write_webhook_state(state, path=st_path)
        except Exception:
            pass

    return {
        "ok": ok,
        "skipped": False,
        "dry_run": dry_run,
        "alerts": [a.get("alertname") for a in alerts],
        "planned_actions": actions,
        "results": results,
    }


def handle_alertmanager_webhook_http(
    body: bytes,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """HTTP adapter for collab_http POST /alertmanager."""
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except Exception:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_json"}).encode("utf-8"),
        )
    if not isinstance(payload, dict):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "payload_not_object"}).encode("utf-8"),
        )

    # Optional shared token
    token = os.environ.get("RAG_ALERTMANAGER_WEBHOOK_TOKEN", "").strip()
    if token:
        hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        auth = hdrs.get("authorization", "")
        provided = hdrs.get("x-webhook-token", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        if provided != token:
            return (
                401,
                {"Content-Type": "application/json"},
                json.dumps({"ok": False, "error": "unauthorized"}).encode("utf-8"),
            )

    dry = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    report = handle_dual_write_alertmanager_webhook(payload, dry_run=dry)
    status = 200 if report.get("ok") else 500
    if report.get("skipped") and report.get("reason") == "cooldown":
        status = 200
    return (
        status,
        {"Content-Type": "application/json"},
        json.dumps(report, ensure_ascii=False, default=str).encode("utf-8"),
    )
