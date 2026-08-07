"""Alertmanager webhook → dual-write catch-up / shadow-compare auto-trigger."""

from __future__ import annotations

import hashlib
import hmac
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

WEBHOOK_SIG_VERSION = "v0"

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


def dual_write_webhook_dlq_path(base: Optional[str] = None) -> str:
    env = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_DLQ", "").strip()
    if env:
        return env
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    return os.path.join(root, "dual_write_webhook_dlq.jsonl")


def dual_write_webhook_dlq_quarantine_path(base: Optional[str] = None) -> str:
    env = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_DLQ_QUARANTINE", "").strip()
    if env:
        return env
    primary = dual_write_webhook_dlq_path(base=base)
    if primary.endswith(".jsonl"):
        return primary[:-6] + "_quarantine.jsonl"
    return primary + ".quarantine.jsonl"


def enqueue_dual_write_dlq(
    *,
    payload: Dict[str, Any],
    report: Dict[str, Any],
    reason: str = "failed",
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Başarısız webhook tetiklerini JSONL DLQ'ya yaz."""
    import hashlib

    out = path or dual_write_webhook_dlq_path()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    entry = {
        "ts": _utcnow_iso(),
        "reason": reason,
        "payload_digest": digest,
        "alerts": report.get("alerts") or [],
        "planned_actions": report.get("planned_actions") or [],
        "results": report.get("results") or [],
        "ok": report.get("ok"),
        "payload": payload,
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return {"ok": True, "path": out, "digest": digest, "reason": reason}


def read_dual_write_dlq(
    *,
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    p = path or dual_write_webhook_dlq_path()
    if not os.path.isfile(p):
        return []
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    if limit is not None:
        rows = rows[-max(0, int(limit)) :]
    return rows


def dual_write_dlq_depth(*, path: Optional[str] = None) -> int:
    return len(read_dual_write_dlq(path=path))


def read_dual_write_dlq_quarantine(
    *,
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return read_dual_write_dlq(
        path=path or dual_write_webhook_dlq_quarantine_path(), limit=limit
    )


def _dlq_quarantine_after() -> int:
    try:
        return max(1, int(os.environ.get("RAG_DUAL_WRITE_DLQ_QUARANTINE_AFTER", "3") or 3))
    except Exception:
        return 3


def _dlq_replay_max_per_run() -> Optional[int]:
    raw = os.environ.get("RAG_DUAL_WRITE_DLQ_REPLAY_MAX_PER_RUN", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except Exception:
        return None


def _dlq_replay_max_per_hour() -> Optional[int]:
    raw = os.environ.get("RAG_DUAL_WRITE_DLQ_REPLAY_MAX_PER_HOUR", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except Exception:
        return None


def check_dual_write_dlq_replay_budget(
    *,
    state_path: Optional[str] = None,
    want: int = 1,
) -> Dict[str, Any]:
    """Enforce per-run / per-hour auto-replay budgets (state-backed)."""
    max_run = _dlq_replay_max_per_run()
    max_hour = _dlq_replay_max_per_hour()
    if max_run is None and max_hour is None:
        return {
            "ok": True,
            "allowed": int(want),
            "max_per_run": None,
            "max_per_hour": None,
            "used_run": 0,
            "used_hour": 0,
            "budget_exhausted": False,
        }
    st = load_dual_write_webhook_state(path=state_path)
    budget = (
        st.get("dlq_replay_budget")
        if isinstance(st.get("dlq_replay_budget"), dict)
        else {}
    )
    now = time.time()
    window_start = float(budget.get("hour_window_start") or 0) or 0.0
    used_hour = int(budget.get("hour_count") or 0)
    if not window_start or now - window_start >= 3600.0:
        window_start = now
        used_hour = 0
    used_run = int(budget.get("run_count") or 0)
    run_id = os.environ.get("RAG_DUAL_WRITE_DLQ_REPLAY_RUN_ID", "").strip() or None
    prev_run = str(budget.get("run_id") or "") or None
    if run_id is not None and run_id != prev_run:
        used_run = 0
        prev_run = run_id
    allowed = int(want)
    if max_run is not None:
        allowed = min(allowed, max(0, max_run - used_run))
    if max_hour is not None:
        allowed = min(allowed, max(0, max_hour - used_hour))
    return {
        "ok": allowed > 0 or int(want) <= 0,
        "allowed": allowed,
        "max_per_run": max_run,
        "max_per_hour": max_hour,
        "used_run": used_run,
        "used_hour": used_hour,
        "hour_window_start": window_start,
        "run_id": prev_run,
        "budget_exhausted": allowed <= 0 and int(want) > 0,
    }


def record_dual_write_dlq_replay_budget(
    count: int,
    *,
    state_path: Optional[str] = None,
    budget_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = max(0, int(count))
    snap = budget_snapshot or check_dual_write_dlq_replay_budget(
        state_path=state_path, want=n
    )
    st = load_dual_write_webhook_state(path=state_path)
    budget = {
        "hour_window_start": snap.get("hour_window_start") or time.time(),
        "hour_count": int(snap.get("used_hour") or 0) + n,
        "run_count": int(snap.get("used_run") or 0) + n,
        "run_id": snap.get("run_id"),
        "updated_at": _utcnow_iso(),
    }
    st["dlq_replay_budget"] = budget
    save_dual_write_webhook_state(st, path=state_path)
    return budget


def maybe_alert_dual_write_dlq_quarantine(
    *,
    entry: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    webhook = (
        os.environ.get("RAG_DUAL_WRITE_DLQ_QUARANTINE_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_DUAL_WRITE_DLQ_SLACK_WEBHOOK", "").strip()
    )
    if not webhook:
        return {"ok": True, "skipped": True, "reason": "webhook_missing"}
    row = entry or {}
    qpath = path or dual_write_webhook_dlq_quarantine_path()
    depth = len(read_dual_write_dlq_quarantine(path=qpath))
    text = (
        "*Dual-write DLQ quarantine*\n"
        f"digest=`{row.get('payload_digest')}` · attempts=`{row.get('attempts')}`\n"
        f"reason=`{row.get('quarantine_reason') or row.get('reason')}`\n"
        f"quarantine depth: *{depth}* · path=`{qpath}`"
    )
    try:
        from rag.judge_alert import post_slack

        ok = post_slack(webhook, {"text": text})
        return {"ok": bool(ok), "skipped": False, "posted": bool(ok), "depth": depth}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def enqueue_dual_write_dlq_quarantine(
    entry: Dict[str, Any],
    *,
    path: Optional[str] = None,
    reason: str = "replay_exhausted",
) -> Dict[str, Any]:
    """Append a DLQ entry to quarantine JSONL and Slack-notify channel."""
    out = path or dual_write_webhook_dlq_quarantine_path()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    row = dict(entry or {})
    row["quarantined_at"] = _utcnow_iso()
    row["quarantine_reason"] = reason
    row["attempts"] = int(row.get("attempts") or 0)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    notify = maybe_alert_dual_write_dlq_quarantine(entry=row, path=out)
    try:
        emit_dual_write_webhook_metric(
            result="dlq_quarantined",
            reason=reason,
            actions=1,
            dlq_depth=dual_write_dlq_depth(),
        )
    except Exception:
        pass
    return {
        "ok": True,
        "path": out,
        "digest": row.get("payload_digest"),
        "notify": notify,
    }



def _parse_dlq_ts(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        text = str(ts).strip()
        if not text:
            return None
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def prune_dual_write_dlq(
    *,
    path: Optional[str] = None,
    days: float = 7.0,
    dry_run: bool = False,
    notify: bool = False,
) -> Dict[str, Any]:
    """Age-based DLQ drop: ts < now-days olanları sil."""
    from datetime import datetime, timezone

    p = path or dual_write_webhook_dlq_path()
    rows = read_dual_write_dlq(path=p)
    if days is None or float(days) < 0:
        return {"ok": False, "error": "days_invalid", "path": p}
    cutoff = time.time() - float(days) * 86400.0
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    oldest_age_hours: Optional[float] = None
    now = time.time()
    for row in rows:
        ts = _parse_dlq_ts(row.get("ts"))
        if ts is None:
            kept.append(row)
            continue
        age_h = (now - ts) / 3600.0
        if oldest_age_hours is None or age_h > oldest_age_hours:
            oldest_age_hours = age_h
        if ts < cutoff:
            dropped.append(
                {
                    "ts": row.get("ts"),
                    "digest": row.get("payload_digest"),
                    "reason": row.get("reason"),
                    "age_hours": round(age_h, 2),
                }
            )
        else:
            kept.append(row)
    if not dry_run and dropped:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    report = {
        "ok": True,
        "path": p,
        "days": float(days),
        "before": len(rows),
        "after": len(kept),
        "removed": len(dropped),
        "dry_run": bool(dry_run),
        "oldest_age_hours": oldest_age_hours,
        "dropped": dropped[:50],
        "cutoff": datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(),
    }
    if notify:
        report["notify"] = maybe_alert_dual_write_dlq(
            depth=len(kept) if not dry_run else len(rows),
            removed=len(dropped),
            oldest_age_hours=oldest_age_hours,
            prune_report=report,
        )
    try:
        emit_dual_write_webhook_metric(
            result="dlq_pruned" if dropped and not dry_run else "dlq_prune_dry",
            reason="age_drop",
            actions=len(dropped),
            dlq_depth=len(kept) if not dry_run else len(rows),
        )
    except Exception:
        pass
    return report


def maybe_alert_dual_write_dlq(
    *,
    depth: Optional[int] = None,
    removed: int = 0,
    oldest_age_hours: Optional[float] = None,
    prune_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Slack alert on DLQ drop and/or depth/age thresholds."""
    webhook = (
        os.environ.get("RAG_DUAL_WRITE_DLQ_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_WEBHOOK", "").strip()
    )
    if not webhook:
        return {"ok": True, "skipped": True, "reason": "webhook_missing"}
    depth_n = int(depth if depth is not None else dual_write_dlq_depth())
    try:
        alert_depth = int(os.environ.get("RAG_DUAL_WRITE_DLQ_ALERT_DEPTH", "10") or 10)
    except Exception:
        alert_depth = 10
    try:
        alert_age_days = float(
            os.environ.get("RAG_DUAL_WRITE_DLQ_ALERT_AGE_DAYS", "3") or 3
        )
    except Exception:
        alert_age_days = 3.0
    reasons: List[str] = []
    if removed > 0:
        reasons.append(f"pruned `{removed}` aged entries")
    if depth_n >= alert_depth:
        reasons.append(f"depth `{depth_n}` ≥ `{alert_depth}`")
    if oldest_age_hours is not None and oldest_age_hours >= alert_age_days * 24:
        reasons.append(
            f"oldest `{oldest_age_hours:.1f}h` ≥ `{alert_age_days:g}d`"
        )
    if not reasons:
        return {"ok": True, "skipped": True, "reason": "below_threshold"}
    text = (
        "*Dual-write webhook DLQ alert*\n"
        + " · ".join(reasons)
        + f"\nRemaining depth: *{depth_n}*"
    )
    if prune_report and prune_report.get("cutoff"):
        text += f"\nCutoff: `{prune_report.get('cutoff')}`"
    try:
        from rag.judge_alert import post_slack

        ok = post_slack(webhook, {"text": text})
        return {"ok": bool(ok), "skipped": False, "posted": bool(ok), "reasons": reasons}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def replay_dual_write_dlq(
    *,
    path: Optional[str] = None,
    limit: int = 5,
    dry_run: bool = False,
    force: bool = True,
    state_path: Optional[str] = None,
    respect_budget: bool = True,
) -> Dict[str, Any]:
    """DLQ'daki son N kaydı yeniden işle (başarılı olanları bırakır).

    Failed-after-N → quarantine JSONL + Slack channel.
    Optional per-run / per-hour auto-replay budget via env + webhook state.
    """
    p = path or dual_write_webhook_dlq_path()
    rows = read_dual_write_dlq(path=p)
    if not rows:
        return {
            "ok": True,
            "replayed": 0,
            "remaining": 0,
            "quarantined": 0,
            "results": [],
            "path": p,
        }
    want = max(1, int(limit))
    budget: Dict[str, Any] = {"ok": True, "allowed": want, "budget_exhausted": False}
    if respect_budget and not dry_run:
        budget = check_dual_write_dlq_replay_budget(state_path=state_path, want=want)
        if budget.get("budget_exhausted"):
            return {
                "ok": True,
                "skipped": True,
                "reason": "replay_budget_exhausted",
                "budget": budget,
                "replayed": 0,
                "remaining": len(rows),
                "quarantined": 0,
                "results": [],
                "path": p,
            }
        want = max(0, int(budget.get("allowed") or 0))
        if want <= 0:
            return {
                "ok": True,
                "skipped": True,
                "reason": "replay_budget_exhausted",
                "budget": budget,
                "replayed": 0,
                "remaining": len(rows),
                "quarantined": 0,
                "results": [],
                "path": p,
            }
    take = rows[-want:]
    keep = rows[: -len(take)] if len(rows) > len(take) else []
    results: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    threshold = _dlq_quarantine_after()
    attempted = 0
    for entry in take:
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if dry_run:
            results.append(
                {
                    "ok": True,
                    "dry_run": True,
                    "digest": entry.get("payload_digest"),
                    "alerts": entry.get("alerts"),
                    "attempts": int(entry.get("attempts") or 0),
                }
            )
            continue
        attempted += 1
        report = handle_dual_write_alertmanager_webhook(
            payload,
            dry_run=False,
            force=force,
            state_path=state_path,
            enqueue_dlq=False,
        )
        row_out = {
            "ok": bool(report.get("ok")),
            "digest": entry.get("payload_digest"),
            "report": {
                k: report.get(k)
                for k in ("ok", "skipped", "reason", "planned_actions", "alerts")
            },
        }
        if not report.get("ok"):
            attempts = int(entry.get("attempts") or 0) + 1
            entry = {**entry, "attempts": attempts, "last_error": report.get("reason")}
            if attempts >= threshold:
                q = enqueue_dual_write_dlq_quarantine(
                    entry, reason="replay_exhausted"
                )
                row_out["quarantined"] = True
                row_out["attempts"] = attempts
                row_out["quarantine"] = {
                    "ok": q.get("ok"),
                    "path": q.get("path"),
                    "notify": q.get("notify"),
                }
                quarantined.append(
                    {"digest": entry.get("payload_digest"), "attempts": attempts}
                )
            else:
                row_out["attempts"] = attempts
                keep.append(entry)
        results.append(row_out)
    if not dry_run:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for row in keep:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if respect_budget and attempted:
            record_dual_write_dlq_replay_budget(
                attempted, state_path=state_path, budget_snapshot=budget
            )
    return {
        "ok": all(r.get("ok") or r.get("quarantined") for r in results)
        if results
        else True,
        "replayed": len(results),
        "remaining": len(keep) if not dry_run else len(rows),
        "quarantined": len(quarantined),
        "quarantine_after": threshold,
        "quarantine_entries": quarantined,
        "budget": budget,
        "results": results,
        "path": p,
    }


def emit_dual_write_webhook_metric(
    *,
    result: str,
    reason: Optional[str] = None,
    actions: int = 0,
    circuit_open: bool = False,
    dlq_enqueued: bool = False,
    dlq_depth: Optional[int] = None,
) -> None:
    try:
        from rag.metrics import record_metric

        values: Dict[str, Any] = {
            "result": result,
            "actions": int(actions),
            "circuit_open": bool(circuit_open),
            "dlq_enqueued": bool(dlq_enqueued),
        }
        if reason:
            values["reason"] = reason
        if dlq_depth is not None:
            values["dlq_depth"] = int(dlq_depth)
        record_metric("dual_write_webhook", values=values)
    except Exception:
        pass


def webhook_cooldown_sec() -> float:
    raw = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_COOLDOWN_SEC", "900").strip()
    try:
        return max(0.0, float(raw or 900))
    except Exception:
        return 900.0


def webhook_circuit_failure_threshold() -> int:
    raw = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_CB_FAILURES", "3").strip()
    try:
        return max(1, int(raw or 3))
    except Exception:
        return 3


def webhook_circuit_open_sec() -> float:
    raw = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_CB_OPEN_SEC", "1800").strip()
    try:
        return max(0.0, float(raw or 1800))
    except Exception:
        return 1800.0


def webhook_strict_rate_limit() -> bool:
    return os.environ.get("RAG_DUAL_WRITE_WEBHOOK_STRICT_RL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def check_webhook_circuit(
    state: Dict[str, Any],
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Circuit breaker: açıkken trigger atlanır."""
    now_f = float(now if now is not None else time.time())
    open_until = float(state.get("circuit_open_until") or 0)
    if open_until and now_f < open_until:
        return {
            "open": True,
            "retry_after_sec": max(0, int(open_until - now_f)),
            "open_until": open_until,
            "consecutive_failures": int(state.get("consecutive_failures") or 0),
        }
    return {
        "open": False,
        "consecutive_failures": int(state.get("consecutive_failures") or 0),
    }


def record_webhook_circuit_result(
    state: Dict[str, Any],
    *,
    ok: bool,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Başarı/başarısızlığa göre circuit state güncelle."""
    now_f = float(now if now is not None else time.time())
    if ok:
        state["consecutive_failures"] = 0
        state["circuit_open_until"] = 0
        state["circuit_opened_at"] = None
        state["last_ok"] = True
        return state
    failures = int(state.get("consecutive_failures") or 0) + 1
    state["consecutive_failures"] = failures
    state["last_ok"] = False
    threshold = webhook_circuit_failure_threshold()
    open_sec = webhook_circuit_open_sec()
    if failures >= threshold and open_sec > 0:
        state["circuit_open_until"] = now_f + open_sec
        state["circuit_opened_at"] = _utcnow_iso()
    return state


def webhook_rate_limit_headers(
    *,
    limit: float,
    remaining: int,
    reset_at: float,
    retry_after: Optional[int] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """IETF-style RateLimit-* (+ optional Retry-After)."""
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "RateLimit-Limit": str(int(max(0, limit))),
        "RateLimit-Remaining": str(max(0, int(remaining))),
        "RateLimit-Reset": str(int(max(0, reset_at))),
    }
    if retry_after is not None:
        headers["Retry-After"] = str(max(0, int(retry_after)))
    if extra:
        headers.update(extra)
    return headers


def build_webhook_http_headers(
    report: Dict[str, Any],
    *,
    now: Optional[float] = None,
) -> Dict[str, str]:
    """Cooldown/circuit bilgisinden RateLimit header seti üret."""
    now_f = float(now if now is not None else time.time())
    cooldown = webhook_cooldown_sec()
    reason = str(report.get("reason") or "")
    retry = report.get("retry_after_sec")
    retry_i = int(retry) if retry is not None else None
    if reason in {"cooldown", "circuit_open"} and retry_i is not None:
        remaining = 0
        reset_at = now_f + max(0, retry_i)
    else:
        remaining = 1
        last = float(report.get("last_trigger_ts") or 0)
        if last and cooldown > 0:
            reset_at = last + cooldown
            if now_f < reset_at:
                remaining = 0
                retry_i = max(0, int(reset_at - now_f))
            else:
                reset_at = now_f + cooldown
        else:
            reset_at = now_f + cooldown
    return webhook_rate_limit_headers(
        limit=1 if cooldown <= 0 else max(1, int(cooldown)),
        remaining=remaining,
        reset_at=reset_at,
        retry_after=retry_i if remaining == 0 else None,
    )


def webhook_signing_secret() -> str:
    return os.environ.get("RAG_ALERTMANAGER_WEBHOOK_SIGNING_SECRET", "").strip()


def webhook_signature_max_age_sec() -> int:
    raw = os.environ.get("RAG_ALERTMANAGER_WEBHOOK_MAX_AGE_SEC", "").strip()
    try:
        return max(30, int(raw or 300))
    except Exception:
        return 300


def build_webhook_signature(
    body: bytes,
    *,
    timestamp: str,
    secret: str,
) -> str:
    """Slack-benzeri v0 HMAC-SHA256 imza."""
    try:
        raw = body.decode("utf-8")
    except UnicodeDecodeError:
        raw = body.decode("utf-8", errors="replace")
    base = f"{WEBHOOK_SIG_VERSION}:{timestamp}:{raw}"
    digest = hmac.new(
        secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{WEBHOOK_SIG_VERSION}={digest}"


def verify_webhook_signature(
    body: bytes,
    *,
    timestamp: str,
    signature: str,
    signing_secret: Optional[str] = None,
    max_age_sec: Optional[int] = None,
    now: Optional[float] = None,
) -> bool:
    """X-Webhook-Timestamp + X-Webhook-Signature doğrulama."""
    secret = (
        signing_secret
        if signing_secret is not None
        else webhook_signing_secret()
    )
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(str(timestamp).strip())
    except (TypeError, ValueError):
        return False
    age = abs(int(now if now is not None else time.time()) - ts)
    limit = int(max_age_sec if max_age_sec is not None else webhook_signature_max_age_sec())
    if age > limit:
        return False
    expected = build_webhook_signature(body, timestamp=str(ts), secret=secret)
    return hmac.compare_digest(expected, str(signature).strip())


def _normalize_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


def extract_webhook_auth_headers(
    headers: Optional[Dict[str, str]],
) -> Dict[str, str]:
    hdrs = _normalize_headers(headers)
    nonce = (
        hdrs.get("x-webhook-nonce")
        or hdrs.get("x-request-id")
        or ""
    ).strip()
    return {
        "timestamp": (
            hdrs.get("x-webhook-timestamp") or hdrs.get("x-timestamp") or ""
        ).strip(),
        "signature": (
            hdrs.get("x-webhook-signature") or hdrs.get("x-signature") or ""
        ).strip(),
        "nonce": nonce,
        "token": (
            hdrs.get("x-webhook-token")
            or (
                hdrs.get("authorization", "")[7:].strip()
                if hdrs.get("authorization", "").lower().startswith("bearer ")
                else ""
            )
        ).strip(),
    }


def purge_seen_nonces(
    seen: Dict[str, Any],
    *,
    max_age_sec: int,
    now: Optional[float] = None,
) -> Dict[str, float]:
    """Eski nonce kayıtlarını temizle; yalnızca geçerli float ts tut."""
    cutoff = float(now if now is not None else time.time()) - max(float(max_age_sec), 1.0)
    out: Dict[str, float] = {}
    if not isinstance(seen, dict):
        return out
    for key, val in seen.items():
        try:
            ts = float(val)
        except (TypeError, ValueError):
            continue
        if ts >= cutoff:
            out[str(key)] = ts
    return out


def check_webhook_replay(
    *,
    nonce: str,
    timestamp: str,
    state: Optional[Dict[str, Any]] = None,
    state_path: Optional[str] = None,
    max_age_sec: Optional[int] = None,
    now: Optional[float] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Nonce + timestamp replay koruması.
    Aynı nonce yeniden gelirse rejected; başarılı doğrulamada seen_nonces'a yazılır.
    """
    n = (nonce or "").strip()
    if not n:
        return {"ok": False, "error": "nonce_missing"}
    try:
        ts = int(str(timestamp).strip())
    except (TypeError, ValueError):
        return {"ok": False, "error": "timestamp_invalid"}
    limit = int(max_age_sec if max_age_sec is not None else webhook_signature_max_age_sec())
    now_f = float(now if now is not None else time.time())
    if abs(now_f - ts) > limit:
        return {"ok": False, "error": "timestamp_expired", "max_age_sec": limit}

    st_path = state_path or dual_write_webhook_state_path()
    st = dict(state) if isinstance(state, dict) else load_dual_write_webhook_state(st_path)
    # Keep nonces a bit longer than signature window to catch delayed replays
    retain = max(limit * 2, limit + 60)
    seen = purge_seen_nonces(st.get("seen_nonces") or {}, max_age_sec=retain, now=now_f)
    if n in seen:
        return {"ok": False, "error": "replay", "nonce": n}
    seen[n] = now_f
    st["seen_nonces"] = seen
    st["last_nonce"] = n
    st["last_nonce_at"] = _utcnow_iso()
    if persist:
        try:
            save_dual_write_webhook_state(st, path=st_path)
        except Exception:
            pass
    return {"ok": True, "nonce": n, "seen_count": len(seen), "state": st}


def authorize_alertmanager_webhook(
    body: bytes,
    *,
    headers: Optional[Dict[str, str]] = None,
    state_path: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Auth sırası:
    1) Signing secret varsa → HMAC + nonce replay zorunlu
    2) Shared token varsa → Bearer / X-Webhook-Token
    3) Hiçbiri yoksa → açık (ok)
    """
    auth = extract_webhook_auth_headers(headers)
    secret = webhook_signing_secret()
    if secret:
        if not verify_webhook_signature(
            body,
            timestamp=auth["timestamp"],
            signature=auth["signature"],
            signing_secret=secret,
            now=now,
        ):
            return {
                "ok": False,
                "error": "invalid_signature",
                "auth": "hmac",
            }
        replay = check_webhook_replay(
            nonce=auth["nonce"],
            timestamp=auth["timestamp"],
            state_path=state_path,
            now=now,
            persist=True,
        )
        if not replay.get("ok"):
            return {
                "ok": False,
                "error": str(replay.get("error") or "replay"),
                "auth": "hmac",
                "replay": replay,
            }
        return {"ok": True, "auth": "hmac", "replay": replay}

    token = os.environ.get("RAG_ALERTMANAGER_WEBHOOK_TOKEN", "").strip()
    if token:
        if auth["token"] != token:
            return {"ok": False, "error": "unauthorized", "auth": "token"}
        return {"ok": True, "auth": "token"}

    return {"ok": True, "auth": "none"}


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
    enqueue_dlq: bool = True,
) -> Dict[str, Any]:
    """Firing dual-write alert → catch-up / shadow-compare (debounce'lu)."""
    alerts = extract_firing_dual_write_alerts(payload)
    if not alerts:
        emit_dual_write_webhook_metric(
            result="skipped", reason="no_matching_firing_alerts"
        )
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

    circuit = check_webhook_circuit(state, now=now)
    if not force and circuit.get("open"):
        emit_dual_write_webhook_metric(
            result="skipped_circuit",
            reason="circuit_open",
            actions=len(actions),
            circuit_open=True,
            dlq_depth=dual_write_dlq_depth(),
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "circuit_open",
            "retry_after_sec": circuit.get("retry_after_sec"),
            "circuit": circuit,
            "alerts": [a.get("alertname") for a in alerts],
            "planned_actions": actions,
            "last_trigger_ts": float(state.get("last_trigger_ts") or 0) or None,
        }

    cooldown = webhook_cooldown_sec()
    last = float(state.get("last_trigger_ts") or 0)
    if not force and cooldown > 0 and last and (now - last) < cooldown:
        emit_dual_write_webhook_metric(
            result="skipped_cooldown",
            reason="cooldown",
            actions=len(actions),
            circuit_open=False,
            dlq_depth=dual_write_dlq_depth(),
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "cooldown",
            "retry_after_sec": max(0, int(cooldown - (now - last))),
            "alerts": [a.get("alertname") for a in alerts],
            "planned_actions": actions,
            "last_trigger_ts": last,
            "circuit": circuit,
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
    dlq_info: Optional[Dict[str, Any]] = None
    if not dry_run:
        state.update(
            {
                "last_trigger_ts": now,
                "last_trigger_at": _utcnow_iso(),
                "last_alerts": [a.get("alertname") for a in alerts],
                "last_actions": actions,
            }
        )
        record_webhook_circuit_result(state, ok=ok, now=now)
        try:
            save_dual_write_webhook_state(state, path=st_path)
        except Exception:
            pass
        if not ok and enqueue_dlq:
            report_partial = {
                "ok": ok,
                "alerts": [a.get("alertname") for a in alerts],
                "planned_actions": actions,
                "results": results,
            }
            try:
                dlq_info = enqueue_dual_write_dlq(
                    payload=payload, report=report_partial, reason="action_failed"
                )
            except Exception as exc:
                dlq_info = {"ok": False, "error": type(exc).__name__}

    depth = dual_write_dlq_depth()
    emit_dual_write_webhook_metric(
        result="ok" if ok else "failed",
        reason=None if ok else "action_failed",
        actions=len(actions),
        circuit_open=bool(check_webhook_circuit(state, now=now).get("open")),
        dlq_enqueued=bool(dlq_info and dlq_info.get("ok")),
        dlq_depth=depth,
    )

    out = {
        "ok": ok,
        "skipped": False,
        "dry_run": dry_run,
        "alerts": [a.get("alertname") for a in alerts],
        "planned_actions": actions,
        "results": results,
        "last_trigger_ts": now,
        "circuit": check_webhook_circuit(state, now=now),
        "dlq_depth": depth,
    }
    if dlq_info is not None:
        out["dlq"] = dlq_info
    return out


def handle_alertmanager_webhook_http(
    body: bytes,
    *,
    headers: Optional[Dict[str, str]] = None,
    state_path: Optional[str] = None,
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

    auth = authorize_alertmanager_webhook(
        body, headers=headers, state_path=state_path
    )
    if not auth.get("ok"):
        err = str(auth.get("error") or "unauthorized")
        code = 401
        if err == "replay":
            code = 409
        elif err in {"timestamp_expired", "timestamp_invalid", "nonce_missing"}:
            code = 401
        return (
            code,
            {"Content-Type": "application/json"},
            json.dumps(
                {"ok": False, "error": err, "auth": auth.get("auth")},
                ensure_ascii=False,
            ).encode("utf-8"),
        )

    dry = os.environ.get("RAG_DUAL_WRITE_WEBHOOK_DRY_RUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    report = handle_dual_write_alertmanager_webhook(
        payload, dry_run=dry, state_path=state_path
    )
    report["auth"] = auth.get("auth")
    status = 200 if report.get("ok") else 500
    if report.get("skipped") and report.get("reason") in {"cooldown", "circuit_open"}:
        # Alertmanager prefers 2xx; optional strict mode → 429 + Retry-After
        status = 429 if webhook_strict_rate_limit() else 200
    resp_headers = build_webhook_http_headers(report)
    return (
        status,
        resp_headers,
        json.dumps(report, ensure_ascii=False, default=str).encode("utf-8"),
    )
