"""Judge soft-fail çok kanallı alert (Slack / PagerDuty / Opsgenie) + auto-resolve."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


_ACK_RATE: Dict[str, float] = {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def judge_alert_state_path(base: Optional[str] = None) -> str:
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    return os.path.join(root, "judge_alert_state.json")


def judge_ack_audit_path(base: Optional[str] = None) -> str:
    env = os.environ.get("RAG_JUDGE_ACK_AUDIT", "").strip()
    if env:
        return env
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    return os.path.join(root, "judge_ack_audit.jsonl")


def append_judge_ack_audit(
    event: str,
    *,
    actor: Optional[str] = None,
    note: Optional[str] = None,
    source: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Judge ack/alert/resolve audit satırı (JSONL append)."""
    st = state or {}
    record: Dict[str, Any] = {
        "ts": _utcnow_iso(),
        "event": str(event or "unknown"),
        "actor": actor,
        "note": (note or "")[:500] if note is not None else None,
        "source": source or st.get("source"),
        "soft_fail": st.get("soft_fail"),
        "acknowledged": st.get("acknowledged"),
        "acknowledged_by": st.get("acknowledged_by"),
        "acknowledged_at": st.get("acknowledged_at"),
        "ack_note": st.get("ack_note"),
    }
    if extra:
        record.update(extra)
    out = path or judge_ack_audit_path()
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            from rag.metrics import record_metric

            record_metric(
                "judge_ack_audit",
                values={
                    "event": record.get("event"),
                    "source": record.get("source") or "unknown",
                },
            )
        except Exception:
            pass
    except Exception as exc:
        record["_write_error"] = type(exc).__name__
    return record


def read_judge_ack_audit(
    *,
    path: Optional[str] = None,
    limit: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    event: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Audit JSONL oku (eskiden yeniye); limit varsa son N."""
    out_path = path or judge_ack_audit_path()
    if not os.path.isfile(out_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if event and str(row.get("event") or "") != str(event):
                continue
            ts = str(row.get("ts") or "")
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            rows.append(row)
    if limit is not None and limit >= 0:
        rows = rows[-int(limit) :]
    return rows


def purge_judge_ack_audit(
    *,
    path: Optional[str] = None,
    days: Optional[float] = None,
    keep: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Judge ack audit JSONL retention: eski satırları sil (days ve/veya keep).

    - days: ts < now-days olanları at
    - keep: kalanlardan sadece son N'i tut
    En az biri gerekli. dry_run=True ise dosyaya yazmaz.
    """
    out_path = path or judge_ack_audit_path()
    if days is None and keep is None:
        return {
            "ok": False,
            "error": "days_or_keep_required",
            "path": out_path,
            "dry_run": bool(dry_run),
        }
    if days is not None and float(days) < 0:
        return {"ok": False, "error": "days_negative", "path": out_path}
    if keep is not None and int(keep) < 0:
        return {"ok": False, "error": "keep_negative", "path": out_path}

    if not os.path.isfile(out_path):
        return {
            "ok": True,
            "path": out_path,
            "before": 0,
            "after": 0,
            "removed": 0,
            "dry_run": bool(dry_run),
            "missing": True,
        }

    raw_rows: List[Dict[str, Any]] = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                raw_rows.append(row)

    before = len(raw_rows)
    kept = list(raw_rows)
    cutoff: Optional[str] = None
    if days is not None:
        cutoff_ts = time.time() - float(days) * 86400.0
        cutoff = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()
        kept = [r for r in kept if str(r.get("ts") or "") >= cutoff]
    if keep is not None:
        kept = kept[-int(keep) :]

    after = len(kept)
    removed = before - after
    if not dry_run and removed > 0:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif not dry_run and before == 0:
        # touch empty file stays empty
        pass

    return {
        "ok": True,
        "path": out_path,
        "before": before,
        "after": after,
        "removed": removed,
        "dry_run": bool(dry_run),
        "days": float(days) if days is not None else None,
        "keep": int(keep) if keep is not None else None,
        "cutoff": cutoff,
    }


def maybe_purge_judge_ack_audit(
    *,
    path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Env ile retention: RAG_JUDGE_ACK_AUDIT_RETENTION_DAYS / _KEEP / _PRUNE."""
    prune_flag = os.environ.get("RAG_JUDGE_ACK_AUDIT_PRUNE", "").strip().lower()
    if prune_flag in {"0", "false", "no", "off"}:
        return {"ok": True, "skipped": True, "reason": "prune_disabled"}

    days_raw = os.environ.get("RAG_JUDGE_ACK_AUDIT_RETENTION_DAYS", "").strip()
    keep_raw = os.environ.get("RAG_JUDGE_ACK_AUDIT_KEEP", "").strip()
    days: Optional[float] = None
    keep: Optional[int] = None
    if days_raw:
        try:
            days = float(days_raw)
        except Exception:
            return {"ok": False, "skipped": True, "reason": "days_invalid"}
    if keep_raw:
        try:
            keep = int(keep_raw)
        except Exception:
            return {"ok": False, "skipped": True, "reason": "keep_invalid"}
    if days is None and keep is None:
        if prune_flag not in {"1", "true", "yes", "on"}:
            return {"ok": True, "skipped": True, "reason": "retention_not_configured"}
        days = 90.0
        keep = 10000
    return purge_judge_ack_audit(path=path, days=days, keep=keep, dry_run=dry_run)


def export_judge_ack_audit(
    *,
    fmt: str = "jsonl",
    path: Optional[str] = None,
    output: Optional[str] = None,
    limit: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    event: Optional[str] = None,
    tenant_id: Optional[str] = None,
    include_state: bool = False,
    state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit trail'i JSONL veya CSV olarak dışa aktar."""
    rows = read_judge_ack_audit(
        path=path, limit=limit, since=since, until=until, event=event
    )
    want_tenant = (tenant_id or "").strip()
    if want_tenant:
        rows = [
            r
            for r in rows
            if str(r.get("tenant_id") or "").strip() == want_tenant
        ]
    if include_state:
        st = load_judge_alert_state(state_path)
        rows = list(rows) + [
            {
                "ts": _utcnow_iso(),
                "event": "state_snapshot",
                "actor": None,
                "note": None,
                "source": st.get("source"),
                "soft_fail": st.get("soft_fail"),
                "acknowledged": st.get("acknowledged"),
                "acknowledged_by": st.get("acknowledged_by"),
                "acknowledged_at": st.get("acknowledged_at"),
                "ack_note": st.get("ack_note"),
                "tenant_id": want_tenant or None,
            }
        ]
    kind = (fmt or "jsonl").strip().lower()
    if kind == "csv":
        import csv
        import io

        fields = [
            "ts",
            "event",
            "actor",
            "note",
            "source",
            "tenant_id",
            "soft_fail",
            "acknowledged",
            "acknowledged_by",
            "acknowledged_at",
            "ack_note",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
        text = buf.getvalue()
    else:
        kind = "jsonl"
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
    return {
        "ok": True,
        "format": kind,
        "count": len(rows),
        "output": output,
        "text": text if not output else None,
        "tenant_id": want_tenant or None,
    }


def summarize_judge_ack_audit(
    *,
    path: Optional[str] = None,
    since_hours: float = 168.0,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Ack audit özeti (varsayılan son 7 gün)."""
    hours = max(0.0, float(since_hours))
    cutoff = datetime.fromtimestamp(
        time.time() - hours * 3600.0, tz=timezone.utc
    ).isoformat()
    rows = read_judge_ack_audit(path=path, since=cutoff)
    want_tenant = (tenant_id or "").strip()
    if want_tenant:
        rows = [
            r
            for r in rows
            if str(r.get("tenant_id") or "").strip() == want_tenant
        ]
    by_event: Dict[str, int] = {}
    by_tenant: Dict[str, int] = {}
    actors: set = set()
    last_by_event: Dict[str, str] = {}
    for row in rows:
        ev = str(row.get("event") or "unknown")
        by_event[ev] = by_event.get(ev, 0) + 1
        tid = str(row.get("tenant_id") or "").strip()
        if tid:
            by_tenant[tid] = by_tenant.get(tid, 0) + 1
        actor = str(row.get("actor") or row.get("acknowledged_by") or "").strip()
        if actor:
            actors.add(actor)
        ts = str(row.get("ts") or "")
        if ts and (ev not in last_by_event or ts > last_by_event[ev]):
            last_by_event[ev] = ts
    return {
        "ok": True,
        "since_hours": hours,
        "cutoff": cutoff,
        "tenant_id": want_tenant or None,
        "total": len(rows),
        "by_event": by_event,
        "by_tenant": by_tenant,
        "actors": sorted(actors),
        "actor_count": len(actors),
        "last_by_event": last_by_event,
    }


def build_ack_heatmap(
    *,
    path: Optional[str] = None,
    since_hours: float = 168.0,
    tenant_id: Optional[str] = None,
    bucket: str = "day",
    max_buckets: int = 14,
) -> Dict[str, Any]:
    """Ack audit event × time-bucket heatmap (digest canvas)."""
    hours = max(0.0, float(since_hours))
    cutoff = datetime.fromtimestamp(
        time.time() - hours * 3600.0, tz=timezone.utc
    ).isoformat()
    rows = read_judge_ack_audit(path=path, since=cutoff)
    want = (tenant_id or "").strip()
    if want:
        rows = [r for r in rows if str(r.get("tenant_id") or "").strip() == want]
    kind = (bucket or "day").strip().lower()
    if kind not in {"day", "hour"}:
        kind = "day"
    cells: Dict[str, Dict[str, int]] = {}
    events: set = set()
    for row in rows:
        ts = str(row.get("ts") or "")
        if not ts:
            continue
        try:
            # Accept both Z and +00:00
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
        except Exception:
            continue
        if kind == "hour":
            key = parsed.strftime("%m-%d %H")
        else:
            key = parsed.strftime("%Y-%m-%d")
        ev = str(row.get("event") or "unknown")
        events.add(ev)
        cells.setdefault(key, {})
        cells[key][ev] = cells[key].get(ev, 0) + 1
    buckets = sorted(cells.keys())
    if max_buckets and len(buckets) > int(max_buckets):
        buckets = buckets[-int(max_buckets) :]
        cells = {b: cells[b] for b in buckets}
    event_list = sorted(events)
    matrix = {
        ev: [int((cells.get(b) or {}).get(ev, 0)) for b in buckets] for ev in event_list
    }
    max_val = 0
    for ev in event_list:
        for n in matrix[ev]:
            if n > max_val:
                max_val = n
    return {
        "ok": True,
        "bucket": kind,
        "buckets": buckets,
        "events": event_list,
        "matrix": matrix,
        "max": max_val,
        "total": len(rows),
        "tenant_id": want or None,
    }


def format_ack_heatmap_mrkdwn(heatmap: Optional[Dict[str, Any]]) -> str:
    """Slack mrkdwn ascii heatmap (░▒▓█)."""
    if not heatmap or not heatmap.get("buckets"):
        return "*Ack heatmap*\n_No activity in window._"
    blocks = " ░▒▓█"
    max_v = max(1, int(heatmap.get("max") or 1))
    lines = [
        f"*Ack heatmap* ({heatmap.get('bucket')} · last {len(heatmap.get('buckets') or [])} buckets)",
    ]
    # Compact header of bucket labels (last 8 chars)
    labels = [str(b)[-5:] for b in (heatmap.get("buckets") or [])]
    lines.append("`" + " ".join(f"{lb:>5}" for lb in labels) + "`")
    for ev in heatmap.get("events") or []:
        series = (heatmap.get("matrix") or {}).get(ev) or []
        chars = []
        for n in series:
            idx = min(len(blocks) - 1, int(round((int(n) / max_v) * (len(blocks) - 1))))
            chars.append(blocks[idx] if n else "·")
        lines.append(f"`{ev[:12]:<12}` {''.join(chars)} ({sum(series)})")
    return "\n".join(lines)


def _append_url_query(url: str, **params: Any) -> str:
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    base = (url or "").strip()
    if not base:
        return ""
    parts = urlsplit(base)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    for k, v in params.items():
        if v is None or v == "":
            continue
        q[str(k)] = str(v)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment)
    )


def judge_ack_canvas_urls(
    *,
    tenant_id: Optional[str] = None,
    hours: Optional[float] = None,
) -> Dict[str, str]:
    """Tenant-scoped canvas deep-links (ack form + export)."""
    form = judge_ack_public_url()
    export = judge_ack_export_public_url()
    tid = (tenant_id or "").strip() or None
    hrs = None
    if hours is not None:
        try:
            hrs = f"{float(hours):g}"
        except Exception:
            hrs = None
    return {
        "form": _append_url_query(form, tenant=tid, hours=hrs) if form else "",
        "export": _append_url_query(export, tenant=tid, hours=hrs) if export else "",
        "export_csv": _append_url_query(
            export, tenant=tid, hours=hrs, format="csv"
        )
        if export
        else "",
        "tenant_id": tid or "",
    }


def judge_ack_digest_snapshot_path(
    *,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
) -> str:
    """Son digest özeti snapshot yolu (digest-diff için)."""
    env = os.environ.get("RAG_JUDGE_ACK_DIGEST_SNAPSHOT", "").strip()
    if env and not tenant_id:
        return env
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    tid = (tenant_id or "").strip()
    if tid:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tid)[:64]
        return os.path.join(root, f"judge_ack_digest_last_{safe}.json")
    return os.path.join(root, "judge_ack_digest_last.json")


def load_judge_ack_digest_snapshot(
    *,
    path: Optional[str] = None,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    p = path or judge_ack_digest_snapshot_path(tenant_id=tenant_id, base=base)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_judge_ack_digest_snapshot(
    summary: Dict[str, Any],
    *,
    path: Optional[str] = None,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
    diff: Optional[Dict[str, Any]] = None,
) -> str:
    p = path or judge_ack_digest_snapshot_path(tenant_id=tenant_id, base=base)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    payload = {
        "saved_at": _utcnow_iso(),
        "tenant_id": (tenant_id or summary.get("tenant_id") or None),
        "summary": {
            "total": summary.get("total"),
            "since_hours": summary.get("since_hours"),
            "by_event": summary.get("by_event") or {},
            "by_tenant": summary.get("by_tenant") or {},
            "actors": summary.get("actors") or [],
            "actor_count": summary.get("actor_count"),
            "last_by_event": summary.get("last_by_event") or {},
        },
    }
    if diff is not None:
        payload["last_diff"] = diff
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def diff_judge_ack_digest(
    prev: Optional[Dict[str, Any]],
    curr: Dict[str, Any],
) -> Dict[str, Any]:
    """Önceki digest özeti ile güncel özet arasındaki delta."""
    prev_sum = prev or {}
    if "summary" in prev_sum and isinstance(prev_sum.get("summary"), dict):
        prev_sum = prev_sum["summary"]
    pe = prev_sum.get("by_event") or {}
    ce = curr.get("by_event") or {}
    if not isinstance(pe, dict):
        pe = {}
    if not isinstance(ce, dict):
        ce = {}
    keys = set(pe) | set(ce)
    by_event_delta = {
        k: int(ce.get(k, 0) or 0) - int(pe.get(k, 0) or 0) for k in sorted(keys)
    }
    prev_actors = set(prev_sum.get("actors") or [])
    curr_actors = set(curr.get("actors") or [])
    has_prev = bool(prev_sum.get("total") is not None or pe or prev_actors)
    return {
        "ok": True,
        "has_previous": has_prev,
        "delta_total": int(curr.get("total") or 0) - int(prev_sum.get("total") or 0),
        "prev_total": int(prev_sum.get("total") or 0) if has_prev else None,
        "curr_total": int(curr.get("total") or 0),
        "new_events": sorted(k for k in ce if k not in pe),
        "removed_events": sorted(k for k in pe if k not in ce),
        "by_event_delta": by_event_delta,
        "actors_added": sorted(curr_actors - prev_actors),
        "actors_removed": sorted(prev_actors - curr_actors),
    }


def format_judge_ack_digest_diff_text(diff: Optional[Dict[str, Any]]) -> str:
    """Slack mrkdwn: vs last digest canvas-style özet."""
    if not diff or not diff.get("has_previous"):
        base = "*vs last digest*\n_No previous snapshot — baseline recorded._"
        muted = diff.get("muted_tenants") if isinstance(diff, dict) else None
        if muted:
            return (
                base
                + "\nMuted tenants: "
                + ", ".join(f"`{t}`" for t in list(muted)[:12])
            )
        return base
    delta = int(diff.get("delta_total") or 0)
    sign = f"+{delta}" if delta > 0 else str(delta)
    lines = [
        "*vs last digest* (scheduled digest-diff)",
        f"Δ total: *{sign}* · prev `{diff.get('prev_total')}` → curr `{diff.get('curr_total')}`",
    ]
    by_delta = diff.get("by_event_delta") or {}
    nonzero = {k: v for k, v in by_delta.items() if v}
    if nonzero:
        parts = [
            f"`{k}`={'+' if v > 0 else ''}{v}" for k, v in sorted(nonzero.items())
        ]
        lines.append("Δ by event: " + " · ".join(parts[:12]))
    added = diff.get("new_events") or []
    removed = diff.get("removed_events") or []
    if added:
        lines.append("New events: " + ", ".join(f"`{e}`" for e in added[:8]))
    if removed:
        lines.append("Removed events: " + ", ".join(f"`{e}`" for e in removed[:8]))
    actors_added = diff.get("actors_added") or []
    actors_removed = diff.get("actors_removed") or []
    if actors_added:
        lines.append("Actors +: " + ", ".join(f"`{a}`" for a in actors_added[:10]))
    if actors_removed:
        lines.append("Actors −: " + ", ".join(f"`{a}`" for a in actors_removed[:10]))
    muted = diff.get("muted_tenants") or []
    if muted:
        lines.append(
            "Muted tenants: " + ", ".join(f"`{t}`" for t in list(muted)[:12])
        )
    return "\n".join(lines)


def digest_diff_enabled() -> bool:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_DIFF", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def parse_judge_ack_digest_webhooks(
    spec: Optional[str] = None,
    *,
    base: Optional[str] = None,
) -> Dict[str, str]:
    """Tenant → Slack webhook. Env JSON veya metadata/judge_ack_digest_webhooks.json."""
    try:
        from app.config import JUDGE_ACK_DIGEST_WEBHOOKS_JSON, METADATA_DIR
    except ImportError:
        JUDGE_ACK_DIGEST_WEBHOOKS_JSON = ""
        METADATA_DIR = "metadata"
    raw = spec if spec is not None else JUDGE_ACK_DIGEST_WEBHOOKS_JSON
    mapping: Dict[str, str] = {}
    text = (raw or "").strip()
    if text:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for k, v in data.items():
                    tid = str(k).strip()
                    url = str(v or "").strip()
                    if tid and url:
                        mapping[tid] = url
        except json.JSONDecodeError:
            pass
    if mapping:
        return mapping
    path = os.path.join(base or METADATA_DIR, "judge_ack_digest_webhooks.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    tid = str(k).strip()
                    url = str(v or "").strip()
                    if tid and url:
                        mapping[tid] = url
        except Exception:
            pass
    return mapping


def resolve_judge_ack_digest_webhook(
    tenant_id: Optional[str] = None,
    *,
    url: Optional[str] = None,
    base: Optional[str] = None,
) -> str:
    """url arg > tenant map > RAG_JUDGE_SLACK_WEBHOOK."""
    if url and str(url).strip():
        return str(url).strip()
    tid = (tenant_id or "").strip()
    if tid:
        mapping = parse_judge_ack_digest_webhooks(base=base)
        if tid in mapping:
            return mapping[tid]
    return os.environ.get("RAG_JUDGE_SLACK_WEBHOOK", "").strip()


def parse_judge_ack_digest_quiet_hours(
    spec: Optional[str] = None,
    *,
    base: Optional[str] = None,
) -> Dict[str, str]:
    """Tenant → quiet hours HH:MM-HH:MM. Env JSON veya metadata dosyası."""
    try:
        from app.config import JUDGE_ACK_DIGEST_QUIET_HOURS_JSON, METADATA_DIR
    except ImportError:
        JUDGE_ACK_DIGEST_QUIET_HOURS_JSON = ""
        METADATA_DIR = "metadata"
    raw = spec if spec is not None else JUDGE_ACK_DIGEST_QUIET_HOURS_JSON
    mapping: Dict[str, str] = {}
    text = (raw or "").strip()
    if text:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for k, v in data.items():
                    tid = str(k).strip()
                    qh = str(v or "").strip()
                    if tid and qh:
                        mapping[tid] = qh
        except json.JSONDecodeError:
            pass
    if mapping:
        return mapping
    path = os.path.join(base or METADATA_DIR, "judge_ack_digest_quiet_hours.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    tid = str(k).strip()
                    qh = str(v or "").strip()
                    if tid and qh:
                        mapping[tid] = qh
        except Exception:
            pass
    return mapping


def _mute_value_active(value: Any, *, now_ts: Optional[float] = None) -> bool:
    """True if mute entry is active (supports bool/str and {muted, expires_at})."""
    if isinstance(value, dict):
        flag = value.get("muted", value.get("mute", True))
        if flag in {False, 0, "0", "false", "no", "off"}:
            return False
        if flag not in {True, 1, "1", "true", "yes", "on", "mute", "muted", None}:
            if not bool(flag):
                return False
        exp = value.get("expires_at")
        if not exp:
            return True
        try:
            exp_ts = datetime.fromisoformat(str(exp).replace("Z", "+00:00")).timestamp()
        except Exception:
            return True
        now = now_ts if now_ts is not None else time.time()
        return exp_ts > now
    if value in {True, 1, "1", "true", "yes", "on", "mute", "muted"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {
        "mute",
        "muted",
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return False


def parse_judge_ack_digest_mutes(
    spec: Optional[str] = None,
    *,
    base: Optional[str] = None,
) -> Set[str]:
    """Muted tenant id set. Env CSV/JSON list veya metadata/judge_ack_digest_mutes.json."""
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    muted: Set[str] = set()
    now_ts = time.time()
    raw = (
        spec
        if spec is not None
        else os.environ.get("RAG_JUDGE_ACK_DIGEST_MUTE", "").strip()
    )
    text = (raw or "").strip()
    if text:
        if text.startswith("[") or text.startswith("{"):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    muted.update(str(x).strip() for x in data if str(x).strip())
                elif isinstance(data, dict):
                    for k, v in data.items():
                        tid = str(k).strip()
                        if tid and _mute_value_active(v, now_ts=now_ts):
                            muted.add(tid)
            except json.JSONDecodeError:
                pass
        if not muted:
            for part in text.replace(";", ",").split(","):
                tid = part.strip()
                if tid:
                    muted.add(tid)
    if muted:
        return muted
    path = os.path.join(base or METADATA_DIR, "judge_ack_digest_mutes.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                muted.update(str(x).strip() for x in data if str(x).strip())
            elif isinstance(data, dict):
                for k, v in data.items():
                    tid = str(k).strip()
                    if tid and _mute_value_active(v, now_ts=now_ts):
                        muted.add(tid)
        except Exception:
            pass
    return muted


def is_judge_ack_digest_muted(
    tenant_id: Optional[str],
    *,
    mutes: Optional[Set[str]] = None,
    base: Optional[str] = None,
) -> bool:
    tid = (tenant_id or "").strip()
    if not tid:
        return False
    pool = mutes if mutes is not None else parse_judge_ack_digest_mutes(base=base)
    return tid in pool


def judge_ack_digest_mutes_path(*, base: Optional[str] = None) -> str:
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    return os.path.join(base or METADATA_DIR, "judge_ack_digest_mutes.json")


def judge_ack_digest_prefs_path(*, base: Optional[str] = None) -> str:
    env = os.environ.get("RAG_JUDGE_ACK_DIGEST_PREFS", "").strip()
    if env:
        return env
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    return os.path.join(base or METADATA_DIR, "judge_ack_digest_prefs.json")


def judge_ack_digest_messages_path(*, base: Optional[str] = None) -> str:
    env = os.environ.get("RAG_JUDGE_ACK_DIGEST_MESSAGES", "").strip()
    if env:
        return env
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    return os.path.join(base or METADATA_DIR, "judge_ack_digest_messages.json")


def load_judge_ack_digest_messages(*, base: Optional[str] = None) -> Dict[str, Any]:
    p = judge_ack_digest_messages_path(base=base)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_judge_ack_digest_messages(
    data: Dict[str, Any], *, base: Optional[str] = None
) -> str:
    p = judge_ack_digest_messages_path(base=base)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    payload = dict(data or {})
    payload["updated_at"] = _utcnow_iso()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def list_judge_ack_digest_message_refs(
    tenant_id: str, *, base: Optional[str] = None
) -> List[Dict[str, str]]:
    tid = (tenant_id or "").strip()
    if not tid:
        return []
    data = load_judge_ack_digest_messages(base=base)
    row = data.get(tid)
    if not isinstance(row, dict):
        return []
    out: List[Dict[str, str]] = []
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for item in msgs:
            if not isinstance(item, dict):
                continue
            ch = str(item.get("channel_id") or "").strip()
            ts = str(item.get("message_ts") or "").strip()
            if ch and ts:
                out.append({"channel_id": ch, "message_ts": ts})
    # Legacy single-ref shape
    ch = str(row.get("channel_id") or "").strip()
    ts = str(row.get("message_ts") or "").strip()
    if ch and ts and not any(
        r["channel_id"] == ch and r["message_ts"] == ts for r in out
    ):
        out.append({"channel_id": ch, "message_ts": ts})
    return out


def save_judge_ack_digest_message(
    tenant_id: str,
    *,
    channel_id: str,
    message_ts: str,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist last digest Slack message ref(s) for fan-out chat.update sync."""
    tid = (tenant_id or "").strip()
    ch = (channel_id or "").strip()
    ts = (message_ts or "").strip()
    if not tid or not ch or not ts:
        return {"ok": False, "error": "ref_incomplete"}
    data = load_judge_ack_digest_messages(base=base)
    row = dict(data.get(tid) or {}) if isinstance(data.get(tid), dict) else {}
    msgs = list(row.get("messages") or []) if isinstance(row.get("messages"), list) else []
    # Deduplicate + keep latest first
    msgs = [
        m
        for m in msgs
        if isinstance(m, dict)
        and not (
            str(m.get("channel_id") or "").strip() == ch
            and str(m.get("message_ts") or "").strip() == ts
        )
    ]
    msgs.insert(
        0,
        {
            "channel_id": ch,
            "message_ts": ts,
            "updated_at": _utcnow_iso(),
        },
    )
    # Cap stored refs per tenant
    msgs = msgs[:20]
    row["messages"] = msgs
    row["channel_id"] = ch
    row["message_ts"] = ts
    row["updated_at"] = _utcnow_iso()
    data[tid] = row
    path = save_judge_ack_digest_messages(data, base=base)
    return {"ok": True, "tenant_id": tid, "path": path, "count": len(msgs)}


def load_judge_ack_digest_prefs(*, base: Optional[str] = None) -> Dict[str, Any]:
    p = judge_ack_digest_prefs_path(base=base)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_judge_ack_digest_prefs(
    prefs: Dict[str, Any], *, base: Optional[str] = None
) -> str:
    p = judge_ack_digest_prefs_path(base=base)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    payload = dict(prefs or {})
    payload["updated_at"] = _utcnow_iso()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def resolve_heatmap_bucket(
    tenant_id: Optional[str] = None,
    *,
    base: Optional[str] = None,
    default: str = "day",
) -> str:
    """Prefs (tenant → global) > env > default."""
    prefs = load_judge_ack_digest_prefs(base=base)
    tid = (tenant_id or "").strip()
    if tid:
        tenants = prefs.get("tenants") if isinstance(prefs.get("tenants"), dict) else {}
        tprefs = tenants.get(tid) if isinstance(tenants.get(tid), dict) else {}
        b = str((tprefs or {}).get("heatmap_bucket") or "").strip().lower()
        if b in {"day", "hour"}:
            return b
    b = str(prefs.get("heatmap_bucket") or "").strip().lower()
    if b in {"day", "hour"}:
        return b
    env = os.environ.get("RAG_JUDGE_ACK_DIGEST_HEATMAP_BUCKET", "").strip().lower()
    if env in {"day", "hour"}:
        return env
    return default if default in {"day", "hour"} else "day"


def set_heatmap_bucket_pref(
    bucket: str,
    *,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    b = (bucket or "").strip().lower()
    if b not in {"day", "hour"}:
        return {"ok": False, "error": "bucket_invalid"}
    prefs = load_judge_ack_digest_prefs(base=base)
    tid = (tenant_id or "").strip()
    if tid:
        tenants = prefs.get("tenants") if isinstance(prefs.get("tenants"), dict) else {}
        row = dict(tenants.get(tid) or {}) if isinstance(tenants.get(tid), dict) else {}
        row["heatmap_bucket"] = b
        row["updated_at"] = _utcnow_iso()
        tenants[tid] = row
        prefs["tenants"] = tenants
    else:
        prefs["heatmap_bucket"] = b
    path = save_judge_ack_digest_prefs(prefs, base=base)
    return {"ok": True, "bucket": b, "tenant_id": tid or None, "path": path}


def _mute_ttl_days() -> Optional[float]:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_MUTE_TTL_DAYS", "").strip()
    if not raw:
        return None
    try:
        days = float(raw)
        return days if days > 0 else None
    except Exception:
        return None


def list_judge_ack_digest_mute_entries(
    *,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rich mute store rows (active flag + TTL metadata)."""
    path = judge_ack_digest_mutes_path(base=base)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []
    now_ts = time.time()
    rows: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            tid = str(item).strip()
            if not tid:
                continue
            rows.append(
                {
                    "tenant_id": tid,
                    "muted": True,
                    "active": True,
                    "muted_at": "",
                    "expires_at": "",
                    "ttl_days": "",
                }
            )
        return rows
    if not isinstance(raw, dict):
        return []
    for k, v in raw.items():
        tid = str(k).strip()
        if not tid:
            continue
        if isinstance(v, dict):
            rows.append(
                {
                    "tenant_id": tid,
                    "muted": bool(_mute_value_active(v, now_ts=now_ts))
                    or bool(v.get("muted", True)),
                    "active": bool(_mute_value_active(v, now_ts=now_ts)),
                    "muted_at": str(v.get("muted_at") or ""),
                    "expires_at": str(v.get("expires_at") or ""),
                    "ttl_days": v.get("ttl_days") if v.get("ttl_days") is not None else "",
                }
            )
        else:
            active = _mute_value_active(v, now_ts=now_ts)
            rows.append(
                {
                    "tenant_id": tid,
                    "muted": bool(active),
                    "active": bool(active),
                    "muted_at": "",
                    "expires_at": "",
                    "ttl_days": "",
                }
            )
    rows.sort(key=lambda r: str(r.get("tenant_id") or ""))
    return rows


def list_judge_ack_digest_snapshots(
    *,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load global + per-tenant keep-on-mute digest snapshots."""
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = base or METADATA_DIR
    out: List[Dict[str, Any]] = []
    seen_paths: Set[str] = set()
    candidates = [
        judge_ack_digest_snapshot_path(base=base),
    ]
    try:
        for name in os.listdir(root):
            if name == "judge_ack_digest_last.json" or (
                name.startswith("judge_ack_digest_last_") and name.endswith(".json")
            ):
                candidates.append(os.path.join(root, name))
    except Exception:
        pass
    for p in candidates:
        ap = os.path.abspath(p)
        if ap in seen_paths or not os.path.isfile(p):
            continue
        seen_paths.add(ap)
        data = load_judge_ack_digest_snapshot(path=p, base=base)
        if not data:
            continue
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        tid = str(data.get("tenant_id") or summary.get("tenant_id") or "").strip()
        if not tid and os.path.basename(p).startswith("judge_ack_digest_last_"):
            tid = os.path.basename(p)[
                len("judge_ack_digest_last_") : -len(".json")
            ]
        last_diff = data.get("last_diff") if isinstance(data.get("last_diff"), dict) else {}
        out.append(
            {
                "tenant_id": tid,
                "snapshot_path": p,
                "saved_at": str(data.get("saved_at") or ""),
                "total": summary.get("total"),
                "since_hours": summary.get("since_hours"),
                "actor_count": summary.get("actor_count"),
                "by_event": json.dumps(
                    summary.get("by_event") or {}, ensure_ascii=False, sort_keys=True
                ),
                "diff_total_delta": (last_diff or {}).get("total_delta"),
                "has_diff": bool(last_diff),
            }
        )
    out.sort(key=lambda r: (str(r.get("tenant_id") or ""), str(r.get("saved_at") or "")))
    return out


def upload_judge_ack_digest_mute_snapshots_slack(
    *,
    payload: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
    bot_token: Optional[str] = None,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    fmt: str = "csv",
) -> Dict[str, Any]:
    """Export mute+snapshot CSV and Slack files.upload (interactive / CLI)."""
    payload = payload if isinstance(payload, dict) else {}
    kind = (fmt or "csv").strip().lower() or "csv"
    if kind not in {"csv", "jsonl"}:
        kind = "csv"
    root = base or os.environ.get("RAG_JUDGE_ACK_DIGEST_BASE", "").strip() or None
    exported = export_judge_ack_digest_mute_snapshots(
        fmt=kind, base=root, tenant_id=tenant_id
    )
    content = exported.get("text") or ""
    count = int(exported.get("count") or 0)
    ch = str(
        (channel_id or "").strip()
        or ((payload.get("channel") or {}).get("id") or "")
        or ((payload.get("container") or {}).get("channel_id") or "")
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
        or ""
    ).strip() or None
    ts = str(
        (thread_ts or "").strip()
        or ((payload.get("message") or {}).get("ts") or "")
        or ((payload.get("container") or {}).get("thread_ts") or "")
        or ((payload.get("container") or {}).get("message_ts") or "")
        or ""
    ).strip() or None
    user_id = slack_interactive_user_id(payload) if payload else None
    token = (
        (bot_token or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    filename = f"judge_ack_digest_mute_snapshots.{kind}"
    tid_label = (tenant_id or "").strip() or "all"
    progress: Dict[str, Any] = {"ok": False, "skipped": True}
    upload: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "not_attempted"}
    thread_reply: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "not_attempted"}
    text = f"Mute snapshot export · {count} rows · `{kind}` · tenant `{tid_label}`"
    if content and token:
        if ch and user_id:
            progress = slack_api(
                "chat.postEphemeral",
                bot_token=token,
                json_body={
                    "channel": ch,
                    "user": user_id,
                    "text": f"Uploading mute snapshots (`{count}` rows, `{kind}`)…",
                },
            )
        upload = slack_files_upload(
            content=content,
            filename=filename,
            title=f"Mute snapshots ({count} rows, {kind})",
            channels=ch,
            thread_ts=ts,
            initial_comment=text,
            bot_token=token,
        )
        if upload.get("ok"):
            text = (
                f"{text}\nAttached `{filename}`"
                + (
                    f" (<{upload.get('permalink')}|open>)"
                    if upload.get("permalink")
                    else ""
                )
            )
            if ch and ts:
                reply_text = (
                    f"Mute snapshot export ready · *{count}* rows · `{kind}` "
                    f"· tenant `{tid_label}`"
                )
                if upload.get("permalink"):
                    reply_text += f" · <{upload.get('permalink')}|open file>"
                thread_reply = post_slack_thread_message(
                    text=reply_text,
                    channel_id=ch,
                    thread_ts=ts,
                    bot_token=token,
                )
        else:
            text = f"{text}\nFile upload skipped: `{upload.get('error')}`"
    elif not token:
        upload = {"ok": False, "skipped": True, "reason": "bot_token_missing"}
    elif not content:
        upload = {"ok": False, "skipped": True, "reason": "empty_export"}
    return {
        "ok": True,
        "mode": "digest_mute_snapshot_upload",
        "tenant_id": (tenant_id or "").strip() or None,
        "export": {
            "ok": exported.get("ok"),
            "count": count,
            "format": exported.get("format") or kind,
            "muted_count": exported.get("muted_count"),
            "snapshot_count": exported.get("snapshot_count"),
        },
        "upload": upload,
        "progress": {
            "ok": bool(progress.get("ok")),
            "skipped": bool(progress.get("skipped")),
            "error": progress.get("error"),
        },
        "thread_reply": thread_reply,
        "text": text,
        "channel_id": ch,
        "message_ts": ts,
    }


def export_judge_ack_digest_mute_snapshots(
    *,
    fmt: str = "csv",
    output: Optional[str] = None,
    base: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Join mute store + digest snapshots → CSV/JSONL export."""
    want = (tenant_id or "").strip()
    mutes = {
        str(r.get("tenant_id") or ""): r
        for r in list_judge_ack_digest_mute_entries(base=base)
        if str(r.get("tenant_id") or "").strip()
    }
    snaps = list_judge_ack_digest_snapshots(base=base)
    snap_by_tid: Dict[str, Dict[str, Any]] = {}
    for s in snaps:
        tid = str(s.get("tenant_id") or "").strip()
        if tid:
            # Prefer newest saved_at per tenant
            prev = snap_by_tid.get(tid)
            if not prev or str(s.get("saved_at") or "") >= str(prev.get("saved_at") or ""):
                snap_by_tid[tid] = s
    tenant_ids = sorted(set(mutes) | set(snap_by_tid))
    if want:
        tenant_ids = [t for t in tenant_ids if t == want]
    rows: List[Dict[str, Any]] = []
    for tid in tenant_ids:
        m = mutes.get(tid) or {}
        s = snap_by_tid.get(tid) or {}
        rows.append(
            {
                "tenant_id": tid,
                "muted": m.get("muted", ""),
                "active": m.get("active", ""),
                "muted_at": m.get("muted_at", ""),
                "expires_at": m.get("expires_at", ""),
                "ttl_days": m.get("ttl_days", ""),
                "snapshot_saved_at": s.get("saved_at", ""),
                "snapshot_total": s.get("total", ""),
                "snapshot_since_hours": s.get("since_hours", ""),
                "snapshot_actor_count": s.get("actor_count", ""),
                "snapshot_by_event": s.get("by_event", ""),
                "snapshot_diff_total_delta": s.get("diff_total_delta", ""),
                "snapshot_has_diff": s.get("has_diff", ""),
                "snapshot_path": s.get("snapshot_path", ""),
            }
        )
    kind = (fmt or "csv").strip().lower()
    if kind == "jsonl":
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    else:
        import csv
        import io

        kind = "csv"
        fields = [
            "tenant_id",
            "muted",
            "active",
            "muted_at",
            "expires_at",
            "ttl_days",
            "snapshot_saved_at",
            "snapshot_total",
            "snapshot_since_hours",
            "snapshot_actor_count",
            "snapshot_by_event",
            "snapshot_diff_total_delta",
            "snapshot_has_diff",
            "snapshot_path",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
        text = buf.getvalue()
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
    archive = archive_judge_ack_digest_mute_export(
        text=text, fmt=kind, base=base, source_output=output
    )
    signed: Optional[Dict[str, Any]] = None
    if archive.get("filename"):
        signed = build_mute_export_signed_url(
            str(archive["filename"]), base=base, public_base=None
        )
        if not signed.get("ok"):
            signed = None
    return {
        "ok": True,
        "format": kind,
        "count": len(rows),
        "output": output,
        "text": text if not output else None,
        "tenant_id": want or None,
        "muted_count": sum(1 for r in rows if r.get("active") in {True, "True", "true", 1, "1"}),
        "snapshot_count": sum(1 for r in rows if r.get("snapshot_saved_at")),
        "archive": archive,
        "signed_url": (signed or {}).get("url"),
        "signed": signed,
    }


def judge_ack_digest_mute_exports_dir(*, base: Optional[str] = None) -> str:
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = (
        base
        or os.environ.get("RAG_JUDGE_ACK_DIGEST_BASE", "").strip()
        or METADATA_DIR
    )
    return os.path.join(root, "mute_exports")


def archive_judge_ack_digest_mute_export(
    *,
    text: str,
    fmt: str = "csv",
    base: Optional[str] = None,
    source_output: Optional[str] = None,
) -> Dict[str, Any]:
    """Timestamped copy under metadata/mute_exports/ for retention + signed URL."""
    kind = (fmt or "csv").strip().lower()
    if kind not in {"csv", "jsonl"}:
        kind = "csv"
    export_dir = judge_ack_digest_mute_exports_dir(base=base)
    os.makedirs(export_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"judge_ack_digest_mute_snapshots_{stamp}.{kind}"
    path = os.path.join(export_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    # Also refresh stable latest name for convenience
    latest = os.path.join(export_dir, f"judge_ack_digest_mute_snapshots.{kind}")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(text)
    return {
        "ok": True,
        "dir": export_dir,
        "path": path,
        "filename": filename,
        "latest": latest,
        "source_output": source_output,
        "format": kind,
        "bytes": len(text.encode("utf-8")),
    }


def mute_export_signing_secret() -> str:
    return (
        os.environ.get("RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SIGNING_SECRET", "").strip()
        or os.environ.get("RAG_JUDGE_ACK_TOKEN", "").strip()
    )


def mute_export_signed_url_ttl_sec() -> int:
    raw = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_URL_TTL_SEC", ""
    ).strip()
    try:
        return max(60, int(raw or 86400))
    except Exception:
        return 86400


def mute_export_signed_urls_path(*, base: Optional[str] = None) -> str:
    try:
        from app.config import METADATA_DIR
    except ImportError:
        METADATA_DIR = "metadata"
    root = (
        base
        or os.environ.get("RAG_JUDGE_ACK_DIGEST_BASE", "").strip()
        or METADATA_DIR
    )
    return os.path.join(root, "mute_export_signed_urls.json")


def load_mute_export_signed_urls(*, base: Optional[str] = None) -> Dict[str, Any]:
    path = mute_export_signed_urls_path(base=base)
    if not os.path.isfile(path):
        return {"links": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"links": {}}
    if not isinstance(data, dict):
        return {"links": {}}
    data.setdefault("links", {})
    if not isinstance(data["links"], dict):
        data["links"] = {}
    return data


def save_mute_export_signed_urls(
    data: Dict[str, Any], *, base: Optional[str] = None
) -> str:
    path = mute_export_signed_urls_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(data or {})
    payload.setdefault("links", {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def build_mute_export_signature(
    filename: str,
    expires: int,
    *,
    secret: str,
    jti: Optional[str] = None,
) -> str:
    jti_s = (jti or "").strip()
    if jti_s:
        base = f"v1:{int(expires)}:{filename}:{jti_s}"
        digest = hmac.new(
            secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"v1={digest}"
    base = f"v0:{int(expires)}:{filename}"
    digest = hmac.new(
        secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"v0={digest}"


def is_mute_export_revoked(
    *,
    jti: Optional[str] = None,
    filename: Optional[str] = None,
    base: Optional[str] = None,
) -> bool:
    data = load_mute_export_signed_urls(base=base)
    links = data.get("links") or {}
    jti_s = (jti or "").strip()
    if jti_s:
        row = links.get(jti_s)
        if isinstance(row, dict) and row.get("revoked"):
            return True
        return False
    name = os.path.basename((filename or "").strip())
    if not name:
        return False
    for row in links.values():
        if isinstance(row, dict) and row.get("filename") == name and row.get("revoked"):
            return True
    return False


def verify_mute_export_signature(
    filename: str,
    expires: str | int,
    signature: str,
    *,
    jti: Optional[str] = None,
    secret: Optional[str] = None,
    now: Optional[float] = None,
    base: Optional[str] = None,
    check_revoke: bool = True,
) -> bool:
    sec = secret if secret is not None else mute_export_signing_secret()
    if not sec or not filename or not signature:
        return False
    try:
        exp = int(str(expires).strip())
    except (TypeError, ValueError):
        return False
    ts_now = int(now if now is not None else time.time())
    if exp < ts_now:
        return False
    jti_s = (jti or "").strip() or None
    if check_revoke and is_mute_export_revoked(jti=jti_s, filename=filename, base=base):
        return False
    sig = str(signature).strip()
    if jti_s or sig.startswith("v1="):
        if not jti_s:
            return False
        expected = build_mute_export_signature(
            filename, exp, secret=sec, jti=jti_s
        )
        return hmac.compare_digest(expected, sig)
    expected = build_mute_export_signature(filename, exp, secret=sec)
    return hmac.compare_digest(expected, sig)


def register_mute_export_signed_url(
    *,
    jti: str,
    filename: str,
    expires: int,
    actor: Optional[str] = None,
    base: Optional[str] = None,
    audit: bool = True,
) -> Dict[str, Any]:
    data = load_mute_export_signed_urls(base=base)
    links = data.setdefault("links", {})
    row = {
        "jti": jti,
        "filename": filename,
        "expires": int(expires),
        "created_at": _utcnow_iso(),
        "actor": actor,
        "revoked": False,
        "revoked_at": None,
    }
    links[jti] = row
    # Cap store size (keep newest ~500)
    if len(links) > 500:
        ordered = sorted(
            links.items(),
            key=lambda kv: str((kv[1] or {}).get("created_at") or ""),
            reverse=True,
        )
        data["links"] = dict(ordered[:500])
    path = save_mute_export_signed_urls(data, base=base)
    if audit:
        append_judge_ack_audit(
            "mute_export_sign",
            actor=actor or "system",
            source="mute_export",
            extra={
                "jti": jti,
                "filename": filename,
                "expires": int(expires),
            },
        )
    return {"ok": True, "path": path, "link": row}


def revoke_mute_export_signed_url(
    ref: str,
    *,
    actor: Optional[str] = None,
    base: Optional[str] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Revoke by jti or filename (all matching links)."""
    key = (ref or "").strip()
    if not key:
        return {"ok": False, "error": "ref_required"}
    data = load_mute_export_signed_urls(base=base)
    links = data.get("links") or {}
    matched: List[str] = []
    if key in links:
        matched = [key]
    else:
        name = os.path.basename(key)
        for jti, row in links.items():
            if isinstance(row, dict) and row.get("filename") == name:
                matched.append(jti)
    if not matched:
        return {"ok": False, "error": "not_found", "ref": key}
    now = _utcnow_iso()
    revoked_rows: List[Dict[str, Any]] = []
    for jti in matched:
        row = dict(links.get(jti) or {})
        row["revoked"] = True
        row["revoked_at"] = now
        row["revoked_by"] = actor or "system"
        if note:
            row["revoke_note"] = str(note)[:500]
        links[jti] = row
        revoked_rows.append(row)
        append_judge_ack_audit(
            "mute_export_revoke",
            actor=actor or "system",
            note=note,
            source="mute_export",
            extra={
                "jti": jti,
                "filename": row.get("filename"),
                "expires": row.get("expires"),
            },
        )
    data["links"] = links
    path = save_mute_export_signed_urls(data, base=base)
    return {
        "ok": True,
        "path": path,
        "revoked": len(revoked_rows),
        "jtis": matched,
        "links": revoked_rows,
    }


def list_mute_export_signed_urls(
    *,
    base: Optional[str] = None,
    include_revoked: bool = True,
    include_expired: bool = True,
    now: Optional[float] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Admin UI listing for mute export signed URLs."""
    data = load_mute_export_signed_urls(base=base)
    links = data.get("links") or {}
    ts_now = int(now if now is not None else time.time())
    rows: List[Dict[str, Any]] = []
    for jti, row in links.items():
        if not isinstance(row, dict):
            continue
        try:
            exp = int(row.get("expires") or 0)
        except (TypeError, ValueError):
            exp = 0
        revoked = bool(row.get("revoked"))
        expired = bool(exp and exp < ts_now)
        if revoked and not include_revoked:
            continue
        if expired and not include_expired and not revoked:
            continue
        rows.append(
            {
                "jti": jti,
                "filename": row.get("filename"),
                "expires": exp,
                "expired": expired,
                "revoked": revoked,
                "created_at": row.get("created_at"),
                "actor": row.get("actor"),
                "revoked_at": row.get("revoked_at"),
                "revoked_by": row.get("revoked_by"),
            }
        )
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    if limit > 0:
        rows = rows[: int(limit)]
    return {
        "ok": True,
        "count": len(rows),
        "links": rows,
        "path": mute_export_signed_urls_path(base=base),
    }


def sweep_mute_export_signed_urls(
    *,
    base: Optional[str] = None,
    now: Optional[float] = None,
    dry_run: bool = False,
    drop_revoked: bool = True,
    revoked_grace_days: float = 7.0,
    audit: bool = True,
) -> Dict[str, Any]:
    """TTL sweep: remove expired (and optionally aged revoked) signed URL rows."""
    data = load_mute_export_signed_urls(base=base)
    links = dict(data.get("links") or {})
    before = len(links)
    ts_now = int(now if now is not None else time.time())
    grace = max(0.0, float(revoked_grace_days)) * 86400.0
    removed: List[str] = []
    kept: Dict[str, Any] = {}
    for jti, row in links.items():
        if not isinstance(row, dict):
            removed.append(str(jti))
            continue
        try:
            exp = int(row.get("expires") or 0)
        except (TypeError, ValueError):
            exp = 0
        drop = False
        if exp and exp < ts_now:
            drop = True
        if drop_revoked and row.get("revoked"):
            revoked_at = str(row.get("revoked_at") or "")
            try:
                ra = datetime.fromisoformat(revoked_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                ra = 0.0
            if ra and (ts_now - ra) >= grace:
                drop = True
            elif not exp:
                drop = True
        if drop:
            removed.append(str(jti))
        else:
            kept[jti] = row
    if not dry_run and removed:
        data["links"] = kept
        path = save_mute_export_signed_urls(data, base=base)
        if audit:
            append_judge_ack_audit(
                "mute_export_sweep",
                actor="system",
                source="mute_export",
                extra={
                    "removed": len(removed),
                    "before": before,
                    "after": len(kept),
                    "jtis": removed[:50],
                },
            )
    else:
        path = mute_export_signed_urls_path(base=base)
    return {
        "ok": True,
        "path": path,
        "before": before,
        "after": len(kept),
        "removed": len(removed),
        "removed_jtis": removed,
        "dry_run": bool(dry_run),
    }


def maybe_sweep_mute_export_signed_urls(
    *,
    base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Env: RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SWEEP=1."""
    flag = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SWEEP", ""
    ).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return {"ok": True, "skipped": True, "reason": "sweep_disabled"}
    if flag not in {"1", "true", "yes", "on"}:
        return {"ok": True, "skipped": True, "reason": "sweep_not_configured"}
    grace_raw = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_SWEEP_REVOKED_DAYS", "7"
    ).strip()
    try:
        grace = float(grace_raw or 7)
    except Exception:
        grace = 7.0
    return sweep_mute_export_signed_urls(
        base=base, dry_run=dry_run, revoked_grace_days=grace
    )


def build_mute_export_signed_url(
    filename: str,
    *,
    base: Optional[str] = None,
    public_base: Optional[str] = None,
    ttl_sec: Optional[int] = None,
    secret: Optional[str] = None,
    now: Optional[float] = None,
    actor: Optional[str] = None,
    jti: Optional[str] = None,
    audit: bool = True,
) -> Dict[str, Any]:
    """HMAC signed download URL for /judge/mute-snapshots (jti + audit)."""
    from urllib.parse import quote

    name = os.path.basename((filename or "").strip())
    if not name or name != os.path.basename(name) or ".." in name:
        return {"ok": False, "error": "filename_invalid"}
    sec = secret if secret is not None else mute_export_signing_secret()
    if not sec:
        return {"ok": False, "error": "signing_secret_missing", "filename": name}
    ttl = int(ttl_sec if ttl_sec is not None else mute_export_signed_url_ttl_sec())
    exp = int(now if now is not None else time.time()) + max(60, ttl)
    token = (jti or "").strip() or secrets.token_urlsafe(16)
    sig = build_mute_export_signature(name, exp, secret=sec, jti=token)
    pub = (public_base or "").strip()
    if not pub:
        pub = os.environ.get("RAG_JUDGE_ACK_PUBLIC_URL", "").strip()
        if pub.endswith("/judge/ack-form"):
            pub = pub[: -len("/judge/ack-form")]
        elif pub.endswith("/judge/ack-ui"):
            pub = pub[: -len("/judge/ack-ui")]
    if not pub:
        try:
            from app.config import COLLAB_HTTP_HOST, COLLAB_HTTP_PORT

            host = COLLAB_HTTP_HOST or "127.0.0.1"
            port = int(COLLAB_HTTP_PORT or 8765)
            pub = f"http://{host}:{port}"
        except Exception:
            pub = "http://127.0.0.1:8765"
    url = (
        f"{pub.rstrip('/')}/judge/mute-snapshots"
        f"?file={quote(name)}&expires={exp}&jti={quote(token)}&sig={quote(sig)}"
    )
    reg = register_mute_export_signed_url(
        jti=token,
        filename=name,
        expires=exp,
        actor=actor,
        base=base,
        audit=audit,
    )
    return {
        "ok": True,
        "url": url,
        "filename": name,
        "expires": exp,
        "ttl_sec": ttl,
        "sig": sig,
        "jti": token,
        "registered": reg.get("ok"),
    }


def prune_judge_ack_digest_mute_exports(
    *,
    days: Optional[float] = None,
    keep: Optional[int] = None,
    base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Remove aged timestamped mute export artifacts under mute_exports/."""
    export_dir = judge_ack_digest_mute_exports_dir(base=base)
    if days is None and keep is None:
        return {
            "ok": False,
            "error": "days_or_keep_required",
            "dir": export_dir,
            "dry_run": bool(dry_run),
        }
    if not os.path.isdir(export_dir):
        return {
            "ok": True,
            "dir": export_dir,
            "before": 0,
            "after": 0,
            "removed": 0,
            "dry_run": bool(dry_run),
            "missing": True,
        }
    entries: List[Tuple[float, str]] = []
    for name in os.listdir(export_dir):
        if not name.startswith("judge_ack_digest_mute_snapshots_"):
            continue
        if not (name.endswith(".csv") or name.endswith(".jsonl")):
            continue
        path = os.path.join(export_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        entries.append((mtime, path))
    entries.sort(key=lambda x: x[0], reverse=True)  # newest first
    before = len(entries)
    cutoff_ts: Optional[float] = None
    if days is not None:
        cutoff_ts = time.time() - float(days) * 86400.0
    kept: List[Tuple[float, str]] = []
    for mtime, path in entries:
        if cutoff_ts is not None and mtime < cutoff_ts:
            continue
        kept.append((mtime, path))
    if keep is not None and len(kept) > int(keep):
        kept = kept[: int(keep)]
    kept_set = {p for _, p in kept}
    uniq_removed = [p for _, p in entries if p not in kept_set]
    if not dry_run:
        for path in uniq_removed:
            try:
                os.remove(path)
            except OSError:
                pass
    return {
        "ok": True,
        "dir": export_dir,
        "before": before,
        "after": len(kept_set),
        "removed": len(uniq_removed),
        "removed_paths": [os.path.basename(p) for p in uniq_removed],
        "dry_run": bool(dry_run),
        "days": float(days) if days is not None else None,
        "keep": int(keep) if keep is not None else None,
        "cutoff": (
            datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()
            if cutoff_ts is not None
            else None
        ),
    }


def maybe_prune_judge_ack_digest_mute_exports(
    *,
    base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Env: RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_RETENTION_DAYS / _KEEP / _PRUNE."""
    prune_flag = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_PRUNE", ""
    ).strip().lower()
    if prune_flag in {"0", "false", "no", "off"}:
        return {"ok": True, "skipped": True, "reason": "prune_disabled"}
    days_raw = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_RETENTION_DAYS", ""
    ).strip()
    keep_raw = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_KEEP", ""
    ).strip()
    days: Optional[float] = None
    keep: Optional[int] = None
    if days_raw:
        try:
            days = float(days_raw)
        except Exception:
            return {"ok": False, "skipped": True, "reason": "days_invalid"}
    if keep_raw:
        try:
            keep = int(keep_raw)
        except Exception:
            return {"ok": False, "skipped": True, "reason": "keep_invalid"}
    if days is None and keep is None:
        if prune_flag not in {"1", "true", "yes", "on"}:
            return {"ok": True, "skipped": True, "reason": "retention_not_configured"}
        days = 30.0
        keep = 50
    return prune_judge_ack_digest_mute_exports(
        days=days, keep=keep, base=base, dry_run=dry_run
    )


def read_judge_ack_digest_mute_export_file(
    filename: str,
    *,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a mute export artifact by basename from mute_exports/."""
    name = os.path.basename((filename or "").strip())
    if not name or name != os.path.basename(name) or ".." in name:
        return {"ok": False, "error": "filename_invalid"}
    if not name.startswith("judge_ack_digest_mute_snapshots"):
        return {"ok": False, "error": "filename_not_allowed"}
    path = os.path.join(judge_ack_digest_mute_exports_dir(base=base), name)
    if not os.path.isfile(path):
        return {"ok": False, "error": "not_found", "path": path}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    kind = "jsonl" if name.endswith(".jsonl") else "csv"
    return {
        "ok": True,
        "filename": name,
        "path": path,
        "text": text,
        "format": kind,
        "bytes": len(text.encode("utf-8")),
    }


def set_judge_ack_digest_mute(
    tenant_id: str,
    *,
    muted: bool = True,
    base: Optional[str] = None,
    ttl_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Persist mute flag to metadata JSON (dict form with optional TTL).

    Env mute still overrides reads. File entries may be bool or
    ``{muted, muted_at, expires_at}``.
    """
    tid = (tenant_id or "").strip()
    if not tid:
        return {"ok": False, "error": "tenant_required"}
    path = judge_ack_digest_mutes_path(base=base)
    data: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                data = dict(raw)
            elif isinstance(raw, list):
                data = {str(x).strip(): True for x in raw if str(x).strip()}
        except Exception:
            data = {}
    if muted:
        ttl = ttl_days if ttl_days is not None else _mute_ttl_days()
        entry: Dict[str, Any] = {
            "muted": True,
            "muted_at": _utcnow_iso(),
        }
        if ttl is not None and float(ttl) > 0:
            exp = datetime.now(timezone.utc).timestamp() + float(ttl) * 86400.0
            entry["expires_at"] = datetime.fromtimestamp(
                exp, tz=timezone.utc
            ).isoformat()
            entry["ttl_days"] = float(ttl)
        data[tid] = entry
    else:
        data.pop(tid, None)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    active = sorted(
        k for k, v in data.items() if _mute_value_active(v)
    )
    return {
        "ok": True,
        "tenant_id": tid,
        "muted": bool(muted),
        "path": path,
        "muted_tenants": active,
        "expires_at": (data.get(tid) or {}).get("expires_at")
        if muted and isinstance(data.get(tid), dict)
        else None,
    }


def prune_judge_ack_digest_mutes(
    *,
    base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Drop expired mute entries from the JSON store."""
    path = judge_ack_digest_mutes_path(base=base)
    if not os.path.isfile(path):
        return {
            "ok": True,
            "path": path,
            "before": 0,
            "after": 0,
            "removed": 0,
            "dry_run": bool(dry_run),
            "missing": True,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {"ok": False, "error": "read_failed", "path": path}
    if isinstance(raw, list):
        data = {str(x).strip(): True for x in raw if str(x).strip()}
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        return {"ok": False, "error": "invalid_format", "path": path}
    now_ts = time.time()
    kept: Dict[str, Any] = {}
    removed: List[str] = []
    for k, v in data.items():
        tid = str(k).strip()
        if not tid:
            continue
        if _mute_value_active(v, now_ts=now_ts):
            kept[tid] = v
        else:
            removed.append(tid)
    if not dry_run and removed:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
    return {
        "ok": True,
        "path": path,
        "before": len(data),
        "after": len(kept),
        "removed": len(removed),
        "removed_tenants": sorted(removed),
        "dry_run": bool(dry_run),
    }


def prune_judge_ack_digest_prefs(
    *,
    base: Optional[str] = None,
    days: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Remove stale per-tenant prefs older than ``days`` (updated_at)."""
    path = judge_ack_digest_prefs_path(base=base)
    if days is None:
        raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_PREFS_RETENTION_DAYS", "").strip()
        if raw:
            try:
                days = float(raw)
            except Exception:
                return {"ok": False, "error": "days_invalid", "path": path}
    if days is None or float(days) < 0:
        return {
            "ok": True,
            "skipped": True,
            "reason": "days_not_configured",
            "path": path,
            "dry_run": bool(dry_run),
        }
    prefs = load_judge_ack_digest_prefs(base=base)
    tenants = prefs.get("tenants") if isinstance(prefs.get("tenants"), dict) else {}
    if not tenants:
        return {
            "ok": True,
            "path": path,
            "before": 0,
            "after": 0,
            "removed": 0,
            "dry_run": bool(dry_run),
        }
    cutoff = time.time() - float(days) * 86400.0
    kept: Dict[str, Any] = {}
    removed: List[str] = []
    for tid, row in tenants.items():
        if not isinstance(row, dict):
            kept[tid] = row
            continue
        ts_raw = row.get("updated_at") or row.get("muted_at") or ""
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
        except Exception:
            kept[tid] = row
            continue
        if ts < cutoff:
            removed.append(str(tid))
        else:
            kept[tid] = row
    if not dry_run and removed:
        prefs["tenants"] = kept
        save_judge_ack_digest_prefs(prefs, base=base)
    return {
        "ok": True,
        "path": path,
        "days": float(days),
        "before": len(tenants),
        "after": len(kept) if not dry_run or not removed else len(tenants) - len(removed),
        "removed": len(removed),
        "removed_tenants": sorted(removed),
        "dry_run": bool(dry_run),
    }


def _messages_ttl_days() -> Optional[float]:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_MESSAGES_TTL_DAYS", "").strip()
    if not raw:
        return None
    try:
        days = float(raw)
        return days if days > 0 else None
    except Exception:
        return None


def _parse_iso_ts(raw: Any) -> Optional[float]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def prune_judge_ack_digest_messages(
    *,
    base: Optional[str] = None,
    days: Optional[float] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Remove stale digest Slack message refs older than ``days`` (updated_at)."""
    path = judge_ack_digest_messages_path(base=base)
    if days is None:
        days = _messages_ttl_days()
    if days is None or float(days) < 0:
        return {
            "ok": True,
            "skipped": True,
            "reason": "days_not_configured",
            "path": path,
            "dry_run": bool(dry_run),
        }
    data = load_judge_ack_digest_messages(base=base)
    tenants = {
        k: v
        for k, v in data.items()
        if k != "updated_at" and isinstance(v, dict)
    }
    if not tenants:
        return {
            "ok": True,
            "path": path,
            "days": float(days),
            "before": 0,
            "after": 0,
            "removed": 0,
            "removed_refs": 0,
            "removed_tenants": [],
            "dry_run": bool(dry_run),
            "missing": not os.path.isfile(path),
        }
    cutoff = time.time() - float(days) * 86400.0
    kept_tenants: Dict[str, Any] = {}
    removed_tenants: List[str] = []
    removed_refs = 0
    before_refs = 0
    for tid, row in tenants.items():
        msgs_in = list(row.get("messages") or []) if isinstance(row.get("messages"), list) else []
        # Count legacy single ref as one
        legacy_ch = str(row.get("channel_id") or "").strip()
        legacy_ts = str(row.get("message_ts") or "").strip()
        if not msgs_in and legacy_ch and legacy_ts:
            msgs_in = [
                {
                    "channel_id": legacy_ch,
                    "message_ts": legacy_ts,
                    "updated_at": row.get("updated_at"),
                }
            ]
        before_refs += len(msgs_in)
        kept_msgs: List[Dict[str, Any]] = []
        for item in msgs_in:
            if not isinstance(item, dict):
                continue
            ts = _parse_iso_ts(item.get("updated_at"))
            if ts is None:
                # Keep unparseable ages (don't drop blindly)
                kept_msgs.append(item)
                continue
            if ts < cutoff:
                removed_refs += 1
                continue
            kept_msgs.append(item)
        row_ts = _parse_iso_ts(row.get("updated_at"))
        if not kept_msgs:
            # Drop tenant if row also stale or no usable timestamps left
            if row_ts is None or row_ts < cutoff:
                removed_tenants.append(str(tid))
                continue
            # Row still fresh but no msgs — keep empty shell? Drop for cleanliness.
            removed_tenants.append(str(tid))
            continue
        new_row = dict(row)
        new_row["messages"] = kept_msgs
        new_row["channel_id"] = str(kept_msgs[0].get("channel_id") or "").strip()
        new_row["message_ts"] = str(kept_msgs[0].get("message_ts") or "").strip()
        new_row["updated_at"] = kept_msgs[0].get("updated_at") or row.get("updated_at")
        kept_tenants[str(tid)] = new_row
    if not dry_run and (removed_refs or removed_tenants):
        out = dict(kept_tenants)
        save_judge_ack_digest_messages(out, base=base)
    after_refs = sum(
        len(r.get("messages") or [])
        for r in kept_tenants.values()
        if isinstance(r, dict)
    )
    return {
        "ok": True,
        "path": path,
        "days": float(days),
        "before": before_refs,
        "after": after_refs if not dry_run else before_refs - removed_refs,
        "removed": removed_refs + len(removed_tenants),
        "removed_refs": removed_refs,
        "removed_tenants": sorted(removed_tenants),
        "dry_run": bool(dry_run),
    }


def digest_history_reconcile_enabled() -> bool:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_HISTORY_RECONCILE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def reconcile_judge_ack_digest_message_refs(
    *,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
    bot_token: Optional[str] = None,
    dry_run: bool = False,
    drop_missing: bool = True,
    repair_channel: bool = True,
) -> Dict[str, Any]:
    """Slack conversations.history ile digest message-ref'leri doğrula / onar / düşür."""
    path = judge_ack_digest_messages_path(base=base)
    data = load_judge_ack_digest_messages(base=base)
    tenants = {
        k: v
        for k, v in data.items()
        if k != "updated_at" and isinstance(v, dict)
    }
    want = (tenant_id or "").strip()
    if want:
        tenants = {want: tenants[want]} if want in tenants else {}
    details: List[Dict[str, Any]] = []
    checked = 0
    kept = 0
    dropped = 0
    repaired = 0
    errors = 0
    new_data: Dict[str, Any] = {
        k: v
        for k, v in data.items()
        if k == "updated_at" or (isinstance(v, dict) and k not in tenants)
    }
    for tid, row in tenants.items():
        refs = list_judge_ack_digest_message_refs(tid, base=base)
        kept_msgs: List[Dict[str, Any]] = []
        # Preserve updated_at from store when available
        store_msgs = (
            list(row.get("messages") or [])
            if isinstance(row.get("messages"), list)
            else []
        )
        meta_by_ts = {
            str(m.get("message_ts") or "").strip(): m
            for m in store_msgs
            if isinstance(m, dict) and str(m.get("message_ts") or "").strip()
        }
        for ref in refs:
            checked += 1
            ch = str(ref.get("channel_id") or "").strip()
            ts = str(ref.get("message_ts") or "").strip()
            meta = dict(meta_by_ts.get(ts) or {"channel_id": ch, "message_ts": ts})
            fetched = fetch_slack_message_by_ts(
                channel_id=ch, message_ts=ts, bot_token=bot_token
            )
            action = "keep"
            err = None
            if fetched.get("ok"):
                kept += 1
                kept_msgs.append(meta)
            elif fetched.get("error") == "message_not_found" and repair_channel:
                looked = lookup_slack_channel_for_ts(
                    thread_ts=ts, bot_token=bot_token, channel_hint=ch
                )
                if looked.get("ok") and looked.get("channel"):
                    new_ch = str(looked.get("channel") or "").strip()
                    meta = dict(meta)
                    meta["channel_id"] = new_ch
                    meta["updated_at"] = _utcnow_iso()
                    meta["repaired_from"] = ch
                    kept_msgs.append(meta)
                    repaired += 1
                    kept += 1
                    action = "repaired"
                elif drop_missing:
                    dropped += 1
                    action = "dropped"
                    err = "message_not_found"
                else:
                    kept += 1
                    kept_msgs.append(meta)
                    action = "keep_missing"
                    err = "message_not_found"
            elif fetched.get("error") == "message_not_found" and drop_missing:
                dropped += 1
                action = "dropped"
                err = "message_not_found"
            elif fetched.get("error") in {"bot_token_missing", "ref_incomplete"}:
                # Don't mutate store when we can't talk to Slack
                errors += 1
                kept += 1
                kept_msgs.append(meta)
                action = "keep_error"
                err = str(fetched.get("error"))
            else:
                # Transient API errors — keep
                errors += 1
                kept += 1
                kept_msgs.append(meta)
                action = "keep_error"
                err = str(fetched.get("error") or "history_failed")
            details.append(
                {
                    "tenant_id": tid,
                    "channel_id": ch,
                    "message_ts": ts,
                    "action": action,
                    "error": err,
                }
            )
        if kept_msgs:
            new_row = dict(row)
            new_row["messages"] = kept_msgs
            new_row["channel_id"] = str(kept_msgs[0].get("channel_id") or "").strip()
            new_row["message_ts"] = str(kept_msgs[0].get("message_ts") or "").strip()
            new_row["updated_at"] = kept_msgs[0].get("updated_at") or row.get(
                "updated_at"
            )
            new_data[tid] = new_row
        # else: tenant dropped entirely (all refs missing)
    mutated = repaired > 0 or dropped > 0
    if not dry_run and mutated:
        save_judge_ack_digest_messages(new_data, base=base)
    report = {
        "ok": True,
        "path": path,
        "dry_run": bool(dry_run),
        "checked": checked,
        "kept": kept,
        "dropped": dropped,
        "repaired": repaired,
        "errors": errors,
        "tenants": len(tenants),
        "details": details,
    }
    emit_judge_ack_digest_msgref_reconcile_metric(report)
    return report


def emit_judge_ack_digest_msgref_reconcile_metric(
    report: Optional[Dict[str, Any]] = None,
) -> None:
    """Prometheus/JSONL: rag_judge_ack_digest_msgref_reconcile_* ."""
    try:
        from rag.metrics import record_metric

        r = report or {}
        if r.get("skipped"):
            record_metric(
                "judge_ack_digest_msgref_reconcile",
                values={"result": "skipped", "skipped": True},
            )
            return
        record_metric(
            "judge_ack_digest_msgref_reconcile",
            values={
                "result": "ok" if r.get("ok") else "fail",
                "checked": int(r.get("checked") or 0),
                "kept": int(r.get("kept") or 0),
                "dropped": int(r.get("dropped") or 0),
                "repaired": int(r.get("repaired") or 0),
                "errors": int(r.get("errors") or 0),
                "dry_run": bool(r.get("dry_run")),
            },
        )
    except Exception:
        pass


def maybe_reconcile_judge_ack_digest_message_refs(
    *,
    base: Optional[str] = None,
    dry_run: bool = False,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Env-gated history reconcile (RAG_JUDGE_ACK_DIGEST_HISTORY_RECONCILE)."""
    if not digest_history_reconcile_enabled():
        out = {"ok": True, "skipped": True, "reason": "reconcile_disabled"}
        emit_judge_ack_digest_msgref_reconcile_metric(out)
        return out
    return reconcile_judge_ack_digest_message_refs(
        base=base, dry_run=dry_run, bot_token=bot_token
    )


def maybe_prune_judge_ack_digest_mutes(
    *,
    base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Env-gated mute/prefs/messages retention (RAG_JUDGE_ACK_DIGEST_MUTE_PRUNE)."""
    flag = os.environ.get("RAG_JUDGE_ACK_DIGEST_MUTE_PRUNE", "").strip().lower()
    msg_flag = os.environ.get("RAG_JUDGE_ACK_DIGEST_MESSAGES_PRUNE", "").strip().lower()
    if (
        flag in {"0", "false", "no", "off"}
        and msg_flag in {"0", "false", "no", "off"}
        and not digest_history_reconcile_enabled()
    ):
        return {"ok": True, "skipped": True, "reason": "prune_disabled"}
    # Auto when TTL configured or explicit prune=1
    ttl = _mute_ttl_days()
    prefs_days_raw = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_PREFS_RETENTION_DAYS", ""
    ).strip()
    messages_ttl = _messages_ttl_days()
    explicit = flag in {"1", "true", "yes", "on"} or msg_flag in {
        "1",
        "true",
        "yes",
        "on",
    }
    recon_on = digest_history_reconcile_enabled()
    if (
        not explicit
        and ttl is None
        and not prefs_days_raw
        and messages_ttl is None
        and not recon_on
    ):
        return {"ok": True, "skipped": True, "reason": "retention_not_configured"}
    mute_report = prune_judge_ack_digest_mutes(base=base, dry_run=dry_run)
    prefs_report = prune_judge_ack_digest_prefs(base=base, dry_run=dry_run)
    messages_report = prune_judge_ack_digest_messages(base=base, dry_run=dry_run)
    reconcile_report = maybe_reconcile_judge_ack_digest_message_refs(
        base=base, dry_run=dry_run
    )
    return {
        "ok": bool(
            mute_report.get("ok")
            and prefs_report.get("ok")
            and messages_report.get("ok")
            and reconcile_report.get("ok")
        ),
        "mutes": mute_report,
        "prefs": prefs_report,
        "messages": messages_report,
        "reconcile": reconcile_report,
        "dry_run": bool(dry_run),
    }



def resolve_judge_ack_digest_quiet_hours(
    tenant_id: Optional[str] = None,
    *,
    quiet_hours: Optional[str] = None,
    base: Optional[str] = None,
) -> Optional[str]:
    """Açık arg > tenant override dosya/env > global quiet hours."""
    if quiet_hours is not None:
        return quiet_hours
    tid = (tenant_id or "").strip()
    if tid:
        mapping = parse_judge_ack_digest_quiet_hours(base=base)
        if tid in mapping:
            return mapping[tid]
        if tid.lower() in mapping:
            return mapping[tid.lower()]
    return _judge_ack_digest_quiet_spec(None)


def build_judge_ack_digest_slack_blocks(
    summary: Dict[str, Any],
    *,
    tenant_id: Optional[str] = None,
    diff: Optional[Dict[str, Any]] = None,
    heatmap: Optional[Dict[str, Any]] = None,
    heatmap_bucket: Optional[str] = None,
    muted: Optional[bool] = None,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Haftalık ack digest için Slack Block Kit (+ digest-diff + heatmap canvas)."""
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
    bucket = (
        str(heatmap_bucket).strip().lower()
        if heatmap_bucket
        else resolve_heatmap_bucket(tid, base=base)
    )
    if bucket not in {"day", "hour"}:
        bucket = "day"
    is_muted = (
        bool(muted)
        if muted is not None
        else (bool(tid) and is_judge_ack_digest_muted(tid, base=base))
    )
    title = "RAG judge ack audit digest"
    if tid:
        title += f" · tenant `{tid}`"
        if is_muted:
            title += " · MUTED"
    header = (
        f"*{title}* (last {summary.get('since_hours')}h)\n"
        f"Total events: *{summary.get('total', 0)}* · "
        f"actors: *{summary.get('actor_count', 0)}* · heatmap=`{bucket}`"
    )
    blocks: List[Dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Judge ack digest"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
    ]
    if diff is not None:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": format_judge_ack_digest_diff_text(diff),
                },
            }
        )
        blocks.append({"type": "divider"})
    if heatmap is not None:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": format_ack_heatmap_mrkdwn(heatmap),
                },
            }
        )
    by_event = summary.get("by_event") or {}
    if by_event:
        parts = [f"`{k}` = *{v}*" for k, v in sorted(by_event.items())]
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*By event*\n" + " · ".join(parts)},
            }
        )
    by_tenant = summary.get("by_tenant") or {}
    muted_set = parse_judge_ack_digest_mutes(base=base) if not tid else set()
    if by_tenant and not tid:
        parts = []
        for k, v in sorted(by_tenant.items())[:12]:
            label = f"`{k}`={v}"
            if str(k) in muted_set:
                label += " _(muted)_"
            parts.append(label)
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*By tenant*\n" + ", ".join(parts)},
            }
        )
    actors = summary.get("actors") or []
    if actors:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Actors: " + ", ".join(f"`{a}`" for a in actors[:20]),
                    }
                ],
            }
        )
    last = summary.get("last_by_event") or {}
    if last:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Last: "
                        + ", ".join(f"{k}@{v}" for k, v in sorted(last.items())[:6]),
                    }
                ],
            }
        )

    elements: List[Dict[str, Any]] = []
    interactive = os.environ.get("RAG_JUDGE_SLACK_SIGNING_SECRET", "").strip() or (
        os.environ.get("RAG_JUDGE_SLACK_INTERACTIVE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    zoom_target = "day" if bucket == "hour" else "hour"
    zoom_value = f"{tid}|{zoom_target}" if tid else zoom_target
    if interactive:
        if tid:
            elements.append(
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Unmute tenant" if is_muted else "Mute tenant",
                    },
                    "action_id": (
                        "judge_ack_digest_unmute"
                        if is_muted
                        else "judge_ack_digest_mute"
                    ),
                    "value": tid,
                }
            )
            if is_muted:
                elements.append(
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Catch-up digest",
                        },
                        "action_id": "judge_ack_digest_catch_up",
                        "value": tid,
                        "style": "primary",
                    }
                )
        if is_muted:
            # Prefer mute+snapshot CSV upload when tenant is muted (keep-on-mute).
            export_btn = {
                "type": "button",
                "text": {"type": "plain_text", "text": "Mute snapshots"},
                "action_id": "judge_ack_digest_export_mute_snapshots",
                "value": tid or "all",
                "style": "primary",
            }
            elements.append(export_btn)
            elements.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Revoke export"},
                    "action_id": "judge_ack_digest_revoke_mute_export",
                    "value": tid or "latest",
                    "style": "danger",
                }
            )
        else:
            export_btn = {
                "type": "button",
                "text": {"type": "plain_text", "text": "Export JSONL"},
                "action_id": "judge_ack_digest_reexport",
                "value": "jsonl",
                "style": "primary",
            }
            elements.append(export_btn)
        # CSV only when mute button is absent (room under 5-cap with form+zoom).
        if not tid and not is_muted:
            elements.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Export CSV"},
                    "action_id": "judge_ack_digest_reexport_csv",
                    "value": "csv",
                }
            )
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Ack soft-fail"},
                "action_id": "judge_ack_interactive",
                "value": "ack",
            }
        )
        # Heatmap zoom: drop when muted+revoke already fills the 5-cap
        if not (is_muted and tid):
            elements.append(
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": f"Heatmap {zoom_target}",
                    },
                    "action_id": "judge_ack_digest_heatmap_zoom",
                    "value": zoom_value,
                }
            )
    ack_urls = judge_ack_canvas_urls(
        tenant_id=tid, hours=summary.get("since_hours")
    )
    ack_url = ack_urls.get("form") or judge_ack_public_url()
    if ack_url:
        # Prefer mute/catch-up/mute-snapshots/revoke + form: drop CSV / audit export
        if len(elements) >= 4:
            drop_ids = {
                "judge_ack_digest_reexport_csv",
            }
            if any(
                e.get("action_id")
                in {
                    "judge_ack_digest_catch_up",
                    "judge_ack_digest_revoke_mute_export",
                }
                for e in elements
            ):
                # Keep mute-snapshots/revoke when muted; drop audit JSONL reexport
                drop_ids.add("judge_ack_digest_reexport")
            elements = [
                e for e in elements if e.get("action_id") not in drop_ids
            ]
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Ack form"},
                "url": ack_url,
                "action_id": "judge_ack_digest_open",
            }
        )
    # Download link stays as context (actions capped at 5)
    export_url = ack_urls.get("export") or judge_ack_export_public_url()
    export_csv = ack_urls.get("export_csv") or (
        f"{export_url}?format=csv" if export_url else ""
    )
    if elements:
        blocks.append(
            {
                "type": "actions",
                "block_id": "judge_ack_digest_actions",
                "elements": elements[:5],
            }
        )
    ctx_bits = []
    if ack_url:
        label = "Canvas" if tid else "Ack form"
        ctx_bits.append(f"{label}: <{ack_url}|open>")
    if export_url:
        ctx_bits.append(
            f"Download: <{export_url}|jsonl> · <{export_csv or export_url}|csv>"
        )
    if tid:
        ctx_bits.append(f"tenant `{tid}`")
        if is_muted:
            ctx_bits.append("*muted*")
            revoke_ui = mute_export_revoke_public_url()
            if revoke_ui:
                ctx_bits.append(f"Revoke UI: <{revoke_ui}|open>")
    ctx_bits.append(f"heatmap=`{bucket}`")
    if ctx_bits:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": " · ".join(ctx_bits)}],
            }
        )
    return blocks



def judge_ack_export_public_url() -> str:
    """Digest re-export deep-link (JSONL download)."""
    explicit = os.environ.get("RAG_JUDGE_ACK_EXPORT_PUBLIC_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    base = judge_ack_public_url()
    if base:
        # /judge/ack-form → /judge/ack-export
        if base.endswith("/judge/ack-form"):
            return base[: -len("/judge/ack-form")] + "/judge/ack-export"
        return base.rstrip("/") + "/judge/ack-export"
    try:
        from app.config import COLLAB_HTTP_PORT, COLLAB_WS_PUBLIC_HOST

        host = (COLLAB_WS_PUBLIC_HOST or "localhost").strip() or "localhost"
        port = int(COLLAB_HTTP_PORT or 8766)
        return f"http://{host}:{port}/judge/ack-export"
    except Exception:
        pub = os.environ.get("RAG_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if pub:
            return f"{pub}/judge/ack-export"
    return ""


def mute_export_revoke_public_url() -> str:
    """Admin UI deep-link for mute export signed URL revoke."""
    explicit = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_REVOKE_PUBLIC_URL", ""
    ).strip()
    if explicit:
        return explicit.rstrip("/")
    base = judge_ack_public_url()
    if base:
        if base.endswith("/judge/ack-form"):
            return base[: -len("/judge/ack-form")] + "/judge/mute-export-revoke"
        if base.endswith("/judge/ack-ui"):
            return base[: -len("/judge/ack-ui")] + "/judge/mute-export-revoke"
        return base.rstrip("/") + "/judge/mute-export-revoke"
    try:
        from app.config import COLLAB_HTTP_PORT, COLLAB_WS_PUBLIC_HOST

        host = (COLLAB_WS_PUBLIC_HOST or "localhost").strip() or "localhost"
        port = int(COLLAB_HTTP_PORT or 8766)
        return f"http://{host}:{port}/judge/mute-export-revoke"
    except Exception:
        pub = os.environ.get("RAG_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if pub:
            return f"{pub}/judge/mute-export-revoke"
    return ""


def resolve_mute_export_revoke_ref(
    value: str = "",
    *,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Map Slack button value → jti/filename for revoke (latest active if empty)."""
    raw = (value or "").strip()
    if raw and raw.lower() not in {"latest", "all", "*", "csv"}:
        return {"ok": True, "ref": raw, "resolved": "explicit"}
    listed = list_mute_export_signed_urls(
        base=base, include_revoked=False, include_expired=False, limit=20
    )
    for row in listed.get("links") or []:
        if row.get("revoked") or row.get("expired"):
            continue
        jti = str(row.get("jti") or "").strip()
        if jti:
            return {
                "ok": True,
                "ref": jti,
                "resolved": "latest",
                "filename": row.get("filename"),
            }
    return {"ok": False, "error": "no_active_signed_url"}


def handle_slack_mute_export_revoke(
    payload: Dict[str, Any],
    *,
    value: str = "",
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Slack Block Kit / modal submit: revoke latest (or explicit) mute export signed URL."""
    actor = slack_interactive_actor(payload)
    resolved = resolve_mute_export_revoke_ref(value, base=base)
    if not resolved.get("ok"):
        text = "No active mute export signed URL to revoke."
        return {
            "ok": False,
            "mode": "digest_mute_export_revoke",
            "error": resolved.get("error") or "not_found",
            "text": text,
            "actor": actor,
        }
    ref = str(resolved.get("ref") or "")
    note = "slack_block_kit"
    # Optional note from confirm modal
    try:
        view = payload.get("view") or {}
        state = (view.get("state") or {}).get("values") or {}
        block = state.get("revoke_note_block") or {}
        field = block.get("revoke_note") or {}
        modal_note = str(field.get("value") or "").strip()
        if modal_note:
            note = f"slack_confirm_modal:{modal_note[:400]}"
        elif str(view.get("callback_id") or "") == "judge_mute_export_revoke_modal":
            note = "slack_confirm_modal"
    except Exception:
        pass
    result = revoke_mute_export_signed_url(
        ref,
        actor=actor or "slack",
        note=note,
        base=base,
    )
    text = (
        f"Mute export signed URL revoked · jti=`{ref}` · by *{actor or 'slack'}*"
        if result.get("ok")
        else f"Mute export revoke failed: `{result.get('error')}`"
    )
    channel_id = str(
        ((payload.get("channel") or {}).get("id"))
        or ((payload.get("container") or {}).get("channel_id"))
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
        or ""
    ).strip() or None
    user_id = slack_interactive_user_id(payload)
    token = os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    ephemeral: Dict[str, Any] = {"ok": False, "skipped": True}
    if token and channel_id and user_id:
        ephemeral = slack_api(
            "chat.postEphemeral",
            bot_token=token,
            json_body={"channel": channel_id, "user": user_id, "text": text},
        )
    return {
        "ok": bool(result.get("ok")),
        "mode": "digest_mute_export_revoke",
        "actor": actor,
        "ref": ref,
        "resolved": resolved.get("resolved"),
        "revoke": result,
        "text": text,
        "ephemeral": {
            "ok": bool(ephemeral.get("ok")),
            "skipped": bool(ephemeral.get("skipped")),
            "error": ephemeral.get("error"),
        },
        "channel_id": channel_id,
    }


def build_mute_export_revoke_modal_view(
    *,
    ref: str,
    filename: Optional[str] = None,
    channel_id: Optional[str] = None,
    message_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Slack confirm modal before mute export signed URL revoke."""
    meta: Dict[str, Any] = {"ref": str(ref or "").strip()}
    if filename:
        meta["filename"] = str(filename)
    if channel_id:
        meta["channel_id"] = str(channel_id)
    if message_ts:
        meta["message_ts"] = str(message_ts)
    fname = str(filename or "").strip()
    body = f"Revoke mute export signed URL jti=`{ref}`"
    if fname:
        body += f" · file=`{fname}`"
    body += "?\nThis invalidates the download link immediately."
    return {
        "type": "modal",
        "callback_id": "judge_mute_export_revoke_modal",
        "title": {"type": "plain_text", "text": "Revoke mute export"},
        "submit": {"type": "plain_text", "text": "Revoke"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "private_metadata": json.dumps(meta, ensure_ascii=False),
        "blocks": [
            {
                "type": "section",
                "block_id": "revoke_confirm_block",
                "text": {"type": "mrkdwn", "text": body},
            },
            {
                "type": "input",
                "block_id": "revoke_note_block",
                "optional": True,
                "label": {"type": "plain_text", "text": "Note"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "revoke_note",
                    "multiline": False,
                    "placeholder": {
                        "type": "plain_text",
                        "text": "optional revoke reason",
                    },
                },
            },
        ],
    }


def parse_mute_export_revoke_modal_metadata(payload: Dict[str, Any]) -> Dict[str, str]:
    view = payload.get("view") or {}
    raw = str(view.get("private_metadata") or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for key in ("ref", "filename", "channel_id", "message_ts"):
        val = str(data.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def open_mute_export_revoke_confirm_modal(
    payload: Dict[str, Any],
    *,
    value: str = "",
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """block_actions → views.open confirm modal (does not revoke yet)."""
    resolved = resolve_mute_export_revoke_ref(value, base=base)
    channel_id = str(
        ((payload.get("channel") or {}).get("id"))
        or ((payload.get("container") or {}).get("channel_id"))
        or ""
    ).strip() or None
    message_ts = str(
        ((payload.get("message") or {}).get("ts"))
        or ((payload.get("container") or {}).get("message_ts"))
        or ""
    ).strip() or None
    if not resolved.get("ok"):
        text = "No active mute export signed URL to revoke."
        return {
            "ok": False,
            "mode": "digest_mute_export_revoke",
            "error": resolved.get("error") or "not_found",
            "text": text,
            "channel_id": channel_id,
        }
    ref = str(resolved.get("ref") or "")
    trigger_id = str(payload.get("trigger_id") or "").strip()
    # Allow skipping modal when explicitly disabled (tests / break-glass)
    confirm = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_MUTE_EXPORT_REVOKE_CONFIRM", "1"
    ).strip().lower()
    if confirm in {"0", "false", "no", "off"}:
        out = handle_slack_mute_export_revoke(payload, value=ref, base=base)
        return out
    opened = open_slack_modal(
        trigger_id=trigger_id,
        view=build_mute_export_revoke_modal_view(
            ref=ref,
            filename=str(resolved.get("filename") or "") or None,
            channel_id=channel_id,
            message_ts=message_ts,
        ),
    )
    return {
        "ok": bool(opened.get("ok")),
        "mode": "modal_open",
        "confirm": "mute_export_revoke",
        "ref": ref,
        "resolved": resolved.get("resolved"),
        "error": opened.get("error"),
        "opened": opened,
        "channel_id": channel_id,
        "message_ts": message_ts,
        "text": f"Confirm revoke modal for jti=`{ref}`",
    }


def build_judge_ack_digest_text(
    summary: Dict[str, Any],
    *,
    tenant_id: Optional[str] = None,
    diff: Optional[Dict[str, Any]] = None,
    heatmap: Optional[Dict[str, Any]] = None,
    base: Optional[str] = None,
) -> str:
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
    title = "*RAG judge ack audit digest*"
    if tid:
        title += f" · tenant `{tid}`"
    lines = [
        f"{title} (last {summary.get('since_hours')}h)",
        f"Total events: *{summary.get('total', 0)}* · actors: *{summary.get('actor_count', 0)}*",
    ]
    if diff is not None:
        lines.append(format_judge_ack_digest_diff_text(diff))
    if heatmap is not None:
        lines.append(format_ack_heatmap_mrkdwn(heatmap))
    canvas = judge_ack_canvas_urls(tenant_id=tid, hours=summary.get("since_hours"))
    if canvas.get("form"):
        lines.append(f"Canvas: {canvas['form']}")
    by_event = summary.get("by_event") or {}
    if by_event:
        parts = [f"`{k}`={v}" for k, v in sorted(by_event.items())]
        lines.append("By event: " + ", ".join(parts))
    by_tenant = summary.get("by_tenant") or {}
    muted_set = parse_judge_ack_digest_mutes(base=base) if not tid else set()
    if by_tenant and not tid:
        parts = []
        for k, v in sorted(by_tenant.items())[:12]:
            label = f"`{k}`={v}"
            if str(k) in muted_set:
                label += " (muted)"
            parts.append(label)
        lines.append("By tenant: " + ", ".join(parts))
    actors = summary.get("actors") or []
    if actors:
        lines.append("Actors: " + ", ".join(f"`{a}`" for a in actors[:20]))
    last = summary.get("last_by_event") or {}
    if last:
        lines.append(
            "Last: " + ", ".join(f"{k}@{v}" for k, v in sorted(last.items())[:6])
        )
    return "\n".join(lines)


def _judge_ack_digest_quiet_spec(quiet_hours: Optional[str] = None) -> Optional[str]:
    if quiet_hours is not None:
        return quiet_hours
    try:
        from app.config import JUDGE_ACK_DIGEST_QUIET_HOURS, NOTIFY_DIGEST_QUIET_HOURS
    except ImportError:
        JUDGE_ACK_DIGEST_QUIET_HOURS = ""
        NOTIFY_DIGEST_QUIET_HOURS = ""
    return (JUDGE_ACK_DIGEST_QUIET_HOURS or NOTIFY_DIGEST_QUIET_HOURS or None)


def digest_snapshot_keep_on_mute_enabled() -> bool:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_SNAPSHOT_KEEP_ON_MUTE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def digest_catch_up_on_unmute_enabled() -> bool:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_CATCH_UP_ON_UNMUTE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def dispatch_judge_ack_digest_catch_up(
    tenant_id: str,
    *,
    base: Optional[str] = None,
    actor: str = "slack",
    webhook: Optional[str] = None,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    bot_token: Optional[str] = None,
    unmute: bool = True,
) -> Dict[str, Any]:
    """Force-dispatch a tenant digest (optionally unmute first) for catch-up."""
    tid = (tenant_id or "").strip()
    if not tid:
        return {"ok": False, "error": "tenant_required", "mode": "digest_catch_up"}
    unmuted = False
    mute_saved: Optional[Dict[str, Any]] = None
    if unmute and is_judge_ack_digest_muted(tid, base=base):
        mute_saved = set_judge_ack_digest_mute(tid, muted=False, base=base)
        unmuted = bool(mute_saved.get("ok"))
        if unmuted:
            append_judge_ack_audit(
                "digest_unmute",
                actor=actor,
                note=f"Tenant `{tid}` unmuted via catch-up digest",
                source="slack_interactive",
                extra={"tenant_id": tid, "muted": False, "catch_up": True},
            )
    try:
        hours = float(os.environ.get("RAG_JUDGE_ACK_DIGEST_HOURS", "168") or 168)
    except Exception:
        hours = 168.0
    audit_path = os.environ.get("RAG_JUDGE_ACK_AUDIT", "").strip() or None
    summary = summarize_judge_ack_audit(
        path=audit_path, since_hours=hours, tenant_id=tid
    )
    summary = {**summary, "tenant_id": tid}
    dispatched = dispatch_judge_ack_digest(
        summary,
        webhook=webhook,
        tenant_id=tid,
        base=base,
        ignore_quiet_hours=True,
    )
    text = (
        f"Catch-up digest for tenant `{tid}`: "
        f"{'sent' if dispatched.get('ok') and not dispatched.get('skipped') else 'skipped'}"
    )
    if dispatched.get("reason"):
        text += f" · reason=`{dispatched.get('reason')}`"
    if unmuted:
        text += " · unmuted"
    thread_reply = post_slack_thread_message(
        text=text,
        channel_id=channel_id,
        thread_ts=thread_ts,
        bot_token=bot_token,
    )
    return {
        "ok": bool(dispatched.get("ok")),
        "mode": "digest_catch_up",
        "tenant_id": tid,
        "unmuted": unmuted,
        "mute": mute_saved,
        "dispatch": dispatched,
        "text": text,
        "thread_reply": thread_reply,
        "channel_id": channel_id,
        "message_ts": thread_ts,
    }


def maybe_keep_digest_snapshot_on_mute(
    summary: Dict[str, Any],
    *,
    tenant_id: Optional[str] = None,
    base: Optional[str] = None,
    include_diff: Optional[bool] = None,
) -> Dict[str, Any]:
    """Muted tenant digests still refresh snapshot so unmute Δ is not inflated."""
    use_diff = digest_diff_enabled() if include_diff is None else bool(include_diff)
    if not use_diff or not digest_snapshot_keep_on_mute_enabled():
        return {
            "ok": True,
            "kept_snapshot": False,
            "skipped": True,
            "reason": "keep_on_mute_disabled" if use_diff else "diff_disabled",
            "snapshot_path": None,
        }
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
    prev = load_judge_ack_digest_snapshot(tenant_id=tid, base=base)
    diff = diff_judge_ack_digest(prev, summary)
    path = save_judge_ack_digest_snapshot(
        summary, tenant_id=tid, base=base, diff=diff
    )
    return {
        "ok": True,
        "kept_snapshot": True,
        "skipped": False,
        "snapshot_path": path,
        "diff": diff,
    }


def dispatch_judge_ack_digest(
    summary: Dict[str, Any],
    *,
    webhook: Optional[str] = None,
    dry_run: bool = False,
    tenant_id: Optional[str] = None,
    quiet_hours: Optional[str] = None,
    ignore_quiet_hours: bool = False,
    timezone_name: Optional[str] = None,
    base: Optional[str] = None,
    block_kit: Optional[bool] = None,
    include_diff: Optional[bool] = None,
) -> Dict[str, Any]:
    """Haftalık ack audit özetini Slack webhook'a gönder (Block Kit + quiet hours + digest-diff)."""
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
    if tid and is_judge_ack_digest_muted(tid, base=base):
        kept = maybe_keep_digest_snapshot_on_mute(
            summary, tenant_id=tid, base=base, include_diff=include_diff
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "muted",
            "tenant_id": tid,
            "configured": bool(
                resolve_judge_ack_digest_webhook(tid, url=webhook, base=base)
            ),
            "kept_snapshot": bool(kept.get("kept_snapshot")),
            "snapshot_path": kept.get("snapshot_path"),
            "diff": kept.get("diff"),
        }
    if not ignore_quiet_hours:
        try:
            from rag.collab_notify_digest import is_quiet_hours

            # Explicit quiet_hours arg wins; else tenant override file then global
            if quiet_hours is not None:
                qspec = quiet_hours
            else:
                qspec = resolve_judge_ack_digest_quiet_hours(
                    tid, quiet_hours=None, base=base
                )
            tz = timezone_name
            if not tz:
                try:
                    from app.config import JUDGE_ACK_DIGEST_TIMEZONE
                except ImportError:
                    JUDGE_ACK_DIGEST_TIMEZONE = ""
                tz = JUDGE_ACK_DIGEST_TIMEZONE or None
            if is_quiet_hours(
                quiet_spec=qspec,
                timezone_name=tz,
                tenant_id=tid,
            ):
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "quiet_hours",
                    "tenant_id": tid,
                    "quiet_hours": qspec,
                    "configured": bool(
                        resolve_judge_ack_digest_webhook(
                            tid, url=webhook, base=base
                        )
                    ),
                }
        except Exception:
            pass

    use_diff = digest_diff_enabled() if include_diff is None else bool(include_diff)
    diff: Optional[Dict[str, Any]] = None
    snapshot_path = None
    muted_tenants = sorted(parse_judge_ack_digest_mutes(base=base))
    if use_diff:
        prev = load_judge_ack_digest_snapshot(tenant_id=tid, base=base)
        diff = diff_judge_ack_digest(prev, summary)
        # Annotate global digests with currently muted tenants.
        if not tid and muted_tenants:
            diff["muted_tenants"] = muted_tenants

    heatmap_bucket = resolve_heatmap_bucket(tid, base=base)
    heatmap: Optional[Dict[str, Any]] = None
    if os.environ.get("RAG_JUDGE_ACK_DIGEST_HEATMAP", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        try:
            max_buckets_raw = os.environ.get(
                "RAG_JUDGE_ACK_DIGEST_HEATMAP_MAX_BUCKETS", ""
            ).strip()
            max_buckets = int(max_buckets_raw) if max_buckets_raw else (
                24 if heatmap_bucket == "hour" else 14
            )
            heatmap = build_ack_heatmap(
                since_hours=float(summary.get("since_hours") or 168),
                tenant_id=tid,
                bucket=heatmap_bucket,
                max_buckets=max_buckets,
            )
        except Exception:
            heatmap = None

    url = resolve_judge_ack_digest_webhook(tid, url=webhook, base=base)
    text = build_judge_ack_digest_text(
        summary, tenant_id=tid, diff=diff, heatmap=heatmap, base=base
    )
    use_blocks = block_kit
    if use_blocks is None:
        try:
            from app.config import JUDGE_ACK_DIGEST_BLOCK_KIT

            use_blocks = bool(JUDGE_ACK_DIGEST_BLOCK_KIT)
        except ImportError:
            use_blocks = True
    payload: Dict[str, Any] = {"text": text}
    if use_blocks:
        payload["blocks"] = build_judge_ack_digest_slack_blocks(
            summary,
            tenant_id=tid,
            diff=diff,
            heatmap=heatmap,
            heatmap_bucket=heatmap_bucket,
            base=base,
        )
    canvas = judge_ack_canvas_urls(
        tenant_id=tid, hours=summary.get("since_hours")
    )
    if dry_run or not url:
        if use_diff and not dry_run:
            # no webhook but still baseline? skip persist on dry_run only
            pass
        if use_diff and dry_run:
            snapshot_path = save_judge_ack_digest_snapshot(
                summary, tenant_id=tid, base=base, diff=diff
            ) if os.environ.get("RAG_JUDGE_ACK_DIGEST_SNAPSHOT_ON_DRY_RUN", "").strip().lower() in {
                "1", "true", "yes", "on"
            } else None
        return {
            "ok": True,
            "dry_run": True,
            "skipped": not bool(url),
            "payload": payload,
            "configured": bool(url),
            "tenant_id": tid,
            "block_kit": bool(use_blocks),
            "diff": diff,
            "heatmap": heatmap,
            "heatmap_bucket": heatmap_bucket,
            "canvas": canvas,
            "snapshot_path": snapshot_path,
        }
    ok = post_slack(url, payload)
    if ok and use_diff:
        snapshot_path = save_judge_ack_digest_snapshot(
            summary, tenant_id=tid, base=base, diff=diff
        )
    return {
        "ok": ok,
        "dry_run": False,
        "configured": True,
        "payload": payload,
        "tenant_id": tid,
        "block_kit": bool(use_blocks),
        "diff": diff,
        "heatmap": heatmap,
        "heatmap_bucket": heatmap_bucket,
        "canvas": canvas,
        "snapshot_path": snapshot_path,
    }


def dispatch_judge_ack_digest_fanout(
    *,
    path: Optional[str] = None,
    since_hours: float = 168.0,
    dry_run: bool = False,
    quiet_hours: Optional[str] = None,
    ignore_quiet_hours: bool = False,
    timezone_name: Optional[str] = None,
    base: Optional[str] = None,
    webhook: Optional[str] = None,
    block_kit: Optional[bool] = None,
) -> Dict[str, Any]:
    """Tenant webhook haritasına göre ack digest fan-out (per-tenant quiet hours + mute)."""
    mapping = parse_judge_ack_digest_webhooks(base=base)
    quiet_map = parse_judge_ack_digest_quiet_hours(base=base)
    mutes = parse_judge_ack_digest_mutes(base=base)
    global_summary = summarize_judge_ack_audit(
        path=path, since_hours=since_hours
    )
    if not mapping:
        dispatched = dispatch_judge_ack_digest(
            global_summary,
            webhook=webhook,
            dry_run=dry_run,
            quiet_hours=quiet_hours,
            ignore_quiet_hours=ignore_quiet_hours,
            timezone_name=timezone_name,
            base=base,
            block_kit=block_kit,
        )
        return {
            "ok": bool(global_summary.get("ok") and dispatched.get("ok")),
            "fanout": False,
            "summary": global_summary,
            "dispatch": dispatched,
            "results": [{"tenant_id": None, **dispatched}],
            "muted": sorted(mutes),
        }

    results: List[Dict[str, Any]] = []
    for tid, url in sorted(mapping.items()):
        if is_judge_ack_digest_muted(tid, mutes=mutes):
            tenant_summary = summarize_judge_ack_audit(
                path=path, since_hours=since_hours, tenant_id=tid
            )
            summary = tenant_summary if tenant_summary.get("total", 0) > 0 else {
                **global_summary,
                "tenant_id": tid,
            }
            kept = maybe_keep_digest_snapshot_on_mute(
                summary, tenant_id=tid, base=base
            )
            results.append(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "muted",
                    "tenant_id": tid,
                    "configured": True,
                    "kept_snapshot": bool(kept.get("kept_snapshot")),
                    "snapshot_path": kept.get("snapshot_path"),
                }
            )
            continue
        tenant_summary = summarize_judge_ack_audit(
            path=path, since_hours=since_hours, tenant_id=tid
        )
        summary = tenant_summary if tenant_summary.get("total", 0) > 0 else {
            **global_summary,
            "tenant_id": tid,
        }
        # Explicit CLI quiet_hours overrides all; else per-tenant file then global
        tenant_quiet = quiet_hours
        if tenant_quiet is None and tid in quiet_map:
            tenant_quiet = quiet_map[tid]
        dispatched = dispatch_judge_ack_digest(
            summary,
            webhook=url,
            dry_run=dry_run,
            tenant_id=tid,
            quiet_hours=tenant_quiet,
            ignore_quiet_hours=ignore_quiet_hours,
            timezone_name=timezone_name,
            base=base,
            block_kit=block_kit,
        )
        results.append({"tenant_id": tid, "summary": summary, **dispatched})

    ok = all(r.get("ok") for r in results) if results else True
    return {
        "ok": ok,
        "fanout": True,
        "tenant_count": len(results),
        "summary": global_summary,
        "results": results,
        "quiet_overrides": len(quiet_map),
        "muted": sorted(mutes),
        "muted_count": sum(1 for r in results if r.get("reason") == "muted"),
    }


def load_judge_alert_state(path: Optional[str] = None) -> Dict[str, Any]:
    # Explicit path → sadece dosya (test/izolasyon); aksi halde remote öncelikli
    if path is None:
        remote = _load_remote_judge_state()
        if remote is not None:
            return remote
    p = path or judge_alert_state_path()
    if not os.path.isfile(p):
        return {"soft_fail": False, "source": None, "updated_at": None}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"soft_fail": False}
    except Exception:
        return {"soft_fail": False, "source": None, "updated_at": None}


def save_judge_alert_state(state: Dict[str, Any], path: Optional[str] = None) -> str:
    p = path or judge_alert_state_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    _save_remote_judge_state(state)
    return p


def judge_alert_state_url() -> str:
    return (
        os.environ.get("RAG_JUDGE_ALERT_STATE_URL", "").strip()
        or os.environ.get("RAG_JUDGE_STATE_URL", "").strip()
    )


def _load_remote_judge_state() -> Optional[Dict[str, Any]]:
    url = judge_alert_state_url()
    if not url:
        return None
    try:
        import requests

        r = requests.get(url, timeout=10)
        if r.status_code >= 400:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_remote_judge_state(state: Dict[str, Any]) -> bool:
    url = judge_alert_state_url()
    if not url:
        return False
    try:
        import requests

        r = requests.put(url, json=state, timeout=10)
        if r.status_code >= 400:
            r = requests.post(url, json=state, timeout=10)
        return r.status_code < 400
    except Exception:
        return False


def judge_ack_public_url() -> str:
    """Slack deep-link hedefi (ack form)."""
    explicit = (
        os.environ.get("RAG_JUDGE_ACK_PUBLIC_URL", "").strip()
        or os.environ.get("RAG_JUDGE_ACK_URL", "").strip()
    )
    if explicit:
        return explicit.rstrip("/")
    try:
        from app.config import COLLAB_HTTP_PORT, COLLAB_WS_PUBLIC_HOST

        host = (COLLAB_WS_PUBLIC_HOST or "localhost").strip() or "localhost"
        port = int(COLLAB_HTTP_PORT or 8766)
        return f"http://{host}:{port}/judge/ack-form"
    except Exception:
        base = os.environ.get("RAG_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if base:
            return f"{base}/judge/ack-form"
    return ""


def load_judge_report(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def soft_fail_from_report(
    report: Dict[str, Any],
    *,
    soft_fail_env: bool,
) -> bool:
    if not soft_fail_env:
        return False
    summary = report.get("summary") or {}
    return not bool(summary.get("ok"))


def soft_fail_from_metrics(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    last: Optional[Dict[str, Any]] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "judge_run":
                    last = row
    except Exception:
        return False
    if not last:
        return False
    vals = last.get("values") or {}
    return bool(vals.get("soft_fail"))


def detect_soft_fail(
    *,
    report_path: str,
    metrics_path: str,
    soft_fail_env: bool,
) -> Tuple[bool, str, Dict[str, Any]]:
    report = load_judge_report(report_path)
    if not report:
        alt = os.environ.get("RAG_JUDGE_OUTPUT_LLM", "metadata/judge_report_llm.json")
        report = load_judge_report(alt)
    from_report = soft_fail_from_report(report, soft_fail_env=soft_fail_env)
    from_metrics = soft_fail_from_metrics(metrics_path)
    if from_report:
        return True, "report", report
    if from_metrics:
        return True, "metrics", report or {"summary": {}}
    return False, "", report or {}


def _summary_text(report: Dict[str, Any], *, source: str) -> str:
    summary = report.get("summary") or {}
    mode = summary.get("mode") or report.get("mode") or "?"
    return (
        f"RAG judge soft-fail ({source}): mode={mode} "
        f"accuracy={summary.get('accuracy')} passed={summary.get('passed')} "
        f"failed={summary.get('failed')} total={summary.get('total')}"
    )


def _resolve_text(report: Dict[str, Any], *, source: str) -> str:
    summary = report.get("summary") or {}
    return (
        f"RAG judge soft-fail RESOLVED ({source}): "
        f"accuracy={summary.get('accuracy')} ok={summary.get('ok')}"
    )


def build_slack_payload(report: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    text = _summary_text(report, source=source)
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail*\n{text}"},
        }
    ]
    elements: List[Dict[str, Any]] = []
    interactive = os.environ.get("RAG_JUDGE_SLACK_SIGNING_SECRET", "").strip() or (
        os.environ.get("RAG_JUDGE_SLACK_INTERACTIVE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    bot_token = os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    if interactive:
        if bot_token:
            elements.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Ack + note"},
                    "action_id": "judge_ack_modal",
                    "value": "ack_modal",
                    "style": "primary",
                }
            )
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Ack now"},
                "action_id": "judge_ack_interactive",
                "value": "ack",
                "style": "danger" if bot_token else "primary",
            }
        )
    ack_url = judge_ack_public_url()
    if ack_url:
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Ack form"},
                "url": ack_url,
                "action_id": "judge_ack_open",
            }
        )
    if elements:
        blocks.append({"type": "actions", "elements": elements})
    if ack_url:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Ack form: <{ack_url}|open>"},
                ],
            }
        )
    return {"text": text, "blocks": blocks}


def build_ack_modal_view(
    *,
    source: str = "report",
    channel_id: Optional[str] = None,
    message_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Slack Block Kit modal: note alanı ile acknowledge."""
    meta: Dict[str, Any] = {"source": source}
    if channel_id:
        meta["channel_id"] = str(channel_id)
    if message_ts:
        meta["message_ts"] = str(message_ts)
    return {
        "type": "modal",
        "callback_id": "judge_ack_modal",
        "title": {"type": "plain_text", "text": "Judge soft-fail ACK"},
        "submit": {"type": "plain_text", "text": "Acknowledge"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "private_metadata": json.dumps(meta, ensure_ascii=False),
        "blocks": [
            {
                "type": "input",
                "block_id": "ack_note_block",
                "optional": True,
                "label": {"type": "plain_text", "text": "Note"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "ack_note",
                    "multiline": True,
                    "placeholder": {
                        "type": "plain_text",
                        "text": "inceleme notu (opsiyonel)",
                    },
                },
            }
        ],
    }


def parse_ack_modal_private_metadata(payload: Dict[str, Any]) -> Dict[str, str]:
    """view.private_metadata → channel_id / message_ts / source."""
    view = payload.get("view") or {}
    raw = str(view.get("private_metadata") or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for key in ("source", "channel_id", "message_ts"):
        val = str(data.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def open_slack_modal(
    *,
    trigger_id: str,
    view: Dict[str, Any],
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    token = (
        bot_token
        if bot_token is not None
        else os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    if not token:
        return {"ok": False, "error": "bot_token_missing"}
    if not (trigger_id or "").strip():
        return {"ok": False, "error": "trigger_id_missing"}
    try:
        import requests

        r = requests.post(
            "https://slack.com/api/views.open",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json={"trigger_id": trigger_id.strip(), "view": view},
            timeout=10,
        )
        data = r.json() if r.content else {}
        if r.status_code >= 400 or not data.get("ok"):
            return {
                "ok": False,
                "error": str(data.get("error") or f"http_{r.status_code}"),
                "response": data,
            }
        return {"ok": True, "response": data}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def extract_modal_ack_note(payload: Dict[str, Any]) -> str:
    view = payload.get("view") or {}
    state = (view.get("state") or {}).get("values") or {}
    block = state.get("ack_note_block") or {}
    field = block.get("ack_note") or {}
    return str(field.get("value") or "")[:500]


def verify_slack_request_signature(
    body: bytes,
    *,
    timestamp: str,
    signature: str,
    signing_secret: Optional[str] = None,
    max_age_sec: int = 60 * 5,
) -> bool:
    """Slack Signing Secret doğrulama (v0 HMAC-SHA256)."""
    import hashlib
    import hmac
    import time

    secret = (
        signing_secret
        if signing_secret is not None
        else os.environ.get("RAG_JUDGE_SLACK_SIGNING_SECRET", "").strip()
    )
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(int(time.time()) - ts) > int(max_age_sec):
        return False
    try:
        raw = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    base = f"v0:{timestamp}:{raw}"
    digest = (
        "v0="
        + hmac.new(secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(digest, signature.strip())


def parse_slack_interactive_payload(body: bytes) -> Dict[str, Any]:
    """application/x-www-form-urlencoded `payload=` veya JSON body."""
    from urllib.parse import parse_qs, unquote_plus

    text = body.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    qs = parse_qs(text, keep_blank_values=True)
    raw = (qs.get("payload") or [""])[0]
    if not raw:
        return {}
    try:
        data = json.loads(unquote_plus(raw))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def slack_interactive_actor(payload: Dict[str, Any]) -> str:
    user = payload.get("user") or {}
    if isinstance(user, dict):
        for key in ("username", "name", "id"):
            val = str(user.get(key) or "").strip()
            if val:
                return val
    return ""


def slack_interactive_user_id(payload: Dict[str, Any]) -> str:
    user = payload.get("user") or {}
    if isinstance(user, dict):
        return str(user.get("id") or "").strip()
    return ""


def judge_ack_rate_limit_sec() -> float:
    raw = os.environ.get("RAG_JUDGE_ACK_RATE_LIMIT_SEC", "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def check_judge_ack_rate_limit(
    actor: str,
    *,
    state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Ack rate limit; limit=0 → kapalı."""
    limit = judge_ack_rate_limit_sec()
    who = (actor or "").strip() or "anonymous"
    if limit <= 0:
        return {"ok": True, "limited": False, "limit_sec": 0}
    now = time.time()
    last = _ACK_RATE.get(who)
    if last is None:
        try:
            st = load_judge_alert_state(state_path)
            by = st.get("ack_rate_by_actor") or {}
            if isinstance(by, dict) and who in by:
                last = float(by[who])
        except Exception:
            last = None
    if last is not None and (now - float(last)) < limit:
        retry = max(0.0, limit - (now - float(last)))
        return {
            "ok": False,
            "limited": True,
            "error": "rate_limited",
            "retry_after_sec": round(retry, 2),
            "limit_sec": limit,
            "actor": who,
        }
    return {"ok": True, "limited": False, "limit_sec": limit, "actor": who}


def record_judge_ack_rate(
    actor: str,
    *,
    state_path: Optional[str] = None,
) -> None:
    who = (actor or "").strip() or "anonymous"
    now = time.time()
    _ACK_RATE[who] = now
    try:
        st = load_judge_alert_state(state_path)
        by = dict(st.get("ack_rate_by_actor") or {})
        by[who] = now
        # son 50 aktör
        if len(by) > 50:
            keep = sorted(by.items(), key=lambda x: float(x[1]), reverse=True)[:50]
            by = dict(keep)
        st["ack_rate_by_actor"] = by
        save_judge_alert_state(st, state_path)
    except Exception:
        pass


def handle_slack_interactive_ack(payload: Dict[str, Any]) -> Dict[str, Any]:
    """block_actions / view_submission → modal open veya acknowledge_judge_alert."""
    ptype = str(payload.get("type") or "")

    if ptype == "view_submission":
        view = payload.get("view") or {}
        callback_id = str(view.get("callback_id") or "")
        if callback_id == "judge_mute_export_revoke_modal":
            meta = parse_mute_export_revoke_modal_metadata(payload)
            ref = str(meta.get("ref") or "").strip()
            result = handle_slack_mute_export_revoke(payload, value=ref)
            result["mode"] = "modal_submit"
            result["confirm"] = "mute_export_revoke"
            result["callback_id"] = callback_id
            return result
        if callback_id != "judge_ack_modal":
            return {"ok": False, "error": "unknown_view"}
        actor = slack_interactive_actor(payload) or "slack"
        note = extract_modal_ack_note(payload) or "slack modal ack"
        state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
        rate = check_judge_ack_rate_limit(actor, state_path=state_path)
        if rate.get("limited"):
            return {
                "ok": False,
                "error": "rate_limited",
                "mode": "modal_submit",
                "retry_after_sec": rate.get("retry_after_sec"),
            }
        result = acknowledge_judge_alert(
            actor=actor,
            note=note,
            state_path=state_path,
            notify=False,
        )
        result["mode"] = "modal_submit"
        if result.get("ok"):
            record_judge_ack_rate(actor, state_path=state_path)
            meta = parse_ack_modal_private_metadata(payload)
            channel_id = str(
                meta.get("channel_id")
                or ((payload.get("container") or {}).get("channel_id"))
                or ((payload.get("channel") or {}).get("id"))
                or ""
            ) or None
            thread = post_judge_ack_thread_reply(
                actor=actor,
                note=note,
                already=bool(result.get("already")),
                channel_id=channel_id,
                thread_ts=str(
                    meta.get("message_ts")
                    or ((payload.get("container") or {}).get("thread_ts"))
                    or ((payload.get("message") or {}).get("ts"))
                    or ((payload.get("container") or {}).get("message_ts"))
                    or ""
                )
                or None,
            )
            result["thread_reply"] = thread
            result["ephemeral"] = post_judge_ack_ephemeral(
                user_id=slack_interactive_user_id(payload),
                actor=actor,
                note=note,
                already=bool(result.get("already")),
                channel_id=channel_id,
            )
        return result

    actions = payload.get("actions") or []
    action_ids = {
        str((a or {}).get("action_id") or "")
        for a in actions
        if isinstance(a, dict)
    }

    if "judge_ack_modal" in action_ids:
        trigger_id = str(payload.get("trigger_id") or "").strip()
        source = "report"
        try:
            st = load_judge_alert_state(
                os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
            )
            source = str(st.get("source") or "report")
        except Exception:
            pass
        channel_id = str(
            ((payload.get("channel") or {}).get("id"))
            or ((payload.get("container") or {}).get("channel_id"))
            or ""
        ).strip() or None
        message_ts = str(
            ((payload.get("message") or {}).get("ts"))
            or ((payload.get("container") or {}).get("message_ts"))
            or ""
        ).strip() or None
        opened = open_slack_modal(
            trigger_id=trigger_id,
            view=build_ack_modal_view(
                source=source,
                channel_id=channel_id,
                message_ts=message_ts,
            ),
        )
        return {
            "ok": bool(opened.get("ok")),
            "mode": "modal_open",
            "error": opened.get("error"),
            "opened": opened,
            "channel_id": channel_id,
            "message_ts": message_ts,
        }

    if (
        "judge_ack_digest_mute" in action_ids
        or "judge_ack_digest_unmute" in action_ids
    ):
        mute = "judge_ack_digest_mute" in action_ids
        tid = ""
        for act in actions:
            if not isinstance(act, dict):
                continue
            aid = str(act.get("action_id") or "")
            if aid in {"judge_ack_digest_mute", "judge_ack_digest_unmute"}:
                tid = str(act.get("value") or "").strip()
                mute = aid == "judge_ack_digest_mute"
                break
        if not tid:
            return {"ok": False, "error": "tenant_required", "mode": "digest_mute"}
        actor = slack_interactive_actor(payload) or "slack"
        state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
        rate = check_judge_ack_rate_limit(actor, state_path=state_path)
        if rate.get("limited"):
            return {
                "ok": False,
                "error": "rate_limited",
                "mode": "digest_mute",
                "retry_after_sec": rate.get("retry_after_sec"),
                "tenant_id": tid,
            }
        base = os.environ.get("RAG_JUDGE_ACK_DIGEST_BASE", "").strip() or None
        saved = set_judge_ack_digest_mute(tid, muted=mute, base=base)
        channel_id = str(
            ((payload.get("channel") or {}).get("id"))
            or ((payload.get("container") or {}).get("channel_id"))
            or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
            or ""
        ).strip() or None
        message_ts = str(
            ((payload.get("message") or {}).get("ts"))
            or ((payload.get("container") or {}).get("thread_ts"))
            or ((payload.get("container") or {}).get("message_ts"))
            or ""
        ).strip() or None
        user_id = slack_interactive_user_id(payload)
        verb = "muted" if mute else "unmuted"
        text = f"Tenant `{tid}` {verb} for ack digests."
        if mute and saved.get("expires_at"):
            text += f" · expires `{saved.get('expires_at')}`"
        if saved.get("ok"):
            record_judge_ack_rate(actor, state_path=state_path)
        audit = append_judge_ack_audit(
            "digest_mute" if mute else "digest_unmute",
            actor=actor,
            note=text,
            source="slack_interactive",
            extra={
                "tenant_id": tid,
                "muted": bool(mute),
                "expires_at": saved.get("expires_at"),
                "channel_id": channel_id,
                "message_ts": message_ts,
            },
        )
        ephemeral: Dict[str, Any] = {"ok": False, "skipped": True}
        bot_token = os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
        if bot_token and channel_id and user_id:
            ephemeral = slack_api(
                "chat.postEphemeral",
                bot_token=bot_token,
                json_body={
                    "channel": channel_id,
                    "user": user_id,
                    "text": text,
                    "mrkdwn": True,
                },
            )
        # Prefer thread confirmation on unmute; also post on mute for audit trail.
        thread_reply: Dict[str, Any] = {"ok": False, "skipped": True}
        if saved.get("ok") and (not mute or os.environ.get(
            "RAG_JUDGE_ACK_DIGEST_MUTE_THREAD", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}):
            thread_reply = post_judge_digest_mute_thread_reply(
                tenant_id=tid,
                muted=bool(mute),
                actor=actor,
                channel_id=channel_id,
                thread_ts=message_ts,
                expires_at=str(saved.get("expires_at") or "") or None,
                bot_token=bot_token or None,
            )
        catch_up: Optional[Dict[str, Any]] = None
        if (
            saved.get("ok")
            and (not mute)
            and digest_catch_up_on_unmute_enabled()
        ):
            catch_up = dispatch_judge_ack_digest_catch_up(
                tid,
                base=base,
                actor=actor,
                channel_id=channel_id,
                thread_ts=message_ts,
                bot_token=bot_token or None,
                unmute=False,
            )
            if catch_up.get("text"):
                text = f"{text}\n{catch_up['text']}"
        message_update: Dict[str, Any] = {"ok": False, "skipped": True}
        fanout_sync: Dict[str, Any] = {"ok": True, "skipped": True}
        if saved.get("ok"):
            if channel_id and message_ts:
                save_judge_ack_digest_message(
                    tid,
                    channel_id=channel_id,
                    message_ts=message_ts,
                    base=base,
                )
            message_update = refresh_judge_ack_digest_message_actions(
                payload,
                muted=bool(mute),
                tenant_id=tid,
                base=base,
                bot_token=bot_token or None,
            )
            fanout_sync = sync_judge_ack_digest_mute_chat_updates(
                tid,
                muted=bool(mute),
                base=base,
                bot_token=bot_token or None,
                exclude_message_ts=message_ts,
            )
        return {
            "ok": bool(saved.get("ok")),
            "mode": "digest_mute",
            "muted": bool(mute),
            "tenant_id": tid,
            "mute": saved,
            "audit": {
                "ok": "_write_error" not in audit,
                "event": audit.get("event"),
                "actor": audit.get("actor"),
            },
            "text": text,
            "ephemeral": {
                "ok": bool(ephemeral.get("ok")),
                "skipped": bool(ephemeral.get("skipped")),
                "error": ephemeral.get("error"),
            },
            "thread_reply": thread_reply,
            "catch_up": catch_up,
            "message_update": message_update,
            "fanout_sync": fanout_sync,
            "channel_id": channel_id,
            "message_ts": message_ts,
            "error": saved.get("error"),
        }

    if "judge_ack_digest_catch_up" in action_ids:
        tid = ""
        for act in actions:
            if not isinstance(act, dict):
                continue
            if str(act.get("action_id") or "") == "judge_ack_digest_catch_up":
                tid = str(act.get("value") or "").strip()
                break
        if not tid:
            return {"ok": False, "error": "tenant_required", "mode": "digest_catch_up"}
        actor = slack_interactive_actor(payload) or "slack"
        state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
        rate = check_judge_ack_rate_limit(actor, state_path=state_path)
        if rate.get("limited"):
            return {
                "ok": False,
                "error": "rate_limited",
                "mode": "digest_catch_up",
                "retry_after_sec": rate.get("retry_after_sec"),
                "tenant_id": tid,
            }
        base = os.environ.get("RAG_JUDGE_ACK_DIGEST_BASE", "").strip() or None
        channel_id = str(
            ((payload.get("channel") or {}).get("id"))
            or ((payload.get("container") or {}).get("channel_id"))
            or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
            or ""
        ).strip() or None
        message_ts = str(
            ((payload.get("message") or {}).get("ts"))
            or ((payload.get("container") or {}).get("thread_ts"))
            or ((payload.get("container") or {}).get("message_ts"))
            or ""
        ).strip() or None
        bot_token = os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip() or None
        out = dispatch_judge_ack_digest_catch_up(
            tid,
            base=base,
            actor=actor,
            channel_id=channel_id,
            thread_ts=message_ts,
            bot_token=bot_token,
            unmute=True,
        )
        if out.get("ok"):
            record_judge_ack_rate(actor, state_path=state_path)
        user_id = slack_interactive_user_id(payload)
        ephemeral: Dict[str, Any] = {"ok": False, "skipped": True}
        if bot_token and channel_id and user_id:
            ephemeral = slack_api(
                "chat.postEphemeral",
                bot_token=bot_token,
                json_body={
                    "channel": channel_id,
                    "user": user_id,
                    "text": out.get("text") or f"Catch-up digest for `{tid}`",
                    "mrkdwn": True,
                },
            )
        out["ephemeral"] = {
            "ok": bool(ephemeral.get("ok")),
            "skipped": bool(ephemeral.get("skipped")),
            "error": ephemeral.get("error"),
        }
        # Catch-up unmutes → refresh actions to Mute (no Catch-up button).
        muted_now = is_judge_ack_digest_muted(tid, base=base)
        if channel_id and message_ts:
            save_judge_ack_digest_message(
                tid,
                channel_id=channel_id,
                message_ts=message_ts,
                base=base,
            )
        out["message_update"] = refresh_judge_ack_digest_message_actions(
            payload,
            muted=muted_now,
            tenant_id=tid,
            base=base,
            bot_token=bot_token,
        )
        out["fanout_sync"] = sync_judge_ack_digest_mute_chat_updates(
            tid,
            muted=muted_now,
            base=base,
            bot_token=bot_token,
            exclude_message_ts=message_ts,
        )
        return out

    if "judge_ack_digest_heatmap_zoom" in action_ids:
        hours = 168.0
        try:
            hours = float(os.environ.get("RAG_JUDGE_ACK_DIGEST_HOURS", "168") or 168)
        except Exception:
            hours = 168.0
        bucket = "hour"
        tid = None
        for act in actions:
            if not isinstance(act, dict):
                continue
            if str(act.get("action_id") or "") == "judge_ack_digest_heatmap_zoom":
                raw = str(act.get("value") or "hour").strip()
                if "|" in raw:
                    left, right = raw.split("|", 1)
                    left, right = left.strip(), right.strip().lower()
                    if right in {"hour", "day"}:
                        bucket = right
                        tid = left or None
                    elif left in {"hour", "day"}:
                        bucket = left
                        tid = right or None
                else:
                    val = raw.lower()
                    if val in {"hour", "day"}:
                        bucket = val
                break
        base = os.environ.get("RAG_JUDGE_ACK_DIGEST_BASE", "").strip() or None
        pref = set_heatmap_bucket_pref(bucket, tenant_id=tid, base=base)
        channel_id = str(
            ((payload.get("channel") or {}).get("id"))
            or ((payload.get("container") or {}).get("channel_id"))
            or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
            or ""
        ).strip() or None
        user_id = slack_interactive_user_id(payload)
        heatmap = build_ack_heatmap(
            since_hours=hours,
            tenant_id=tid,
            bucket=bucket,
            max_buckets=24 if bucket == "hour" else 14,
        )
        text = format_ack_heatmap_mrkdwn(heatmap)
        if tid:
            text = f"{text}\n_Persisted heatmap=`{bucket}` for tenant `{tid}`._"
        else:
            text = f"{text}\n_Persisted heatmap=`{bucket}` (global)._"
        ephemeral: Dict[str, Any] = {"ok": False, "skipped": True}
        bot_token = os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
        if bot_token and channel_id and user_id:
            ephemeral = slack_api(
                "chat.postEphemeral",
                bot_token=bot_token,
                json_body={
                    "channel": channel_id,
                    "user": user_id,
                    "text": text,
                    "mrkdwn": True,
                },
            )
        return {
            "ok": True,
            "mode": "heatmap_zoom",
            "bucket": bucket,
            "tenant_id": tid,
            "pref": pref,
            "heatmap": heatmap,
            "text": text,
            "ephemeral": {
                "ok": bool(ephemeral.get("ok")),
                "skipped": bool(ephemeral.get("skipped")),
                "error": ephemeral.get("error"),
            },
            "channel_id": channel_id,
        }

    if "judge_ack_digest_export_mute_snapshots" in action_ids:
        tid = ""
        for act in actions:
            if not isinstance(act, dict):
                continue
            if str(act.get("action_id") or "") == "judge_ack_digest_export_mute_snapshots":
                tid = str(act.get("value") or "").strip()
                break
        if tid.lower() in {"", "all", "csv", "*"}:
            tid = ""
        return upload_judge_ack_digest_mute_snapshots_slack(
            payload=payload, tenant_id=tid or None
        )

    if "judge_ack_digest_revoke_mute_export" in action_ids:
        ref_val = ""
        for act in actions:
            if not isinstance(act, dict):
                continue
            if str(act.get("action_id") or "") == "judge_ack_digest_revoke_mute_export":
                ref_val = str(act.get("value") or "").strip()
                break
        return open_mute_export_revoke_confirm_modal(payload, value=ref_val)

    if (
        "judge_ack_digest_reexport" in action_ids
        or "judge_ack_digest_reexport_csv" in action_ids
    ):
        hours = 168.0
        try:
            hours = float(os.environ.get("RAG_JUDGE_ACK_DIGEST_HOURS", "168") or 168)
        except Exception:
            hours = 168.0
        # Format from button value (jsonl|csv) or action_id
        fmt = "jsonl"
        for act in actions:
            if not isinstance(act, dict):
                continue
            aid = str(act.get("action_id") or "")
            val = str(act.get("value") or "").strip().lower()
            if aid == "judge_ack_digest_reexport_csv" or val == "csv":
                fmt = "csv"
                break
            if aid == "judge_ack_digest_reexport" and val in {"jsonl", "csv"}:
                fmt = val
                break
        summary = summarize_judge_ack_audit(since_hours=hours)
        exported = export_judge_ack_audit(fmt=fmt, limit=200)
        text = build_judge_ack_digest_text(summary)
        export_url = judge_ack_export_public_url()
        if export_url:
            q = "" if fmt == "jsonl" else f"?format={fmt}"
            text = f"{text}\nExport link: {export_url}{q}"
        channel_id = str(
            ((payload.get("channel") or {}).get("id"))
            or ((payload.get("container") or {}).get("channel_id"))
            or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
            or ""
        ).strip() or None
        message_ts = str(
            ((payload.get("message") or {}).get("ts"))
            or ((payload.get("container") or {}).get("thread_ts"))
            or ((payload.get("container") or {}).get("message_ts"))
            or ""
        ).strip() or None
        user_id = slack_interactive_user_id(payload)
        upload: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "not_attempted"}
        progress: Dict[str, Any] = {"ok": False, "skipped": True}
        thread_reply: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "not_attempted"}
        content = exported.get("text") or ""
        count = int(exported.get("count") or 0)
        filename = f"judge_ack_audit.{fmt}"
        bot_token = os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
        if content and bot_token:
            # Progress ephemeral before upload
            if channel_id and user_id:
                progress = slack_api(
                    "chat.postEphemeral",
                    bot_token=bot_token,
                    json_body={
                        "channel": channel_id,
                        "user": user_id,
                        "text": (
                            f"Exporting *{count}* ack audit rows as `{fmt}`…"
                        ),
                    },
                )
            upload = slack_files_upload(
                content=content,
                filename=filename,
                title=f"Judge ack audit ({count} rows, {fmt})",
                channels=channel_id,
                thread_ts=message_ts,
                initial_comment=(
                    f"Ack audit re-export · {count} events · {fmt} · last {hours:g}h"
                ),
            )
            if upload.get("ok"):
                text = (
                    f"{text}\nAttached `{filename}`"
                    + (
                        f" (<{upload.get('permalink')}|open>)"
                        if upload.get("permalink")
                        else ""
                    )
                    + " · upload done"
                )
            else:
                text = f"{text}\nFile upload skipped: `{upload.get('error')}`"
            # Channel thread confirmation under the digest message
            if channel_id and message_ts:
                thread_reply = post_judge_digest_reexport_thread_reply(
                    fmt=fmt,
                    count=count,
                    hours=hours,
                    channel_id=channel_id,
                    thread_ts=message_ts,
                    permalink=str(upload.get("permalink") or "") or None,
                    bot_token=bot_token,
                    upload_ok=bool(upload.get("ok")),
                )
                if thread_reply.get("ok"):
                    text = f"{text}\nThread reply posted"
        elif not bot_token:
            upload = {"ok": False, "skipped": True, "reason": "bot_token_missing"}
        elif not content:
            upload = {"ok": False, "skipped": True, "reason": "empty_export"}
        return {
            "ok": True,
            "mode": "digest_reexport",
            "summary": summary,
            "export": {
                "ok": exported.get("ok"),
                "count": exported.get("count"),
                "format": exported.get("format") or fmt,
            },
            "upload": upload,
            "progress": {
                "ok": bool(progress.get("ok")),
                "skipped": bool(progress.get("skipped")),
                "error": progress.get("error"),
            },
            "thread_reply": thread_reply,
            "channel_id": channel_id,
            "message_ts": message_ts,
            "text": text,
            "export_url": export_url or None,
            "format": fmt,
        }

    if "judge_ack_interactive" not in action_ids and ptype == "block_actions":
        if not any(
            str((a or {}).get("value") or "") == "ack" for a in actions if isinstance(a, dict)
        ):
            return {"ok": False, "error": "unknown_action"}
    actor = slack_interactive_actor(payload) or "slack"
    note = "slack interactive ack"
    state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
    rate = check_judge_ack_rate_limit(actor, state_path=state_path)
    if rate.get("limited"):
        return {
            "ok": False,
            "error": "rate_limited",
            "mode": "instant",
            "retry_after_sec": rate.get("retry_after_sec"),
        }
    result = acknowledge_judge_alert(
        actor=actor,
        note=note,
        state_path=state_path,
        notify=False,
    )
    result["mode"] = "instant"
    if result.get("ok"):
        record_judge_ack_rate(actor, state_path=state_path)
        channel_id = str(((payload.get("channel") or {}).get("id")) or "") or None
        result["thread_reply"] = post_judge_ack_thread_reply(
            actor=actor,
            note=note,
            already=bool(result.get("already")),
            channel_id=channel_id,
            thread_ts=str(
                ((payload.get("message") or {}).get("ts"))
                or ((payload.get("container") or {}).get("message_ts"))
                or ""
            )
            or None,
        )
        result["ephemeral"] = post_judge_ack_ephemeral(
            user_id=slack_interactive_user_id(payload),
            actor=actor,
            note=note,
            already=bool(result.get("already")),
            channel_id=channel_id,
        )
    return result


def slack_api(
    method: str,
    *,
    bot_token: str,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Slack Web API (Bearer) ince sarmalayıcı."""
    token = (bot_token or "").strip()
    if not token:
        return {"ok": False, "error": "bot_token_missing"}
    name = (method or "").strip().lstrip("/")
    if not name:
        return {"ok": False, "error": "method_missing"}
    try:
        import requests

        r = requests.post(
            f"https://slack.com/api/{name}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=json_body or {},
            params=params,
            timeout=15,
        )
        data = r.json() if r.content else {}
        if r.status_code >= 400:
            return {
                "ok": False,
                "error": str(data.get("error") or f"http_{r.status_code}"),
                "response": data,
            }
        if not isinstance(data, dict):
            return {"ok": False, "error": "invalid_response"}
        return data
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def slack_files_upload(
    *,
    bot_token: Optional[str] = None,
    content: str,
    filename: str = "judge_ack_audit.jsonl",
    title: Optional[str] = None,
    channels: Optional[str] = None,
    initial_comment: Optional[str] = None,
    thread_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Slack files.upload (multipart) — digest re-export eki."""
    token = (
        (bot_token or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    if not token:
        return {"ok": False, "error": "bot_token_missing"}
    channel = (
        (channels or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "").strip()
    )
    if not channel:
        return {"ok": False, "error": "channel_missing"}
    try:
        import requests

        data: Dict[str, Any] = {
            "filename": filename,
            "title": title or filename,
            "channels": channel,
        }
        if initial_comment:
            data["initial_comment"] = initial_comment
        ts = (thread_ts or "").strip()
        if ts:
            data["thread_ts"] = ts
        files = {
            "file": (
                filename,
                (content or "").encode("utf-8"),
                "application/octet-stream",
            )
        }
        r = requests.post(
            "https://slack.com/api/files.upload",
            headers={"Authorization": f"Bearer {token}"},
            data=data,
            files=files,
            timeout=30,
        )
        body = r.json() if r.content else {}
        if r.status_code >= 400:
            return {
                "ok": False,
                "error": str(body.get("error") or f"http_{r.status_code}"),
                "response": body,
            }
        if not isinstance(body, dict):
            return {"ok": False, "error": "invalid_response"}
        if not body.get("ok"):
            return {
                "ok": False,
                "error": str(body.get("error") or "files_upload_failed"),
                "response": body,
            }
        file_obj = body.get("file") if isinstance(body.get("file"), dict) else {}
        return {
            "ok": True,
            "file_id": file_obj.get("id"),
            "permalink": file_obj.get("permalink"),
            "filename": filename,
            "channels": channel,
            "thread_ts": ts or None,
            "response": body,
        }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def lookup_slack_channel_for_ts(
    *,
    thread_ts: str,
    bot_token: Optional[str] = None,
    channel_hint: Optional[str] = None,
    max_channels: int = 40,
) -> Dict[str, Any]:
    """conversations.history / list ile message_ts için kanal bul."""
    token = (
        bot_token
        if bot_token is not None
        else os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    ts = (thread_ts or "").strip()
    if not token:
        return {"ok": False, "error": "bot_token_missing"}
    if not ts:
        return {"ok": False, "error": "thread_ts_missing"}

    def _history_has(channel: str) -> bool:
        hist = slack_api(
            "conversations.history",
            bot_token=token,
            json_body={
                "channel": channel,
                "latest": ts,
                "oldest": ts,
                "inclusive": True,
                "limit": 1,
            },
        )
        if not hist.get("ok"):
            return False
        messages = hist.get("messages") or []
        return any(str((m or {}).get("ts") or "") == ts for m in messages if isinstance(m, dict))

    hint = (channel_hint or "").strip()
    if hint and _history_has(hint):
        return {"ok": True, "channel": hint, "via": "hint"}

    cursor = None
    scanned = 0
    while scanned < max(1, int(max_channels)):
        body: Dict[str, Any] = {
            "types": "public_channel,private_channel",
            "exclude_archived": True,
            "limit": min(100, max(1, int(max_channels) - scanned)),
        }
        if cursor:
            body["cursor"] = cursor
        listed = slack_api("conversations.list", bot_token=token, json_body=body)
        if not listed.get("ok"):
            return {
                "ok": False,
                "error": str(listed.get("error") or "conversations_list_failed"),
                "response": listed,
            }
        channels = listed.get("channels") or []
        for ch in channels:
            if not isinstance(ch, dict):
                continue
            cid = str(ch.get("id") or "").strip()
            if not cid:
                continue
            scanned += 1
            if _history_has(cid):
                return {"ok": True, "channel": cid, "via": "conversations.history"}
            if scanned >= max_channels:
                break
        cursor = ((listed.get("response_metadata") or {}).get("next_cursor") or "").strip()
        if not cursor:
            break
    return {"ok": False, "error": "channel_not_found", "scanned": scanned}


def resolve_slack_thread_parent_ts(
    *,
    channel_id: str,
    thread_ts: str,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """conversations.replies ile reply ts → root parent thread_ts."""
    token = (
        bot_token
        if bot_token is not None
        else os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    channel = (channel_id or "").strip()
    ts = (thread_ts or "").strip()
    if not token:
        return {"ok": False, "error": "bot_token_missing"}
    if not channel:
        return {"ok": False, "error": "channel_missing"}
    if not ts:
        return {"ok": False, "error": "thread_ts_missing"}
    data = slack_api(
        "conversations.replies",
        bot_token=token,
        json_body={"channel": channel, "ts": ts, "limit": 1, "inclusive": True},
    )
    if not data.get("ok"):
        return {
            "ok": False,
            "error": str(data.get("error") or "conversations_replies_failed"),
            "response": data,
            "input_ts": ts,
        }
    messages = data.get("messages") or []
    if not messages or not isinstance(messages[0], dict):
        return {"ok": False, "error": "empty_replies", "input_ts": ts}
    msg = messages[0]
    parent = str(msg.get("thread_ts") or msg.get("ts") or ts).strip() or ts
    return {
        "ok": True,
        "parent_ts": parent,
        "input_ts": ts,
        "changed": parent != ts,
        "via": "conversations.replies",
    }


def post_judge_ack_thread_reply(
    *,
    actor: str,
    note: str = "",
    already: bool = False,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Modal/instant ack sonrası Slack thread reply (chat.postMessage)."""
    status = "already acknowledged" if already else "acknowledged"
    text = f"Judge soft-fail {status} by *{actor}*"
    if note:
        text += f"\n> {note}"
    return post_slack_thread_message(
        text=text,
        channel_id=channel_id,
        thread_ts=thread_ts,
        bot_token=bot_token,
    )


def post_judge_digest_reexport_thread_reply(
    *,
    fmt: str,
    count: int,
    hours: float,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    permalink: Optional[str] = None,
    bot_token: Optional[str] = None,
    upload_ok: bool = True,
) -> Dict[str, Any]:
    """Digest re-export sonrası kanal thread'ine onay mesajı."""
    kind = (fmt or "jsonl").strip().lower() or "jsonl"
    if upload_ok:
        text = (
            f"Ack audit re-export ready · *{count}* rows · `{kind}` · last {hours:g}h"
        )
        if permalink:
            text += f" · <{permalink}|open file>"
    else:
        text = (
            f"Ack audit re-export attempted · *{count}* rows · `{kind}` "
            f"(file upload failed or skipped)"
        )
    return post_slack_thread_message(
        text=text,
        channel_id=channel_id,
        thread_ts=thread_ts,
        bot_token=bot_token,
    )


def post_judge_digest_mute_thread_reply(
    *,
    tenant_id: str,
    muted: bool,
    actor: str = "slack",
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    expires_at: Optional[str] = None,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Mute/unmute sonrası digest mesajı thread'ine onay."""
    tid = (tenant_id or "").strip() or "unknown"
    verb = "muted" if muted else "unmuted"
    text = f"Tenant `{tid}` {verb} for ack digests by *{actor or 'slack'}*"
    if muted and expires_at:
        text += f" · expires `{expires_at}`"
    return post_slack_thread_message(
        text=text,
        channel_id=channel_id,
        thread_ts=thread_ts,
        bot_token=bot_token,
    )


def digest_chat_update_enabled() -> bool:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_CHAT_UPDATE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def digest_fanout_chat_update_enabled() -> bool:
    raw = os.environ.get("RAG_JUDGE_ACK_DIGEST_FANOUT_CHAT_UPDATE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def fetch_slack_message_by_ts(
    *,
    channel_id: str,
    message_ts: str,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """conversations.history ile tek digest mesajını (blocks dahil) getir."""
    token = (
        (bot_token or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    ch = (channel_id or "").strip()
    ts = (message_ts or "").strip()
    if not token:
        return {"ok": False, "error": "bot_token_missing"}
    if not ch or not ts:
        return {"ok": False, "error": "ref_incomplete"}
    hist = slack_api(
        "conversations.history",
        bot_token=token,
        json_body={
            "channel": ch,
            "latest": ts,
            "oldest": ts,
            "inclusive": True,
            "limit": 1,
        },
    )
    if not hist.get("ok"):
        return {
            "ok": False,
            "error": str(hist.get("error") or "history_failed"),
            "response": hist,
        }
    messages = hist.get("messages") or []
    for m in messages:
        if isinstance(m, dict) and str(m.get("ts") or "") == ts:
            return {"ok": True, "message": m, "channel_id": ch, "message_ts": ts}
    return {"ok": False, "error": "message_not_found", "channel_id": ch, "message_ts": ts}


def sync_judge_ack_digest_mute_chat_updates(
    tenant_id: str,
    *,
    muted: bool,
    base: Optional[str] = None,
    bot_token: Optional[str] = None,
    exclude_message_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Fan-out: stored digest message refs için chat.update (mute state sync)."""
    if not digest_fanout_chat_update_enabled():
        return {"ok": True, "skipped": True, "reason": "fanout_chat_update_disabled"}
    tid = (tenant_id or "").strip()
    if not tid:
        return {"ok": False, "error": "tenant_required"}
    refs = list_judge_ack_digest_message_refs(tid, base=base)
    if not refs:
        return {"ok": True, "skipped": True, "reason": "no_message_refs", "updates": []}
    skip_ts = (exclude_message_ts or "").strip()
    updates: List[Dict[str, Any]] = []
    for ref in refs:
        if skip_ts and ref.get("message_ts") == skip_ts:
            continue
        fetched = fetch_slack_message_by_ts(
            channel_id=ref["channel_id"],
            message_ts=ref["message_ts"],
            bot_token=bot_token,
        )
        if not fetched.get("ok"):
            updates.append(
                {
                    "ok": False,
                    "channel_id": ref["channel_id"],
                    "message_ts": ref["message_ts"],
                    "error": fetched.get("error"),
                }
            )
            continue
        msg = fetched.get("message") or {}
        payload = {
            "channel": {"id": ref["channel_id"]},
            "message": msg,
        }
        out = refresh_judge_ack_digest_message_actions(
            payload,
            muted=bool(muted),
            tenant_id=tid,
            base=base,
            bot_token=bot_token,
        )
        updates.append(
            {
                **out,
                "channel_id": ref["channel_id"],
                "message_ts": ref["message_ts"],
            }
        )
    ok_any = any(u.get("ok") and not u.get("skipped") for u in updates)
    return {
        "ok": True if updates else True,
        "skipped": not bool(updates),
        "tenant_id": tid,
        "muted": bool(muted),
        "updates": updates,
        "updated": sum(1 for u in updates if u.get("ok") and not u.get("skipped")),
        "failed": sum(1 for u in updates if not u.get("ok")),
        "fanout_ok": ok_any or not updates,
    }


def refresh_judge_ack_digest_message_actions(
    payload: Dict[str, Any],
    *,
    muted: bool,
    tenant_id: str,
    base: Optional[str] = None,
    bot_token: Optional[str] = None,
    channel_id: Optional[str] = None,
    message_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """chat.update original digest so Mute/Unmute/Catch-up match mute state."""
    if not digest_chat_update_enabled():
        return {"ok": True, "skipped": True, "reason": "chat_update_disabled"}
    token = (
        (bot_token or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    if not token:
        return {"ok": False, "skipped": True, "error": "bot_token_missing"}
    msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    channel_id = str(
        (channel_id or "").strip()
        or ((payload.get("channel") or {}).get("id"))
        or ((payload.get("container") or {}).get("channel_id"))
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "")
        or ""
    ).strip()
    message_ts = str(
        (message_ts or "").strip()
        or (msg.get("ts") if isinstance(msg, dict) else None)
        or ((payload.get("container") or {}).get("message_ts"))
        or ((payload.get("container") or {}).get("thread_ts"))
        or ""
    ).strip()
    if not channel_id:
        return {"ok": False, "skipped": True, "error": "channel_missing"}
    if not message_ts:
        return {"ok": False, "skipped": True, "error": "message_ts_missing"}
    tid = (tenant_id or "").strip()
    if not tid:
        return {"ok": False, "skipped": True, "error": "tenant_required"}
    old_blocks = list(msg.get("blocks") or []) if isinstance(msg, dict) else []
    text = str((msg.get("text") if isinstance(msg, dict) else None) or "").strip()
    try:
        hours = float(os.environ.get("RAG_JUDGE_ACK_DIGEST_HOURS", "168") or 168)
    except Exception:
        hours = 168.0
    summary = {
        "ok": True,
        "since_hours": hours,
        "total": 0,
        "actor_count": 0,
        "by_event": {},
        "actors": [],
        "tenant_id": tid,
    }
    fresh = build_judge_ack_digest_slack_blocks(
        summary, tenant_id=tid, muted=bool(muted), base=base
    )
    fresh_actions = next(
        (
            b
            for b in fresh
            if b.get("type") == "actions"
            and b.get("block_id") == "judge_ack_digest_actions"
        ),
        None,
    )
    if not fresh_actions:
        return {"ok": False, "skipped": True, "error": "actions_missing"}
    fresh_header_section = next(
        (b for b in fresh if b.get("type") == "section"), None
    )
    new_blocks: List[Dict[str, Any]] = []
    replaced_actions = False
    replaced_header = False
    for b in old_blocks:
        if (
            isinstance(b, dict)
            and b.get("type") == "actions"
            and b.get("block_id") == "judge_ack_digest_actions"
        ):
            new_blocks.append(fresh_actions)
            replaced_actions = True
            continue
        if (
            not replaced_header
            and isinstance(b, dict)
            and b.get("type") == "section"
            and fresh_header_section is not None
        ):
            body = str(((b.get("text") or {}).get("text")) or "")
            if "ack audit digest" in body.lower() or "MUTED" in body or (
                tid and f"tenant `{tid}`" in body
            ):
                new_blocks.append(fresh_header_section)
                replaced_header = True
                continue
        new_blocks.append(b)
    if not replaced_actions:
        new_blocks.append(fresh_actions)
    if not text:
        text = f"Judge ack digest · tenant `{tid}`" + (
            " · MUTED" if muted else ""
        )
    try:
        data = slack_api(
            "chat.update",
            bot_token=token,
            json_body={
                "channel": channel_id,
                "ts": message_ts,
                "text": text[:3900],
                "blocks": new_blocks,
            },
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "detail": str(exc),
            "channel_id": channel_id,
            "message_ts": message_ts,
        }
    if not data.get("ok"):
        return {
            "ok": False,
            "error": str(data.get("error") or "chat_update_failed"),
            "response": data,
            "channel_id": channel_id,
            "message_ts": message_ts,
        }
    action_ids = [
        str(e.get("action_id") or "")
        for e in (fresh_actions.get("elements") or [])
        if isinstance(e, dict)
    ]
    return {
        "ok": True,
        "skipped": False,
        "channel_id": channel_id,
        "message_ts": message_ts,
        "muted": bool(muted),
        "action_ids": action_ids,
        "replaced_actions": replaced_actions,
        "replaced_header": replaced_header,
    }


def post_slack_thread_message(
    *,
    text: str,
    channel_id: Optional[str] = None,
    thread_ts: Optional[str] = None,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Genel chat.postMessage (opsiyonel thread_ts + parent resolve)."""
    token = (
        bot_token
        if bot_token is not None
        else os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    channel = (
        (channel_id or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "").strip()
    )
    ts = (
        (thread_ts or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_THREAD_TS", "").strip()
    )
    if not token:
        return {"ok": False, "error": "bot_token_missing", "skipped": True}
    lookup: Optional[Dict[str, Any]] = None
    resolve: Optional[Dict[str, Any]] = None
    if not channel and ts:
        lookup = lookup_slack_channel_for_ts(thread_ts=ts, bot_token=token)
        if lookup.get("ok"):
            channel = str(lookup.get("channel") or "").strip()
    if not channel:
        out = {"ok": False, "error": "channel_missing", "skipped": True}
        if lookup is not None:
            out["lookup"] = lookup
        return out
    if ts:
        resolve = resolve_slack_thread_parent_ts(
            channel_id=channel, thread_ts=ts, bot_token=token
        )
        if resolve.get("ok"):
            ts = str(resolve.get("parent_ts") or ts).strip() or ts
    body: Dict[str, Any] = {
        "channel": channel,
        "text": text,
        "mrkdwn": True,
    }
    if ts:
        body["thread_ts"] = ts
    try:
        data = slack_api("chat.postMessage", bot_token=token, json_body=body)
        if not data.get("ok"):
            return {
                "ok": False,
                "error": str(data.get("error") or "post_failed"),
                "response": data,
                "lookup": lookup,
                "resolve": resolve,
            }
        return {
            "ok": True,
            "ts": data.get("ts"),
            "channel": data.get("channel") or channel,
            "thread_ts": ts or None,
            "lookup": lookup,
            "resolve": resolve,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "lookup": lookup,
            "resolve": resolve,
        }


def post_judge_ack_ephemeral(
    *,
    user_id: str,
    actor: str,
    note: str = "",
    already: bool = False,
    channel_id: Optional[str] = None,
    bot_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Ack sonrası kullanıcıya chat.postEphemeral onayı."""
    token = (
        bot_token
        if bot_token is not None
        else os.environ.get("RAG_JUDGE_SLACK_BOT_TOKEN", "").strip()
    )
    channel = (
        (channel_id or "").strip()
        or os.environ.get("RAG_JUDGE_SLACK_CHANNEL", "").strip()
    )
    uid = (user_id or "").strip()
    if not token:
        return {"ok": False, "error": "bot_token_missing", "skipped": True}
    if not channel:
        return {"ok": False, "error": "channel_missing", "skipped": True}
    if not uid:
        return {"ok": False, "error": "user_id_missing", "skipped": True}
    status = "already acknowledged" if already else "acknowledged"
    text = f"Judge soft-fail {status} (by *{actor}*)"
    if note:
        text += f"\n> {note}"
    data = slack_api(
        "chat.postEphemeral",
        bot_token=token,
        json_body={
            "channel": channel,
            "user": uid,
            "text": text,
            "mrkdwn": True,
        },
    )
    if not data.get("ok"):
        return {
            "ok": False,
            "error": str(data.get("error") or "ephemeral_failed"),
            "response": data,
        }
    return {
        "ok": True,
        "channel": channel,
        "user": uid,
        "message_ts": data.get("message_ts"),
    }


def build_slack_resolve_payload(report: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    text = _resolve_text(report, source=source)
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail resolved*\n{text}"},
        }
    ]
    return {"text": text, "blocks": blocks}


def post_slack(webhook: str, payload: Dict[str, Any]) -> bool:
    url = (webhook or "").strip()
    if not url:
        return False
    try:
        import requests

        r = requests.post(url, json=payload, timeout=10)
        return r.status_code < 400
    except Exception:
        return False


def silence_burn_ops_public_url(
    *,
    public_base: Optional[str] = None,
) -> str:
    """Absolute (or path) URL for silence burn runbook page."""
    explicit = (
        os.environ.get("INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL", "").strip()
        or os.environ.get("RAG_INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL", "").strip()
        or os.environ.get("RAG_SILENCE_BURN_RUNBOOK_URL", "").strip()
    )
    if explicit:
        return explicit
    pub = (public_base or "").strip()
    if not pub:
        pub = os.environ.get("RAG_JUDGE_ACK_PUBLIC_URL", "").strip()
        if pub.endswith("/judge/ack-form"):
            pub = pub[: -len("/judge/ack-form")]
        elif pub.endswith("/judge/ack-ui"):
            pub = pub[: -len("/judge/ack-ui")]
    if not pub:
        try:
            from app.config import COLLAB_HTTP_HOST, COLLAB_HTTP_PORT

            host = COLLAB_HTTP_HOST or "127.0.0.1"
            port = int(COLLAB_HTTP_PORT or 8765)
            pub = f"http://{host}:{port}"
        except Exception:
            return "/ops/silence-burn"
    return f"{pub.rstrip('/')}/ops/silence-burn"


def opsgenie_alert_deep_link(
    *,
    source: str = "inhibit-equal-canary",
    region: Optional[str] = None,
) -> str:
    """Web UI deep-link to Opsgenie alert list filtered by soft-fail alias."""
    explicit = (
        os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_ALERT_URL", "").strip()
        or os.environ.get("RAG_OPSGENIE_ALERT_URL", "").strip()
    )
    if explicit:
        return explicit
    reg = (
        region
        or os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_REGION", "").strip()
        or os.environ.get("RAG_OPSGENIE_REGION", "").strip()
        or "us"
    ).strip().lower()
    host = (
        "https://eu.app.opsgenie.com"
        if reg in {"eu", "europe"}
        else "https://app.opsgenie.com"
    )
    alias = f"rag-judge-soft-fail/{source or 'inhibit-equal-canary'}"
    from urllib.parse import quote

    return f"{host}/alert/list?query={quote(f'alias:\"{alias}\"')}"


def post_pagerduty(
    *,
    routing_key: str,
    report: Dict[str, Any],
    source: str,
    severity: str = "warning",
    event_action: str = "trigger",
    runbook_url: Optional[str] = None,
) -> bool:
    key = (routing_key or "").strip()
    if not key:
        return False
    action = event_action if event_action in {"trigger", "acknowledge", "resolve"} else "trigger"
    text = (
        _resolve_text(report, source=source)
        if action == "resolve"
        else _summary_text(report, source=source)
    )
    summary = report.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    sev = severity if severity in {"info", "warning", "error", "critical"} else "warning"
    body: Dict[str, Any] = {
        "routing_key": key,
        "event_action": action,
        "dedup_key": f"rag-judge-soft-fail/{source}",
    }
    if action == "trigger":
        details: Dict[str, Any] = {
            "accuracy": summary.get("accuracy"),
            "passed": summary.get("passed"),
            "failed": summary.get("failed"),
            "total": summary.get("total"),
            "source": source,
            "mode": summary.get("mode") or report.get("mode"),
        }
        rb = (
            runbook_url
            or report.get("runbook_url")
            or summary.get("runbook_url")
            or os.environ.get("RAG_PAGERDUTY_RUNBOOK_URL", "").strip()
            or None
        )
        if rb:
            details["runbook_url"] = str(rb)
        body["payload"] = {
            "summary": text[:1024],
            "severity": "rag-judge",
            "source": "rag-ci",
            "severity": sev,
            "component": "judge",
            "group": "ci",
            "class": "soft_fail",
            "custom_details": details,
        }
    try:
        import requests

        r = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=body,
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def opsgenie_api_base(
    *,
    region: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Resolve Opsgenie API host (us / eu / custom URL)."""
    explicit = (base_url or os.environ.get("RAG_OPSGENIE_API_URL", "")).strip()
    if explicit:
        return explicit.rstrip("/")
    reg = (
        (region or os.environ.get("RAG_OPSGENIE_REGION", "") or "us")
        .strip()
        .lower()
    )
    if reg in {"eu", "europe"}:
        return "https://eu.api.opsgenie.com"
    if reg.startswith("http://") or reg.startswith("https://"):
        return reg.rstrip("/")
    return "https://api.opsgenie.com"


def post_opsgenie(
    *,
    api_key: str,
    report: Dict[str, Any],
    source: str,
    priority: str = "P3",
    region: Optional[str] = None,
    base_url: Optional[str] = None,
    runbook_url: Optional[str] = None,
) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    text = _summary_text(report, source=source)
    summary = report.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    pri = priority if priority in {"P1", "P2", "P3", "P4", "P5"} else "P3"
    host = opsgenie_api_base(region=region, base_url=base_url)
    tags = ["rag", "judge", "soft-fail", source]
    if region:
        tags.append(f"region:{region}")
    details: Dict[str, Any] = {
        "accuracy": str(summary.get("accuracy")),
        "passed": str(summary.get("passed")),
        "failed": str(summary.get("failed")),
        "total": str(summary.get("total")),
        "mode": str(summary.get("mode") or report.get("mode") or ""),
        "region": str(region or ""),
    }
    rb = (
        runbook_url
        or report.get("runbook_url")
        or summary.get("runbook_url")
        or os.environ.get("INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL", "").strip()
        or os.environ.get("RAG_OPSGENIE_RUNBOOK_URL", "").strip()
        or None
    )
    if rb:
        details["runbook_url"] = str(rb)
        # Silence-burn multi-region runbook tags (alongside region:*)
        for t in ("silence-burn", "runbook"):
            if t not in tags:
                tags.append(t)
        rb_l = str(rb).lower()
        if "silence-burn" in rb_l or "/ops/silence-burn" in rb_l:
            if "runbook:silence-burn" not in tags:
                tags.append("runbook:silence-burn")
    # Grafana annotation ↔ Opsgenie deep-link (alias list)
    og_link = opsgenie_alert_deep_link(source=source, region=region)
    if og_link:
        details["opsgenie_url"] = og_link
        details["opsgenie_deep_link"] = og_link
    extra_tags = (
        os.environ.get("INHIBIT_EQUAL_CANARY_OPSGENIE_TAGS", "").strip()
        or os.environ.get("RAG_OPSGENIE_EXTRA_TAGS", "").strip()
    )
    if extra_tags:
        for part in extra_tags.replace(";", ",").split(","):
            p = part.strip()
            if p and p not in tags:
                tags.append(p)
    body = {
        "message": text[:130],
        "alias": f"rag-judge-soft-fail/{source}",
        "description": text,
        "priority": pri,
        "tags": tags,
        "details": details,
        "entity": "rag-judge",
        "source": "rag-ci",
    }
    try:
        import requests

        r = requests.post(
            f"{host}/v2/alerts",
            headers={
                "Authorization": f"GenieKey {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def post_opsgenie_close(
    *,
    api_key: str,
    source: str,
    region: Optional[str] = None,
    base_url: Optional[str] = None,
) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    alias = f"rag-judge-soft-fail/{source}"
    host = opsgenie_api_base(region=region, base_url=base_url)
    try:
        import requests
        from urllib.parse import quote

        r = requests.post(
            f"{host}/v2/alerts/{quote(alias, safe='')}/close",
            params={"identifierType": "alias"},
            headers={
                "Authorization": f"GenieKey {key}",
                "Content-Type": "application/json",
            },
            json={"note": "RAG judge soft-fail resolved in CI"},
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def _channel_creds(
    *,
    slack_webhook: Optional[str],
    pagerduty_routing_key: Optional[str],
    opsgenie_api_key: Optional[str],
) -> Tuple[str, str, str]:
    slack_url = (
        slack_webhook
        if slack_webhook is not None
        else (
            os.environ.get("RAG_JUDGE_SLACK_WEBHOOK", "").strip()
            or os.environ.get("RAG_DIGEST_ALERT_WEBHOOK_URL", "").strip()
            or os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        )
    )
    pd_key = (
        pagerduty_routing_key
        if pagerduty_routing_key is not None
        else os.environ.get("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", "").strip()
    )
    og_key = (
        opsgenie_api_key
        if opsgenie_api_key is not None
        else os.environ.get("RAG_JUDGE_OPSGENIE_API_KEY", "").strip()
    )
    return slack_url, pd_key, og_key


def dispatch_judge_alerts(
    report: Dict[str, Any],
    *,
    source: str,
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    slack_url, pd_key, og_key = _channel_creds(
        slack_webhook=slack_webhook,
        pagerduty_routing_key=pagerduty_routing_key,
        opsgenie_api_key=opsgenie_api_key,
    )
    channels: Dict[str, Any] = {}
    any_ok = False
    if slack_url:
        ok = post_slack(slack_url, build_slack_payload(report, source=source))
        channels["slack"] = ok
        any_ok = any_ok or ok
    if pd_key:
        ok = post_pagerduty(routing_key=pd_key, report=report, source=source)
        channels["pagerduty"] = ok
        any_ok = any_ok or ok
    if og_key:
        ok = post_opsgenie(api_key=og_key, report=report, source=source)
        channels["opsgenie"] = ok
        any_ok = any_ok or ok
    return {
        "alerted": any_ok,
        "channels": channels,
        "source": source,
        "soft_fail": True,
        "configured": bool(slack_url or pd_key or og_key),
    }


def dispatch_judge_resolve(
    report: Dict[str, Any],
    *,
    source: str,
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    slack_url, pd_key, og_key = _channel_creds(
        slack_webhook=slack_webhook,
        pagerduty_routing_key=pagerduty_routing_key,
        opsgenie_api_key=opsgenie_api_key,
    )
    channels: Dict[str, Any] = {}
    any_ok = False
    if slack_url:
        ok = post_slack(slack_url, build_slack_resolve_payload(report, source=source))
        channels["slack"] = ok
        any_ok = any_ok or ok
    if pd_key:
        ok = post_pagerduty(
            routing_key=pd_key,
            report=report,
            source=source,
            event_action="resolve",
        )
        channels["pagerduty"] = ok
        any_ok = any_ok or ok
    if og_key:
        ok = post_opsgenie_close(api_key=og_key, source=source)
        channels["opsgenie"] = ok
        any_ok = any_ok or ok
    return {
        "resolved": any_ok,
        "channels": channels,
        "source": source,
        "soft_fail": False,
        "configured": bool(slack_url or pd_key or og_key),
    }


def build_slack_ack_payload(
    report: Dict[str, Any],
    *,
    source: str,
    actor: str,
    note: str = "",
) -> Dict[str, Any]:
    text = (
        f"RAG judge soft-fail ACK ({source}): by={actor}"
        + (f" note={note}" if note else "")
    )
    blocks: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*RAG judge soft-fail acknowledged*\n{text}"},
        }
    ]
    return {"text": text, "blocks": blocks}


def post_opsgenie_ack(
    *,
    api_key: str,
    source: str,
    note: str = "",
    region: Optional[str] = None,
    base_url: Optional[str] = None,
) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    alias = f"rag-judge-soft-fail/{source}"
    host = opsgenie_api_base(region=region, base_url=base_url)
    body: Dict[str, Any] = {"user": "rag-ci"}
    if note:
        body["note"] = note[:15000]
    try:
        import requests
        from urllib.parse import quote

        r = requests.post(
            f"{host}/v2/alerts/{quote(alias, safe='')}/acknowledge",
            params={"identifierType": "alias"},
            headers={
                "Authorization": f"GenieKey {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=10,
        )
        return r.status_code < 300
    except Exception:
        return False


def dispatch_judge_ack(
    report: Dict[str, Any],
    *,
    source: str,
    actor: str,
    note: str = "",
    slack_webhook: Optional[str] = None,
    pagerduty_routing_key: Optional[str] = None,
    opsgenie_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    slack_url, pd_key, og_key = _channel_creds(
        slack_webhook=slack_webhook,
        pagerduty_routing_key=pagerduty_routing_key,
        opsgenie_api_key=opsgenie_api_key,
    )
    channels: Dict[str, Any] = {}
    any_ok = False
    if slack_url:
        ok = post_slack(
            slack_url,
            build_slack_ack_payload(report, source=source, actor=actor, note=note),
        )
        channels["slack"] = ok
        any_ok = any_ok or ok
    if pd_key:
        ok = post_pagerduty(
            routing_key=pd_key,
            report=report,
            source=source,
            event_action="acknowledge",
        )
        channels["pagerduty"] = ok
        any_ok = any_ok or ok
    if og_key:
        ok = post_opsgenie_ack(api_key=og_key, source=source, note=note or f"acked by {actor}")
        channels["opsgenie"] = ok
        any_ok = any_ok or ok
    return {
        "acked": any_ok,
        "channels": channels,
        "source": source,
        "soft_fail": True,
        "configured": bool(slack_url or pd_key or og_key),
        "actor": actor,
        "note": note,
    }


def acknowledge_judge_alert(
    *,
    actor: str,
    note: str = "",
    state_path: Optional[str] = None,
    notify: bool = True,
    report: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Manuel soft-fail acknowledge; state'e yazar, opsiyonel kanal bildirimi."""
    who = (actor or "").strip()
    if not who:
        return {"ok": False, "error": "actor_required"}
    state = load_judge_alert_state(state_path)
    tid = (tenant_id or "").strip() or None
    extra = {"tenant_id": tid} if tid else None
    if not state.get("soft_fail"):
        return {
            "ok": False,
            "error": "no_active_soft_fail",
            "state": state,
        }
    if state.get("acknowledged"):
        append_judge_ack_audit(
            "ack_already",
            actor=who,
            note=note or "",
            source=str(state.get("source") or "report"),
            state=state,
            extra=extra,
        )
        return {
            "ok": True,
            "already": True,
            "state": state,
            "tenant_id": tid,
            "notify": {"acked": False, "reason": "already_acknowledged"},
        }
    source = str(state.get("source") or "report")
    new_state = {
        **state,
        "soft_fail": True,
        "source": source,
        "acknowledged": True,
        "acknowledged_at": _utcnow_iso(),
        "acknowledged_by": who,
        "ack_note": (note or "")[:500],
        "updated_at": _utcnow_iso(),
        "last_action": "ack",
    }
    if tid:
        new_state["tenant_id"] = tid
    save_judge_alert_state(new_state, state_path)
    append_judge_ack_audit(
        "ack",
        actor=who,
        note=note or "",
        source=source,
        state=new_state,
        extra=extra,
    )
    notify_result: Dict[str, Any] = {"acked": False, "skipped": True}
    if notify:
        notify_result = dispatch_judge_ack(
            report or {"summary": {"ok": False}},
            source=source,
            actor=who,
            note=note or "",
        )
    return {
        "ok": True,
        "already": False,
        "state": new_state,
        "tenant_id": tid,
        "notify": notify_result,
    }


def detect_and_dispatch(
    *,
    report_path: str,
    metrics_path: str,
    soft_fail_env: bool,
    state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    soft_fail → alert + state; ack'lı soft-fail → yeniden alert yok;
    önceki soft_fail + şimdi ok → resolve + state temizle.
    """
    fired, source, report = detect_soft_fail(
        report_path=report_path,
        metrics_path=metrics_path,
        soft_fail_env=soft_fail_env,
    )
    state = load_judge_alert_state(state_path)
    prev = bool(state.get("soft_fail"))
    prev_source = str(state.get("source") or "report")
    acked = bool(state.get("acknowledged"))

    if fired:
        if prev and acked:
            append_judge_ack_audit(
                "ack_hold",
                actor=str(state.get("acknowledged_by") or ""),
                source=source or prev_source,
                state=state,
            )
            return {
                "action": "ack_hold",
                "soft_fail": True,
                "alerted": False,
                "reason": "acknowledged",
                "acknowledged_by": state.get("acknowledged_by"),
                "acknowledged_at": state.get("acknowledged_at"),
                "source": source or prev_source,
            }
        result = dispatch_judge_alerts(report or {"summary": {}}, source=source or "unknown")
        new_state = {
            "soft_fail": True,
            "source": source or "unknown",
            "updated_at": _utcnow_iso(),
            "last_action": "alert",
            "acknowledged": False,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "ack_note": None,
        }
        save_judge_alert_state(new_state, state_path)
        append_judge_ack_audit(
            "alert",
            source=source or "unknown",
            state=new_state,
        )
        return {
            "action": "alert",
            "soft_fail": True,
            **result,
        }

    if prev:
        result = dispatch_judge_resolve(
            report or {"summary": {"ok": True}},
            source=prev_source,
        )
        new_state = {
            "soft_fail": False,
            "source": None,
            "updated_at": _utcnow_iso(),
            "last_action": "resolve",
            "acknowledged": False,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "ack_note": None,
        }
        save_judge_alert_state(new_state, state_path)
        append_judge_ack_audit(
            "resolve",
            source=prev_source,
            state={**new_state, "source": prev_source},
        )
        return {
            "action": "resolve",
            "soft_fail": False,
            **result,
        }

    return {
        "action": "noop",
        "alerted": False,
        "resolved": False,
        "reason": "no_soft_fail",
        "soft_fail": False,
    }
