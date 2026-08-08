"""Judge soft-fail çok kanallı alert (Slack / PagerDuty / Opsgenie) + auto-resolve."""

from __future__ import annotations

import json
import os
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


def maybe_prune_judge_ack_digest_mutes(
    *,
    base: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Env-gated mute/prefs retention (RAG_JUDGE_ACK_DIGEST_MUTE_PRUNE)."""
    flag = os.environ.get("RAG_JUDGE_ACK_DIGEST_MUTE_PRUNE", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return {"ok": True, "skipped": True, "reason": "prune_disabled"}
    # Auto when TTL configured or explicit prune=1
    ttl = _mute_ttl_days()
    prefs_days_raw = os.environ.get(
        "RAG_JUDGE_ACK_DIGEST_PREFS_RETENTION_DAYS", ""
    ).strip()
    if flag not in {"1", "true", "yes", "on"} and ttl is None and not prefs_days_raw:
        return {"ok": True, "skipped": True, "reason": "retention_not_configured"}
    mute_report = prune_judge_ack_digest_mutes(base=base, dry_run=dry_run)
    prefs_report = prune_judge_ack_digest_prefs(base=base, dry_run=dry_run)
    return {
        "ok": bool(mute_report.get("ok") and prefs_report.get("ok")),
        "mutes": mute_report,
        "prefs": prefs_report,
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
        export_btn: Dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": "Export JSONL"},
            "action_id": "judge_ack_digest_reexport",
            "value": "jsonl",
        }
        if not is_muted:
            export_btn["style"] = "primary"
        elements.append(export_btn)
        # CSV only when mute button is absent (room under 5-cap with form+zoom).
        if not tid:
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
        # Prefer mute/catch-up/zoom + form: drop CSV / export if we would exceed 5
        if len(elements) >= 4:
            drop_ids = {"judge_ack_digest_reexport_csv"}
            if any(
                e.get("action_id") == "judge_ack_digest_catch_up" for e in elements
            ):
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
        if str(view.get("callback_id") or "") != "judge_ack_modal":
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


def post_pagerduty(
    *,
    routing_key: str,
    report: Dict[str, Any],
    source: str,
    severity: str = "warning",
    event_action: str = "trigger",
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
    sev = severity if severity in {"info", "warning", "error", "critical"} else "warning"
    body: Dict[str, Any] = {
        "routing_key": key,
        "event_action": action,
        "dedup_key": f"rag-judge-soft-fail/{source}",
    }
    if action == "trigger":
        body["payload"] = {
            "summary": text[:1024],
            "severity": "rag-judge",
            "source": "rag-ci",
            "severity": sev,
            "component": "judge",
            "group": "ci",
            "class": "soft_fail",
            "custom_details": {
                "accuracy": summary.get("accuracy"),
                "passed": summary.get("passed"),
                "failed": summary.get("failed"),
                "total": summary.get("total"),
                "source": source,
                "mode": summary.get("mode") or report.get("mode"),
            },
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
) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    text = _summary_text(report, source=source)
    summary = report.get("summary") or {}
    pri = priority if priority in {"P1", "P2", "P3", "P4", "P5"} else "P3"
    host = opsgenie_api_base(region=region, base_url=base_url)
    tags = ["rag", "judge", "soft-fail", source]
    if region:
        tags.append(f"region:{region}")
    body = {
        "message": text[:130],
        "alias": f"rag-judge-soft-fail/{source}",
        "description": text,
        "priority": pri,
        "tags": tags,
        "details": {
            "accuracy": str(summary.get("accuracy")),
            "passed": str(summary.get("passed")),
            "failed": str(summary.get("failed")),
            "total": str(summary.get("total")),
            "mode": str(summary.get("mode") or report.get("mode") or ""),
            "region": str(region or ""),
        },
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
