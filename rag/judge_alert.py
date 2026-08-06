"""Judge soft-fail çok kanallı alert (Slack / PagerDuty / Opsgenie) + auto-resolve."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


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


def export_judge_ack_audit(
    *,
    fmt: str = "jsonl",
    path: Optional[str] = None,
    output: Optional[str] = None,
    limit: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    event: Optional[str] = None,
    include_state: bool = False,
    state_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit trail'i JSONL veya CSV olarak dışa aktar."""
    rows = read_judge_ack_audit(
        path=path, limit=limit, since=since, until=until, event=event
    )
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
) -> List[Dict[str, Any]]:
    """Haftalık ack digest için Slack Block Kit."""
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
    title = "RAG judge ack audit digest"
    if tid:
        title += f" · tenant `{tid}`"
    header = (
        f"*{title}* (last {summary.get('since_hours')}h)\n"
        f"Total events: *{summary.get('total', 0)}* · "
        f"actors: *{summary.get('actor_count', 0)}*"
    )
    blocks: List[Dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Judge ack digest"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
    ]
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
    if by_tenant and not tid:
        parts = [f"`{k}`={v}" for k, v in sorted(by_tenant.items())[:12]]
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
    if interactive:
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Re-export audit"},
                "action_id": "judge_ack_digest_reexport",
                "value": "reexport",
                "style": "primary",
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
    ack_url = judge_ack_public_url()
    if ack_url:
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Ack form"},
                "url": ack_url,
                "action_id": "judge_ack_digest_open",
            }
        )
    export_url = judge_ack_export_public_url()
    if export_url:
        elements.append(
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Download export"},
                "url": export_url,
                "action_id": "judge_ack_digest_export_link",
            }
        )
    if elements:
        blocks.append({"type": "actions", "block_id": "judge_ack_digest_actions", "elements": elements[:5]})
    if ack_url:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Ack form: <{ack_url}|open>"},
                ],
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
) -> str:
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
    title = "*RAG judge ack audit digest*"
    if tid:
        title += f" · tenant `{tid}`"
    lines = [
        f"{title} (last {summary.get('since_hours')}h)",
        f"Total events: *{summary.get('total', 0)}* · actors: *{summary.get('actor_count', 0)}*",
    ]
    by_event = summary.get("by_event") or {}
    if by_event:
        parts = [f"`{k}`={v}" for k, v in sorted(by_event.items())]
        lines.append("By event: " + ", ".join(parts))
    by_tenant = summary.get("by_tenant") or {}
    if by_tenant and not tid:
        parts = [f"`{k}`={v}" for k, v in sorted(by_tenant.items())[:12]]
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
) -> Dict[str, Any]:
    """Haftalık ack audit özetini Slack webhook'a gönder (Block Kit + quiet hours)."""
    tid = (tenant_id or summary.get("tenant_id") or "").strip() or None
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

    url = resolve_judge_ack_digest_webhook(tid, url=webhook, base=base)
    text = build_judge_ack_digest_text(summary, tenant_id=tid)
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
            summary, tenant_id=tid
        )
    if dry_run or not url:
        return {
            "ok": True,
            "dry_run": True,
            "skipped": not bool(url),
            "payload": payload,
            "configured": bool(url),
            "tenant_id": tid,
            "block_kit": bool(use_blocks),
        }
    ok = post_slack(url, payload)
    return {
        "ok": ok,
        "dry_run": False,
        "configured": True,
        "payload": payload,
        "tenant_id": tid,
        "block_kit": bool(use_blocks),
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
    """Tenant webhook haritasına göre ack digest fan-out (per-tenant quiet hours)."""
    mapping = parse_judge_ack_digest_webhooks(base=base)
    quiet_map = parse_judge_ack_digest_quiet_hours(base=base)
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
        }

    results: List[Dict[str, Any]] = []
    for tid, url in sorted(mapping.items()):
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

    if "judge_ack_digest_reexport" in action_ids:
        hours = 168.0
        try:
            hours = float(os.environ.get("RAG_JUDGE_ACK_DIGEST_HOURS", "168") or 168)
        except Exception:
            hours = 168.0
        summary = summarize_judge_ack_audit(since_hours=hours)
        exported = export_judge_ack_audit(fmt="jsonl", limit=200)
        text = build_judge_ack_digest_text(summary)
        export_url = judge_ack_export_public_url()
        if export_url:
            text = f"{text}\nExport link: {export_url}"
        return {
            "ok": True,
            "mode": "digest_reexport",
            "summary": summary,
            "export": {
                "ok": exported.get("ok"),
                "count": exported.get("count"),
                "format": exported.get("format"),
            },
            "text": text,
            "export_url": export_url or None,
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
    status = "already acknowledged" if already else "acknowledged"
    text = f"Judge soft-fail {status} by *{actor}*"
    if note:
        text += f"\n> {note}"
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


def post_opsgenie(
    *,
    api_key: str,
    report: Dict[str, Any],
    source: str,
    priority: str = "P3",
) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    text = _summary_text(report, source=source)
    summary = report.get("summary") or {}
    pri = priority if priority in {"P1", "P2", "P3", "P4", "P5"} else "P3"
    body = {
        "message": text[:130],
        "alias": f"rag-judge-soft-fail/{source}",
        "description": text,
        "priority": pri,
        "tags": ["rag", "judge", "soft-fail", source],
        "details": {
            "accuracy": str(summary.get("accuracy")),
            "passed": str(summary.get("passed")),
            "failed": str(summary.get("failed")),
            "total": str(summary.get("total")),
            "mode": str(summary.get("mode") or report.get("mode") or ""),
        },
        "entity": "rag-judge",
        "source": "rag-ci",
    }
    try:
        import requests

        r = requests.post(
            "https://api.opsgenie.com/v2/alerts",
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


def post_opsgenie_close(*, api_key: str, source: str) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    alias = f"rag-judge-soft-fail/{source}"
    try:
        import requests
        from urllib.parse import quote

        r = requests.post(
            f"https://api.opsgenie.com/v2/alerts/{quote(alias, safe='')}/close",
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


def post_opsgenie_ack(*, api_key: str, source: str, note: str = "") -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    alias = f"rag-judge-soft-fail/{source}"
    body: Dict[str, Any] = {"user": "rag-ci"}
    if note:
        body["note"] = note[:15000]
    try:
        import requests
        from urllib.parse import quote

        r = requests.post(
            f"https://api.opsgenie.com/v2/alerts/{quote(alias, safe='')}/acknowledge",
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
) -> Dict[str, Any]:
    """Manuel soft-fail acknowledge; state'e yazar, opsiyonel kanal bildirimi."""
    who = (actor or "").strip()
    if not who:
        return {"ok": False, "error": "actor_required"}
    state = load_judge_alert_state(state_path)
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
        )
        return {
            "ok": True,
            "already": True,
            "state": state,
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
    save_judge_alert_state(new_state, state_path)
    append_judge_ack_audit(
        "ack",
        actor=who,
        note=note or "",
        source=source,
        state=new_state,
    )
    notify_result: Dict[str, Any] = {"acked": False, "skipped": True}
    if notify:
        notify_result = dispatch_judge_ack(
            report or {"summary": {"ok": False}},
            source=source,
            actor=who,
            note=note or "",
        )
    return {"ok": True, "already": False, "state": new_state, "notify": notify_result}


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
