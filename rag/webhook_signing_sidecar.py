"""Alertmanager → RAG webhook HMAC signing sidecar.

Alertmanager cannot attach X-Webhook-* HMAC headers natively. Point AM at this
proxy; it signs the body and forwards to the real catch-up / quarantine hook.
"""

from __future__ import annotations

import json
import os
import secrets
import ssl
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag.dual_write_webhook import (
    build_webhook_signature,
    webhook_signing_secret,
)


def sidecar_upstream_url(*, mode: str = "dlq_quarantine") -> str:
    explicit = os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM", "").strip()
    if explicit:
        return explicit
    if mode == "catch_up":
        return os.environ.get(
            "RAG_WEBHOOK_SIGNING_SIDECAR_CATCHUP_UPSTREAM",
            "http://127.0.0.1:8766/hooks/dual-write-catch-up",
        ).strip()
    return os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_QUARANTINE_UPSTREAM",
        "http://127.0.0.1:8766/hooks/dual-write-dlq-quarantine",
    ).strip()


def sidecar_mode_from_path(path: str) -> str:
    p = (path or "/").split("?", 1)[0].rstrip("/") or "/"
    if p.endswith("catch-up") or p.endswith("/catchup"):
        return "catch_up"
    return "dlq_quarantine"


def sidecar_listen_host() -> str:
    return os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_HOST", "0.0.0.0").strip() or "0.0.0.0"


def sidecar_listen_port() -> int:
    raw = os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_PORT", "8777").strip()
    try:
        return max(1, int(raw or 8777))
    except Exception:
        return 8777


def sidecar_forward_timeout_sec() -> float:
    raw = os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_TIMEOUT_SEC", "30").strip()
    try:
        return max(1.0, float(raw or 30))
    except Exception:
        return 30.0


def sidecar_tls_cert_path() -> str:
    return os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT", "").strip()


def sidecar_tls_key_path() -> str:
    return os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_KEY", "").strip()


def sidecar_tls_ca_path() -> str:
    return os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CA", "").strip()


def sidecar_tls_enabled() -> bool:
    return bool(sidecar_tls_cert_path() and sidecar_tls_key_path())


def sidecar_mtls_required() -> bool:
    raw = os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_TLS_REQUIRE_CLIENT_CERT", ""
    ).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    mtls = os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_MTLS", "").strip().lower()
    if mtls in {"1", "true", "yes", "on"}:
        return True
    # CA without explicit off → require client cert
    return bool(sidecar_tls_ca_path())


def build_sidecar_ssl_context() -> Optional[ssl.SSLContext]:
    """TLS server context; optional mTLS when CA / REQUIRE_CLIENT_CERT / MTLS set."""
    if not sidecar_tls_enabled():
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(sidecar_tls_cert_path(), sidecar_tls_key_path())
    if sidecar_mtls_required():
        ca = sidecar_tls_ca_path()
        if not ca:
            raise ValueError("mtls_requires_ca")
        ctx.load_verify_locations(ca)
        ctx.verify_mode = ssl.CERT_REQUIRED
        try:
            ctx.check_hostname = False
        except Exception:
            pass
    else:
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def sidecar_upstream_client_cert_path() -> str:
    return os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT", ""
    ).strip()


def sidecar_upstream_client_key_path() -> str:
    return os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY", ""
    ).strip()


def sidecar_upstream_ca_path() -> str:
    return os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CA", "").strip()


def sidecar_upstream_client_cert_enabled() -> bool:
    return bool(
        sidecar_upstream_client_cert_path() and sidecar_upstream_client_key_path()
    )


def build_upstream_ssl_context() -> Optional[ssl.SSLContext]:
    """TLS client context for upstream forward (optional client-cert + CA)."""
    cert = sidecar_upstream_client_cert_path()
    key = sidecar_upstream_client_key_path()
    ca = sidecar_upstream_ca_path()
    if not cert and not key and not ca:
        return None
    if bool(cert) ^ bool(key):
        raise ValueError("upstream_client_cert_incomplete")
    ctx = ssl.create_default_context()
    if ca:
        ctx.load_verify_locations(ca)
    if cert and key:
        ctx.load_cert_chain(cert, key)
    return ctx


def inspect_pem_cert_expiry(
    path: str,
    *,
    role: str = "server",
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Parse PEM cert notAfter → days_left for Grafana cert-expiry alerts."""
    out: Dict[str, Any] = {
        "ok": False,
        "role": role,
        "path": path or "",
        "configured": bool(path),
    }
    if not path:
        out["error"] = "not_configured"
        return out
    if not os.path.isfile(path):
        out["error"] = "missing_file"
        return out
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
    except Exception as exc:
        out["error"] = f"cryptography_unavailable:{exc}"
        return out
    try:
        raw = open(path, "rb").read()
        cert = x509.load_pem_x509_certificate(raw, default_backend())
    except Exception as exc:
        out["error"] = f"parse_failed:{exc}"
        return out
    try:
        not_after = cert.not_valid_after_utc  # type: ignore[attr-defined]
    except Exception:
        not_after = cert.not_valid_after.replace(tzinfo=timezone.utc)
    try:
        not_before = cert.not_valid_before_utc  # type: ignore[attr-defined]
    except Exception:
        not_before = cert.not_valid_before.replace(tzinfo=timezone.utc)
    ts_now = float(now if now is not None else time.time())
    days_left = (not_after.timestamp() - ts_now) / 86400.0
    out.update(
        {
            "ok": True,
            "not_before": not_before.isoformat(),
            "not_after": not_after.isoformat(),
            "days_left": round(days_left, 3),
            "expired": days_left < 0,
        }
    )
    return out


def inspect_sidecar_certs(*, now: Optional[float] = None) -> Dict[str, Any]:
    """Inspect server + upstream client cert expiry (mTLS / upstream client-cert)."""
    server = inspect_pem_cert_expiry(
        sidecar_tls_cert_path(), role="server", now=now
    )
    upstream = inspect_pem_cert_expiry(
        sidecar_upstream_client_cert_path(),
        role="upstream_client",
        now=now,
    )
    ca = inspect_pem_cert_expiry(sidecar_tls_ca_path(), role="ca", now=now)
    certs = [c for c in (server, upstream, ca) if c.get("configured")]
    days_values = [
        float(c["days_left"])
        for c in certs
        if c.get("ok") and c.get("days_left") is not None
    ]
    min_days = min(days_values) if days_values else None
    return {
        "ok": True,
        "server": server,
        "upstream_client": upstream,
        "ca": ca,
        "min_days_left": min_days,
        "certs": certs,
    }


def emit_sidecar_cert_metrics(report: Optional[Dict[str, Any]] = None) -> None:
    """Prometheus: rag_webhook_signing_sidecar_cert_expiry_days{role}."""
    try:
        from rag.metrics import record_metric
    except Exception:
        return
    data = report or inspect_sidecar_certs()
    for role_key in ("server", "upstream_client", "ca"):
        item = data.get(role_key) or {}
        if not item.get("ok"):
            continue
        try:
            days = float(item.get("days_left"))
        except (TypeError, ValueError):
            continue
        try:
            record_metric(
                "webhook_signing_sidecar_cert_expiry",
                values={"role": role_key, "days_left": days},
            )
        except Exception:
            pass


def sidecar_tls_status() -> Dict[str, Any]:
    enabled = sidecar_tls_enabled()
    mtls = bool(enabled and sidecar_mtls_required())
    upstream_client = sidecar_upstream_client_cert_enabled()
    certs = inspect_sidecar_certs()
    emit_sidecar_cert_metrics(certs)
    return {
        "tls": enabled,
        "mtls": mtls,
        "cert": sidecar_tls_cert_path() if enabled else "",
        "ca": sidecar_tls_ca_path() if mtls else "",
        "upstream_client_cert": upstream_client,
        "upstream_ca": bool(sidecar_upstream_ca_path()),
        "cert_expiry": {
            "server_days_left": (certs.get("server") or {}).get("days_left"),
            "upstream_client_days_left": (certs.get("upstream_client") or {}).get(
                "days_left"
            ),
            "ca_days_left": (certs.get("ca") or {}).get("days_left"),
            "min_days_left": certs.get("min_days_left"),
        },
    }


def sidecar_tls_dir() -> str:
    explicit = os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_DIR", "").strip()
    if explicit:
        return explicit
    cert = sidecar_tls_cert_path()
    if cert:
        return os.path.dirname(os.path.abspath(cert)) or "metadata/sidecar-tls"
    return "metadata/sidecar-tls"


def generate_sidecar_self_signed_pair(
    *,
    common_name: str = "webhook-signing-sidecar",
    days: int = 90,
    now: Optional[float] = None,
) -> Dict[str, bytes]:
    """Create self-signed server cert+key (+ identical CA PEM for demo mTLS)."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    ts = float(now if now is not None else time.time())
    start = datetime.datetime.fromtimestamp(ts, tz=timezone.utc) - datetime.timedelta(
        minutes=1
    )
    end = start + datetime.timedelta(days=max(1, int(days)))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, common_name)]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(start)
        .not_valid_after(end)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.DNSName("webhook-signing-sidecar"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return {"cert": cert_pem, "key": key_pem, "ca": cert_pem}


def _archive_sidecar_pem(path: str, *, stamp: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    archive_dir = os.path.join(os.path.dirname(path) or ".", "archive")
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, f"{os.path.basename(path)}.{stamp}")
    try:
        with open(path, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())
        return dest
    except OSError:
        return None


def rotate_sidecar_certs(
    *,
    roles: Optional[Tuple[str, ...]] = None,
    days: int = 90,
    out_dir: Optional[str] = None,
    dry_run: bool = False,
    if_expiring_days: Optional[float] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Auto-rotate sidecar server (+ optional upstream client) PEMs under tls dir.

    When ``if_expiring_days`` is set, skip roles whose days_left >= threshold.
    """
    want = roles or ("server", "upstream_client")
    tls_dir = out_dir or sidecar_tls_dir()
    before = inspect_sidecar_certs(now=now)
    warn = if_expiring_days
    if warn is None:
        raw = os.environ.get(
            "RAG_WEBHOOK_SIGNING_SIDECAR_CERT_EXPIRY_WARN_DAYS", ""
        ).strip()
        # only used when caller passes rotate-if-expiring; keep None here
        _ = raw
    to_rotate: List[str] = []
    for role in want:
        item = before.get(role) or {}
        if warn is not None:
            if not item.get("configured"):
                to_rotate.append(role)
                continue
            try:
                left = float(item.get("days_left"))
            except (TypeError, ValueError):
                to_rotate.append(role)
                continue
            if left < float(warn):
                to_rotate.append(role)
        else:
            to_rotate.append(role)
    if not to_rotate:
        return {
            "ok": True,
            "skipped": True,
            "reason": "not_expiring",
            "before": before,
            "roles": list(want),
            "dry_run": bool(dry_run),
        }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written: Dict[str, Any] = {}
    archived: List[str] = []
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "would_rotate": to_rotate,
            "out_dir": tls_dir,
            "before": before,
        }
    os.makedirs(tls_dir, exist_ok=True)
    if "server" in to_rotate:
        pair = generate_sidecar_self_signed_pair(
            common_name="webhook-signing-sidecar", days=days, now=now
        )
        cert_path = sidecar_tls_cert_path() or os.path.join(tls_dir, "server.crt")
        key_path = sidecar_tls_key_path() or os.path.join(tls_dir, "server.key")
        ca_path = sidecar_tls_ca_path() or os.path.join(tls_dir, "ca.crt")
        for p in (cert_path, key_path, ca_path):
            arch = _archive_sidecar_pem(p, stamp=stamp)
            if arch:
                archived.append(arch)
        with open(cert_path, "wb") as f:
            f.write(pair["cert"])
        with open(key_path, "wb") as f:
            f.write(pair["key"])
        with open(ca_path, "wb") as f:
            f.write(pair["ca"])
        os.environ.setdefault("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT", cert_path)
        os.environ.setdefault("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_KEY", key_path)
        os.environ.setdefault("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CA", ca_path)
        written["server"] = {
            "cert": cert_path,
            "key": key_path,
            "ca": ca_path,
            "days": days,
        }
    if "upstream_client" in to_rotate:
        pair = generate_sidecar_self_signed_pair(
            common_name="am-client", days=days, now=now
        )
        cert_path = sidecar_upstream_client_cert_path() or os.path.join(
            tls_dir, "client.crt"
        )
        key_path = sidecar_upstream_client_key_path() or os.path.join(
            tls_dir, "client.key"
        )
        for p in (cert_path, key_path):
            arch = _archive_sidecar_pem(p, stamp=stamp)
            if arch:
                archived.append(arch)
        with open(cert_path, "wb") as f:
            f.write(pair["cert"])
        with open(key_path, "wb") as f:
            f.write(pair["key"])
        os.environ.setdefault(
            "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT", cert_path
        )
        os.environ.setdefault(
            "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY", key_path
        )
        written["upstream_client"] = {
            "cert": cert_path,
            "key": key_path,
            "days": days,
        }
    after = inspect_sidecar_certs(now=now)
    emit_sidecar_cert_metrics(after)
    return {
        "ok": True,
        "dry_run": False,
        "rotated": list(written.keys()),
        "written": written,
        "archived": archived,
        "out_dir": tls_dir,
        "before": before,
        "after": after,
        "stamp": stamp,
    }


def maybe_notify_sidecar_cert_rotate(
    report: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Slack notify after successful sidecar PEM rotate (DLQ quarantine path)."""
    flag = os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY", ""
    ).strip().lower()
    if not force and flag in {"0", "false", "no", "off"}:
        return {"ok": True, "skipped": True, "reason": "notify_disabled"}
    if not force and flag not in {"1", "true", "yes", "on", ""}:
        # empty = allow when webhook present
        pass
    if report.get("dry_run"):
        return {"ok": True, "skipped": True, "reason": "dry_run"}
    if report.get("skipped") and not report.get("rotated"):
        return {"ok": True, "skipped": True, "reason": "not_rotated"}
    rotated = report.get("rotated") or []
    if not rotated and not report.get("written"):
        return {"ok": True, "skipped": True, "reason": "no_roles"}
    webhook = (
        os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_DUAL_WRITE_DLQ_QUARANTINE_SLACK_WEBHOOK", "").strip()
        or os.environ.get("RAG_DUAL_WRITE_DLQ_SLACK_WEBHOOK", "").strip()
    )
    if not webhook:
        return {"ok": True, "skipped": True, "reason": "webhook_missing"}
    if not isinstance(rotated, list):
        rotated = [str(rotated)]
    stamp = report.get("stamp") or ""
    after = report.get("after") or {}
    before = report.get("before") or {}
    min_days = after.get("min_days_left")
    role_lines: List[str] = []
    for role in rotated:
        item = after.get(role) if isinstance(after, dict) else None
        if not isinstance(item, dict):
            item = {}
        days = item.get("days_left")
        prev = None
        if isinstance(before, dict) and isinstance(before.get(role), dict):
            prev = (before.get(role) or {}).get("days_left")
        bit = f"`{role}` days_left=`{days if days is not None else '—'}`"
        if prev is not None:
            bit += f" (was `{prev}`)"
        role_lines.append(bit)
    roles_text = "\n".join(f"• {line}" for line in role_lines) if role_lines else "• —"
    text = (
        "*Webhook signing sidecar cert rotate digest*\n"
        f"stamp=`{stamp or '—'}`"
        + (f" · min_days_left=`{min_days}`" if min_days is not None else "")
        + f"\n{roles_text}\n"
        "Dual-write DLQ quarantine HMAC path — restart sidecar if live."
    )
    digest_flag = os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "1"
    ).strip().lower()
    use_digest = digest_flag not in {"0", "false", "no", "off"}
    if use_digest:
        payload = {
            "text": text,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Sidecar cert rotate digest",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text[:2900]},
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                "mode=`rotate` · service=`rag-ingest` · "
                                "quarantine HMAC · "
                                f"roles=`{len(rotated)}`"
                            ),
                        }
                    ],
                },
            ],
        }
    else:
        payload = {
            "text": text,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Sidecar cert rotate",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                },
            ],
        }
    try:
        from rag.judge_alert import post_slack

        ok = post_slack(webhook, payload)
        return {
            "ok": bool(ok),
            "skipped": False,
            "posted": bool(ok),
            "roles": rotated,
            "stamp": stamp,
            "digest": bool(use_digest),
        }
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "posted": False,
            "error": type(exc).__name__,
        }


def resolve_sidecar_cert_rotate_pd_severity(
    report: Optional[Dict[str, Any]] = None,
) -> str:
    """PagerDuty severity for sidecar rotate fail, keyed by error/env.

    Precedence:
    1. ``RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY`` (explicit)
    2. ``RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY_BY_ERROR`` JSON map
    3. Error-class defaults (PermissionError/OSError → critical, …)
    4. fallback ``error``
    """
    allowed = {"info", "warning", "error", "critical"}
    explicit = (
        os.environ.get("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY", "")
        .strip()
        .lower()
    )
    if explicit in allowed:
        return explicit

    err = str(
        (report or {}).get("error")
        or (report or {}).get("detail")
        or (report or {}).get("reason")
        or ""
    ).strip()
    err_l = err.lower()
    raw_map = os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY_BY_ERROR", ""
    ).strip()
    if raw_map:
        try:
            data = json.loads(raw_map)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and err:
            # exact key, then case-insensitive, then substring
            mapped = str(data.get(err) or data.get(err_l) or "").strip().lower()
            if mapped in allowed:
                return mapped
            for key, val in data.items():
                k = str(key or "").strip().lower()
                if k and k in err_l:
                    mv = str(val or "").strip().lower()
                    if mv in allowed:
                        return mv

    defaults = (
        ("permissionerror", "critical"),
        ("permission", "critical"),
        ("oserror", "critical"),
        ("filenotfound", "error"),
        ("certificate", "error"),
        ("cryptography", "error"),
        ("timeout", "warning"),
        ("valueerror", "warning"),
    )
    for needle, sev in defaults:
        if needle in err_l:
            return sev
    return "error"


def maybe_notify_sidecar_cert_rotate_fail(
    report: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """PagerDuty when sidecar cert rotate fails (DLQ quarantine HMAC path)."""
    if report.get("ok") and not report.get("error"):
        return {"ok": True, "skipped": True, "reason": "rotate_ok"}
    flag = os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_NOTIFY", ""
    ).strip().lower()
    if not force and flag in {"0", "false", "no", "off"}:
        return {"ok": True, "skipped": True, "reason": "pd_notify_disabled"}
    key = (
        os.environ.get(
            "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PAGERDUTY_ROUTING_KEY", ""
        ).strip()
        or os.environ.get("RAG_JUDGE_PAGERDUTY_ROUTING_KEY", "").strip()
        or os.environ.get("RAG_PAGERDUTY_ROUTING_KEY", "").strip()
    )
    if not key:
        return {"ok": True, "skipped": True, "reason": "pd_key_missing"}
    severity = resolve_sidecar_cert_rotate_pd_severity(report)
    err = str(report.get("error") or report.get("detail") or "rotate_failed")
    pd_report = {
        "summary": {
            "ok": False,
            "mode": "sidecar_cert_rotate",
            "failed": 1,
            "total": 1,
            "error": err,
            "rotated": report.get("rotated"),
            "pd_severity": severity,
        },
        "mode": "sidecar_cert_rotate",
        "error": err,
    }
    try:
        from rag.judge_alert import post_pagerduty

        ok = post_pagerduty(
            routing_key=key,
            report=pd_report,
            source="webhook-signing-sidecar-rotate",
            severity=severity,
        )
        return {
            "ok": bool(ok),
            "skipped": False,
            "pagerduty": bool(ok),
            "severity": severity,
            "error": err,
        }
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "pagerduty": False,
            "error": type(exc).__name__,
        }


def build_signed_webhook_headers(
    body: bytes,
    *,
    secret: str,
    now: Optional[float] = None,
    nonce: Optional[str] = None,
) -> Dict[str, str]:
    """Produce X-Webhook-Timestamp / Signature / Nonce for an upstream body."""
    ts = str(int(now if now is not None else time.time()))
    n = (nonce or secrets.token_hex(16)).strip()
    sig = build_webhook_signature(body, timestamp=ts, secret=secret)
    return {
        "Content-Type": "application/json",
        "X-Webhook-Timestamp": ts,
        "X-Webhook-Signature": sig,
        "X-Webhook-Nonce": n,
    }


def emit_sidecar_forward_metric(report: Dict[str, Any]) -> None:
    """Prometheus/JSONL: rag_webhook_signing_sidecar_forward_total{mode,result}."""
    try:
        from rag.metrics import record_metric

        result = "ok" if report.get("ok") else str(
            report.get("error") or report.get("result") or "fail"
        )
        if report.get("dry_run"):
            result = "dry_run"
        record_metric(
            "webhook_signing_sidecar_forward",
            values={
                "mode": str(report.get("mode") or "dlq_quarantine"),
                "result": result,
                "status": report.get("status"),
                "upstream_client_cert": bool(report.get("upstream_client_cert")),
            },
        )
    except Exception:
        pass


def sign_and_forward_webhook(
    body: bytes,
    *,
    upstream: Optional[str] = None,
    mode: str = "dlq_quarantine",
    secret: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    now: Optional[float] = None,
    nonce: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Sign body and POST to upstream; return status + response snippet."""
    sec = (
        secret
        if secret is not None
        else webhook_signing_secret(mode=mode)
    )
    up = (upstream or sidecar_upstream_url(mode=mode)).strip()
    if not sec:
        out = {
            "ok": False,
            "error": "signing_secret_missing",
            "mode": mode,
            "upstream": up,
        }
        emit_sidecar_forward_metric(out)
        return out
    if not up:
        out = {
            "ok": False,
            "error": "upstream_missing",
            "mode": mode,
        }
        emit_sidecar_forward_metric(out)
        return out
    headers = build_signed_webhook_headers(
        body, secret=sec, now=now, nonce=nonce
    )
    if extra_headers:
        for k, v in extra_headers.items():
            if k.lower() in {
                "host",
                "content-length",
                "x-webhook-timestamp",
                "x-webhook-signature",
                "x-webhook-nonce",
            }:
                continue
            headers[k] = v
    timeout = float(
        timeout_sec if timeout_sec is not None else sidecar_forward_timeout_sec()
    )
    try:
        ssl_ctx = build_upstream_ssl_context()
    except ValueError as exc:
        out = {
            "ok": False,
            "error": str(exc),
            "mode": mode,
            "upstream": up,
        }
        emit_sidecar_forward_metric(out)
        return out
    req = Request(up, data=body, headers=headers, method="POST")
    open_kw: Dict[str, Any] = {"timeout": timeout}
    if ssl_ctx is not None:
        open_kw["context"] = ssl_ctx
    try:
        with urlopen(req, **open_kw) as resp:
            resp_body = resp.read()
            status = int(getattr(resp, "status", 200) or 200)
            out = {
                "ok": 200 <= status < 300,
                "status": status,
                "mode": mode,
                "upstream": up,
                "upstream_client_cert": bool(ssl_ctx and sidecar_upstream_client_cert_enabled()),
                "signed": {
                    "timestamp": headers["X-Webhook-Timestamp"],
                    "nonce": headers["X-Webhook-Nonce"],
                    "signature": headers["X-Webhook-Signature"],
                },
                "response": resp_body.decode("utf-8", errors="replace")[:4000],
            }
            if not out["ok"]:
                out["error"] = f"upstream_status_{status}"
            emit_sidecar_forward_metric(out)
            return out
    except HTTPError as exc:
        try:
            resp_body = exc.read()
        except Exception:
            resp_body = b""
        out = {
            "ok": False,
            "error": "upstream_http_error",
            "status": int(exc.code),
            "mode": mode,
            "upstream": up,
            "upstream_client_cert": bool(ssl_ctx and sidecar_upstream_client_cert_enabled()),
            "signed": {
                "timestamp": headers["X-Webhook-Timestamp"],
                "nonce": headers["X-Webhook-Nonce"],
                "signature": headers["X-Webhook-Signature"],
            },
            "response": resp_body.decode("utf-8", errors="replace")[:4000],
        }
        emit_sidecar_forward_metric(out)
        return out
    except URLError as exc:
        out = {
            "ok": False,
            "error": "upstream_unreachable",
            "detail": str(exc.reason or exc),
            "mode": mode,
            "upstream": up,
            "upstream_client_cert": bool(ssl_ctx and sidecar_upstream_client_cert_enabled()),
            "signed": {
                "timestamp": headers["X-Webhook-Timestamp"],
                "nonce": headers["X-Webhook-Nonce"],
                "signature": headers["X-Webhook-Signature"],
            },
        }
        emit_sidecar_forward_metric(out)
        return out
    except Exception as exc:
        out = {
            "ok": False,
            "error": "forward_failed",
            "detail": str(exc),
            "mode": mode,
            "upstream": up,
            "upstream_client_cert": bool(
                ssl_ctx is not None and sidecar_upstream_client_cert_enabled()
            ),
        }
        emit_sidecar_forward_metric(out)
        return out


def handle_signing_sidecar_http(
    body: bytes,
    *,
    path: str = "/",
    headers: Optional[Dict[str, str]] = None,
    upstream: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[int, Dict[str, str], bytes]:
    """HTTP adapter used by the sidecar server and tests."""
    mode = sidecar_mode_from_path(path)
    if dry_run or os.environ.get(
        "RAG_WEBHOOK_SIGNING_SIDECAR_DRY_RUN", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        secret = webhook_signing_secret(mode=mode)
        if not secret:
            payload = {"ok": False, "error": "signing_secret_missing", "mode": mode}
            emit_sidecar_forward_metric(payload)
            return (
                500,
                {"Content-Type": "application/json"},
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )
        signed = build_signed_webhook_headers(body, secret=secret)
        payload = {
            "ok": True,
            "dry_run": True,
            "mode": mode,
            "upstream": upstream or sidecar_upstream_url(mode=mode),
            "signed": {
                "timestamp": signed["X-Webhook-Timestamp"],
                "nonce": signed["X-Webhook-Nonce"],
                "signature": signed["X-Webhook-Signature"],
            },
        }
        emit_sidecar_forward_metric(payload)
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    report = sign_and_forward_webhook(
        body,
        upstream=upstream,
        mode=mode,
        extra_headers={
            k: v
            for k, v in (headers or {}).items()
            if k.lower().startswith("x-") or k.lower() == "authorization"
        },
    )
    status = int(report.get("status") or (200 if report.get("ok") else 502))
    if report.get("error") == "signing_secret_missing":
        status = 500
    elif report.get("error") in {"upstream_missing", "upstream_unreachable", "forward_failed"}:
        status = 502
    return (
        status,
        {"Content-Type": "application/json"},
        json.dumps(report, ensure_ascii=False).encode("utf-8"),
    )


class _SigningSidecarHandler(BaseHTTPRequestHandler):
    server_version = "RagWebhookSigningSidecar/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except Exception:
            length = 0
        if length <= 0:
            return b"{}"
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802
        path = (self.path or "/").split("?", 1)[0]
        if path in {"/", "/healthz", "/readyz"}:
            tls = sidecar_tls_status()
            body = json.dumps(
                {
                    "ok": True,
                    "service": "webhook-signing-sidecar",
                    "upstream_quarantine": sidecar_upstream_url(mode="dlq_quarantine"),
                    "upstream_catch_up": sidecar_upstream_url(mode="catch_up"),
                    "tls": tls.get("tls"),
                    "mtls": tls.get("mtls"),
                    "upstream_client_cert": tls.get("upstream_client_cert"),
                    "upstream_ca": tls.get("upstream_ca"),
                    "cert_expiry": tls.get("cert_expiry"),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in {"/certs", "/check-certs"}:
            report = inspect_sidecar_certs()
            emit_sidecar_cert_metrics(report)
            body = json.dumps(report, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()
        headers = {k: v for k, v in self.headers.items()}
        code, resp_headers, resp_body = handle_signing_sidecar_http(
            body, path=self.path or "/", headers=headers
        )
        self.send_response(code)
        for k, v in resp_headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)


def run_webhook_signing_sidecar(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    ssl_context: Optional[ssl.SSLContext] = None,
) -> ThreadingHTTPServer:
    bind_host = host or sidecar_listen_host()
    bind_port = int(port if port is not None else sidecar_listen_port())
    server = ThreadingHTTPServer((bind_host, bind_port), _SigningSidecarHandler)
    ctx = ssl_context if ssl_context is not None else build_sidecar_ssl_context()
    if ctx is not None:
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    return server
