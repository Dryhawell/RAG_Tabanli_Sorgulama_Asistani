"""Alertmanager → RAG webhook HMAC signing sidecar.

Alertmanager cannot attach X-Webhook-* HMAC headers natively. Point AM at this
proxy; it signs the body and forwards to the real catch-up / quarantine hook.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
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
        return {
            "ok": False,
            "error": "signing_secret_missing",
            "mode": mode,
            "upstream": up,
        }
    if not up:
        return {
            "ok": False,
            "error": "upstream_missing",
            "mode": mode,
        }
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
    req = Request(up, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
            status = int(getattr(resp, "status", 200) or 200)
            return {
                "ok": 200 <= status < 300,
                "status": status,
                "mode": mode,
                "upstream": up,
                "signed": {
                    "timestamp": headers["X-Webhook-Timestamp"],
                    "nonce": headers["X-Webhook-Nonce"],
                    "signature": headers["X-Webhook-Signature"],
                },
                "response": resp_body.decode("utf-8", errors="replace")[:4000],
            }
    except HTTPError as exc:
        try:
            resp_body = exc.read()
        except Exception:
            resp_body = b""
        return {
            "ok": False,
            "error": "upstream_http_error",
            "status": int(exc.code),
            "mode": mode,
            "upstream": up,
            "signed": {
                "timestamp": headers["X-Webhook-Timestamp"],
                "nonce": headers["X-Webhook-Nonce"],
                "signature": headers["X-Webhook-Signature"],
            },
            "response": resp_body.decode("utf-8", errors="replace")[:4000],
        }
    except URLError as exc:
        return {
            "ok": False,
            "error": "upstream_unreachable",
            "detail": str(exc.reason or exc),
            "mode": mode,
            "upstream": up,
            "signed": {
                "timestamp": headers["X-Webhook-Timestamp"],
                "nonce": headers["X-Webhook-Nonce"],
                "signature": headers["X-Webhook-Signature"],
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": "forward_failed",
            "detail": str(exc),
            "mode": mode,
            "upstream": up,
        }


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
            body = json.dumps(
                {
                    "ok": True,
                    "service": "webhook-signing-sidecar",
                    "upstream_quarantine": sidecar_upstream_url(mode="dlq_quarantine"),
                    "upstream_catch_up": sidecar_upstream_url(mode="catch_up"),
                },
                ensure_ascii=False,
            ).encode("utf-8")
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
) -> ThreadingHTTPServer:
    bind_host = host or sidecar_listen_host()
    bind_port = int(port if port is not None else sidecar_listen_port())
    server = ThreadingHTTPServer((bind_host, bind_port), _SigningSidecarHandler)
    return server
