"""Webhook HMAC signing sidecar tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_build_signed_headers_and_dry_run(monkeypatch) -> None:
    from rag.dual_write_webhook import build_webhook_signature, verify_webhook_signature
    from rag.webhook_signing_sidecar import (
        build_signed_webhook_headers,
        handle_signing_sidecar_http,
    )

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_WEBHOOK_SIGNING_SECRET", "sidecar-secret"
    )
    body = b'{"alerts":[{"status":"firing"}]}'
    headers = build_signed_webhook_headers(body, secret="sidecar-secret", now=1_700_000_000.0)
    assert headers["X-Webhook-Timestamp"] == "1700000000"
    assert headers["X-Webhook-Signature"].startswith("v0=")
    assert len(headers["X-Webhook-Nonce"]) >= 16
    assert verify_webhook_signature(
        body,
        timestamp=headers["X-Webhook-Timestamp"],
        signature=headers["X-Webhook-Signature"],
        signing_secret="sidecar-secret",
        now=1_700_000_000.0,
    )
    expected = build_webhook_signature(
        body, timestamp="1700000000", secret="sidecar-secret"
    )
    assert headers["X-Webhook-Signature"] == expected

    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_DRY_RUN", "1")
    code, _, resp = handle_signing_sidecar_http(
        body, path="/hooks/dual-write-dlq-quarantine"
    )
    assert code == 200
    data = json.loads(resp.decode())
    assert data["ok"] is True
    assert data["dry_run"] is True
    assert data["mode"] == "dlq_quarantine"
    assert data["signed"]["signature"].startswith("v0=")


def test_sign_and_forward_uses_upstream(monkeypatch, tmp_path: Path) -> None:
    from rag.webhook_signing_sidecar import sign_and_forward_webhook

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_WEBHOOK_SIGNING_SECRET", "fwd-secret"
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM",
        "http://example.test/hooks/dual-write-dlq-quarantine",
    )
    body = b'{"status":"firing","alerts":[]}'
    captured = {}
    metrics: list[dict] = []
    monkeypatch.setattr(
        "rag.metrics.record_metric",
        lambda kind, values=None, **kw: metrics.append(
            {"kind": kind, "values": values or {}}
        ),
    )

    class _Resp:
        status = 200

        def read(self):
            return b'{"ok":true,"auth":"hmac"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen(req, timeout=30, context=None):
        captured["url"] = req.full_url
        captured["headers"] = {k: v for k, v in req.header_items()}
        captured["data"] = req.data
        return _Resp()

    with patch("rag.webhook_signing_sidecar.urlopen", side_effect=_urlopen):
        out = sign_and_forward_webhook(body, mode="dlq_quarantine")
    assert out["ok"] is True
    assert out["status"] == 200
    assert "dual-write-dlq-quarantine" in captured["url"]
    hdrs = {k.lower(): v for k, v in captured["headers"].items()}
    assert "x-webhook-signature" in hdrs
    assert "x-webhook-timestamp" in hdrs
    assert "x-webhook-nonce" in hdrs
    assert captured["data"] == body
    assert any(
        m["kind"] == "webhook_signing_sidecar_forward"
        and m["values"].get("result") == "ok"
        for m in metrics
    )


def test_cli_sign_once(monkeypatch, capsys) -> None:
    from rag.cli import main

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_WEBHOOK_SIGNING_SECRET", "cli-secret"
    )
    monkeypatch.setattr(
        "sys.stdin",
        MagicMock(buffer=MagicMock(read=MagicMock(return_value=b'{"a":1}'))),
    )
    code = main(
        [
            "webhook-signing-sidecar",
            "--sign-once",
            "--mode",
            "dlq_quarantine",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["headers"]["X-Webhook-Signature"].startswith("v0=")


def test_compose_and_readme_document_sidecar() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "webhook-signing-sidecar:" in compose
    assert "rag.cli" in compose
    assert "webhook-signing-sidecar" in compose
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT" in compose
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_MTLS" in compose
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "webhook-signing-sidecar" in readme
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM" in readme
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_MTLS" in readme


def _write_self_signed_pair(tmp_path: Path):
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "sidecar.test")]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "server.crt"
    key_path = tmp_path / "server.key"
    ca_path = tmp_path / "ca.crt"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    # Self-signed: CA == server cert for client trust / mTLS demo
    ca_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client_subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "am-client")]
    )
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_subject)
        .issuer_name(issuer)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    client_cert_path = tmp_path / "client.crt"
    client_key_path = tmp_path / "client.key"
    client_cert_path.write_bytes(client_cert.public_bytes(serialization.Encoding.PEM))
    client_key_path.write_bytes(
        client_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path, ca_path, client_cert_path, client_key_path


def test_sidecar_tls_and_mtls_context(tmp_path: Path, monkeypatch) -> None:
    cryptography = pytest.importorskip("cryptography")
    _ = cryptography

    from rag.webhook_signing_sidecar import (
        build_sidecar_ssl_context,
        sidecar_mtls_required,
        sidecar_tls_enabled,
        sidecar_tls_status,
    )

    cert, key, ca, _cc, _ck = _write_self_signed_pair(tmp_path)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT", raising=False)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_KEY", raising=False)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CA", raising=False)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_MTLS", raising=False)
    assert sidecar_tls_enabled() is False
    assert build_sidecar_ssl_context() is None

    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT", str(cert))
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_KEY", str(key))
    assert sidecar_tls_enabled() is True
    assert sidecar_mtls_required() is False
    ctx = build_sidecar_ssl_context()
    assert ctx is not None

    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CA", str(ca))
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_MTLS", "1")
    assert sidecar_mtls_required() is True
    mtls_ctx = build_sidecar_ssl_context()
    assert mtls_ctx is not None
    assert mtls_ctx.verify_mode.name == "CERT_REQUIRED"
    status = sidecar_tls_status()
    assert status["tls"] is True
    assert status["mtls"] is True


def test_upstream_client_cert_context(tmp_path: Path, monkeypatch) -> None:
    cryptography = pytest.importorskip("cryptography")
    _ = cryptography
    from rag.webhook_signing_sidecar import (
        build_upstream_ssl_context,
        sidecar_tls_status,
        sidecar_upstream_client_cert_enabled,
        sign_and_forward_webhook,
    )

    cert, key, ca, client_cert, client_key = _write_self_signed_pair(tmp_path)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT", raising=False)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY", raising=False)
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CA", raising=False)
    assert sidecar_upstream_client_cert_enabled() is False
    assert build_upstream_ssl_context() is None

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT", str(client_cert)
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY", str(client_key)
    )
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CA", str(ca))
    assert sidecar_upstream_client_cert_enabled() is True
    ctx = build_upstream_ssl_context()
    assert ctx is not None
    status = sidecar_tls_status()
    assert status["upstream_client_cert"] is True
    assert status["upstream_ca"] is True

    monkeypatch.setenv(
        "RAG_DUAL_WRITE_DLQ_QUARANTINE_WEBHOOK_SIGNING_SECRET", "fwd-secret"
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM",
        "https://example.test/hooks/dual-write-dlq-quarantine",
    )
    captured = {}

    class _Resp:
        status = 200

        def read(self):
            return b'{"ok":true}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen(req, timeout=30, context=None):
        captured["context"] = context
        captured["url"] = req.full_url
        return _Resp()

    with patch("rag.webhook_signing_sidecar.urlopen", side_effect=_urlopen):
        out = sign_and_forward_webhook(b'{"alerts":[]}', mode="dlq_quarantine")
    assert out["ok"] is True
    assert out.get("upstream_client_cert") is True
    assert captured["context"] is not None

    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY", raising=False)
    with pytest.raises(ValueError, match="upstream_client_cert_incomplete"):
        build_upstream_ssl_context()
    _ = (cert, key)  # server pair unused here


def test_sidecar_mtls_requires_ca(monkeypatch, tmp_path: Path) -> None:
    cryptography = pytest.importorskip("cryptography")
    _ = cryptography
    from rag.webhook_signing_sidecar import build_sidecar_ssl_context

    cert, key, _ca, _cc, _ck = _write_self_signed_pair(tmp_path)
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT", str(cert))
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_KEY", str(key))
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CA", raising=False)
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_MTLS", "1")
    with pytest.raises(ValueError, match="mtls_requires_ca"):
        build_sidecar_ssl_context()


def test_sidecar_cert_expiry_inspect(tmp_path: Path, monkeypatch) -> None:
    cryptography = pytest.importorskip("cryptography")
    _ = cryptography
    from rag.webhook_signing_sidecar import (
        inspect_sidecar_certs,
        sidecar_tls_status,
    )

    cert, key, ca, client_cert, client_key = _write_self_signed_pair(tmp_path)
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CERT", str(cert))
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_KEY", str(key))
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_TLS_CA", str(ca))
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_MTLS", "1")
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_CERT", str(client_cert)
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM_CLIENT_KEY", str(client_key)
    )
    report = inspect_sidecar_certs()
    assert report["ok"] is True
    assert report["server"]["ok"] is True
    assert report["server"]["days_left"] > 0
    assert report["upstream_client"]["ok"] is True
    assert report["min_days_left"] is not None
    status = sidecar_tls_status()
    assert status["cert_expiry"]["server_days_left"] is not None
    assert status["cert_expiry"]["min_days_left"] is not None

    from rag.cli import build_parser
    from rag.webhook_signing_sidecar import rotate_sidecar_certs

    args = build_parser().parse_args(
        ["webhook-signing-sidecar", "--check-certs", "--tls-cert", str(cert)]
    )
    assert args.check_certs is True
    rot_args = build_parser().parse_args(
        ["webhook-signing-sidecar", "--rotate-certs-if-expiring", "--cert-days", "60"]
    )
    assert rot_args.rotate_certs_if_expiring is True
    # Force rotate: certs are 1d validity; warn=14 → rotate
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_CERT_EXPIRY_WARN_DAYS", "14")
    rotated = rotate_sidecar_certs(
        if_expiring_days=14,
        days=60,
        out_dir=str(tmp_path / "tls-rot"),
    )
    assert rotated["ok"] is True
    assert "server" in (rotated.get("rotated") or rotated.get("would_rotate") or [])
    assert (rotated.get("after") or {}).get("server", {}).get("days_left", 0) > 30

    from rag.cli import build_parser as _bp
    from rag.webhook_signing_sidecar import maybe_notify_sidecar_cert_rotate

    notify_args = _bp().parse_args(
        ["webhook-signing-sidecar", "--rotate-certs", "--notify"]
    )
    assert notify_args.notify is True

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK",
        "https://hooks.slack.test/rotate-notify",
    )
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-rotate")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_CHANNEL", "C-rot")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_TS", "99.1")
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_STATE",
        str(tmp_path / "rotate_thread.json"),
    )
    posts: list = []
    threads: list = []

    def _fake_slack(url, payload):
        posts.append((url, payload))
        return True

    def _fake_thread(**kwargs):
        threads.append(kwargs)
        return {"ok": True, "ts": "99.2", "thread_ts": kwargs.get("thread_ts")}

    monkeypatch.setattr("rag.judge_alert.post_slack", _fake_slack)
    monkeypatch.setattr("rag.judge_alert.post_slack_thread_message", _fake_thread)
    notified = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server", "upstream_client"],
            "stamp": "20990101T000000Z",
            "before": {
                "server": {"days_left": 3.0},
                "upstream_client": {"days_left": 2.0},
            },
            "after": {
                "min_days_left": 60.0,
                "server": {"days_left": 60.0},
                "upstream_client": {"days_left": 60.0},
            },
        },
        force=True,
    )
    assert notified.get("ok") is True
    assert notified.get("skipped") is not True
    assert notified.get("posted") is True
    assert notified.get("digest") is True
    assert posts and posts[0][0].endswith("/rotate-notify")
    body = posts[0][1]
    assert body.get("thread_ts") == "99.1"
    assert "digest" in str(body.get("text") or "").lower()
    assert "days_left" in str(body.get("text") or "")
    assert "was" in str(body.get("text") or "")
    headers = [
        b.get("text", {}).get("text")
        for b in (body.get("blocks") or [])
        if b.get("type") == "header"
    ]
    assert any("digest" in str(h).lower() for h in headers)
    ctx = [
        el.get("text")
        for b in (body.get("blocks") or [])
        if b.get("type") == "context"
        for el in (b.get("elements") or [])
    ]
    assert any("rag-ingest" in str(c) for c in ctx)
    assert notified.get("thread") is True
    assert notified.get("thread_reply", {}).get("ok") is True
    assert threads and "digest thread" in str(threads[0].get("text") or "").lower()
    assert (tmp_path / "rotate_thread.json").is_file()

    posts.clear()
    threads.clear()
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "0")
    legacy = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server"],
            "stamp": "20990101T000001Z",
            "after": {"min_days_left": 60.0, "server": {"days_left": 60.0}},
        },
        force=True,
    )
    assert legacy.get("digest") is False
    assert legacy.get("thread") is not True
    legacy_headers = [
        b.get("text", {}).get("text")
        for b in (posts[0][1].get("blocks") or [])
        if b.get("type") == "header"
    ]
    assert any(h == "Sidecar cert rotate" for h in legacy_headers)
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "1")

    from rag.webhook_signing_sidecar import maybe_notify_sidecar_cert_rotate_fail

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PAGERDUTY_ROUTING_KEY", "pd-rotate-key"
    )
    pd_calls: list = []

    def _fake_pd(**kwargs):
        pd_calls.append(kwargs)
        return True

    monkeypatch.setattr("rag.judge_alert.post_pagerduty", _fake_pd)
    skipped_ok = maybe_notify_sidecar_cert_rotate_fail(
        {"ok": True, "rotated": ["server"]}, force=True
    )
    assert skipped_ok.get("skipped") is True
    failed = maybe_notify_sidecar_cert_rotate_fail(
        {"ok": False, "error": "RotateBoom", "rotated": []},
        force=True,
    )
    assert failed.get("ok") is True
    assert failed.get("pagerduty") is True
    assert failed.get("skipped") is not True
    assert pd_calls
    assert pd_calls[0].get("routing_key") == "pd-rotate-key"
    assert pd_calls[0].get("source") == "webhook-signing-sidecar-rotate"
    assert failed.get("severity") == "error"

    from rag.webhook_signing_sidecar import resolve_sidecar_cert_rotate_pd_severity

    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY", raising=False)
    assert (
        resolve_sidecar_cert_rotate_pd_severity({"error": "PermissionError"})
        == "critical"
    )
    assert resolve_sidecar_cert_rotate_pd_severity({"error": "Timeout"}) == "warning"
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY_BY_ERROR",
        '{"RotateBoom":"critical","timeout":"info"}',
    )
    assert (
        resolve_sidecar_cert_rotate_pd_severity({"error": "RotateBoom"}) == "critical"
    )
    pd_calls.clear()
    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_PD_SEVERITY", raising=False)
    sev_out = maybe_notify_sidecar_cert_rotate_fail(
        {"ok": False, "error": "RotateBoom", "rotated": []},
        force=True,
    )
    assert sev_out.get("severity") == "critical"
    assert pd_calls[0].get("severity") == "critical"

    # Dry-run CI gate: would_rotate → exit 2
    from rag.cli import cmd_webhook_signing_sidecar
    from argparse import Namespace

    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_DRY_RUN_GATE", "1")
    monkeypatch.setattr(
        "rag.webhook_signing_sidecar.rotate_sidecar_certs",
        lambda **kw: {
            "ok": True,
            "dry_run": True,
            "would_rotate": ["server"],
            "rotated": [],
        },
    )
    monkeypatch.setattr(
        "rag.webhook_signing_sidecar.maybe_notify_sidecar_cert_rotate",
        lambda *a, **k: {"ok": True, "skipped": True},
    )
    monkeypatch.setattr(
        "rag.webhook_signing_sidecar.maybe_notify_sidecar_cert_rotate_fail",
        lambda *a, **k: {"ok": True, "skipped": True},
    )
    rc = cmd_webhook_signing_sidecar(
        Namespace(
            rotate_certs=False,
            rotate_certs_if_expiring=True,
            dry_run=True,
            cert_days=90,
            notify=False,
            check_certs=False,
            sign_once=False,
            upstream=None,
            tls_cert=None,
            tls_key=None,
            tls_ca=None,
            mtls=False,
            upstream_client_cert=None,
            upstream_client_key=None,
            upstream_ca=None,
        )
    )
    assert rc == 2


def test_sidecar_cert_rotate_notify_digest_thread(tmp_path: Path, monkeypatch) -> None:
    from rag.webhook_signing_sidecar import maybe_notify_sidecar_cert_rotate

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK",
        "https://hooks.slack.test/rotate-notify",
    )
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-rotate")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_CHANNEL", "C-rot")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_TS", "99.1")
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_STATE",
        str(tmp_path / "rotate_thread.json"),
    )
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "1")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD", "1")
    posts: list = []
    threads: list = []

    def _fake_slack(url, payload):
        posts.append((url, payload))
        return True

    def _fake_thread(**kwargs):
        threads.append(kwargs)
        return {"ok": True, "ts": "99.2", "thread_ts": kwargs.get("thread_ts")}

    monkeypatch.setattr("rag.judge_alert.post_slack", _fake_slack)
    monkeypatch.setattr("rag.judge_alert.post_slack_thread_message", _fake_thread)
    monkeypatch.setattr(
        "rag.judge_alert.slack_api",
        lambda method, **kwargs: {"ok": True, "method": method},
    )
    notified = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server"],
            "stamp": "20990101T000000Z",
            "before": {"server": {"days_left": 3.0}},
            "after": {"min_days_left": 60.0, "server": {"days_left": 60.0}},
        },
        force=True,
    )
    assert notified.get("ok") is True
    assert notified.get("digest") is True
    assert notified.get("thread") is True
    assert notified.get("thread_reply", {}).get("ok") is True
    assert posts and posts[0][1].get("thread_ts") == "99.1"
    assert posts[0][1].get("channel") == "C-rot"
    assert notified.get("channel_id") == "C-rot"
    ctx = [
        el.get("text")
        for b in (posts[0][1].get("blocks") or [])
        if b.get("type") == "context"
        for el in (b.get("elements") or [])
    ]
    assert any("channel=`C-rot`" in str(c) for c in ctx)
    assert threads and "digest thread" in str(threads[0].get("text") or "").lower()
    assert threads[0].get("channel_id") == "C-rot"
    state = json.loads((tmp_path / "rotate_thread.json").read_text(encoding="utf-8"))
    assert state.get("thread_ts") == "99.1"
    assert state.get("parent_ts") == "99.1"
    assert state.get("last_reply_ts") == "99.2"
    assert notified.get("persist", {}).get("ok") is True
    hist = state.get("history") or []
    assert len(hist) == 1
    assert hist[0].get("stamp") == "20990101T000000Z"

    posts.clear()
    threads.clear()
    monkeypatch.delenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_TS", raising=False
    )
    again = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["upstream_client"],
            "stamp": "20990101T000010Z",
            "after": {
                "min_days_left": 60.0,
                "upstream_client": {"days_left": 60.0},
            },
        },
        force=True,
    )
    assert again.get("thread") is True
    assert posts and posts[0][1].get("thread_ts") == "99.1"
    state2 = json.loads((tmp_path / "rotate_thread.json").read_text(encoding="utf-8"))
    assert state2.get("parent_ts") == "99.1"
    assert len(state2.get("history") or []) == 2

    posts.clear()
    threads.clear()
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "0")
    legacy = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server"],
            "stamp": "20990101T000001Z",
            "after": {"min_days_left": 60.0, "server": {"days_left": 60.0}},
        },
        force=True,
    )
    assert legacy.get("digest") is False
    assert legacy.get("thread") is not True
    assert not threads


def test_resolve_sidecar_rotate_notify_channel(tmp_path: Path, monkeypatch) -> None:
    from rag.webhook_signing_sidecar import (
        resolve_sidecar_rotate_notify_channel,
        save_sidecar_rotate_notify_thread_state,
    )

    monkeypatch.delenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_CHANNEL", raising=False)
    monkeypatch.delenv("RAG_DUAL_WRITE_DLQ_QUARANTINE_SLACK_CHANNEL", raising=False)
    monkeypatch.delenv("RAG_JUDGE_SLACK_CHANNEL", raising=False)
    assert resolve_sidecar_rotate_notify_channel({}) == ""
    path = str(tmp_path / "st.json")
    save_sidecar_rotate_notify_thread_state({"channel_id": "C-stored"}, path=path)
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_STATE", path
    )
    from rag.webhook_signing_sidecar import load_sidecar_rotate_notify_thread_state

    stored = load_sidecar_rotate_notify_thread_state()
    assert resolve_sidecar_rotate_notify_channel(stored) == "C-stored"
    monkeypatch.setenv("RAG_JUDGE_SLACK_CHANNEL", "C-judge")
    # stored still wins over judge channel
    assert resolve_sidecar_rotate_notify_channel(stored) == "C-stored"
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_CHANNEL", "C-rot")
    assert resolve_sidecar_rotate_notify_channel(stored) == "C-rot"


def test_sidecar_rotate_notify_thread_reply_broadcast(
    tmp_path: Path, monkeypatch
) -> None:
    from rag.webhook_signing_sidecar import (
        maybe_notify_sidecar_cert_rotate,
        parse_sidecar_rotate_notify_broadcast_channels,
        resolve_sidecar_rotate_notify_broadcast_channels,
    )

    assert parse_sidecar_rotate_notify_broadcast_channels("C-ops, C-sec") == [
        "C-ops",
        "C-sec",
    ]
    assert parse_sidecar_rotate_notify_broadcast_channels('["C-json"]') == ["C-json"]
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_BROADCAST_CHANNELS",
        "C-ops,C-rot",
    )
    chans = resolve_sidecar_rotate_notify_broadcast_channels({}, primary="C-rot")
    assert chans == ["C-ops"]
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_BROADCAST", "0"
    )
    assert resolve_sidecar_rotate_notify_broadcast_channels({}, primary="C-rot") == []

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_BROADCAST", "1"
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK",
        "https://hooks.slack.test/rotate-notify",
    )
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-rotate")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_CHANNEL", "C-rot")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_TS", "99.1")
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_STATE",
        str(tmp_path / "rotate_thread.json"),
    )
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "1")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD", "1")
    threads: list = []

    def _fake_slack(url, payload):
        return True

    def _fake_thread(**kwargs):
        threads.append(kwargs)
        return {"ok": True, "ts": "99.2", "thread_ts": kwargs.get("thread_ts")}

    monkeypatch.setattr("rag.judge_alert.post_slack", _fake_slack)
    monkeypatch.setattr("rag.judge_alert.post_slack_thread_message", _fake_thread)
    monkeypatch.setattr(
        "rag.judge_alert.slack_api",
        lambda method, **kwargs: {"ok": True, "method": method},
    )
    notified = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server"],
            "stamp": "20990101T000000Z",
            "after": {"min_days_left": 60.0, "server": {"days_left": 60.0}},
        },
        force=True,
    )
    assert notified.get("ok") is True
    bcast = notified.get("broadcast") or {}
    assert bcast.get("skipped") is not True
    assert bcast.get("posted") == 1
    assert bcast.get("channels") == ["C-ops"]
    assert any(t.get("channel_id") == "C-ops" for t in threads)
    assert any(t.get("channel_id") == "C-rot" for t in threads)
    assert any(t.get("thread_ts") is None for t in threads)
    state = json.loads((tmp_path / "rotate_thread.json").read_text(encoding="utf-8"))
    assert state.get("broadcast_channels") == ["C-ops"]


def test_sidecar_rotate_notify_thread_reply_ack(tmp_path: Path, monkeypatch) -> None:
    from rag.webhook_signing_sidecar import (
        ack_sidecar_rotate_notify_thread_reply,
        maybe_notify_sidecar_cert_rotate,
        persist_sidecar_rotate_notify_thread_ack,
    )

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_STATE",
        str(tmp_path / "rotate_thread.json"),
    )
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK", "0")
    skipped = ack_sidecar_rotate_notify_thread_reply(
        channel_id="C-rot", timestamp="99.2", bot_token="xoxb"
    )
    assert skipped.get("skipped") is True
    assert skipped.get("reason") == "ack_disabled"

    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK", "1")
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK_EMOJI", "ack"
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK_PERSIST", "1"
    )
    calls: list = []

    def _fake_api(method, **kwargs):
        calls.append((method, kwargs))
        return {"ok": True}

    monkeypatch.setattr("rag.judge_alert.slack_api", _fake_api)
    acked = ack_sidecar_rotate_notify_thread_reply(
        channel_id="C-rot", timestamp="99.2", bot_token="xoxb-rotate"
    )
    assert acked.get("ok") is True
    assert acked.get("skipped") is not True
    assert acked.get("name") == "ack"
    assert acked.get("persist", {}).get("ok") is True
    assert calls and calls[0][0] == "reactions.add"
    body = (calls[0][1].get("json_body") or {})
    assert body.get("channel") == "C-rot"
    assert body.get("timestamp") == "99.2"
    assert body.get("name") == "ack"
    state = json.loads((tmp_path / "rotate_thread.json").read_text(encoding="utf-8"))
    assert state.get("last_ack_ts") == "99.2"
    assert len(state.get("ack_history") or []) == 1

    again = ack_sidecar_rotate_notify_thread_reply(
        channel_id="C-rot", timestamp="99.2", bot_token="xoxb-rotate"
    )
    assert again.get("reason") == "already_acked_persisted"
    assert len(calls) == 1

    persisted = persist_sidecar_rotate_notify_thread_ack(
        channel_id="C-rot", timestamp="99.2", name="ack"
    )
    assert persisted.get("reason") == "ack_already_persisted"

    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_SLACK_WEBHOOK",
        "https://hooks.slack.test/rotate-notify",
    )
    monkeypatch.setenv("RAG_JUDGE_SLACK_BOT_TOKEN", "xoxb-rotate")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_CHANNEL", "C-rot")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_TS", "99.1")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_DIGEST", "1")
    monkeypatch.setenv("RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD", "1")
    monkeypatch.setattr("rag.judge_alert.post_slack", lambda url, payload: True)
    monkeypatch.setattr(
        "rag.judge_alert.post_slack_thread_message",
        lambda **kwargs: {"ok": True, "ts": "99.2", "thread_ts": "99.1"},
    )
    notified = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server"],
            "stamp": "20990101T000000Z",
            "after": {"min_days_left": 60.0, "server": {"days_left": 60.0}},
        },
        force=True,
    )
    assert notified.get("ack", {}).get("ok") is True
    assert notified.get("ack", {}).get("name") == "ack"
    state = json.loads((tmp_path / "rotate_thread.json").read_text(encoding="utf-8"))
    assert state.get("last_ack_ts") == "99.2"
    assert state.get("ack_name") == "ack"
    assert len(state.get("ack_history") or []) == 1
    assert "acks=`1`" in str(notified.get("ack_history") or "")


def test_sidecar_rotate_notify_thread_ack_history(tmp_path: Path, monkeypatch) -> None:
    from rag.webhook_signing_sidecar import (
        format_sidecar_rotate_notify_thread_ack_history,
        list_sidecar_rotate_notify_thread_ack_history,
        persist_sidecar_rotate_notify_thread_ack,
        prune_sidecar_rotate_notify_thread_ack_history,
    )

    path = str(tmp_path / "rotate_thread.json")
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_STATE", path
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK_HISTORY", "1"
    )
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK_KEEP", "2"
    )
    empty = format_sidecar_rotate_notify_thread_ack_history(path=path)
    assert "acks=`0`" in empty
    persist_sidecar_rotate_notify_thread_ack(
        channel_id="C-rot", timestamp="1.1", name="ack", path=path
    )
    persist_sidecar_rotate_notify_thread_ack(
        channel_id="C-rot", timestamp="2.2", name="ack", path=path
    )
    persist_sidecar_rotate_notify_thread_ack(
        channel_id="C-rot", timestamp="3.3", name="white_check_mark", path=path
    )
    listed = list_sidecar_rotate_notify_thread_ack_history(path=path, limit=5)
    assert [h.get("timestamp") for h in listed] == ["2.2", "3.3"]
    fmt = format_sidecar_rotate_notify_thread_ack_history(path=path)
    assert "acks=`2`" in fmt
    assert "last_ack=`3.3`" in fmt
    monkeypatch.setenv(
        "RAG_WEBHOOK_SIGNING_SIDECAR_ROTATE_NOTIFY_THREAD_ACK_HISTORY", "0"
    )
    assert format_sidecar_rotate_notify_thread_ack_history(path=path) == ""
    pruned = prune_sidecar_rotate_notify_thread_ack_history(keep=1, path=path)
    assert pruned.get("ok") is True
    assert pruned.get("before") == 2
    assert pruned.get("after") == 1
    assert list_sidecar_rotate_notify_thread_ack_history(path=path)[-1].get(
        "timestamp"
    ) == "3.3"

