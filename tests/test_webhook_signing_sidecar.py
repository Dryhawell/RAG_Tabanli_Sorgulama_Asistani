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
    posts: list = []

    def _fake_slack(url, payload):
        posts.append((url, payload))
        return True

    monkeypatch.setattr("rag.judge_alert.post_slack", _fake_slack)
    notified = maybe_notify_sidecar_cert_rotate(
        {
            "ok": True,
            "rotated": ["server", "upstream_client"],
            "stamp": "20990101T000000Z",
            "after": {"min_days_left": 60.0},
        },
        force=True,
    )
    assert notified.get("ok") is True
    assert notified.get("skipped") is not True
    assert notified.get("posted") is True
    assert posts and posts[0][0].endswith("/rotate-notify")
    assert "rotated" in str(posts[0][1].get("text") or "").lower()
