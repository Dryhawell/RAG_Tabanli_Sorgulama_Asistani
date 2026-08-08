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

    class _Resp:
        status = 200

        def read(self):
            return b'{"ok":true,"auth":"hmac"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen(req, timeout=30):
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
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "webhook-signing-sidecar" in readme
    assert "RAG_WEBHOOK_SIGNING_SIDECAR_UPSTREAM" in readme
