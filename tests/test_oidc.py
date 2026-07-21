import base64
import json
from unittest.mock import MagicMock

import pytest

from rag.auth import upsert_oidc_user
from rag.oidc import (
    OIDCConfig,
    OIDCError,
    build_authorize_url,
    complete_login,
    decode_jwt_payload,
    extract_username,
    generate_pkce_pair,
    resolve_role,
    resolve_tenant,
)


def _b64url(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fake_jwt(payload: dict) -> str:
    header = _b64url({"alg": "none", "typ": "JWT"})
    body = _b64url(payload)
    return f"{header}.{body}.sig"


def test_pkce_and_decode_jwt():
    verifier, challenge = generate_pkce_pair()
    assert len(verifier) > 20
    assert len(challenge) > 20
    assert "=" not in challenge

    token = _fake_jwt({"sub": "u1", "preferred_username": "alice", "nonce": "n1"})
    claims = decode_jwt_payload(token)
    assert claims["preferred_username"] == "alice"


def test_role_and_tenant_from_claims():
    cfg = OIDCConfig(
        issuer="https://idp.example",
        client_id="rag",
        client_secret="secret",
        redirect_uri="http://localhost:8501",
        admin_groups=("rag-admins",),
        default_role="user",
        tenant_claim="tenant_id",
    )
    assert resolve_role({"groups": ["rag-admins"]}, cfg) == "admin"
    assert resolve_role({"groups": ["viewers"]}, cfg) == "user"
    assert resolve_tenant({"tenant_id": "Acme"}, cfg) == "Acme"
    assert extract_username({"email": "bob@corp.com"}, cfg) == "bob"


def test_build_authorize_url_includes_pkce():
    cfg = OIDCConfig(
        issuer="https://idp.example",
        client_id="rag",
        client_secret="s",
        redirect_uri="http://localhost:8501",
    )
    disc = {"authorization_endpoint": "https://idp.example/auth"}
    url = build_authorize_url(
        disc, cfg, state="st", nonce="nn", code_challenge="ch"
    )
    assert "client_id=rag" in url
    assert "state=st" in url
    assert "code_challenge=ch" in url
    assert "code_challenge_method=S256" in url


def test_complete_login_mocked(monkeypatch):
    cfg = OIDCConfig(
        issuer="https://idp.example",
        client_id="rag",
        client_secret="secret",
        redirect_uri="http://localhost:8501",
        admin_groups=("admins",),
        auto_provision=True,
    )
    id_token = _fake_jwt(
        {
            "sub": "42",
            "preferred_username": "Carol",
            "nonce": "nonce-1",
            "groups": ["admins"],
            "tenant_id": "beta",
            "email": "carol@ex.com",
            "exp": 9_999_999_999,
        }
    )
    discovery = {
        "authorization_endpoint": "https://idp.example/auth",
        "token_endpoint": "https://idp.example/token",
        "userinfo_endpoint": "https://idp.example/userinfo",
    }

    class FakeResp:
        def __init__(self, data, status=200):
            self._data = data
            self.status_code = status
            self.text = json.dumps(data)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")

        def json(self):
            return self._data

    sess = MagicMock()
    sess.post.return_value = FakeResp(
        {"access_token": "at", "id_token": id_token, "token_type": "Bearer"}
    )
    sess.get.return_value = FakeResp({"name": "Carol"})

    result = complete_login(
        cfg,
        code="authcode",
        expected_state="abc",
        received_state="abc",
        expected_nonce="nonce-1",
        code_verifier="verifier",
        discovery=discovery,
        session=sess,
    )
    assert result["username"] == "Carol"
    assert result["role"] == "admin"
    assert result["tenant_id"] == "beta"

    with pytest.raises(OIDCError):
        complete_login(
            cfg,
            code="authcode",
            expected_state="abc",
            received_state="WRONG",
            discovery=discovery,
            session=sess,
        )


def test_upsert_oidc_user(tmp_path, monkeypatch):
    path = str(tmp_path / "users.json")
    monkeypatch.setattr("rag.auth.AUTH_BOOTSTRAP_ADMIN", "")

    u = upsert_oidc_user(
        "Alice.O",
        role="admin",
        tenant_id="acme",
        path=path,
        auto_provision=True,
        claims={"email": "alice@acme.com"},
    )
    assert u.username == "alice.o"
    assert u.role == "admin"
    assert u.tenant_id == "acme"

    # ikinci çağrı günceller
    u2 = upsert_oidc_user(
        "alice.o",
        role="user",
        tenant_id="beta",
        path=path,
        auto_provision=True,
    )
    assert u2.role == "user"
    assert u2.tenant_id == "beta"

    with pytest.raises(ValueError):
        upsert_oidc_user(
            "newuser",
            path=path,
            auto_provision=False,
        )
