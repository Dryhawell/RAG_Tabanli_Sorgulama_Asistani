"""OIDC Authorization Code akışı (SSO).

Keşif (discovery) → authorize → code exchange → userinfo/id_token claims.
JWKS imza doğrulaması isteğe bağlı; varsayılan olarak token endpoint'e
güvenilir (confidential client + TLS).
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests

from app.config import (
    DEFAULT_TENANT,
    OIDC_ADMIN_GROUPS,
    OIDC_AUTO_PROVISION,
    OIDC_CLIENT_ID,
    OIDC_CLIENT_SECRET,
    OIDC_DEFAULT_ROLE,
    OIDC_ENABLED,
    OIDC_ISSUER,
    OIDC_REDIRECT_URI,
    OIDC_SCOPES,
    OIDC_TENANT_CLAIM,
    OIDC_USERNAME_CLAIM,
)


class OIDCError(RuntimeError):
    """OIDC yapılandırma veya protokol hatası."""


@dataclass(frozen=True)
class OIDCConfig:
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: str = "openid profile email"
    username_claim: str = "preferred_username"
    tenant_claim: str = "tenant_id"
    admin_groups: Tuple[str, ...] = ()
    default_role: str = "user"
    auto_provision: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.issuer and self.client_id and self.redirect_uri)


def oidc_enabled() -> bool:
    return bool(OIDC_ENABLED)


def load_oidc_config() -> OIDCConfig:
    groups = tuple(
        g.strip()
        for g in (OIDC_ADMIN_GROUPS or "").replace(";", ",").split(",")
        if g.strip()
    )
    return OIDCConfig(
        issuer=(OIDC_ISSUER or "").rstrip("/"),
        client_id=OIDC_CLIENT_ID or "",
        client_secret=OIDC_CLIENT_SECRET or "",
        redirect_uri=OIDC_REDIRECT_URI or "",
        scopes=OIDC_SCOPES or "openid profile email",
        username_claim=OIDC_USERNAME_CLAIM or "preferred_username",
        tenant_claim=OIDC_TENANT_CLAIM or "tenant_id",
        admin_groups=groups,
        default_role=OIDC_DEFAULT_ROLE if OIDC_DEFAULT_ROLE in {"admin", "user"} else "user",
        auto_provision=OIDC_AUTO_PROVISION,
    )


def fetch_discovery(issuer: str, *, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    if not issuer:
        raise OIDCError("OIDC issuer boş")
    url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
    sess = session or requests
    resp = sess.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise OIDCError("Discovery yanıtı geçersiz")
    for key in ("authorization_endpoint", "token_endpoint"):
        if not data.get(key):
            raise OIDCError(f"Discovery eksik alan: {key}")
    return data


def generate_pkce_pair() -> Tuple[str, str]:
    """(code_verifier, code_challenge) — S256."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def build_authorize_url(
    discovery: Dict[str, Any],
    cfg: OIDCConfig,
    *,
    state: str,
    nonce: str,
    code_challenge: Optional[str] = None,
) -> str:
    params = {
        "response_type": "code",
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "scope": cfg.scopes,
        "state": state,
        "nonce": nonce,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{discovery['authorization_endpoint']}?{urlencode(params)}"


def exchange_code(
    discovery: Dict[str, Any],
    cfg: OIDCConfig,
    *,
    code: str,
    code_verifier: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": cfg.redirect_uri,
        "client_id": cfg.client_id,
    }
    if cfg.client_secret:
        data["client_secret"] = cfg.client_secret
    if code_verifier:
        data["code_verifier"] = code_verifier

    sess = session or requests
    resp = sess.post(discovery["token_endpoint"], data=data, timeout=30)
    if resp.status_code >= 400:
        raise OIDCError(f"Token exchange başarısız: {resp.status_code} {resp.text[:200]}")
    payload = resp.json()
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise OIDCError("Token yanıtı geçersiz")
    return payload


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """İmza doğrulamadan JWT payload okur (token endpoint güvenine dayanır)."""
    parts = token.split(".")
    if len(parts) < 2:
        raise OIDCError("Geçersiz JWT")
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    raw = base64.urlsafe_b64decode(payload_b64 + padding)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise OIDCError("JWT payload obje değil")
    return data


def fetch_userinfo(
    discovery: Dict[str, Any],
    access_token: str,
    *,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    endpoint = discovery.get("userinfo_endpoint")
    if not endpoint:
        return {}
    sess = session or requests
    resp = sess.get(
        endpoint,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if resp.status_code >= 400:
        raise OIDCError(f"Userinfo başarısız: {resp.status_code}")
    data = resp.json()
    return data if isinstance(data, dict) else {}


def merge_claims(
    token_payload: Dict[str, Any],
    id_token_claims: Optional[Dict[str, Any]] = None,
    userinfo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for src in (id_token_claims or {}, userinfo or {}, token_payload):
        merged.update(src)
    return merged


def extract_username(claims: Dict[str, Any], cfg: OIDCConfig) -> str:
    for key in (cfg.username_claim, "preferred_username", "email", "sub"):
        val = claims.get(key)
        if val:
            # email ise @ öncesi
            text = str(val).strip()
            if "@" in text and key == "email":
                text = text.split("@", 1)[0]
            return text
    raise OIDCError("Kullanıcı adı claim'i bulunamadı")


def extract_groups(claims: Dict[str, Any]) -> Tuple[str, ...]:
    raw = claims.get("groups") or claims.get("roles") or []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        return tuple(parts)
    if isinstance(raw, list):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    return ()


def resolve_role(claims: Dict[str, Any], cfg: OIDCConfig) -> str:
    groups = {g.lower() for g in extract_groups(claims)}
    admin_set = {g.lower() for g in cfg.admin_groups}
    if admin_set and groups & admin_set:
        return "admin"
    # explicit role claim
    role = str(claims.get("role") or "").strip().lower()
    if role in {"admin", "user"}:
        return role
    return cfg.default_role


def resolve_tenant(claims: Dict[str, Any], cfg: OIDCConfig) -> str:
    val = claims.get(cfg.tenant_claim) or claims.get("tenant") or DEFAULT_TENANT
    return str(val).strip() or DEFAULT_TENANT


def complete_login(
    cfg: OIDCConfig,
    *,
    code: str,
    expected_state: str,
    received_state: str,
    expected_nonce: Optional[str] = None,
    code_verifier: Optional[str] = None,
    discovery: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Code → claims. Dönüş: {username, role, tenant_id, claims, tokens}."""
    if not cfg.configured:
        raise OIDCError("OIDC yapılandırması eksik (issuer/client_id/redirect_uri)")
    if not secrets.compare_digest(expected_state or "", received_state or ""):
        raise OIDCError("OIDC state uyuşmazlığı (CSRF)")

    disc = discovery or fetch_discovery(cfg.issuer, session=session)
    tokens = exchange_code(
        disc, cfg, code=code, code_verifier=code_verifier, session=session
    )

    id_claims: Dict[str, Any] = {}
    if tokens.get("id_token"):
        id_claims = decode_jwt_payload(tokens["id_token"])
        if expected_nonce:
            if id_claims.get("nonce") != expected_nonce:
                raise OIDCError("OIDC nonce uyuşmazlığı")
        # basit expiry kontrolü
        exp = id_claims.get("exp")
        if exp is not None and float(exp) < time.time() - 30:
            raise OIDCError("id_token süresi dolmuş")

    userinfo: Dict[str, Any] = {}
    if disc.get("userinfo_endpoint") and tokens.get("access_token"):
        try:
            userinfo = fetch_userinfo(disc, tokens["access_token"], session=session)
        except OIDCError:
            userinfo = {}

    claims = merge_claims({}, id_claims, userinfo)
    username = extract_username(claims, cfg)
    role = resolve_role(claims, cfg)
    tenant_id = resolve_tenant(claims, cfg)

    return {
        "username": username,
        "role": role,
        "tenant_id": tenant_id,
        "claims": claims,
        "tokens": {
            "access_token": tokens.get("access_token"),
            "id_token": tokens.get("id_token"),
            "token_type": tokens.get("token_type"),
            "expires_in": tokens.get("expires_in"),
        },
    }
