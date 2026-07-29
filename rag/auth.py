"""Basit çok kullanıcılı kimlik doğrulama (paylaşımlı indeks)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import (
    AUTH_BOOTSTRAP_ADMIN,
    AUTH_SHARED_INDEX,
    AUTH_USER_CAN_INGEST,
    CHAT_DIR,
    DEFAULT_TENANT,
    ENABLE_AUTH,
    USERS_PATH,
)


@dataclass
class User:
    username: str
    role: str = "user"  # admin | user
    tenant_id: str = "default"
    # None veya ["*"] => tümü; liste => kısıtlı ACL
    allowed_folders: Optional[List[str]] = None
    allowed_tags: Optional[List[str]] = None

    @property
    def can_ingest(self) -> bool:
        if self.role == "admin":
            return True
        # Kişisel indekste her kullanıcı kendi dokümanlarını yönetebilir
        if not AUTH_SHARED_INDEX:
            return True
        return AUTH_USER_CAN_INGEST


def _parse_acl_list(raw) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        return parts or None
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        return parts or None
    return None


def _safe_tenant(tenant_id: Optional[str]) -> str:
    t = (tenant_id or DEFAULT_TENANT or "default").strip().lower()
    t = re.sub(r"[^a-z0-9._-]+", "", t)
    return t or "default"


def user_from_meta(username: str, meta: dict) -> User:
    role = meta.get("role") or "user"
    if role not in {"admin", "user"}:
        role = "user"
    return User(
        username=username,
        role=role,
        tenant_id=_safe_tenant(meta.get("tenant_id") or DEFAULT_TENANT),
        allowed_folders=_parse_acl_list(meta.get("allowed_folders")),
        allowed_tags=_parse_acl_list(meta.get("allowed_tags")),
    )


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    check = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return secrets.compare_digest(check, digest)


def _safe_username(username: str) -> str:
    u = (username or "").strip().lower()
    u = re.sub(r"[^a-z0-9._-]+", "", u)
    return u


def user_chat_dir(
    username: str,
    base: str = CHAT_DIR,
    *,
    tenant_id: Optional[str] = None,
) -> str:
    safe = _safe_username(username) or "anon"
    if tenant_id:
        path = os.path.join(base, _safe_tenant(tenant_id), safe)
    else:
        path = os.path.join(base, safe)
    os.makedirs(path, exist_ok=True)
    return path


def load_users(path: str = USERS_PATH) -> Dict[str, dict]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_users(users: Dict[str, dict], path: str = USERS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def ensure_users_file(path: str = USERS_PATH) -> Dict[str, dict]:
    """users.json yoksa bootstrap admin ile oluşturur; plaintext password alanlarını hash'ler."""
    users = load_users(path)
    changed = False

    if not users and AUTH_BOOTSTRAP_ADMIN and ":" in AUTH_BOOTSTRAP_ADMIN:
        raw_user, raw_pass = AUTH_BOOTSTRAP_ADMIN.split(":", 1)
        uname = _safe_username(raw_user) or "admin"
        users[uname] = {
            "password_hash": _hash_password(raw_pass),
            "role": "admin",
            "tenant_id": DEFAULT_TENANT,
        }
        changed = True

    # plaintext "password" alanını bir kez hash'le
    for uname, meta in list(users.items()):
        if not isinstance(meta, dict):
            continue
        if meta.get("password") and not meta.get("password_hash"):
            meta["password_hash"] = _hash_password(str(meta["password"]))
            del meta["password"]
            changed = True
        meta.setdefault("role", "user")
        if "tenant_id" not in meta:
            meta["tenant_id"] = DEFAULT_TENANT
            changed = True

    if changed:
        save_users(users, path)
    return users


def list_usernames(path: str = USERS_PATH) -> List[str]:
    return sorted(ensure_users_file(path).keys())


def authenticate(
    username: str,
    password: str,
    path: str = USERS_PATH,
) -> Optional[User]:
    users = ensure_users_file(path)
    uname = _safe_username(username)
    meta = users.get(uname)
    if not meta:
        return None
    if not verify_password(password, meta.get("password_hash", "")):
        return None
    return user_from_meta(uname, meta)


def add_user(
    username: str,
    password: str,
    *,
    role: str = "user",
    path: str = USERS_PATH,
    tenant_id: Optional[str] = None,
    allowed_folders: Optional[List[str]] = None,
    allowed_tags: Optional[List[str]] = None,
) -> User:
    users = ensure_users_file(path)
    uname = _safe_username(username)
    if not uname:
        raise ValueError("Geçersiz kullanıcı adı")
    if uname in users:
        raise ValueError("Kullanıcı zaten var")
    if role not in {"admin", "user"}:
        role = "user"
    entry = {
        "password_hash": _hash_password(password),
        "role": role,
        "tenant_id": _safe_tenant(tenant_id or DEFAULT_TENANT),
    }
    folders = _parse_acl_list(allowed_folders)
    tags = _parse_acl_list(allowed_tags)
    if folders is not None:
        entry["allowed_folders"] = folders
    if tags is not None:
        entry["allowed_tags"] = tags
    users[uname] = entry
    save_users(users, path)
    return user_from_meta(uname, entry)


def update_user_acl(
    username: str,
    *,
    allowed_folders: Optional[List[str]] = None,
    allowed_tags: Optional[List[str]] = None,
    path: str = USERS_PATH,
    clear_folders: bool = False,
    clear_tags: bool = False,
) -> User:
    """Kullanıcı klasör/etiket ACL'sini günceller.

    clear_*=True => kısıtı kaldır (tümüne izin).
    """
    users = ensure_users_file(path)
    uname = _safe_username(username)
    if uname not in users:
        raise ValueError("Kullanıcı bulunamadı")
    meta = users[uname]
    if not isinstance(meta, dict):
        raise ValueError("Geçersiz kullanıcı kaydı")

    if clear_folders:
        meta.pop("allowed_folders", None)
    elif allowed_folders is not None:
        parsed = _parse_acl_list(allowed_folders)
        if parsed is None:
            meta.pop("allowed_folders", None)
        else:
            meta["allowed_folders"] = parsed

    if clear_tags:
        meta.pop("allowed_tags", None)
    elif allowed_tags is not None:
        parsed = _parse_acl_list(allowed_tags)
        if parsed is None:
            meta.pop("allowed_tags", None)
        else:
            meta["allowed_tags"] = parsed

    users[uname] = meta
    save_users(users, path)
    return user_from_meta(uname, meta)


def get_user_timezone(username: str, path: Optional[str] = None) -> Optional[str]:
    """Kullanıcı profilindeki IANA timezone (yoksa None)."""
    users = ensure_users_file(path or USERS_PATH)
    uname = _safe_username(username)
    meta = users.get(uname)
    if not isinstance(meta, dict):
        return None
    tz = str(meta.get("timezone") or "").strip()
    return tz or None


def update_user_timezone(
    username: str,
    timezone_name: Optional[str],
    *,
    path: Optional[str] = None,
) -> User:
    """Kullanıcı quiet-hours / digest timezone tercihini kaydeder."""
    users_path = path or USERS_PATH
    users = ensure_users_file(users_path)
    uname = _safe_username(username)
    if uname not in users:
        raise ValueError("Kullanıcı bulunamadı")
    meta = users[uname]
    if not isinstance(meta, dict):
        raise ValueError("Geçersiz kullanıcı kaydı")
    tz = (timezone_name or "").strip()
    if tz:
        meta["timezone"] = tz
    else:
        meta.pop("timezone", None)
    users[uname] = meta
    save_users(users, users_path)
    return user_from_meta(uname, meta)


def delete_user(
    username: str,
    *,
    path: str = USERS_PATH,
    protect_last_admin: bool = True,
) -> str:
    """Kullanıcıyı siler; son admin korunur. Dönüş: silinen kullanıcı adı."""
    users = ensure_users_file(path)
    uname = _safe_username(username)
    if uname not in users:
        raise ValueError("Kullanıcı bulunamadı")
    role = users[uname].get("role") or "user"
    if protect_last_admin and role == "admin":
        admins = [u for u, m in users.items() if (m.get("role") or "user") == "admin"]
        if len(admins) <= 1:
            raise ValueError("Son admin kullanıcısı silinemez")
    del users[uname]
    save_users(users, path)
    return uname


def list_users_detail(path: str = USERS_PATH) -> List[dict]:
    users = ensure_users_file(path)
    out = []
    for uname in sorted(users.keys()):
        meta = users[uname] if isinstance(users[uname], dict) else {}
        out.append(
            {
                "username": uname,
                "role": meta.get("role") or "user",
                "tenant_id": meta.get("tenant_id") or DEFAULT_TENANT,
                "allowed_folders": meta.get("allowed_folders"),
                "allowed_tags": meta.get("allowed_tags"),
                "auth_provider": meta.get("auth_provider") or "local",
            }
        )
    return out


def upsert_oidc_user(
    username: str,
    *,
    role: str = "user",
    tenant_id: Optional[str] = None,
    path: str = USERS_PATH,
    auto_provision: bool = True,
    claims: Optional[dict] = None,
) -> User:
    """OIDC claims ile kullanıcı oluşturur veya günceller.

    auto_provision=False ve kullanıcı yoksa ValueError.
    Mevcut kullanıcının ACL alanları korunur; role/tenant OIDC'den güncellenir.
    """
    users = ensure_users_file(path)
    uname = _safe_username(username)
    if not uname:
        raise ValueError("Geçersiz kullanıcı adı")
    if role not in {"admin", "user"}:
        role = "user"
    tid = _safe_tenant(tenant_id or DEFAULT_TENANT)

    if uname not in users:
        if not auto_provision:
            raise ValueError(
                f"OIDC kullanıcısı yerelde yok ve otomatik oluşturma kapalı: {uname}"
            )
        users[uname] = {
            # OIDC kullanıcıları parola ile giriş yapmaz; rastgele hash
            "password_hash": _hash_password(secrets.token_urlsafe(24)),
            "role": role,
            "tenant_id": tid,
            "auth_provider": "oidc",
        }
        if claims and claims.get("email"):
            users[uname]["email"] = str(claims["email"])
    else:
        meta = users[uname]
        if not isinstance(meta, dict):
            raise ValueError("Geçersiz kullanıcı kaydı")
        meta["role"] = role
        meta["tenant_id"] = tid
        meta["auth_provider"] = "oidc"
        if claims and claims.get("email"):
            meta["email"] = str(claims["email"])
        users[uname] = meta

    save_users(users, path)
    return user_from_meta(uname, users[uname])


def auth_enabled() -> bool:
    return bool(ENABLE_AUTH)
