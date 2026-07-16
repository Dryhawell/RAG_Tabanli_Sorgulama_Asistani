"""Basit çok kullanıcılı kimlik doğrulama (paylaşımlı indeks)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import (
    AUTH_BOOTSTRAP_ADMIN,
    AUTH_SHARED_INDEX,
    AUTH_USER_CAN_INGEST,
    CHAT_DIR,
    ENABLE_AUTH,
    USERS_PATH,
)


@dataclass
class User:
    username: str
    role: str = "user"  # admin | user

    @property
    def can_ingest(self) -> bool:
        if self.role == "admin":
            return True
        # Kişisel indekste her kullanıcı kendi dokümanlarını yönetebilir
        if not AUTH_SHARED_INDEX:
            return True
        return AUTH_USER_CAN_INGEST


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


def user_chat_dir(username: str, base: str = CHAT_DIR) -> str:
    safe = _safe_username(username) or "anon"
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
    role = meta.get("role") or "user"
    if role not in {"admin", "user"}:
        role = "user"
    return User(username=uname, role=role)


def add_user(
    username: str,
    password: str,
    *,
    role: str = "user",
    path: str = USERS_PATH,
) -> User:
    users = ensure_users_file(path)
    uname = _safe_username(username)
    if not uname:
        raise ValueError("Geçersiz kullanıcı adı")
    if uname in users:
        raise ValueError("Kullanıcı zaten var")
    if role not in {"admin", "user"}:
        role = "user"
    users[uname] = {"password_hash": _hash_password(password), "role": role}
    save_users(users, path)
    return User(username=uname, role=role)


def auth_enabled() -> bool:
    return bool(ENABLE_AUTH)
