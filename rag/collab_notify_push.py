"""Mobil push bildirim PoC (FCM HTTP / generic webhook) + token TTL."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from app.config import (
        METADATA_DIR,
        NOTIFY_PUSH_API_KEY,
        NOTIFY_PUSH_APNS_KEY_ID,
        NOTIFY_PUSH_APNS_P8_CONTENT,
        NOTIFY_PUSH_APNS_P8_PATH,
        NOTIFY_PUSH_APNS_TEAM_ID,
        NOTIFY_PUSH_APNS_TOPIC,
        NOTIFY_PUSH_APNS_USE_SANDBOX,
        NOTIFY_PUSH_FCM_PROJECT_ID,
        NOTIFY_PUSH_FCM_SERVICE_ACCOUNT_JSON,
        NOTIFY_PUSH_PROVIDER,
        NOTIFY_PUSH_TOKEN_TTL_DAYS,
        NOTIFY_PUSH_URL,
    )
except ImportError:
    METADATA_DIR = "metadata"
    NOTIFY_PUSH_URL = ""
    NOTIFY_PUSH_API_KEY = ""
    NOTIFY_PUSH_PROVIDER = "generic"
    NOTIFY_PUSH_TOKEN_TTL_DAYS = 90
    NOTIFY_PUSH_FCM_PROJECT_ID = ""
    NOTIFY_PUSH_FCM_SERVICE_ACCOUNT_JSON = ""
    NOTIFY_PUSH_APNS_KEY_ID = ""
    NOTIFY_PUSH_APNS_TEAM_ID = ""
    NOTIFY_PUSH_APNS_TOPIC = ""
    NOTIFY_PUSH_APNS_P8_PATH = ""
    NOTIFY_PUSH_APNS_P8_CONTENT = ""
    NOTIFY_PUSH_APNS_USE_SANDBOX = False

_FCM_TOKEN_CACHE: Dict[str, Any] = {"token": None, "expires_at": 0.0}
_APNS_TOKEN_CACHE: Dict[str, Any] = {"token": None, "expires_at": 0.0}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def device_tokens_path(base: Optional[str] = None) -> str:
    root = base or METADATA_DIR
    return os.path.join(root, "collab", "push_tokens.jsonl")


def _normalize_quiet_hours_field(quiet_hours: Optional[str]) -> Optional[str]:
    if quiet_hours is None:
        return None
    qh = str(quiet_hours).strip()
    if not qh:
        return "off"
    return qh


def _geofence_dict(
    lat: Optional[float],
    lon: Optional[float],
    radius_m: Optional[float],
) -> Optional[Dict[str, float]]:
    if lat is None or lon is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "radius_m": float(radius_m if radius_m is not None else 500.0),
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki WGS84 nokta arası metre cinsinden mesafe."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def device_inside_geofence(row: Dict[str, Any]) -> Optional[bool]:
    """Geofence + last_location varsa içeride mi; yapılandırma yoksa None."""
    geo = row.get("geofence")
    loc = row.get("last_location")
    if not isinstance(geo, dict) or not isinstance(loc, dict):
        return None
    try:
        glat = float(geo["lat"])
        glon = float(geo["lon"])
        radius = float(geo.get("radius_m") or 500.0)
        llat = float(loc["lat"])
        llon = float(loc["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    return haversine_m(glat, glon, llat, llon) <= max(1.0, radius)


def resolve_device_quiet_spec(
    row: Dict[str, Any],
    username: str,
) -> Tuple[Optional[str], str]:
    """(quiet_spec, source). source: device|device_off|user_global."""
    if "quiet_hours" in row and row.get("quiet_hours") is not None:
        qh = str(row.get("quiet_hours") or "").strip()
        if not qh or qh.lower() in {"off", "none", "disabled"}:
            return None, "device_off"
        return qh, "device"
    from rag.collab_notify_digest import resolve_quiet_hours_spec

    return resolve_quiet_hours_spec(username=username), "user_global"


def is_device_in_quiet_hours(
    row: Dict[str, Any],
    username: str,
    *,
    now: Optional[datetime] = None,
    timezone_name: Optional[str] = None,
) -> bool:
    """Cihaz sessiz saatte mi?

    Geofence dışı (seyahat): quiet hours uygulanmaz → False.
    Cihaz quiet_hours=off: False.
    Cihaz HH:MM-HH:MM: o aralık.
    Aksi halde kullanıcı/global quiet hours.
    """
    inside = device_inside_geofence(row)
    if inside is False:
        return False
    spec, _src = resolve_device_quiet_spec(row, username)
    from rag.collab_notify_digest import is_quiet_hours

    if spec is None:
        # açık kapalı (device_off) veya profil/global yok
        if "quiet_hours" in row and row.get("quiet_hours") is not None:
            return False
        return is_quiet_hours(
            now=now,
            quiet_spec=None,
            timezone_name=timezone_name,
            username=username,
        )
    return is_quiet_hours(
        now=now,
        quiet_spec=spec,
        timezone_name=timezone_name,
        username=username,
    )


def filter_deliverable_tokens(
    username: str,
    tokens: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    ignore_quiet_hours: bool = False,
) -> List[Dict[str, Any]]:
    if ignore_quiet_hours:
        return list(tokens)
    return [
        row
        for row in tokens
        if not is_device_in_quiet_hours(row, username, now=now)
    ]


def register_device_token(
    username: str,
    token: str,
    *,
    platform: str = "fcm",
    label: Optional[str] = None,
    device_name: Optional[str] = None,
    os_name: Optional[str] = None,
    os_version: Optional[str] = None,
    app_version: Optional[str] = None,
    quiet_hours: Optional[str] = None,
    geofence_lat: Optional[float] = None,
    geofence_lon: Optional[float] = None,
    geofence_radius_m: Optional[float] = None,
    last_lat: Optional[float] = None,
    last_lon: Optional[float] = None,
    ttl_days: Optional[int] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Kullanıcı cihaz token'ını kaydeder (TTL + cihaz meta + quiet/geofence)."""
    user = (username or "").strip()
    tok = (token or "").strip()
    if not user or not tok:
        raise ValueError("username ve token gerekli")
    days = int(ttl_days if ttl_days is not None else NOTIFY_PUSH_TOKEN_TTL_DAYS)
    now = _utcnow()
    expires = now + timedelta(days=max(1, days))
    record: Dict[str, Any] = {
        "username": user,
        "token": tok,
        "platform": (platform or "fcm").strip().lower(),
        "label": (label or "").strip() or None,
        "device_name": (device_name or "").strip() or None,
        "os_name": (os_name or "").strip() or None,
        "os_version": (os_version or "").strip() or None,
        "app_version": (app_version or "").strip() or None,
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "last_seen_at": now.isoformat(),
        "revoked": False,
    }
    if quiet_hours is not None:
        record["quiet_hours"] = _normalize_quiet_hours_field(quiet_hours)
    geo = _geofence_dict(geofence_lat, geofence_lon, geofence_radius_m)
    if geo is not None:
        record["geofence"] = geo
    if last_lat is not None and last_lon is not None:
        record["last_location"] = {
            "lat": float(last_lat),
            "lon": float(last_lon),
            "updated_at": now.isoformat(),
        }
    path = device_tokens_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = _read_all_tokens(base=base)
    updated = False
    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user.lower()
            and str(row.get("token") or "") == tok
        ):
            # created_at koru; quiet/geofence verilmezse eski değer kalsın
            created = row.get("created_at") or record["created_at"]
            preserved_qh = row.get("quiet_hours") if "quiet_hours" not in record else None
            preserved_geo = row.get("geofence") if "geofence" not in record else None
            preserved_loc = row.get("last_location") if "last_location" not in record else None
            row.update(record)
            row["created_at"] = created
            if preserved_qh is not None and "quiet_hours" not in record:
                row["quiet_hours"] = preserved_qh
            if preserved_geo is not None and "geofence" not in record:
                row["geofence"] = preserved_geo
            if preserved_loc is not None and "last_location" not in record:
                row["last_location"] = preserved_loc
            updated = True
            record = row
        new_rows.append(row)
    if not updated:
        new_rows.append(record)
    _write_all_tokens(new_rows, base=base)
    return record


def update_device_location(
    username: str,
    token: str,
    *,
    lat: float,
    lon: float,
    base: Optional[str] = None,
) -> bool:
    """Cihaz son konumunu günceller (geofence değerlendirmesi için)."""
    user_key = (username or "").strip().lower()
    tok = (token or "").strip()
    rows = _read_all_tokens(base=base)
    changed = False
    now = _utcnow_iso()
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user_key
            and str(row.get("token") or "") == tok
            and not row.get("revoked")
        ):
            row["last_location"] = {
                "lat": float(lat),
                "lon": float(lon),
                "updated_at": now,
            }
            row["last_seen_at"] = now
            changed = True
    if changed:
        _write_all_tokens(rows, base=base)
    return changed


def update_device_quiet_hours(
    username: str,
    token: str,
    quiet_hours: Optional[str],
    *,
    base: Optional[str] = None,
) -> bool:
    """Cihaz bazlı quiet hours override (off = kapat)."""
    user_key = (username or "").strip().lower()
    tok = (token or "").strip()
    rows = _read_all_tokens(base=base)
    changed = False
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user_key
            and str(row.get("token") or "") == tok
        ):
            if quiet_hours is None:
                row.pop("quiet_hours", None)
            else:
                row["quiet_hours"] = _normalize_quiet_hours_field(quiet_hours)
            changed = True
    if changed:
        _write_all_tokens(rows, base=base)
    return changed


def touch_device_last_seen(
    username: str,
    token: str,
    *,
    base: Optional[str] = None,
) -> bool:
    """Başarılı push sonrası last_seen_at günceller."""
    user_key = (username or "").strip().lower()
    tok = (token or "").strip()
    rows = _read_all_tokens(base=base)
    changed = False
    now = _utcnow_iso()
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user_key
            and str(row.get("token") or "") == tok
            and not row.get("revoked")
        ):
            row["last_seen_at"] = now
            changed = True
    if changed:
        _write_all_tokens(rows, base=base)
    return changed


def _read_all_tokens(base: Optional[str] = None) -> List[Dict[str, Any]]:
    path = device_tokens_path(base=base)
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_all_tokens(rows: List[Dict[str, Any]], base: Optional[str] = None) -> None:
    path = device_tokens_path(base=base)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_token_expired(row: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    if row.get("revoked"):
        return True
    exp = _parse_iso(str(row.get("expires_at") or ""))
    if exp is None:
        return False
    current = now or _utcnow()
    return exp <= current


def list_device_tokens(
    username: str,
    *,
    include_expired: bool = False,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_key = (username or "").strip().lower()
    rows = [
        row
        for row in _read_all_tokens(base=base)
        if str(row.get("username") or "").strip().lower() == user_key
    ]
    if include_expired:
        return rows
    return [row for row in rows if not is_token_expired(row)]


def revoke_device_token(
    username: str,
    token: str,
    *,
    base: Optional[str] = None,
) -> bool:
    user_key = (username or "").strip().lower()
    tok = (token or "").strip()
    rows = _read_all_tokens(base=base)
    changed = False
    for row in rows:
        if (
            str(row.get("username") or "").strip().lower() == user_key
            and str(row.get("token") or "") == tok
        ):
            row["revoked"] = True
            row["revoked_at"] = _utcnow_iso()
            changed = True
    if changed:
        _write_all_tokens(rows, base=base)
    return changed


def prune_expired_tokens(
    *,
    username: Optional[str] = None,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    """Süresi dolmuş / revoked token'ları dosyadan temizler."""
    user_key = (username or "").strip().lower() if username else None
    rows = _read_all_tokens(base=base)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for row in rows:
        if user_key and str(row.get("username") or "").strip().lower() != user_key:
            kept.append(row)
            continue
        if is_token_expired(row):
            removed += 1
            continue
        kept.append(row)
    if removed:
        _write_all_tokens(kept, base=base)
    return {"removed": removed, "remaining": len(kept)}


def summarize_user_devices(
    username: str,
    *,
    base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """UI için cihaz özeti (token kısaltılmış)."""
    out: List[Dict[str, Any]] = []
    for row in list_device_tokens(username, include_expired=True, base=base):
        tok = str(row.get("token") or "")
        preview = tok[:6] + "…" + tok[-4:] if len(tok) > 12 else tok
        out.append(
            {
                "token": tok,
                "token_preview": preview,
                "platform": row.get("platform") or "generic",
                "label": row.get("label"),
                "device_name": row.get("device_name"),
                "os_name": row.get("os_name"),
                "os_version": row.get("os_version"),
                "app_version": row.get("app_version"),
                "quiet_hours": row.get("quiet_hours"),
                "geofence": row.get("geofence"),
                "last_location": row.get("last_location"),
                "created_at": row.get("created_at"),
                "expires_at": row.get("expires_at"),
                "last_seen_at": row.get("last_seen_at"),
                "expired": is_token_expired(row),
                "revoked": bool(row.get("revoked")),
            }
        )
    return out


def build_fcm_payload(
    token: str,
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """FCM HTTP v1 tarzı message gövdesi (PoC)."""
    return {
        "message": {
            "token": token,
            "notification": {"title": title, "body": body[:200]},
            "data": {k: str(v) for k, v in (data or {}).items()},
        }
    }


_FCM_INVALID_TOKEN_CODES = frozenset(
    {
        "UNREGISTERED",
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "SENDER_ID_MISMATCH",
    }
)
_APNS_INVALID_REASONS = frozenset(
    {
        "BadDeviceToken",
        "Unregistered",
        "DeviceTokenNotForTopic",
        "ExpiredProviderToken",
    }
)


def parse_fcm_error_code(response: Any) -> Optional[str]:
    """FCM HTTP v1 hata gövdesinden errorCode / status çıkarır."""
    try:
        status = int(getattr(response, "status_code", 0) or 0)
    except Exception:
        status = 0
    body: Any = None
    try:
        body = response.json()
    except Exception:
        text = getattr(response, "text", None) or getattr(response, "content", b"")
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        if text:
            try:
                body = json.loads(text)
            except Exception:
                body = None
    if not isinstance(body, dict):
        if status in {400, 404}:
            return "NOT_FOUND" if status == 404 else "INVALID_ARGUMENT"
        return None
    err = body.get("error") if isinstance(body.get("error"), dict) else {}
    details = err.get("details") if isinstance(err, dict) else None
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, dict):
                continue
            code = str(item.get("errorCode") or "").strip().upper()
            if code:
                return code
    status_name = str((err or {}).get("status") or "").strip().upper()
    if status_name:
        return status_name
    if status in {400, 404}:
        return "NOT_FOUND" if status == 404 else "INVALID_ARGUMENT"
    return None


def is_fcm_invalid_token_response(response: Any) -> bool:
    """Kalıcı geçersiz token hatası mı (revoke adayı)."""
    try:
        status = int(getattr(response, "status_code", 0) or 0)
    except Exception:
        return False
    if status < 400:
        return False
    if status >= 500:
        return False
    code = parse_fcm_error_code(response)
    if not code:
        return False
    return code.upper() in _FCM_INVALID_TOKEN_CODES


def parse_apns_error_reason(response: Any) -> Optional[str]:
    try:
        body = response.json()
    except Exception:
        return None
    if isinstance(body, dict):
        reason = str(body.get("reason") or "").strip()
        return reason or None
    return None


def is_apns_invalid_token_response(response: Any) -> bool:
    try:
        status = int(getattr(response, "status_code", 0) or 0)
    except Exception:
        return False
    if status not in {400, 410}:
        return False
    reason = parse_apns_error_reason(response) or ""
    if reason in _APNS_INVALID_REASONS:
        return True
    return status == 410


def load_fcm_service_account_info(
    source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Service account JSON (dosya yolu veya inline JSON)."""
    raw = (source if source is not None else NOTIFY_PUSH_FCM_SERVICE_ACCOUNT_JSON) or ""
    raw = raw.strip()
    if not raw:
        return None
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def get_fcm_access_token(
    *,
    service_account_json: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[str]:
    """FCM v1 OAuth access token (google-auth service account)."""
    import time

    now = time.time()
    cached = _FCM_TOKEN_CACHE.get("token")
    exp = float(_FCM_TOKEN_CACHE.get("expires_at") or 0.0)
    if cached and not force_refresh and now < exp - 60:
        return str(cached)
    info = load_fcm_service_account_info(service_account_json)
    if not info:
        return None
    try:
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/firebase.messaging"],
        )
        creds.refresh(__import__("google.auth.transport.requests", fromlist=["Request"]).Request())
        token = creds.token
        expiry = getattr(creds, "expiry", None)
        expires_at = expiry.timestamp() if expiry is not None else now + 3500
        _FCM_TOKEN_CACHE["token"] = token
        _FCM_TOKEN_CACHE["expires_at"] = expires_at
        return str(token) if token else None
    except Exception:
        return None


def resolve_fcm_endpoint(
    *,
    project_id: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """FCM v1 messages:send URL; açık URL > project_id varsayılanı > NOTIFY_PUSH_URL."""
    explicit = (url if url is not None else NOTIFY_PUSH_URL) or ""
    explicit = explicit.strip()
    if explicit:
        return explicit
    pid = (project_id if project_id is not None else NOTIFY_PUSH_FCM_PROJECT_ID) or ""
    pid = pid.strip()
    if pid:
        return f"https://fcm.googleapis.com/v1/projects/{pid}/messages:send"
    return ""


def resolve_fcm_auth_header(
    *,
    api_key: Optional[str] = None,
    service_account_json: Optional[str] = None,
) -> Optional[str]:
    """Bearer token: OAuth service account > statik API key."""
    oauth = get_fcm_access_token(service_account_json=service_account_json)
    if oauth:
        return f"Bearer {oauth}"
    key = (api_key if api_key is not None else NOTIFY_PUSH_API_KEY) or ""
    key = key.strip()
    if key:
        if key.lower().startswith("bearer "):
            return key
        return f"Bearer {key}"
    return None


def is_push_configured() -> bool:
    """Push kanalı yapılandırılmış mı (URL veya FCM/APNs native)."""
    provider = (NOTIFY_PUSH_PROVIDER or "generic").lower()
    if provider == "fcm":
        return bool(resolve_fcm_endpoint())
    if provider == "apns":
        return apns_native_configured() or bool((NOTIFY_PUSH_URL or "").strip())
    return bool((NOTIFY_PUSH_URL or "").strip())


def load_apns_p8(
    *,
    path: Optional[str] = None,
    content: Optional[str] = None,
) -> Optional[str]:
    """APNs .p8 PEM içeriğini yükler."""
    raw = (content if content is not None else NOTIFY_PUSH_APNS_P8_CONTENT) or ""
    raw = raw.strip()
    if raw and "BEGIN PRIVATE KEY" in raw:
        return raw.replace("\\n", "\n")
    p8_path = (path if path is not None else NOTIFY_PUSH_APNS_P8_PATH) or ""
    p8_path = p8_path.strip()
    if p8_path and os.path.isfile(p8_path):
        with open(p8_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def apns_native_configured() -> bool:
    return bool(
        (NOTIFY_PUSH_APNS_KEY_ID or "").strip()
        and (NOTIFY_PUSH_APNS_TEAM_ID or "").strip()
        and (NOTIFY_PUSH_APNS_TOPIC or "").strip()
        and load_apns_p8()
    )


def get_apns_provider_token(
    *,
    force_refresh: bool = False,
    key_id: Optional[str] = None,
    team_id: Optional[str] = None,
    p8: Optional[str] = None,
) -> Optional[str]:
    """APNs provider authentication token (ES256 JWT, ~1h)."""
    import time

    now = time.time()
    cached = _APNS_TOKEN_CACHE.get("token")
    exp = float(_APNS_TOKEN_CACHE.get("expires_at") or 0.0)
    if cached and not force_refresh and now < exp - 60:
        return str(cached)
    kid = (key_id if key_id is not None else NOTIFY_PUSH_APNS_KEY_ID) or ""
    tid = (team_id if team_id is not None else NOTIFY_PUSH_APNS_TEAM_ID) or ""
    pem = p8 if p8 is not None else load_apns_p8()
    if not (kid.strip() and tid.strip() and pem):
        return None
    try:
        import jwt
    except ImportError:
        return None
    try:
        token = jwt.encode(
            {"iss": tid.strip(), "iat": int(now)},
            pem,
            algorithm="ES256",
            headers={"alg": "ES256", "kid": kid.strip()},
        )
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        _APNS_TOKEN_CACHE["token"] = token
        _APNS_TOKEN_CACHE["expires_at"] = now + 3500
        return str(token)
    except Exception:
        return None


def resolve_apns_host(*, use_sandbox: Optional[bool] = None) -> str:
    sandbox = (
        NOTIFY_PUSH_APNS_USE_SANDBOX if use_sandbox is None else bool(use_sandbox)
    )
    if sandbox:
        return "https://api.sandbox.push.apple.com"
    return "https://api.push.apple.com"


def resolve_apns_endpoint(
    device_token: str,
    *,
    use_sandbox: Optional[bool] = None,
) -> str:
    """Native APNs URL; .p8 yoksa boş (gateway path kullanılsın)."""
    if not apns_native_configured():
        return ""
    tok = (device_token or "").strip()
    if not tok:
        return ""
    return f"{resolve_apns_host(use_sandbox=use_sandbox)}/3/device/{tok}"


def build_apns_http2_body(
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apple HTTP/2 APNs JSON gövdesi (yalnızca aps + custom data)."""
    payload: Dict[str, Any] = {
        "aps": {
            "alert": {"title": title, "body": body[:200]},
            "sound": "default",
        }
    }
    for k, v in (data or {}).items():
        if k == "aps":
            continue
        payload[str(k)] = v
    return payload


def build_apns_payload(
    token: str,
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """APNs gateway PoC payload (HTTP gateway üzerinden)."""
    return {
        "device_token": token,
        "aps": {
            "alert": {"title": title, "body": body[:200]},
            "sound": "default",
        },
        "data": data or {},
    }


def _dispatch_apns_native(
    tokens: List[Dict[str, Any]],
    *,
    username: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]],
    base: Optional[str],
) -> bool:
    jwt_token = get_apns_provider_token()
    topic = (NOTIFY_PUSH_APNS_TOPIC or "").strip()
    if not jwt_token or not topic:
        return False
    try:
        import httpx
    except ImportError:
        return False
    ok_any = False
    headers_base = {
        "authorization": f"bearer {jwt_token}",
        "apns-topic": topic,
        "apns-push-type": "alert",
        "content-type": "application/json",
    }
    payload = build_apns_http2_body(title=title, body=body, data=data)
    try:
        with httpx.Client(http2=True, timeout=15.0) as client:
            for tok in tokens:
                t = str(tok.get("token") or "").strip()
                if not t:
                    continue
                url = resolve_apns_endpoint(t)
                if not url:
                    continue
                r = client.post(url, json=payload, headers=headers_base)
                if r.status_code < 400:
                    ok_any = True
                    touch_device_last_seen(username, t, base=base)
                elif is_apns_invalid_token_response(r):
                    revoke_device_token(username, t, base=base)
    except Exception:
        return False
    return ok_any


def build_generic_push_payload(
    username: str,
    tokens: List[Dict[str, Any]],
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "type": "collab_push",
        "provider": NOTIFY_PUSH_PROVIDER,
        "username": username,
        "title": title,
        "body": body,
        "tokens": tokens,
        "data": data or {},
    }


def dispatch_push(
    username: str,
    *,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None,
    base: Optional[str] = None,
    ignore_quiet_hours: bool = False,
) -> bool:
    """Kayıtlı (süresi dolmamış) cihaz token'larına push gönderir."""
    provider = (NOTIFY_PUSH_PROVIDER or "generic").lower()
    if provider == "fcm":
        url = resolve_fcm_endpoint()
    elif provider == "apns" and apns_native_configured():
        url = "__apns_native__"
    else:
        url = (NOTIFY_PUSH_URL or "").strip()
    if not url:
        return False
    tokens = list_device_tokens(username, include_expired=False, base=base)
    if not tokens:
        tokens = [{"username": username, "token": "", "platform": "generic"}]
    else:
        tokens = filter_deliverable_tokens(
            username,
            tokens,
            ignore_quiet_hours=ignore_quiet_hours,
        )
        if not tokens:
            return False
    try:
        if provider == "apns" and url == "__apns_native__":
            return _dispatch_apns_native(
                tokens,
                username=username,
                title=title,
                body=body,
                data=data,
                base=base,
            )

        import requests

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if provider == "fcm":
            auth = resolve_fcm_auth_header()
            if auth:
                headers["Authorization"] = auth
        elif NOTIFY_PUSH_API_KEY:
            headers["Authorization"] = f"key={NOTIFY_PUSH_API_KEY}"

        ok_any = False
        if provider == "fcm":
            for tok in tokens:
                t = str(tok.get("token") or "").strip()
                if not t:
                    continue
                payload = build_fcm_payload(t, title=title, body=body, data=data)
                r = requests.post(url, json=payload, headers=headers, timeout=10)
                if r.status_code < 400:
                    ok_any = True
                    touch_device_last_seen(username, t, base=base)
                elif is_fcm_invalid_token_response(r):
                    revoke_device_token(username, t, base=base)
        elif provider == "apns":
            for tok in tokens:
                t = str(tok.get("token") or "").strip()
                if not t:
                    continue
                payload = build_apns_payload(t, title=title, body=body, data=data)
                r = requests.post(url, json=payload, headers=headers, timeout=10)
                if r.status_code < 400:
                    ok_any = True
                    touch_device_last_seen(username, t, base=base)
                elif is_apns_invalid_token_response(r):
                    revoke_device_token(username, t, base=base)
        else:
            payload = build_generic_push_payload(
                username,
                tokens,
                title=title,
                body=body,
                data=data,
            )
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            if r.status_code < 400:
                ok_any = True
                for tok in tokens:
                    t = str(tok.get("token") or "").strip()
                    if t:
                        touch_device_last_seen(username, t, base=base)
        return ok_any
    except Exception:
        return False


def dispatch_digest_push(
    username: str,
    events: List[Dict[str, Any]],
    *,
    mentions_only: Optional[bool] = None,
    group_by: Optional[str] = None,
    min_per_workspace: Optional[int] = None,
    base: Optional[str] = None,
) -> bool:
    from rag.collab_notify_digest import prepare_digest_events

    prep = prepare_digest_events(
        events,
        mentions_only=mentions_only,
        group_by=group_by,
        min_per_workspace=min_per_workspace,
    )
    count = prep["total"]
    if count <= 0:
        return False
    title = f"Bildirim özeti ({count})"
    body = f"{username}: {count} okunmamış collab bildirimi"
    return dispatch_push(
        username,
        title=title,
        body=body,
        data={"type": "collab_digest", "count": count},
        base=base,
    )
