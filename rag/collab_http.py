"""Collab HTTP: aynı-origin Web Push SW + abonelik kaydı (PoC)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

_http_started = False
_http_lock = threading.Lock()
_http_server: Optional[ThreadingHTTPServer] = None


def collab_http_public_base(
    *,
    public_host: Optional[str] = None,
    port: Optional[int] = None,
) -> str:
    try:
        from app.config import COLLAB_HTTP_PORT, COLLAB_WS_PUBLIC_HOST
    except ImportError:
        COLLAB_HTTP_PORT = 8766
        COLLAB_WS_PUBLIC_HOST = "localhost"
    host = (public_host or COLLAB_WS_PUBLIC_HOST or "localhost").strip() or "localhost"
    p = int(port if port is not None else COLLAB_HTTP_PORT)
    return f"http://{host}:{p}"


def handle_sw_js() -> Tuple[int, Dict[str, str], bytes]:
    from rag.collab_notify_push import load_service_worker_js

    body = (load_service_worker_js() or "").encode("utf-8")
    if not body:
        return 404, {"Content-Type": "text/plain; charset=utf-8"}, b"sw.js not found\n"
    headers = {
        "Content-Type": "application/javascript; charset=utf-8",
        "Service-Worker-Allowed": "/",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
    }
    return 200, headers, body


def _json_error(code: int, error: str) -> Tuple[int, Dict[str, str], bytes]:
    return (
        code,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps({"ok": False, "error": error}).encode("utf-8"),
    )


def handle_judge_ack(body: bytes, *, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, str], bytes]:
    """POST JSON: {actor, note?, token?} — soft-fail manuel acknowledge."""
    expected = os.environ.get("RAG_JUDGE_ACK_TOKEN", "").strip()
    if not expected:
        return _json_error(403, "ack_token_not_configured")
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _json_error(400, "invalid_json")
    if not isinstance(data, dict):
        return _json_error(400, "object_required")
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    token = str(data.get("token") or "").strip()
    if not token:
        token = hdrs.get("x-judge-ack-token", "").strip()
    if not token:
        auth = hdrs.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    if token != expected:
        return _json_error(401, "invalid_token")
    actor = str(data.get("actor") or data.get("by") or "").strip()
    note = str(data.get("note") or "")[:500]
    state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
    notify = str(data.get("notify", "1")).strip().lower() not in {"0", "false", "no", "off"}
    from rag.judge_alert import acknowledge_judge_alert

    result = acknowledge_judge_alert(
        actor=actor,
        note=note,
        state_path=state_path,
        notify=notify,
    )
    if not result.get("ok") and result.get("error") == "actor_required":
        return _json_error(400, "actor_required")
    if not result.get("ok") and result.get("error") == "no_active_soft_fail":
        return _json_error(409, "no_active_soft_fail")
    payload = {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}
    return (
        200,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def handle_judge_alert_state() -> Tuple[int, Dict[str, str], bytes]:
    """GET soft-fail alert state (token gerekmez; soft_fail bayrağı + ack özeti)."""
    state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
    from rag.judge_alert import load_judge_alert_state

    state = load_judge_alert_state(state_path)
    public = {
        "soft_fail": bool(state.get("soft_fail")),
        "source": state.get("source"),
        "acknowledged": bool(state.get("acknowledged")),
        "acknowledged_by": state.get("acknowledged_by"),
        "acknowledged_at": state.get("acknowledged_at"),
        "last_action": state.get("last_action"),
        "updated_at": state.get("updated_at"),
    }
    return (
        200,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps({"ok": True, "state": public}, ensure_ascii=False).encode("utf-8"),
    )


def handle_webpush_register(body: bytes) -> Tuple[int, Dict[str, str], bytes]:
    """POST JSON: {username, subscription|token, label?} → register_device_token."""
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_json"}).encode("utf-8"),
        )
    if not isinstance(data, dict):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "object_required"}).encode("utf-8"),
        )
    username = str(data.get("username") or "").strip()
    if not username:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "username_required"}).encode("utf-8"),
        )
    sub = data.get("subscription")
    token = data.get("token")
    if isinstance(sub, dict):
        token_raw = json.dumps(sub, ensure_ascii=False)
    elif isinstance(sub, str) and sub.strip():
        token_raw = sub.strip()
    elif isinstance(token, str) and token.strip():
        token_raw = token.strip()
    else:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "subscription_required"}).encode("utf-8"),
        )
    from rag.collab_notify_push import parse_webpush_subscription, register_device_token

    if parse_webpush_subscription(token_raw) is None:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_subscription"}).encode("utf-8"),
        )
    rec = register_device_token(
        username,
        token_raw,
        platform="webpush",
        label=str(data.get("label") or "browser")[:64] or "browser",
        device_name=str(data.get("device_name") or "Browser")[:64] or None,
        os_name=str(data.get("os_name") or "web")[:32] or None,
    )
    payload = {
        "ok": True,
        "username": username,
        "platform": "webpush",
        "token_preview": (rec or {}).get("token", "")[:24],
    }
    return (
        200,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def handle_webpush_page(
    *,
    username: str = "",
    public_base: Optional[str] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """Aynı-origin abonelik sayfası (SW + auto register)."""
    from rag.collab_notify_push import vapid_application_server_key, vapid_configured

    base = public_base or collab_http_public_base()
    vapid = vapid_application_server_key() or ""
    configured = vapid_configured() and bool(vapid)
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>RAG Web Push</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:560px;margin:2rem auto;padding:0 1rem;line-height:1.45}}
button{{padding:8px 14px;margin-right:8px}}
#msg{{margin-top:10px;color:#444}}
code{{font-size:12px;word-break:break-all}}
</style></head><body>
<h1>Web Push abonelik</h1>
<p>Service Worker: <code>{base}/sw.js</code></p>
<p>Kullanıcı: <input id="user" value="{username}" placeholder="username"/></p>
<button id="sub" type="button" {"disabled" if not configured else ""}>Abone ol ve kaydet</button>
<pre id="msg"></pre>
<script>
const VAPID = {json.dumps(vapid)};
const SW_URL = {json.dumps(base + "/sw.js")};
const REG_URL = {json.dumps(base + "/webpush/register")};
function urlBase64ToUint8Array(base64String) {{
  const padding = "=".repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; ++i) arr[i] = raw.charCodeAt(i);
  return arr;
}}
document.getElementById("sub").onclick = async () => {{
  const msg = document.getElementById("msg");
  try {{
    if (!VAPID) throw new Error("VAPID yok");
    if (!window.isSecureContext && location.hostname !== "localhost")
      throw new Error("HTTPS veya localhost gerekli");
    const user = (document.getElementById("user").value || "").trim();
    if (!user) throw new Error("username gerekli");
    const perm = await Notification.requestPermission();
    if (perm !== "granted") throw new Error("izin reddedildi");
    const reg = await navigator.serviceWorker.register(SW_URL, {{ scope: "/" }});
    await navigator.serviceWorker.ready;
    let sub = await reg.pushManager.getSubscription();
    if (!sub) {{
      sub = await reg.pushManager.subscribe({{
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(VAPID)
      }});
    }}
    const r = await fetch(REG_URL, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ username: user, subscription: sub.toJSON(), label: "browser-pwa" }})
    }});
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || ("HTTP " + r.status));
    msg.textContent = "Kayıt OK: " + JSON.stringify(data);
  }} catch (e) {{
    msg.textContent = String(e && e.message ? e.message : e);
  }}
}};
</script>
</body></html>
"""
    return (
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"},
        html.encode("utf-8"),
    )


def _cors_preflight() -> Tuple[int, Dict[str, str], bytes]:
    return (
        204,
        {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Judge-Ack-Token",
            "Access-Control-Max-Age": "86400",
        },
        b"",
    )


class CollabHTTPHandler(BaseHTTPRequestHandler):
    public_base: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, code: int, headers: Dict[str, str], body: bytes) -> None:
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        code, headers, body = _cors_preflight()
        self._send(code, headers, body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/sw.js":
            self._send(*handle_sw_js())
            return
        if path in {"/webpush", "/webpush/"}:
            qs = parse_qs(parsed.query or "")
            user = (qs.get("user") or qs.get("username") or [""])[0]
            self._send(
                *handle_webpush_page(
                    username=user,
                    public_base=self.public_base or None,
                )
            )
            return
        if path in {"/judge/alert", "/judge/alert-state"}:
            self._send(*handle_judge_alert_state())
            return
        if path in {"/", "/health"}:
            body = json.dumps(
                {
                    "ok": True,
                    "service": "collab-http",
                    "routes": [
                        "/sw.js",
                        "/webpush",
                        "/webpush/register",
                        "/judge/ack",
                        "/judge/alert",
                    ],
                }
            ).encode("utf-8")
            self._send(200, {"Content-Type": "application/json"}, body)
            return
        self._send(404, {"Content-Type": "text/plain"}, b"not found\n")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if path == "/webpush/register":
            self._send(*handle_webpush_register(raw))
            return
        if path == "/judge/ack":
            hdrs = {k: v for k, v in self.headers.items()}
            self._send(*handle_judge_ack(raw, headers=hdrs))
            return
        self._send(404, {"Content-Type": "text/plain"}, b"not found\n")


def ensure_collab_http_server(
    *,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    enabled: bool = True,
    public_host: Optional[str] = None,
) -> bool:
    """Daemon thread ile collab HTTP (SW + register) başlatır."""
    global _http_started, _http_server
    if not enabled:
        return False
    try:
        from app.config import COLLAB_HTTP_PORT
    except ImportError:
        COLLAB_HTTP_PORT = 8766
    bind_port = int(port if port is not None else COLLAB_HTTP_PORT)
    with _http_lock:
        if _http_started:
            return True
        public = collab_http_public_base(public_host=public_host, port=bind_port)

        class _Handler(CollabHTTPHandler):
            public_base = public

        server = ThreadingHTTPServer((host, bind_port), _Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="collab-http",
            daemon=True,
        )
        thread.start()
        _http_server = server
        _http_started = True
        return True


def run_collab_http_server(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    *,
    public_host: Optional[str] = None,
) -> None:
    """Bloklayarak HTTP sunucusu (test/CLI)."""
    try:
        from app.config import COLLAB_HTTP_PORT
    except ImportError:
        COLLAB_HTTP_PORT = 8766
    bind_port = int(port if port is not None else COLLAB_HTTP_PORT)
    public = collab_http_public_base(public_host=public_host, port=bind_port)

    class _Handler(CollabHTTPHandler):
        public_base = public

    server = ThreadingHTTPServer((host, bind_port), _Handler)
    print(f"Collab HTTP: {public}  (/sw.js, /webpush, /webpush/register, /judge/ack)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
