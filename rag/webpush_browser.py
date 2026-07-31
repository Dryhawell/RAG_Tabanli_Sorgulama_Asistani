"""Tarayıcı Web Push abonelik PoC (Streamlit components.html)."""

from __future__ import annotations

import json
from typing import Optional

import streamlit.components.v1 as components

from rag.collab_notify_push import (
    load_service_worker_js,
    vapid_application_server_key,
    vapid_configured,
    vapid_public_key,
)


def _js_str(value: Optional[str]) -> str:
    return json.dumps(value or "")


_WEBPUSH_HTML = """
<div style="font-family:system-ui,sans-serif;font-size:13px;line-height:1.4;">
  <p style="margin:0 0 8px;color:#444;">__STATUS__</p>
  <button id="wp-sub" type="button" style="padding:6px 12px;margin-right:6px;">Abone ol</button>
  <button id="wp-copy" type="button" style="padding:6px 12px;">JSON kopyala</button>
  <textarea id="wp-out" rows="6" style="width:100%;margin-top:8px;font-family:ui-monospace,Menlo,monospace;font-size:11px;"
    placeholder="PushSubscription JSON burada görünecek — token alanına yapıştırın"></textarea>
  <p id="wp-msg" style="margin:6px 0 0;color:#666;min-height:18px;"></p>
</div>
<script>
(function() {
  const VAPID = __VAPID__;
  const SW_BLOB = __SW_BLOB__;
  const out = document.getElementById("wp-out");
  const msg = document.getElementById("wp-msg");
  function setMsg(t, ok) {
    msg.textContent = t || "";
    msg.style.color = ok ? "#0a7" : "#a40";
  }
  function urlBase64ToUint8Array(base64String) {
    const padding = "=".repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
    const raw = atob(base64);
    const arr = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; ++i) arr[i] = raw.charCodeAt(i);
    return arr;
  }
  async function ensureSW() {
    if (!("serviceWorker" in navigator)) throw new Error("Service Worker desteklenmiyor");
    // iframe kökeninde blob SW (Streamlit PoC)
    const blob = new Blob([SW_BLOB], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    const reg = await navigator.serviceWorker.register(url, { scope: "/" });
    await navigator.serviceWorker.ready;
    return reg;
  }
  document.getElementById("wp-sub").onclick = async function() {
    try {
      if (!VAPID) throw new Error("VAPID public key yok");
      if (!window.isSecureContext) throw new Error("HTTPS / localhost gerekli");
      const perm = await Notification.requestPermission();
      if (perm !== "granted") throw new Error("Bildirim izni reddedildi");
      const reg = await ensureSW();
      let sub = await reg.pushManager.getSubscription();
      if (!sub) {
        sub = await reg.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: urlBase64ToUint8Array(VAPID)
        });
      }
      out.value = JSON.stringify(sub.toJSON());
      setMsg("Abonelik hazır — JSON'u token alanına yapıştırıp kaydedin", true);
    } catch (err) {
      setMsg(String(err && err.message ? err.message : err), false);
    }
  };
  document.getElementById("wp-copy").onclick = async function() {
    try {
      await navigator.clipboard.writeText(out.value || "");
      setMsg("Panoya kopyalandı", true);
    } catch (err) {
      setMsg("Kopyalama başarısız — metni elle seçin", false);
    }
  };
})();
</script>
"""


def render_webpush_subscribe_widget(*, height: int = 240) -> None:
    """Streamlit içinde Web Push abonelik PoC widget'ı."""
    if not vapid_configured():
        components.html(
            "<p style='font-family:system-ui;color:#a40;'>VAPID anahtarları yapılandırılmamış "
            "(RAG_NOTIFY_PUSH_VAPID_PUBLIC / PRIVATE).</p>",
            height=48,
        )
        return
    app_key = vapid_application_server_key()
    sw_js = load_service_worker_js() or "self.addEventListener('install',()=>self.skipWaiting());"
    status = (
        f"VAPID public: {(vapid_public_key() or '')[:24]}…"
        if vapid_public_key()
        else "VAPID public eksik"
    )
    if not app_key:
        components.html(
            "<p style='font-family:system-ui;color:#a40;'>"
            "VAPID public key URL-safe base64 olmalı (PEM değil).</p>",
            height=48,
        )
        return
    html = (
        _WEBPUSH_HTML.replace("__STATUS__", status)
        .replace("__VAPID__", _js_str(app_key))
        .replace("__SW_BLOB__", _js_str(sw_js))
    )
    components.html(html, height=height)
