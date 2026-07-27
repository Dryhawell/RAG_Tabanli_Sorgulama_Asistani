"""Streamlit gömülü CRDT canlı düzenleyici (WebSocket istemcisi)."""

from __future__ import annotations

import json
from typing import Optional

import streamlit.components.v1 as components


def _js_str(value: Optional[str]) -> str:
    return json.dumps(value or "")


_COLLAB_EDITOR_HTML = """
<div style="font-family: system-ui, sans-serif;">
  <textarea id="collab-editor" style="width:100%;height:170px;padding:8px;border:1px solid #ccc;border-radius:6px;box-sizing:border-box;"></textarea>
  <div id="collab-status" style="font-size:12px;color:#555;margin-top:6px;">Bağlanıyor…</div>
</div>
<script>
(function() {
  const WS_URL = __WS_URL__;
  const WORKSPACE = __WORKSPACE__;
  const USERNAME = __USERNAME__;
  const INITIAL = __INITIAL__;
  const textarea = document.getElementById("collab-editor");
  const status = document.getElementById("collab-status");
  let revision = __REVISION__;
  let ws = null;
  let dirty = false;
  let debounceTimer = null;

  function setStatus(msg) {
    status.textContent = msg;
  }

  function connect() {
    try {
      ws = new WebSocket(WS_URL);
    } catch (e) {
      setStatus("WebSocket hatası: " + e);
      return;
    }
    ws.onopen = function() {
      setStatus("Bağlandı — canlı senkron");
      ws.send(JSON.stringify({op: "join", workspace_key: WORKSPACE, username: USERNAME}));
    };
    ws.onmessage = function(ev) {
      try {
        const data = JSON.parse(ev.data);
        if (data.op === "snapshot" || data.op === "sync") {
          revision = data.revision || revision;
          if (!dirty) {
            textarea.value = data.content || "";
          }
          setStatus("Rev " + revision + (data.updated_by ? " · " + data.updated_by : ""));
        } else if (data.op === "conflict") {
          revision = data.revision || revision;
          textarea.value = data.content || textarea.value;
          setStatus("Çakışma birleştirildi (rev " + revision + ")");
        } else if (data.op === "error") {
          setStatus("Hata: " + (data.message || "?"));
        }
      } catch (e) {
        setStatus("Mesaj hatası");
      }
    };
    ws.onclose = function() {
      setStatus("Bağlantı kapandı — 3s sonra yeniden…");
      setTimeout(connect, 3000);
    };
    ws.onerror = function() {
      setStatus("WebSocket bağlantı hatası");
    };
  }

  textarea.value = INITIAL;
  textarea.addEventListener("input", function() {
    dirty = true;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({
        op: "edit",
        revision: revision,
        content: textarea.value,
        username: USERNAME
      }));
      dirty = false;
    }, 450);
  });

  connect();
})();
</script>
"""


def render_collab_live_editor(
    *,
    ws_url: str,
    workspace_key: str,
    username: str,
    initial_content: str = "",
    revision: int = 0,
    height: int = 260,
) -> None:
    """HTML/JS CRDT istemcisi — düzenlemeler WebSocket üzerinden CRDT store'a yazılır."""
    html = (
        _COLLAB_EDITOR_HTML.replace("__WS_URL__", _js_str(ws_url))
        .replace("__WORKSPACE__", _js_str(workspace_key))
        .replace("__USERNAME__", _js_str(username))
        .replace("__INITIAL__", _js_str(initial_content))
        .replace("__REVISION__", str(int(revision)))
    )
    components.html(html, height=height, scrolling=False)
