"""Streamlit gömülü CRDT canlı düzenleyici (WebSocket istemcisi)."""

from __future__ import annotations

import json
from typing import Optional

import streamlit.components.v1 as components


def _js_str(value: Optional[str]) -> str:
    return json.dumps(value or "")


_COLLAB_EDITOR_HTML = """
<div style="font-family: system-ui, sans-serif;">
  <div id="collab-presence" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;font-size:12px;"></div>
  <div style="position:relative;">
    <textarea id="collab-editor" style="width:100%;height:170px;padding:8px;border:1px solid #ccc;border-radius:6px;box-sizing:border-box;"></textarea>
    <div id="collab-cursors" style="position:absolute;left:8px;top:8px;pointer-events:none;font-size:11px;line-height:1.4;"></div>
  </div>
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
  const presenceEl = document.getElementById("collab-presence");
  const cursorsEl = document.getElementById("collab-cursors");
  let revision = __REVISION__;
  let ws = null;
  let dirty = false;
  let debounceTimer = null;
  let cursorTimer = null;
  const peers = {};

  function setStatus(msg) {
    status.textContent = msg;
  }

  function renderPresence() {
    presenceEl.innerHTML = "";
    const keys = Object.keys(peers);
    if (!keys.length) {
      presenceEl.textContent = "Kimse yok";
      return;
    }
    keys.forEach(function(name) {
      const p = peers[name];
      const chip = document.createElement("span");
      chip.style.cssText = "padding:2px 8px;border-radius:12px;background:#f4f4f4;border:1px solid #ddd;";
      chip.innerHTML = '<span style="color:' + p.color + '">●</span> ' + name +
        (p.cursor != null ? ' @' + p.cursor : '');
      presenceEl.appendChild(chip);
    });
    cursorsEl.innerHTML = "";
    keys.forEach(function(name) {
      if (name === USERNAME) return;
      const p = peers[name];
      const mark = document.createElement("div");
      mark.style.color = p.color;
      mark.textContent = name + " imleç:" + (p.cursor || 0);
      cursorsEl.appendChild(mark);
    });
  }

  function upsertPeer(user) {
    if (!user || !user.username) return;
    if (user.username === USERNAME) return;
    peers[user.username] = {
      color: user.color || "#888",
      cursor: user.cursor || 0,
      selection_end: user.selection_end || 0
    };
    renderPresence();
  }

  function removePeer(user) {
    if (!user || !user.username) return;
    delete peers[user.username];
    renderPresence();
  }

  function sendCursor() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pos = textarea.selectionStart || 0;
    const end = textarea.selectionEnd || pos;
    ws.send(JSON.stringify({
      op: "cursor",
      cursor: pos,
      selection_end: end,
      username: USERNAME
    }));
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
        if (data.op === "snapshot") {
          revision = data.revision || revision;
          if (!dirty) textarea.value = data.content || "";
          if (data.presence) {
            data.presence.forEach(function(u) { upsertPeer(u); });
          }
          setStatus("Rev " + revision);
        } else if (data.op === "sync") {
          revision = data.revision || revision;
          if (!dirty) textarea.value = data.content || "";
          setStatus("Rev " + revision + (data.updated_by ? " · " + data.updated_by : ""));
        } else if (data.op === "conflict") {
          revision = data.revision || revision;
          textarea.value = data.content || textarea.value;
          setStatus("Çakışma birleştirildi (rev " + revision + ")");
        } else if (data.op === "presence_join" || data.op === "presence_update") {
          upsertPeer(data.user);
        } else if (data.op === "presence_leave") {
          removePeer(data.user);
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
  textarea.addEventListener("keyup", function() {
    if (cursorTimer) clearTimeout(cursorTimer);
    cursorTimer = setTimeout(sendCursor, 120);
  });
  textarea.addEventListener("click", sendCursor);
  textarea.addEventListener("select", sendCursor);

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
    height: int = 300,
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
