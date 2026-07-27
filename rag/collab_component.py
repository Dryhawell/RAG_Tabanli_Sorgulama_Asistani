"""Streamlit gömülü CRDT canlı düzenleyici (contenteditable + imleç overlay)."""

from __future__ import annotations

import json
from typing import Optional

import streamlit.components.v1 as components


def _js_str(value: Optional[str]) -> str:
    return json.dumps(value or "")


_COLLAB_EDITOR_HTML = """
<div style="font-family: system-ui, sans-serif;">
  <div id="collab-presence" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;font-size:12px;"></div>
  <div id="editor-wrap" style="position:relative;border:1px solid #ccc;border-radius:6px;min-height:170px;background:#fff;">
    <div id="collab-editor" contenteditable="true" spellcheck="false"
      style="width:100%;min-height:170px;padding:10px 12px;box-sizing:border-box;outline:none;white-space:pre-wrap;word-break:break-word;line-height:1.5;font-size:14px;font-family:ui-monospace,Menlo,monospace;"></div>
    <div id="collab-cursors" style="position:absolute;inset:0;pointer-events:none;overflow:hidden;"></div>
  </div>
  <div id="collab-status" style="font-size:12px;color:#555;margin-top:6px;">Bağlanıyor…</div>
</div>
<script>
(function() {
  const WS_URL = __WS_URL__;
  const WORKSPACE = __WORKSPACE__;
  const USERNAME = __USERNAME__;
  const INITIAL = __INITIAL__;
  const editor = document.getElementById("collab-editor");
  const wrap = document.getElementById("editor-wrap");
  const status = document.getElementById("collab-status");
  const presenceEl = document.getElementById("collab-presence");
  const cursorsEl = document.getElementById("collab-cursors");
  let revision = __REVISION__;
  let ws = null;
  let dirty = false;
  let debounceTimer = null;
  let cursorTimer = null;
  const peers = {};

  function setStatus(msg) { status.textContent = msg; }

  function getText() {
    return editor.innerText.replace(/\\r/g, "");
  }

  function setText(t) {
    editor.innerText = t || "";
  }

  function getCaretOffset() {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return 0;
    const range = sel.getRangeAt(0);
    const pre = range.cloneRange();
    pre.selectNodeContents(editor);
    pre.setEnd(range.startContainer, range.startOffset);
    return pre.toString().length;
  }

  function caretCoordsAt(offset) {
    const text = getText();
    const safe = Math.max(0, Math.min(offset, text.length));
    // Mirror ölçümü
    const mirror = document.createElement("div");
    const style = window.getComputedStyle(editor);
    mirror.style.cssText = [
      "position:absolute", "visibility:hidden", "white-space:pre-wrap", "word-break:break-word",
      "left:0", "top:0", "width:" + editor.clientWidth + "px",
      "padding:" + style.padding, "font:" + style.font, "line-height:" + style.lineHeight,
      "box-sizing:border-box"
    ].join(";");
    const before = document.createTextNode(text.slice(0, safe));
    const marker = document.createElement("span");
    marker.textContent = "|";
    mirror.appendChild(before);
    mirror.appendChild(marker);
    wrap.appendChild(mirror);
    const top = marker.offsetTop;
    const left = marker.offsetLeft;
    wrap.removeChild(mirror);
    return { top: top, left: left };
  }

  function renderPresence() {
    presenceEl.innerHTML = "";
    const keys = Object.keys(peers);
    if (!keys.length) {
      presenceEl.textContent = "Kimse yok";
      cursorsEl.innerHTML = "";
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
    renderCursorOverlay();
  }

  function renderCursorOverlay() {
    cursorsEl.innerHTML = "";
    Object.keys(peers).forEach(function(name) {
      if (name === USERNAME) return;
      const p = peers[name];
      const pos = caretCoordsAt(p.cursor || 0);
      const caret = document.createElement("div");
      caret.style.cssText = [
        "position:absolute",
        "top:" + pos.top + "px",
        "left:" + pos.left + "px",
        "width:2px",
        "height:1.2em",
        "background:" + (p.color || "#e74c3c"),
        "z-index:5"
      ].join(";");
      const label = document.createElement("div");
      label.style.cssText = [
        "position:absolute",
        "top:-14px",
        "left:0",
        "font-size:10px",
        "line-height:1",
        "white-space:nowrap",
        "padding:1px 4px",
        "border-radius:3px",
        "color:#fff",
        "background:" + (p.color || "#e74c3c")
      ].join(";");
      label.textContent = name;
      caret.appendChild(label);
      cursorsEl.appendChild(caret);
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
    const pos = getCaretOffset();
    ws.send(JSON.stringify({
      op: "cursor",
      cursor: pos,
      selection_end: pos,
      username: USERNAME
    }));
  }

  function connect() {
    try { ws = new WebSocket(WS_URL); }
    catch (e) { setStatus("WebSocket hatası: " + e); return; }
    ws.onopen = function() {
      setStatus("Bağlandı — canlı senkron");
      ws.send(JSON.stringify({op: "join", workspace_key: WORKSPACE, username: USERNAME}));
    };
    ws.onmessage = function(ev) {
      try {
        const data = JSON.parse(ev.data);
        if (data.op === "snapshot") {
          revision = data.revision || revision;
          if (!dirty) setText(data.content || "");
          if (data.presence) data.presence.forEach(function(u) { upsertPeer(u); });
          setStatus("Rev " + revision);
        } else if (data.op === "sync") {
          revision = data.revision || revision;
          if (!dirty) setText(data.content || "");
          setStatus("Rev " + revision + (data.updated_by ? " · " + data.updated_by : ""));
          renderCursorOverlay();
        } else if (data.op === "conflict") {
          revision = data.revision || revision;
          setText(data.content || getText());
          setStatus("Çakışma birleştirildi (rev " + revision + ")");
        } else if (data.op === "presence_join" || data.op === "presence_update") {
          upsertPeer(data.user);
        } else if (data.op === "presence_leave") {
          removePeer(data.user);
        } else if (data.op === "error") {
          setStatus("Hata: " + (data.message || "?"));
        }
      } catch (e) { setStatus("Mesaj hatası"); }
    };
    ws.onclose = function() {
      setStatus("Bağlantı kapandı — 3s sonra yeniden…");
      setTimeout(connect, 3000);
    };
    ws.onerror = function() { setStatus("WebSocket bağlantı hatası"); };
  }

  setText(INITIAL);
  editor.addEventListener("input", function() {
    dirty = true;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({
        op: "edit",
        revision: revision,
        content: getText(),
        username: USERNAME
      }));
      dirty = false;
    }, 450);
  });
  ["keyup", "click", "mouseup"].forEach(function(evt) {
    editor.addEventListener(evt, function() {
      if (cursorTimer) clearTimeout(cursorTimer);
      cursorTimer = setTimeout(sendCursor, 80);
    });
  });
  window.addEventListener("resize", renderCursorOverlay);

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
    height: int = 320,
) -> None:
    """Contenteditable CRDT istemcisi — görsel remote imleç overlay."""
    html = (
        _COLLAB_EDITOR_HTML.replace("__WS_URL__", _js_str(ws_url))
        .replace("__WORKSPACE__", _js_str(workspace_key))
        .replace("__USERNAME__", _js_str(username))
        .replace("__INITIAL__", _js_str(initial_content))
        .replace("__REVISION__", str(int(revision)))
    )
    components.html(html, height=height, scrolling=False)
