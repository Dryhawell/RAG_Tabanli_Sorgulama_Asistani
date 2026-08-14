"""Streamlit gömülü CRDT canlı düzenleyici (contenteditable + imleç overlay)."""

from __future__ import annotations

import json
from typing import Optional

import streamlit.components.v1 as components


def _js_str(value: Optional[str]) -> str:
    return json.dumps(value or "")


_COLLAB_EDITOR_HTML = """
<style>
  #collab-editor strong { font-weight: 700; }
  #collab-editor em { font-style: italic; }
  #collab-editor code {
    font-family: ui-monospace, Menlo, monospace;
    background: #f4f4f4;
    padding: 0 2px;
    border-radius: 2px;
  }
  #collab-editor .crdt-comment {
    color: #4a90d9;
    font-size: 10px;
    vertical-align: super;
    margin-right: 1px;
  }
  #collab-notify {
    font-size: 12px;
    color: #444;
    margin: 4px 0 6px;
    padding: 6px 8px;
    background: #fff8e6;
    border: 1px solid #f0d78c;
    border-radius: 4px;
    display: none;
  }
  #collab-layers {
    font-size: 11px;
    color: #555;
    margin-bottom: 4px;
    min-height: 14px;
  }
  #collab-editor .crdt-layer-bold-italic strong em {
    font-weight: 700;
    font-style: italic;
  }
  #collab-editor .crdt-layer-bold-code code {
    font-weight: 700;
    background: #eef6ff;
  }
  #collab-editor .crdt-layer-italic-code em code,
  #collab-editor .crdt-layer-code-italic code {
    font-style: italic;
    background: #f4f4f4;
  }
</style>
<div style="font-family: system-ui, sans-serif;">
  <div id="collab-presence" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;font-size:12px;"></div>
  <div id="collab-notify"></div>
  <div id="collab-layers"></div>
  <div id="collab-toolbar" style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;">
    <button type="button" data-mark="bold" style="padding:4px 10px;font-weight:700;">B</button>
    <button type="button" data-mark="italic" style="padding:4px 10px;font-style:italic;">I</button>
    <button type="button" data-mark="code" style="padding:4px 10px;font-family:monospace;">&lt;/&gt;</button>
    <button type="button" id="collab-comment-btn" style="padding:4px 10px;">Yorum</button>
  </div>
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
  const RICHTEXT = __RICHTEXT__;
  const editor = document.getElementById("collab-editor");
  const wrap = document.getElementById("editor-wrap");
  const status = document.getElementById("collab-status");
  const presenceEl = document.getElementById("collab-presence");
  const cursorsEl = document.getElementById("collab-cursors");
  const toolbar = document.getElementById("collab-toolbar");
  const notifyEl = document.getElementById("collab-notify");
  const layersEl = document.getElementById("collab-layers");
  let revision = __REVISION__;
  let ws = null;
  let dirty = false;
  let debounceTimer = null;
  let cursorTimer = null;
  let composing = false;
  let imeStart = 0;
  const peers = {};
  let richMarks = [];
  if (!RICHTEXT && toolbar) toolbar.style.display = "none";

  function setStatus(msg) { status.textContent = msg; }

  function getText() {
    return editor.innerText.replace(/\\r/g, "");
  }

  function setText(t) {
    editor.innerText = t || "";
  }

  function applyRichHtml(html) {
    if (!RICHTEXT || dirty) return;
    if (html) {
      editor.innerHTML = html;
    } else {
      setText(getText());
    }
    renderCursorOverlay();
  }

  function showNotify(msg) {
    if (!notifyEl) return;
    notifyEl.style.display = "block";
    notifyEl.textContent = msg;
    setTimeout(function() { notifyEl.style.display = "none"; }, 6000);
  }

  function applyContentFromServer(data) {
    if (dirty) return;
    if (RICHTEXT && data.rich && data.rich.html) {
      applyRichHtml(data.rich.html);
    } else if (data.content !== undefined) {
      setText(data.content || "");
    }
    if (RICHTEXT && data.rich && data.rich.marks) {
      richMarks = data.rich.marks;
      updateLayerPreview();
    }
  }

  function marksAtSelection() {
    const sel = getSelectionOffsets();
    if (sel.end <= sel.start) return [];
    const found = new Set();
    richMarks.forEach(function(m) {
      const a = parseInt(m.start || 0, 10);
      const b = parseInt(m.end || 0, 10);
      if (a < sel.end && b > sel.start) found.add(m.mark);
    });
    return Array.from(found).sort();
  }

  function updateLayerPreview() {
    if (!layersEl || !RICHTEXT) return;
    const sel = getSelectionOffsets();
    const layers = marksAtSelection();
    if (layers.length) {
      layersEl.textContent =
        "Katmanlar @" + sel.start + "-" + sel.end + ": " + layers.join(" + ");
    } else if (sel.end > sel.start) {
      layersEl.textContent = "Seçim @" + sel.start + "-" + sel.end + " (biçim yok)";
    } else {
      layersEl.textContent = "";
    }
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
        (p.typing ? ' <em>yazıyor…</em>' : '') +
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
      const start = Math.min(p.cursor || 0, p.selection_end != null ? p.selection_end : (p.cursor || 0));
      const end = Math.max(p.cursor || 0, p.selection_end != null ? p.selection_end : (p.cursor || 0));
      if (end > start) {
        // selection highlight: satır satır yaklaşık kutular
        for (let i = start; i < end; i++) {
          const c = caretCoordsAt(i);
          const n = caretCoordsAt(i + 1);
          const hl = document.createElement("div");
          const w = Math.max(4, (n.left - c.left) || 8);
          hl.style.cssText = [
            "position:absolute",
            "top:" + c.top + "px",
            "left:" + c.left + "px",
            "width:" + w + "px",
            "height:1.2em",
            "background:" + (p.color || "#e74c3c"),
            "opacity:0.22",
            "z-index:3"
          ].join(";");
          cursorsEl.appendChild(hl);
        }
      }
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
      selection_end: user.selection_end || 0,
      typing: !!user.typing
    };
    renderPresence();
  }

  function removePeer(user) {
    if (!user || !user.username) return;
    delete peers[user.username];
    renderPresence();
  }

  function getSelectionOffsets() {
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) {
      const p = getCaretOffset();
      return {start: p, end: p};
    }
    const r0 = sel.getRangeAt(0);
    const preStart = r0.cloneRange();
    preStart.selectNodeContents(editor);
    preStart.setEnd(r0.startContainer, r0.startOffset);
    const start = preStart.toString().length;
    const preEnd = r0.cloneRange();
    preEnd.selectNodeContents(editor);
    preEnd.setEnd(r0.endContainer, r0.endOffset);
    const end = preEnd.toString().length;
    return {start: start, end: end};
  }

  function sendCursor() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const sel = getSelectionOffsets();
    ws.send(JSON.stringify({
      op: "cursor",
      cursor: sel.start,
      selection_end: sel.end,
      username: USERNAME
    }));
  }

  let typingTimer = null;
  let typingActive = false;
  function sendTyping(flag) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!!flag === typingActive && flag) return;
    typingActive = !!flag;
    ws.send(JSON.stringify({
      op: "typing",
      typing: typingActive,
      username: USERNAME
    }));
  }
  function bumpTyping() {
    sendTyping(true);
    if (typingTimer) clearTimeout(typingTimer);
    typingTimer = setTimeout(function() { sendTyping(false); }, 1200);
  }

  function sendUndoRedo(kind) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({op: kind, username: USERNAME}));
  }

  function sendPasteOrIme(kind, start, end, text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      op: kind,
      start: start,
      end: end,
      text: text,
      username: USERNAME
    }));
  }

  function sendRichOps(ops) {
    if (!RICHTEXT || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({op: "rich_ops", ops: ops, username: USERNAME}));
  }

  function flushEdit() {
    if (!ws || ws.readyState !== WebSocket.OPEN || composing) return;
    ws.send(JSON.stringify({
      op: "edit",
      revision: revision,
      content: getText(),
      username: USERNAME
    }));
    dirty = false;
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
          applyContentFromServer(data);
          if (data.presence) data.presence.forEach(function(u) { upsertPeer(u); });
          setStatus("Rev " + revision);
        } else if (data.op === "sync") {
          revision = data.revision || revision;
          applyContentFromServer(data);
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
        } else if (data.op === "rich_sync") {
          if (RICHTEXT && data.rich && data.rich.marks) richMarks = data.rich.marks;
          if (RICHTEXT && data.rich && data.rich.html) applyRichHtml(data.rich.html);
          updateLayerPreview();
          setStatus("Richtext güncellendi");
        } else if (data.op === "mention_notify") {
          const n = data.notification;
          if (n && String(n.target_user || "").toLowerCase() === String(USERNAME || "").toLowerCase()) {
            showNotify((n.from_user || "Biri") + " sizi etiketledi: " + (n.body_preview || ""));
          }
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
  editor.addEventListener("compositionstart", function() {
    composing = true;
    imeStart = getSelectionOffsets().start;
    if (debounceTimer) clearTimeout(debounceTimer);
  });
  editor.addEventListener("compositionend", function() {
    composing = false;
    // IME metni zaten DOM'da; tek seferlik full-doc commit
    dirty = true;
    flushEdit();
  });
  editor.addEventListener("paste", function(e) {
    e.preventDefault();
    if (composing) return;
    const clip = (e.clipboardData || window.clipboardData);
    const text = clip ? String(clip.getData("text/plain") || "") : "";
    const sel = getSelectionOffsets();
    // Sunucu paste op uygular; sync ile DOM güncellenir (çift yazımı önler)
    sendPasteOrIme("paste", sel.start, sel.end, text);
    dirty = false;
  });
  editor.addEventListener("input", function(e) {
    if (composing || (e && e.isComposing)) return;
    dirty = true;
    bumpTyping();
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(flushEdit, 450);
  });
  ["keyup", "click", "mouseup"].forEach(function(evt) {
    editor.addEventListener(evt, function() {
      if (composing) return;
      if (cursorTimer) clearTimeout(cursorTimer);
      cursorTimer = setTimeout(function() {
        sendCursor();
        updateLayerPreview();
      }, 80);
    });
  });
  editor.addEventListener("keydown", function(e) {
    if (composing || e.isComposing) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && !e.shiftKey && (e.key === "z" || e.key === "Z")) {
      e.preventDefault();
      sendUndoRedo("undo");
      return;
    }
    if (mod && ((e.key === "y" || e.key === "Y") || (e.shiftKey && (e.key === "z" || e.key === "Z")))) {
      e.preventDefault();
      sendUndoRedo("redo");
    }
  });
  window.addEventListener("resize", renderCursorOverlay);

  if (toolbar && RICHTEXT) {
    toolbar.querySelectorAll("button[data-mark]").forEach(function(btn) {
      btn.addEventListener("click", function() {
        const sel = getSelectionOffsets();
        if (sel.end <= sel.start) {
          setStatus("Biçim için metin seçin");
          return;
        }
        sendRichOps([{
          type: "add_mark",
          mark: btn.getAttribute("data-mark"),
          start: sel.start,
          end: sel.end
        }]);
      });
    });
    const cmtBtn = document.getElementById("collab-comment-btn");
    if (cmtBtn) {
      cmtBtn.addEventListener("click", function() {
        const sel = getSelectionOffsets();
        const body = window.prompt("Yorum (@kullanici ile etiketleyin)");
        if (!body) return;
        sendRichOps([{
          type: "add_comment",
          start: sel.start,
          end: Math.max(sel.start, sel.end),
          body: body
        }]);
      });
    }
  }

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
    height: int = 340,
    richtext: bool = True,
) -> None:
    """Contenteditable CRDT istemcisi — görsel remote imleç overlay."""
    html = (
        _COLLAB_EDITOR_HTML.replace("__WS_URL__", _js_str(ws_url))
        .replace("__WORKSPACE__", _js_str(workspace_key))
        .replace("__USERNAME__", _js_str(username))
        .replace("__INITIAL__", _js_str(initial_content))
        .replace("__REVISION__", str(int(revision)))
        .replace("__RICHTEXT__", "true" if richtext else "false")
    )
    components.html(html, height=height, scrolling=False)
