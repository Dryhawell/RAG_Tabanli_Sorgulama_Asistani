"""CRDT canlı düzenleyici bileşeni testleri."""

from rag.collab_component import _js_str, _COLLAB_EDITOR_HTML


def test_collab_editor_html_includes_ws_url():
    html = (
        _COLLAB_EDITOR_HTML.replace("__WS_URL__", _js_str("ws://localhost:8765"))
        .replace("__WORKSPACE__", _js_str("tenant:default|shared"))
        .replace("__USERNAME__", _js_str("alice"))
        .replace("__INITIAL__", _js_str("merhaba"))
        .replace("__REVISION__", "3")
        .replace("__RICHTEXT__", "true")
    )
    assert "ws://localhost:8765" in html
    assert "tenant:default|shared" in html
    assert "collab-editor" in html
    assert "contenteditable" in html
    assert "collab-presence" in html
    assert "collab-cursors" in html
    assert "presence_join" in html or "presence_update" in html
    assert "op: \"cursor\"" in html or 'op: "cursor"' in html
    assert "caretCoordsAt" in html
    assert "sendUndoRedo" in html
    assert 'op: "undo"' in html or "sendUndoRedo(\"undo\")" in html
    assert "opacity:0.22" in html  # selection highlight
    assert "compositionstart" in html
    assert "compositionend" in html
    assert "sendPasteOrIme" in html
    assert '"paste"' in html or "paste" in html
    assert "sendRichOps" in html
    assert "collab-toolbar" in html
    assert "data-mark" in html
    assert "applyRichHtml" in html
    assert "mention_notify" in html
    assert "collab-notify" in html