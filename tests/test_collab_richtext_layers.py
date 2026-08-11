"""Mark katman özet testleri."""

from rag.collab_richtext import (
    add_mark,
    marks_in_range,
    summarize_mark_layers,
)


def test_marks_in_range_and_layers(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    from rag.collab_crdt import apply_text_edit

    ws = "layer-ws"
    apply_text_edit(ws, "abcdefgh", author="a")
    add_mark(ws, mark="bold", start=1, end=5, author="a")
    add_mark(ws, mark="italic", start=2, end=6, author="b")
    store_layers = summarize_mark_layers(ws)
    assert any("bold" in r.get("layers", []) and "italic" in r.get("layers", []) for r in store_layers)
    from rag.collab_richtext import load_richtext

    rt = load_richtext(ws)
    layers = marks_in_range(rt, 3, 5)
    assert "bold" in layers and "italic" in layers
