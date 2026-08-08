"""CRDT undo/redo testleri."""

from rag.collab_undo import apply_text_edit_with_undo, redo_edit, undo_edit
from rag.collab_crdt import load_crdt


def test_undo_redo_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    # undo_path crdt_path kullanır
    key = "undo-room"
    apply_text_edit_with_undo(key, "Merhaba", author="alice", base=str(tmp_path))
    apply_text_edit_with_undo(key, "Merhaba dünya", author="alice", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "Merhaba dünya"

    undo_edit(key, author="alice", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "Merhaba"

    redo_edit(key, author="alice", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "Merhaba dünya"


def test_undo_empty_stack(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    key = "empty-undo"
    doc = undo_edit(key, author="x", base=str(tmp_path))
    assert doc.materialize() == ""
