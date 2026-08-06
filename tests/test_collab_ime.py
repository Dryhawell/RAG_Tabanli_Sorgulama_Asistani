"""CRDT paste / IME testleri."""

from rag.collab_crdt import load_crdt
from rag.collab_ime import apply_paste
from rag.collab_undo import apply_text_edit_with_undo, undo_edit


def test_paste_insert_and_replace(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    key = "paste-room"
    apply_text_edit_with_undo(key, "Merhaba", author="a", base=str(tmp_path))
    apply_paste(key, start=7, end=7, text=" dünya", author="a", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "Merhaba dünya"

    apply_paste(key, start=0, end=7, text="Selam", author="a", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "Selam dünya"


def test_paste_undo(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    key = "paste-undo"
    apply_text_edit_with_undo(key, "ABC", author="a", base=str(tmp_path))
    apply_paste(key, start=3, end=3, text="XYZ", author="a", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "ABCXYZ"
    undo_edit(key, author="a", base=str(tmp_path))
    assert load_crdt(key, base=str(tmp_path)).materialize() == "ABC"


def test_replace_range_ops_unit(tmp_path, monkeypatch):
    from rag.collab_ime import replace_range_text

    assert replace_range_text("abcd", 1, 3, "XY") == "aXYd"
    assert replace_range_text("Merhaba", 0, 7, "Selam") == "Selam"
