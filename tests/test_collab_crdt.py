"""CRDT işbirlikçi not testleri."""

from rag.collab_crdt import apply_text_edit, load_crdt, merge_remote_ops


def test_crdt_initial_text_edit(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    key = "crdt-room"
    doc = apply_text_edit(key, "Merhaba", author="alice", base=str(tmp_path))
    assert doc.materialize() == "Merhaba"
    assert doc.revision == 1

    doc2 = apply_text_edit(key, "Merhaba dünya", author="bob", base=str(tmp_path))
    assert doc2.materialize() == "Merhaba dünya"
    assert doc2.revision == 2


def test_crdt_concurrent_merge_via_ops(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    key = "merge-room"
    apply_text_edit(key, "ABC", author="alice", base=str(tmp_path))

    # Alice: ABC -> AXBC (insert X after A)
    doc = load_crdt(key, base=str(tmp_path))
    ids = doc.ordered_visible_ids()
    assert doc.materialize() == "ABC"
    op_x = {
        "type": "ins",
        "id": "node-x",
        "after": ids[0],
        "char": "X",
        "lamport": doc.lamport + 1,
    }
    merge_remote_ops(key, [op_x], author="alice", base=str(tmp_path))

    # Bob: ABC -> ABYC (insert Y after B) — CRDT ile birleşmeli
    doc = load_crdt(key, base=str(tmp_path))
    ids = doc.ordered_visible_ids()
    op_y = {
        "type": "ins",
        "id": "node-y",
        "after": ids[1],
        "char": "Y",
        "lamport": doc.lamport + 1,
    }
    merged = merge_remote_ops(key, [op_y], author="bob", base=str(tmp_path))
    text = merged.materialize()
    assert "X" in text and "Y" in text and "A" in text and "B" in text and "C" in text
    assert len(text) == 5


def test_crdt_delete_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    key = "del-room"
    apply_text_edit(key, "test", author="u", base=str(tmp_path))
    apply_text_edit(key, "te", author="u", base=str(tmp_path))
    reloaded = load_crdt(key, base=str(tmp_path))
    assert reloaded.materialize() == "te"
