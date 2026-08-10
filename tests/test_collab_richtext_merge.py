"""Mark birleştirme testleri."""

from rag.collab_richtext import add_mark, load_richtext, merge_overlapping_marks, TextMark


def test_merge_overlapping_same_type():
    marks = [
        TextMark(id="a", mark="bold", start=0, end=5),
        TextMark(id="b", mark="bold", start=3, end=8),
        TextMark(id="c", mark="italic", start=2, end=6),
    ]
    merged = merge_overlapping_marks(marks)
    bold = [m for m in merged if m.mark == "bold"]
    italic = [m for m in merged if m.mark == "italic"]
    assert len(bold) == 1
    assert bold[0].start == 0 and bold[0].end == 8
    assert len(italic) == 1


def test_add_mark_merges(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_richtext.METADATA_DIR", str(tmp_path))
    ws = "merge-ws"
    add_mark(ws, mark="bold", start=0, end=5, author="a")
    add_mark(ws, mark="bold", start=4, end=10, author="b")
    store = load_richtext(ws)
    assert len(store.marks) == 1
    assert store.marks[0].end == 10
