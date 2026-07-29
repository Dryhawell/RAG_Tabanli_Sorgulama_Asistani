"""Rich-text mark diff testleri."""

from rag.collab_richtext_diff import (
    diff_mark_snapshots,
    format_mark_diff_line,
    html_mark_diff_entry,
    rebuild_mark_timeline,
    summarize_mark_audit_diffs,
)


def test_diff_mark_snapshots():
    before = [{"id": "a", "mark": "bold", "start": 0, "end": 3}]
    after = [
        {"id": "a", "mark": "bold", "start": 0, "end": 3},
        {"id": "b", "mark": "italic", "start": 4, "end": 8},
    ]
    diff = diff_mark_snapshots(before, after)
    assert len(diff["added"]) == 1
    assert diff["added"][0]["id"] == "b"
    assert len(diff["removed"]) == 0


def test_rebuild_mark_timeline():
    rows = [
        {
            "action": "add",
            "mark": {"id": "m1", "mark": "bold", "start": 0, "end": 3},
            "author": "alice",
        },
        {
            "action": "remove",
            "mark": {"id": "m1", "mark": "bold", "start": 0, "end": 3},
            "author": "bob",
        },
    ]
    timeline = rebuild_mark_timeline(rows)
    assert len(timeline) == 2
    assert len(timeline[0]["after"]) == 1
    assert len(timeline[1]["after"]) == 0
    assert timeline[1]["diff"]["removed"]


def test_summarize_mark_audit_diffs_html():
    rows = [
        {
            "ts": "2026-01-01T00:00:00+00:00",
            "action": "add",
            "mark": {"id": "m1", "mark": "code", "start": 1, "end": 4},
            "author": "alice",
        }
    ]
    out = summarize_mark_audit_diffs(rows, "hello world")
    assert len(out) == 1
    assert "add" in out[0]["line"]
    assert "code" in out[0]["html"]
    assert "ell" in out[0]["html"]


def test_format_and_html_mark_diff_line():
    entry = {
        "action": "add",
        "mark": {"mark": "bold", "start": 0, "end": 2},
        "author": "alice",
    }
    line = format_mark_diff_line(entry, "ab")
    assert "bold" in line
    html = html_mark_diff_entry(entry, "ab")
    assert "bold" in html
    assert "ab" in html


def test_summarize_comment_audit_diffs():
    from rag.collab_richtext_diff import summarize_comment_audit_diffs

    rows = [
        {
            "action": "add",
            "thread": {"id": "t1", "body": "yorum", "start": 0, "end": 4},
            "author": "alice",
        },
        {
            "action": "reply",
            "thread": {"id": "t1", "start": 0, "end": 4},
            "reply": {"body": "yanıt"},
            "author": "bob",
        },
    ]
    out = summarize_comment_audit_diffs(rows, "test metin")
    assert len(out) == 2
    assert "add" in out[0]["line"]
    assert "reply" in out[1]["line"]
