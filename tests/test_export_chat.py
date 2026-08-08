from rag.export_chat import export_session_json, export_session_markdown


def test_export_json_and_markdown():
    session = {
        "id": "abc123",
        "title": "İzin sorusu",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:01:00+00:00",
        "messages": [
            {"role": "user", "content": "Kaç gün izin?"},
            {
                "role": "assistant",
                "content": "14 gün",
                "sources": [{"label": "politika.txt#0", "score": 0.9}],
            },
        ],
    }
    js = export_session_json(session)
    assert '"id": "abc123"' in js
    assert "Kaç gün izin?" in js

    md = export_session_markdown(session)
    assert md.startswith("# İzin sorusu")
    assert "## Kullanıcı" in md
    assert "## Asistan" in md
    assert "politika.txt#0" in md
