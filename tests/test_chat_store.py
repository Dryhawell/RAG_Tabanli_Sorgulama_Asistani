from rag.chat_store import (
    append_messages,
    create_session,
    delete_session,
    list_sessions,
    load_session,
)


def test_chat_session_roundtrip(tmp_path):
    chat_dir = str(tmp_path / "chats")
    session = create_session(title="Yeni sohbet", chat_dir=chat_dir)
    sid = session["id"]

    append_messages(
        session,
        [
            {"role": "user", "content": "Merhaba, bu uzun bir soru örneği mi?"},
            {"role": "assistant", "content": "Yanıt"},
        ],
        chat_dir=chat_dir,
    )
    loaded = load_session(sid, chat_dir=chat_dir)
    assert loaded is not None
    assert len(loaded["messages"]) == 2
    assert loaded["title"].startswith("Merhaba")

    items = list_sessions(chat_dir=chat_dir)
    assert any(i["id"] == sid for i in items)
    assert delete_session(sid, chat_dir=chat_dir) is True
    assert load_session(sid, chat_dir=chat_dir) is None
