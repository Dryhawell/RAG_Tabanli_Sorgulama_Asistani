"""Paylaşım linkleri ve işbirlikçi not testleri."""

from datetime import datetime, timedelta, timezone

from rag.collab_notes import load_note, save_note
from rag.share_links import (
    build_share_url,
    create_share_link,
    list_links_for_session,
    load_shared_session,
    resolve_share_token,
    revoke_share_link,
)


def test_share_link_create_resolve_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.share_links.METADATA_DIR", str(tmp_path))
    chat_dir = str(tmp_path / "chats" / "alice")
    session_id = "sess01"
    from rag.chat_store import create_session, save_session

    session = create_session(chat_dir=chat_dir)
    session["id"] = session_id
    save_session(session, chat_dir=chat_dir)

    link = create_share_link(chat_dir, session_id, owner="alice", ttl_days=3, base=str(tmp_path))
    url = build_share_url(link.token, "http://localhost:8501")
    assert "?share=" in url

    resolved = resolve_share_token(link.token, base=str(tmp_path))
    assert resolved is not None
    assert resolved.session_id == session_id

    loaded = load_shared_session(resolved)
    assert loaded is not None
    assert loaded["id"] == session_id

    items = list_links_for_session(chat_dir, session_id, base=str(tmp_path))
    assert len(items) == 1

    assert revoke_share_link(link.token, base=str(tmp_path))
    assert resolve_share_token(link.token, base=str(tmp_path)) is None


def test_share_link_expired(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.share_links.METADATA_DIR", str(tmp_path))
    chat_dir = str(tmp_path / "chats" / "bob")
    link = create_share_link(chat_dir, "x", ttl_days=1, base=str(tmp_path))
    from rag.share_links import load_shares, save_shares

    store = load_shares(str(tmp_path))
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    store["links"][link.token]["expires_at"] = past
    save_shares(store, str(tmp_path))
    assert resolve_share_token(link.token, base=str(tmp_path)) is None


def test_collab_note_save_conflict_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notes.METADATA_DIR", str(tmp_path))
    key = "tenant:default|shared"
    n1 = save_note(key, "İlk not", username="alice", base=str(tmp_path))
    assert n1.revision == 1
    assert n1.updated_by == "alice"

    try:
        save_note(
            key,
            "Çakışma",
            username="bob",
            expected_revision=0,
            base=str(tmp_path),
        )
        assert False, "conflict expected"
    except ValueError as exc:
        assert "Çakışma" in str(exc)

    n2 = save_note(
        key,
        "Güncel not",
        username="bob",
        expected_revision=1,
        base=str(tmp_path),
    )
    assert n2.revision == 2
    reloaded = load_note(key, base=str(tmp_path))
    assert reloaded.content == "Güncel not"
    assert len(reloaded.history) == 2
