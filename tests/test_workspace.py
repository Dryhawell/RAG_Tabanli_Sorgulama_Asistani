import os

from rag.auth import User
from rag.ingest import list_data_files
from rag.workspace import ensure_workspace_dirs, resolve_workspace


def test_shared_workspace_paths():
    ws = resolve_workspace(User("alice", role="user"), shared=True)
    assert ws.shared is True
    assert ws.key == "shared"
    assert ws.data_dir.endswith("data") or ws.data_dir == "data"
    assert "users" not in ws.index_path.replace("\\", "/")


def test_private_workspace_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.workspace.DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("rag.workspace.INDEXES_DIR", str(tmp_path / "indexes"))
    monkeypatch.setattr("rag.workspace.METADATA_DIR", str(tmp_path / "metadata"))
    monkeypatch.setattr("rag.workspace.CHAT_DIR", str(tmp_path / "chats"))
    monkeypatch.setattr("rag.workspace.INDEX_PATH", str(tmp_path / "indexes" / "faiss.index"))
    monkeypatch.setattr("rag.workspace.DOCSTORE_PATH", str(tmp_path / "metadata" / "docstore.json"))

    alice = resolve_workspace(User("Alice", role="user"), shared=False)
    bob = resolve_workspace(User("bob", role="user"), shared=False)
    ensure_workspace_dirs(alice)
    ensure_workspace_dirs(bob)

    assert alice.shared is False
    assert "users/alice" in alice.data_dir.replace("\\", "/")
    assert "users/bob" in bob.data_dir.replace("\\", "/")
    assert alice.index_path != bob.index_path
    assert alice.docstore_path != bob.docstore_path
    assert os.path.isdir(alice.data_dir)


def test_list_data_files_skips_users_namespace(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    users = tmp_path / "users" / "alice"
    users.mkdir(parents=True)
    (users / "secret.txt").write_text("y", encoding="utf-8")

    shared = list_data_files(str(tmp_path), skip_user_namespaces=True)
    rels = {os.path.relpath(p, tmp_path).replace("\\", "/") for p in shared}
    assert rels == {"a.txt"}

    all_files = list_data_files(str(tmp_path), skip_user_namespaces=False)
    rels_all = {os.path.relpath(p, tmp_path).replace("\\", "/") for p in all_files}
    assert "users/alice/secret.txt" in rels_all


def test_private_user_can_ingest(monkeypatch):
    monkeypatch.setattr("rag.auth.AUTH_SHARED_INDEX", False)
    monkeypatch.setattr("rag.auth.AUTH_USER_CAN_INGEST", False)
    u = User("demo", role="user")
    assert u.can_ingest is True
