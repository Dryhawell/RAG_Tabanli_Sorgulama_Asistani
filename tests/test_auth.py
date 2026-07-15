from rag.auth import (
    User,
    add_user,
    authenticate,
    ensure_users_file,
    user_chat_dir,
    verify_password,
    _hash_password,
)


def test_password_hash_roundtrip():
    stored = _hash_password("gizli")
    assert verify_password("gizli", stored)
    assert not verify_password("yanlis", stored)


def test_authenticate_and_roles(tmp_path, monkeypatch):
    path = str(tmp_path / "users.json")
    monkeypatch.setattr("rag.auth.AUTH_BOOTSTRAP_ADMIN", "")
    add_user("Admin", "secret", role="admin", path=path)
    add_user("demo", "demo", role="user", path=path)

    admin = authenticate("admin", "secret", path=path)
    assert admin is not None
    assert admin.role == "admin"
    assert admin.can_ingest is True

    user = authenticate("demo", "demo", path=path)
    assert user is not None
    assert user.role == "user"

    assert authenticate("demo", "nope", path=path) is None


def test_bootstrap_creates_admin(tmp_path, monkeypatch):
    path = str(tmp_path / "users.json")
    monkeypatch.setattr("rag.auth.AUTH_BOOTSTRAP_ADMIN", "root:rootpass")
    monkeypatch.setattr("rag.auth.USERS_PATH", path)
    users = ensure_users_file(path)
    assert "root" in users
    assert authenticate("root", "rootpass", path=path) is not None


def test_user_chat_dir_isolated(tmp_path):
    d1 = user_chat_dir("Alice", base=str(tmp_path / "chats"))
    d2 = user_chat_dir("bob", base=str(tmp_path / "chats"))
    assert d1 != d2
    assert d1.endswith("alice")
    assert d2.endswith("bob")
