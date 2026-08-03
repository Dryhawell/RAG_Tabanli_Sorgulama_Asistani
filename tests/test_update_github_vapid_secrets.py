"""GitHub VAPID secret update script tests."""

from scripts.update_github_vapid_secrets import update_vapid_github_secrets


def test_update_vapid_github_secrets_dry_run(monkeypatch):
    monkeypatch.setenv("GH_PAT", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")
    result = update_vapid_github_secrets(
        public="BNcPublicKeyExample",
        subject="mailto:ops@example.com",
        dry_run=True,
        include_private=False,
    )
    assert result["dry_run"] is True
    assert "RAG_NOTIFY_PUSH_VAPID_PUBLIC" in result["updated"]
    assert "RAG_NOTIFY_PUSH_VAPID_SUBJECT" in result["updated"]
    assert result["failed"] == []


def test_update_vapid_github_secrets_requires_token(monkeypatch):
    monkeypatch.delenv("GH_PAT", raising=False)
    monkeypatch.delenv("VAPID_GH_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")
    result = update_vapid_github_secrets(public="abc", dry_run=False)
    assert "no_token" in result["failed"]
