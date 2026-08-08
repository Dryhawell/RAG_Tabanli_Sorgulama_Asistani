"""GitHub Alertmanager secret update script tests."""

from scripts.update_github_alertmanager_secrets import update_alertmanager_github_secrets


def test_update_alertmanager_github_secrets_dry_run(monkeypatch):
    monkeypatch.setenv("GH_PAT", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")
    result = update_alertmanager_github_secrets(
        slack_webhook="https://hooks.slack.test/T/B/xxx",
        webhook_url="http://example.test/hook",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert "RAG_ALERTMANAGER_SLACK_WEBHOOK" in result["updated"]
    assert "RAG_ALERTMANAGER_WEBHOOK_URL" in result["updated"]
    assert result["failed"] == []


def test_update_alertmanager_github_secrets_environment_dry_run(monkeypatch):
    monkeypatch.setenv("GH_PAT", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")
    result = update_alertmanager_github_secrets(
        slack_webhook="https://hooks.slack.test/T/B/xxx",
        dry_run=True,
        environment="production",
    )
    assert result["environment"] == "production"
    assert "RAG_ALERTMANAGER_SLACK_WEBHOOK" in result["updated"]


def test_update_alertmanager_github_secrets_requires_token(monkeypatch):
    monkeypatch.delenv("GH_PAT", raising=False)
    monkeypatch.delenv("ALERTMANAGER_GH_PAT", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/rag")
    result = update_alertmanager_github_secrets(
        slack_webhook="https://hooks.slack.test/x", dry_run=False
    )
    assert "no_token" in result["failed"]
