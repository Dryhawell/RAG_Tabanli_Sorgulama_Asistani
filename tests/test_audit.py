import json

from rag.audit import read_audit, write_audit


def test_write_and_read_audit(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr("rag.audit.ENABLE_AUDIT", True)

    write_audit(
        "login_success",
        username="alice",
        tenant_id="acme",
        details={"ok": True},
        path=str(path),
    )
    write_audit(
        "query",
        username="alice",
        tenant_id="acme",
        details={"question": "merhaba"},
        path=str(path),
    )

    rows = read_audit(path=str(path), limit=10)
    assert len(rows) == 2
    assert rows[0]["event"] == "query"
    assert rows[1]["event"] == "login_success"
    assert rows[0]["tenant_id"] == "acme"

    only_login = read_audit(path=str(path), event="login_success")
    assert len(only_login) == 1

    # dosya gerçek JSONL
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "login_success"


def test_audit_disabled_skips_disk(tmp_path):
    path = tmp_path / "audit.jsonl"
    rec = write_audit(
        "ingest",
        username="bob",
        details={"n": 1},
        path=str(path),
        enabled=False,
    )
    assert rec["event"] == "ingest"
    assert not path.exists()
