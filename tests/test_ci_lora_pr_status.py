"""LoRA PR comment bot (upsert) testleri."""

from scripts.ci_lora_pr_status import find_bot_comment_id, upsert_pr_comment


def test_find_bot_comment_id():
    marker = "<!-- lora-rollback-bot -->"
    comments = [
        {"id": 1, "body": "hello"},
        {"id": 42, "body": f"{marker}\n### LoRA rollback status\n"},
    ]
    assert find_bot_comment_id(comments, marker) == 42
    assert find_bot_comment_id([], marker) is None


def test_upsert_pr_comment_creates_then_updates():
    calls = []

    def fake_gh(args, payload=None, token=""):
        calls.append({"args": args, "payload": payload})
        path = " ".join(args)
        if "issues/9/comments" in path and "--paginate" in args:
            return {
                "ok": True,
                "stdout": '[{"id": 7, "body": "<!-- lora-rollback-bot -->\\nold"}]',
                "stderr": "",
                "returncode": 0,
            }
        if "issues/comments/7" in path and "PATCH" in args:
            return {"ok": True, "stdout": '{"id": 7}', "stderr": "", "returncode": 0}
        if "issues/9/comments" in path and "POST" in args:
            return {"ok": True, "stdout": '{"id": 99}', "stderr": "", "returncode": 0}
        return {"ok": False, "stdout": "", "stderr": "unexpected", "returncode": 1}

    updated = upsert_pr_comment(
        repo="org/repo",
        pr_number="9",
        body="<!-- lora-rollback-bot -->\nnew",
        marker="<!-- lora-rollback-bot -->",
        token="t",
        gh_api=fake_gh,
    )
    assert updated["posted"] is True
    assert updated["updated"] is True
    assert updated["comment_id"] == 7
    assert any("PATCH" in c["args"] for c in calls)

    calls.clear()

    def fake_gh_empty(args, payload=None, token=""):
        calls.append({"args": args, "payload": payload})
        if "--paginate" in args:
            return {"ok": True, "stdout": "[]", "stderr": "", "returncode": 0}
        return {"ok": True, "stdout": '{"id": 99}', "stderr": "", "returncode": 0}

    created = upsert_pr_comment(
        repo="org/repo",
        pr_number="9",
        body="<!-- lora-rollback-bot -->\nfirst",
        marker="<!-- lora-rollback-bot -->",
        token="t",
        gh_api=fake_gh_empty,
    )
    assert created["posted"] is True
    assert created["updated"] is False
    assert created["comment_id"] == 99
