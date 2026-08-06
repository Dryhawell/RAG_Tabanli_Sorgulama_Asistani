from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rag.alertmanager_ops import (
    apply_inhibit_equal_with_gate,
    canonicalize_inhibit_yaml,
    diff_inhibit_rules,
    render_inhibit_rules_yaml,
    write_inhibit_rules,
)
from scripts.ci_inhibit_equal_apply import resolve_dry_run


def test_diff_inhibit_unchanged(tmp_path: Path) -> None:
    text = render_inhibit_rules_yaml(
        [
            {
                "source_matchers": ["severity = critical"],
                "target_matchers": ["severity = warning"],
                "equal": ["alertname", "service"],
            }
        ]
    )
    cur = tmp_path / "cur.yml"
    cur.write_text(text, encoding="utf-8")
    diff = diff_inhibit_rules(str(cur), text)
    assert diff["ok"] is True
    assert diff["changed"] is False


def test_diff_inhibit_changed(tmp_path: Path) -> None:
    old = render_inhibit_rules_yaml(
        [
            {
                "source_matchers": ["severity = critical"],
                "target_matchers": ["severity = warning"],
                "equal": ["alertname", "service"],
            }
        ]
    )
    new = render_inhibit_rules_yaml(
        [
            {
                "source_matchers": ["severity = critical"],
                "target_matchers": ["severity = warning"],
                "equal": ["alertname", "service", "team"],
            }
        ]
    )
    cur = tmp_path / "cur.yml"
    cur.write_text(old, encoding="utf-8")
    diff = diff_inhibit_rules(str(cur), new)
    assert diff["changed"] is True
    assert "team" in diff["unified_diff"]


def test_canonicalize_strips_comments() -> None:
    raw = "# greeting-line\ninhibit_rules:\n  - equal: [\"a\"]\n\n"
    out = canonicalize_inhibit_yaml(raw)
    assert "greeting-line" not in out
    assert "inhibit_rules:" in out
    assert out.startswith("inhibit_rules:")


def test_apply_inhibit_equal_dry_run_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "inhibit.yml"
    # seed different content so changed=True
    out.write_text("inhibit_rules: []\n", encoding="utf-8")
    with patch(
        "rag.alertmanager_ops.check_alertmanager_config",
        return_value={"ok": True, "method": "structural"},
    ):
        with patch(
            "rag.alertmanager_ops.render_alertmanager_config",
            return_value={"ok": True, "output": str(tmp_path / "am.yml")},
        ):
            report = apply_inhibit_equal_with_gate(
                output=str(out),
                from_live=False,
                dry_run=True,
            )
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["applied"] is False
    # file unchanged because dry_run
    assert out.read_text(encoding="utf-8") == "inhibit_rules: []\n"


def test_apply_inhibit_equal_writes_when_gate_ok(tmp_path: Path) -> None:
    out = tmp_path / "inhibit.yml"
    out.write_text("inhibit_rules: []\n", encoding="utf-8")
    with patch(
        "rag.alertmanager_ops.check_alertmanager_config",
        return_value={"ok": True, "method": "amtool"},
    ):
        with patch(
            "rag.alertmanager_ops.render_alertmanager_config",
            return_value={"ok": True, "output": str(tmp_path / "am.yml")},
        ):
            report = apply_inhibit_equal_with_gate(
                output=str(out),
                from_live=False,
                dry_run=False,
            )
    assert report["ok"] is True
    assert report["applied"] is True
    assert "source_matchers:" in out.read_text(encoding="utf-8")


def test_resolve_dry_run_auto(monkeypatch) -> None:
    monkeypatch.setenv("INHIBIT_EQUAL_APPLY", "auto")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    assert resolve_dry_run() is True
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    assert resolve_dry_run() is False
    monkeypatch.setenv("INHIBIT_EQUAL_APPLY", "dry-run")
    assert resolve_dry_run() is True


def test_ci_workflow_has_apply_job() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "inhibit-equal-apply" in text
    assert "ci_inhibit_equal_apply.py" in text
    assert "INHIBIT_EQUAL_APPLY" in text
    assert "INHIBIT_EQUAL_APPLY_PR_COMMENT_POST" in text
    assert "inhibit-equal-apply-bot" in Path("scripts/ci_inhibit_equal_apply.py").read_text(
        encoding="utf-8"
    )
    assert "--diff-inhibit" in Path("rag/cli.py").read_text(encoding="utf-8") or (
        "--apply-equal" in Path("rag/cli.py").read_text(encoding="utf-8")
    )


def test_build_inhibit_equal_apply_comment() -> None:
    from scripts.ci_inhibit_equal_apply import (
        INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER,
        build_inhibit_equal_apply_comment,
        write_apply_preview_artifacts,
    )

    report = {
        "ok": True,
        "dry_run": True,
        "reason": "dry_run_changed",
        "output": "grafana/inhibit_rules.generated.yml",
        "generated": {"equal": ["alertname", "service"]},
        "diff": {
            "changed": True,
            "unified_diff": "--- a\n+++ b\n+equal: [team]\n",
        },
    }
    md = build_inhibit_equal_apply_comment(report)
    assert INHIBIT_EQUAL_APPLY_PR_COMMENT_MARKER in md
    assert "```diff" in md
    assert "equal: [team]" in md
    assert "dry-run preview" in md.lower()

    unchanged = build_inhibit_equal_apply_comment(
        {
            "ok": True,
            "dry_run": True,
            "reason": "unchanged",
            "generated": {"equal": ["alertname"]},
            "diff": {"changed": False, "unified_diff": ""},
        }
    )
    assert "No inhibit diff" in unchanged
