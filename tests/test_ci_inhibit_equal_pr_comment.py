from __future__ import annotations

import json
from pathlib import Path

from scripts.ci_inhibit_equal_pr_comment import (
    INHIBIT_EQUAL_PR_COMMENT_MARKER,
    build_inhibit_equal_comment,
    write_artifacts,
)


def test_build_inhibit_equal_comment_and_artifacts(tmp_path: Path) -> None:
    tune = {
        "ok": True,
        "equal": ["alertname", "service", "secondary_backend"],
        "source": "static",
        "live_ok": False,
        "live_error": "connection refused",
        "live_alerts": 0,
        "static_label_sets": 4,
        "label_sets": 4,
    }
    md = build_inhibit_equal_comment(tune)
    assert INHIBIT_EQUAL_PR_COMMENT_MARKER in md
    assert "secondary_backend" in md
    assert "live_error" in md

    written = write_artifacts(tune, base=str(tmp_path))
    assert Path(written["tune"]).is_file()
    data = json.loads(Path(written["tune"]).read_text(encoding="utf-8"))
    assert data["equal"] == tune["equal"]
    comment = Path(written["comment"]).read_text(encoding="utf-8")
    assert "inhibit equal tune" in comment.lower()
