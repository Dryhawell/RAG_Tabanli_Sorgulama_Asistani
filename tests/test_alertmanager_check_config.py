from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from rag.alertmanager_ops import check_alertmanager_config, _structural_alertmanager_check
from rag.cli import main


def test_structural_alertmanager_check_ok(tmp_path: Path) -> None:
    cfg = tmp_path / "am.yml"
    cfg.write_text(
        "route:\n  receiver: default\nreceivers:\n  - name: default\n",
        encoding="utf-8",
    )
    result = _structural_alertmanager_check(str(cfg))
    assert result["ok"] is True
    assert result["method"] == "structural"
    assert result["has_route"] is True
    assert result["has_receivers"] is True


def test_structural_alertmanager_check_rejects_missing_route(tmp_path: Path) -> None:
    cfg = tmp_path / "am.yml"
    cfg.write_text("receivers:\n  - name: default\n", encoding="utf-8")
    result = _structural_alertmanager_check(str(cfg))
    assert result["ok"] is False
    assert result["method"] == "structural"
    assert result["error"] == "missing_route_or_receivers"


def test_check_alertmanager_config_falls_back_to_structural(tmp_path: Path) -> None:
    cfg = tmp_path / "am.yml"
    cfg.write_text(
        "route:\n  receiver: default\nreceivers:\n  - name: default\n",
        encoding="utf-8",
    )
    with patch("shutil.which", return_value=None):
        result = check_alertmanager_config(str(cfg))
    assert result["ok"] is True
    assert result["method"] == "structural"


def test_cli_alertmanager_check_config(tmp_path: Path, capsys) -> None:
    cfg = tmp_path / "am.yml"
    cfg.write_text(
        "route:\n  receiver: default\nreceivers:\n  - name: default\n",
        encoding="utf-8",
    )
    with patch("shutil.which", return_value=None):
        code = main(["alertmanager", "--check-config", "--output", str(cfg)])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["method"] == "structural"
