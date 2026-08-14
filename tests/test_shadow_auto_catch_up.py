from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.cli import main
from rag.store import DualWriteIndex


def test_shadow_compare_auto_catch_up_on_fail(tmp_path: Path, monkeypatch, capsys) -> None:
    dual = MagicMock(spec=DualWriteIndex)
    dual.__class__ = DualWriteIndex  # isinstance(index, DualWriteIndex)

    # Make isinstance work: use a real DualWriteIndex subclass instance via type
    class _FakeDual(DualWriteIndex):
        def __init__(self):
            pass

    dual = _FakeDual.__new__(_FakeDual)

    monkeypatch.setattr("rag.cli.INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr("rag.cli.DOCSTORE_PATH", str(tmp_path / "docstore.json"))

    calls: list[str] = []

    def fake_compare(index, **kwargs):
        calls.append("compare")
        if calls.count("compare") == 1:
            return {"ok": False, "mean_overlap": 0.1}
        return {"ok": True, "mean_overlap": 1.0}

    def fake_catch_up(index):
        calls.append("catch_up")
        return {"ok": True, "synced": 1}

    with patch("rag.store.load_index", return_value=dual):
        with patch(
            "rag.store.dual_write_shadow_compare_from_index",
            side_effect=fake_compare,
        ):
            with patch("rag.store.dual_write_catch_up", side_effect=fake_catch_up):
                code = main(
                    [
                        "migrate-vector",
                        "--shadow-compare",
                        "--auto-catch-up-on-fail",
                    ]
                )
    assert code == 0
    assert calls == ["compare", "catch_up", "compare"]
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["auto_catch_up"] is True
    assert out["retry"]["ok"] is True


def test_shadow_compare_auto_catch_up_still_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    class _FakeDual(DualWriteIndex):
        def __init__(self):
            pass

    dual = _FakeDual.__new__(_FakeDual)

    monkeypatch.setattr("rag.cli.INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr("rag.cli.DOCSTORE_PATH", str(tmp_path / "docstore.json"))

    def fake_compare(index, **kwargs):
        return {"ok": False, "mean_overlap": 0.1}

    def fake_catch_up(index):
        return {"ok": True, "synced": 0}

    with patch("rag.store.load_index", return_value=dual):
        with patch(
            "rag.store.dual_write_shadow_compare_from_index",
            side_effect=fake_compare,
        ):
            with patch("rag.store.dual_write_catch_up", side_effect=fake_catch_up):
                code = main(
                    [
                        "migrate-vector",
                        "--shadow-compare",
                        "--auto-catch-up-on-fail",
                    ]
                )
    assert code != 0
    out = json.loads(capsys.readouterr().out)
    assert out["auto_catch_up"] is True
    assert out["ok"] is False
