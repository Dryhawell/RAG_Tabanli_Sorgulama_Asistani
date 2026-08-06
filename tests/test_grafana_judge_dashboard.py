"""Grafana dashboard artifact smoke."""

import json
from pathlib import Path


def test_rag_judge_grafana_dashboard_json():
    path = Path("grafana/dashboards/rag_judge.json")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("uid") == "rag-judge-soft-fail"
    titles = [p.get("title") for p in data.get("panels") or []]
    assert any("accuracy" in (t or "").lower() for t in titles)
    assert any("soft-fail" in (t or "").lower() or "Soft-fail" in (t or "") for t in titles)
    exprs = []
    for panel in data.get("panels") or []:
        for t in panel.get("targets") or []:
            if t.get("expr"):
                exprs.append(t["expr"])
    joined = "\n".join(exprs)
    assert "rag_judge_accuracy" in joined
    assert "rag_judge_soft_fail_total" in joined
    assert "rag_judge_runs_total" in joined
