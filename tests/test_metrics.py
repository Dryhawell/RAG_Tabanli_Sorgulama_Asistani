import json

from rag.metrics import record_metric, read_metrics, summarize_metrics


def test_record_and_summarize_query_metrics(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr("rag.metrics.ENABLE_METRICS", True)

    record_metric(
        "query",
        username="alice",
        tenant_id="acme",
        values={"gate_score": 0.8, "n_sources": 3, "no_answer": False, "latency_ms": 120.5},
        path=str(path),
    )
    record_metric(
        "query",
        username="alice",
        values={"gate_score": 0.1, "n_sources": 0, "no_answer": True, "latency_ms": 40.0},
        path=str(path),
    )
    record_metric(
        "ingest",
        values={"chunks_added": 12, "files": 2},
        path=str(path),
    )

    rows = read_metrics(path=str(path))
    assert len(rows) == 3

    summary = summarize_metrics(path=str(path))
    assert summary["query"]["count"] == 2
    assert summary["query"]["no_answer_count"] == 1
    assert summary["query"]["no_answer_rate"] == 0.5
    assert summary["ingest"]["chunks_added"] == 12
    assert summary["total_events"] == 3


def test_metrics_disabled_skips_disk(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rec = record_metric("query", values={"latency_ms": 1}, path=str(path), enabled=False)
    assert rec["kind"] == "query"
    assert not path.exists()


def test_summarize_empty_file(tmp_path):
    path = tmp_path / "missing.jsonl"
    summary = summarize_metrics(path=str(path))
    assert summary["total_events"] == 0
    assert summary["query"]["count"] == 0
