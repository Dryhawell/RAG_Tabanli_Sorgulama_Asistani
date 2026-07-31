from rag.cli import build_parser
from rag.metrics import record_metric
from rag.prometheus_sink import observe_metric, prometheus_available, render_prometheus


def test_prometheus_observe_and_dump(monkeypatch):
    if not prometheus_available():
        return  # ortamda paket yoksa atla

    monkeypatch.setattr("rag.prometheus_sink.ENABLE_PROMETHEUS", True)
    observe_metric(
        "query",
        values={"latency_ms": 150.0, "gate_score": 0.55, "no_answer": False},
        tenant_id="acme",
        enabled=True,
    )
    observe_metric("ingest", values={"chunks_added": 3}, enabled=True)
    observe_metric(
        "judge_run",
        values={"mode": "heuristic", "accuracy": 0.9, "ok": True, "soft_fail": False},
        enabled=True,
    )
    observe_metric(
        "judge_run",
        values={"mode": "llm", "accuracy": 0.4, "ok": False, "soft_fail": True},
        enabled=True,
    )
    body = render_prometheus().decode("utf-8")
    assert "rag_queries_total" in body
    assert "rag_ingest_total" in body
    assert "rag_judge_accuracy" in body
    assert "rag_judge_soft_fail_total" in body
    assert "rag_judge_runs_total" in body


def test_record_metric_forwards_to_prometheus(tmp_path, monkeypatch):
    if not prometheus_available():
        return
    path = tmp_path / "m.jsonl"
    monkeypatch.setattr("rag.metrics.ENABLE_METRICS", True)
    monkeypatch.setattr("rag.prometheus_sink.ENABLE_PROMETHEUS", True)
    record_metric(
        "query",
        values={"latency_ms": 10, "gate_score": 0.2, "no_answer": True},
        tenant_id="t1",
        path=str(path),
    )
    assert path.exists()
    body = render_prometheus().decode("utf-8")
    assert "rag_events_total" in body


def test_cli_prometheus_parser():
    parser = build_parser()
    args = parser.parse_args(["prometheus", "--dump"])
    assert args.command == "prometheus"
    assert args.dump is True
