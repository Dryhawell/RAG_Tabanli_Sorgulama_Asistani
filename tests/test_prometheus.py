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
    observe_metric(
        "llm_usage",
        values={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
        enabled=True,
    )
    observe_metric(
        "vector_dual_write_lag",
        values={
            "lag": 2,
            "primary_size": 10,
            "secondary_size": 8,
            "secondary_error_count": 1,
            "secondary_backend": "qdrant",
            "ok": False,
        },
        enabled=True,
    )
    # aynı error_count tekrar → counter artmamalı (delta)
    observe_metric(
        "vector_dual_write_lag",
        values={
            "lag": 0,
            "secondary_error_count": 1,
            "secondary_backend": "qdrant",
            "ok": True,
        },
        enabled=True,
    )
    observe_metric(
        "vector_dual_write_catch_up",
        values={
            "result": "fixed",
            "copied_chunks": 3,
            "remaining": 1,
            "secondary_backend": "qdrant",
        },
        enabled=True,
    )
    observe_metric(
        "vector_dual_write_catch_up",
        values={
            "result": "done",
            "remaining": 0,
            "secondary_backend": "qdrant",
        },
        enabled=True,
    )
    observe_metric(
        "vector_dual_write_shadow",
        values={
            "mean_overlap": 0.95,
            "min_overlap": 0.9,
            "secondary_backend": "qdrant",
            "ok": True,
        },
        enabled=True,
    )
    observe_metric(
        "dual_write_webhook",
        values={
            "result": "failed",
            "reason": "action_failed",
            "actions": 1,
            "circuit_open": True,
            "dlq_enqueued": True,
            "dlq_depth": 3,
            "dlq_quarantine_depth": 2,
        },
        enabled=True,
    )
    observe_metric(
        "inhibit_equal_canary_resolve",
        values={"result": "ok", "via": "thread"},
        enabled=True,
    )
    observe_metric(
        "inhibit_equal_canary_silence",
        values={"result": "ok"},
        enabled=True,
    )
    observe_metric(
        "judge_ack_digest_msgref_reconcile",
        values={
            "checked": 3,
            "kept": 1,
            "dropped": 1,
            "repaired": 1,
            "errors": 0,
        },
        enabled=True,
    )
    observe_metric(
        "webhook_signing_sidecar_forward",
        values={"mode": "dlq_quarantine", "result": "ok"},
        enabled=True,
    )
    observe_metric(
        "webhook_signing_sidecar_cert_expiry",
        values={"role": "server", "days_left": 12.5},
        enabled=True,
    )
    body = render_prometheus().decode("utf-8")
    assert "rag_queries_total" in body
    assert "rag_ingest_total" in body
    assert "rag_judge_accuracy" in body
    assert "rag_judge_soft_fail_total" in body
    assert "rag_judge_runs_total" in body
    assert "rag_llm_tokens_total" in body
    assert "rag_llm_calls_total" in body
    assert "rag_vector_dual_write_lag" in body
    assert "rag_vector_dual_write_errors_total" in body
    assert "rag_vector_dual_write_catch_up_sources_total" in body
    assert "rag_vector_dual_write_catch_up_chunks_total" in body
    assert "rag_vector_dual_write_catch_up_remaining" in body
    assert "rag_vector_dual_write_shadow_overlap" in body
    assert "rag_dual_write_webhook_total" in body
    assert "rag_dual_write_webhook_dlq_depth" in body
    assert "rag_dual_write_webhook_dlq_quarantine_depth" in body
    assert "rag_dual_write_webhook_circuit_open" in body
    assert "rag_inhibit_equal_canary_resolve_total" in body
    assert "rag_inhibit_equal_canary_silence_total" in body
    assert "rag_judge_ack_digest_msgref_reconcile_total" in body
    assert "rag_judge_ack_digest_msgref_reconcile_checked" in body
    assert "rag_judge_ack_digest_msgref_reconcile_dropped" in body
    assert "rag_judge_ack_digest_msgref_reconcile_repaired" in body
    assert "rag_webhook_signing_sidecar_forward_total" in body
    assert "rag_webhook_signing_sidecar_cert_expiry_days" in body


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
