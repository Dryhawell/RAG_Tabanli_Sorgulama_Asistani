"""Prometheus metrik sink (opsiyonel).

`RAG_ENABLE_PROMETHEUS=1` iken `record_metric` çağrıları counter/histogram'a yansır.
HTTP scrape endpoint: `python -m rag.cli prometheus` veya `ensure_prometheus_server()`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from app.config import ENABLE_PROMETHEUS, PROMETHEUS_ADDR, PROMETHEUS_PORT

_lock = threading.Lock()
_server_started = False
_prometheus_available: Optional[bool] = None

# Lazy metric objects
_query_total = None
_query_latency = None
_query_gate = None
_ingest_total = None
_ingest_chunks = None
_rebuild_total = None
_delete_total = None
_events_total = None
_push_revoked_total = None
_push_prune_total = None
_judge_accuracy = None
_judge_soft_fail_total = None
_judge_runs_total = None
_judge_ack_audit_total = None
_llm_tokens_total = None
_llm_calls_total = None
_dual_write_lag = None
_dual_write_error_last: dict = {}
_dual_write_errors = None
_dual_write_catch_up_sources = None
_dual_write_catch_up_chunks = None
_dual_write_catch_up_remaining = None
_dual_write_shadow_overlap = None
_dual_write_webhook_total = None
_dual_write_webhook_dlq_depth = None
_dual_write_webhook_dlq_quarantine_depth = None
_dual_write_webhook_circuit_open = None
_inhibit_equal_canary_resolve_total = None


def prometheus_available() -> bool:
    global _prometheus_available
    if _prometheus_available is None:
        try:
            import prometheus_client  # noqa: F401

            _prometheus_available = True
        except Exception:
            _prometheus_available = False
    return bool(_prometheus_available)


def _ensure_metrics():
    global _query_total, _query_latency, _query_gate
    global _ingest_total, _ingest_chunks, _rebuild_total, _delete_total, _events_total
    global _push_revoked_total, _push_prune_total
    global _judge_accuracy, _judge_soft_fail_total, _judge_runs_total
    global _judge_ack_audit_total
    global _llm_tokens_total, _llm_calls_total
    global _dual_write_lag, _dual_write_errors
    global _dual_write_catch_up_sources, _dual_write_catch_up_chunks
    global _dual_write_catch_up_remaining
    global _dual_write_shadow_overlap
    global _dual_write_webhook_total, _dual_write_webhook_dlq_depth
    global _dual_write_webhook_dlq_quarantine_depth
    global _dual_write_webhook_circuit_open
    global _inhibit_equal_canary_resolve_total
    if _events_total is not None:
        return
    if not prometheus_available():
        raise RuntimeError("prometheus_client kurulu değil")

    from prometheus_client import Counter, Gauge, Histogram

    _events_total = Counter(
        "rag_events_total",
        "Toplam RAG olay sayısı",
        ["kind"],
    )
    _query_total = Counter(
        "rag_queries_total",
        "Sorgu sayısı",
        ["no_answer", "tenant"],
    )
    _query_latency = Histogram(
        "rag_query_latency_seconds",
        "Uçtan uca sorgu gecikmesi (saniye)",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _query_gate = Histogram(
        "rag_query_gate_score",
        "Retrieval gate (cosine) skoru",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    _ingest_total = Counter("rag_ingest_total", "Ingest işlem sayısı")
    _ingest_chunks = Counter("rag_ingest_chunks_total", "Ingest edilen chunk sayısı")
    _rebuild_total = Counter("rag_rebuild_total", "Rebuild işlem sayısı")
    _delete_total = Counter("rag_delete_total", "Silme işlem sayısı")
    _push_revoked_total = Counter(
        "rag_push_token_revoked_total",
        "Geçersiz push token revoke sayısı",
        ["provider", "reason"],
    )
    _push_prune_total = Counter(
        "rag_push_token_prune_total",
        "Push token prune ile silinen kayıt sayısı",
        ["kind"],
    )
    _judge_accuracy = Gauge(
        "rag_judge_accuracy",
        "Son judge koşusu accuracy",
        ["mode"],
    )
    _judge_soft_fail_total = Counter(
        "rag_judge_soft_fail_total",
        "Judge soft-fail sayısı",
        ["mode"],
    )
    _judge_runs_total = Counter(
        "rag_judge_runs_total",
        "Judge koşu sayısı",
        ["mode", "ok"],
    )
    _judge_ack_audit_total = Counter(
        "rag_judge_ack_audit_total",
        "Judge soft-fail ack audit olayları",
        ["event"],
    )
    _llm_tokens_total = Counter(
        "rag_llm_tokens_total",
        "LLM token kullanımı",
        ["provider", "model", "token_type"],
    )
    _llm_calls_total = Counter(
        "rag_llm_calls_total",
        "LLM çağrı sayısı",
        ["provider", "model"],
    )
    _dual_write_lag = Gauge(
        "rag_vector_dual_write_lag",
        "Primary-secondary chunk size farkı (dual-write)",
        ["secondary_backend"],
    )
    _dual_write_errors = Counter(
        "rag_vector_dual_write_errors_total",
        "Dual-write secondary hata sayısı",
        ["secondary_backend"],
    )
    _dual_write_catch_up_sources = Counter(
        "rag_vector_dual_write_catch_up_sources_total",
        "Dual-write catch-up kaynak sonuçları",
        ["result", "secondary_backend"],
    )
    _dual_write_catch_up_chunks = Counter(
        "rag_vector_dual_write_catch_up_chunks_total",
        "Dual-write catch-up kopyalanan chunk",
        ["secondary_backend"],
    )
    _dual_write_catch_up_remaining = Gauge(
        "rag_vector_dual_write_catch_up_remaining",
        "Catch-up bekleyen kaynak sayısı",
        ["secondary_backend"],
    )
    _dual_write_shadow_overlap = Gauge(
        "rag_vector_dual_write_shadow_overlap",
        "Dual-write shadow-read mean overlap",
        ["secondary_backend"],
    )
    _dual_write_webhook_total = Counter(
        "rag_dual_write_webhook_total",
        "Alertmanager dual-write webhook sonuçları",
        ["result"],
    )
    _dual_write_webhook_dlq_depth = Gauge(
        "rag_dual_write_webhook_dlq_depth",
        "Dual-write webhook dead-letter queue derinliği",
    )
    _dual_write_webhook_dlq_quarantine_depth = Gauge(
        "rag_dual_write_webhook_dlq_quarantine_depth",
        "Dual-write webhook DLQ quarantine derinliği",
    )
    _dual_write_webhook_circuit_open = Gauge(
        "rag_dual_write_webhook_circuit_open",
        "Dual-write webhook circuit breaker açık mı (1/0)",
    )
    _inhibit_equal_canary_resolve_total = Counter(
        "rag_inhibit_equal_canary_resolve_total",
        "Inhibit-equal canary Slack thread/resolve outcomes",
        ["result", "via"],
    )


def _observe_with_exemplar(metric, value: float) -> None:
    """Histogram.observe; destekleniyorsa OTel trace exemplar ekler."""
    exemplar = None
    try:
        from rag.otel import current_trace_exemplar

        exemplar = current_trace_exemplar()
    except Exception:
        exemplar = None
    if exemplar:
        try:
            metric.observe(value, exemplar=exemplar)
            return
        except TypeError:
            pass
        except Exception:
            pass
    metric.observe(value)


def observe_metric(
    kind: str,
    *,
    values: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> None:
    """JSONL kaydına ek olarak Prometheus metriklerini günceller."""
    use = ENABLE_PROMETHEUS if enabled is None else enabled
    if not use:
        return
    if not prometheus_available():
        return

    vals = values or {}
    tenant = (tenant_id or "default").strip() or "default"

    with _lock:
        _ensure_metrics()
        assert _events_total is not None
        _events_total.labels(kind=kind).inc()

        if kind == "query":
            no_ans = "1" if vals.get("no_answer") else "0"
            _query_total.labels(no_answer=no_ans, tenant=tenant).inc()
            if vals.get("latency_ms") is not None:
                _observe_with_exemplar(
                    _query_latency, float(vals["latency_ms"]) / 1000.0
                )
            if vals.get("gate_score") is not None:
                _observe_with_exemplar(_query_gate, float(vals["gate_score"]))
        elif kind == "ingest":
            _ingest_total.inc()
            chunks = int(vals.get("chunks_added") or 0)
            if chunks:
                _ingest_chunks.inc(chunks)
        elif kind == "rebuild":
            _rebuild_total.inc()
        elif kind == "delete":
            _delete_total.inc()
        elif kind == "push_token_revoked":
            assert _push_revoked_total is not None
            _push_revoked_total.labels(
                provider=str(vals.get("provider") or "unknown"),
                reason=str(vals.get("reason") or "revoked"),
            ).inc()
        elif kind == "push_token_prune":
            assert _push_prune_total is not None
            revoked_n = int(vals.get("revoked_pruned") or 0)
            expired_n = int(vals.get("expired_pruned") or 0)
            if revoked_n:
                _push_prune_total.labels(kind="revoked").inc(revoked_n)
            if expired_n:
                _push_prune_total.labels(kind="expired").inc(expired_n)
        elif kind == "judge_run":
            assert _judge_accuracy is not None
            assert _judge_soft_fail_total is not None
            assert _judge_runs_total is not None
            mode = str(vals.get("mode") or "heuristic")
            ok_label = "1" if vals.get("ok") else "0"
            _judge_runs_total.labels(mode=mode, ok=ok_label).inc()
            if vals.get("accuracy") is not None:
                try:
                    _judge_accuracy.labels(mode=mode).set(float(vals["accuracy"]))
                except (TypeError, ValueError):
                    pass
            if vals.get("soft_fail"):
                _judge_soft_fail_total.labels(mode=mode).inc()
        elif kind == "judge_ack_audit":
            assert _judge_ack_audit_total is not None
            event = str(vals.get("event") or "unknown")
            _judge_ack_audit_total.labels(event=event).inc()
        elif kind == "llm_usage":
            assert _llm_tokens_total is not None
            assert _llm_calls_total is not None
            provider = str(vals.get("provider") or "unknown")
            model = str(vals.get("model") or "unknown")
            _llm_calls_total.labels(provider=provider, model=model).inc()
            for token_type, key in (
                ("prompt", "prompt_tokens"),
                ("completion", "completion_tokens"),
                ("total", "total_tokens"),
            ):
                raw = vals.get(key)
                if raw is None:
                    continue
                try:
                    n = float(raw)
                except (TypeError, ValueError):
                    continue
                if n > 0:
                    _llm_tokens_total.labels(
                        provider=provider, model=model, token_type=token_type
                    ).inc(n)
        elif kind == "vector_dual_write_lag":
            assert _dual_write_lag is not None
            assert _dual_write_errors is not None
            secondary = str(vals.get("secondary_backend") or "secondary")
            try:
                _dual_write_lag.labels(secondary_backend=secondary).set(
                    float(vals.get("lag") or 0)
                )
            except (TypeError, ValueError):
                pass
            try:
                err_n = int(vals.get("secondary_error_count") or 0)
            except (TypeError, ValueError):
                err_n = 0
            last = int(_dual_write_error_last.get(secondary, 0) or 0)
            if err_n > last:
                _dual_write_errors.labels(secondary_backend=secondary).inc(err_n - last)
            _dual_write_error_last[secondary] = err_n
        elif kind == "vector_dual_write_catch_up":
            assert _dual_write_catch_up_sources is not None
            assert _dual_write_catch_up_chunks is not None
            assert _dual_write_catch_up_remaining is not None
            secondary = str(vals.get("secondary_backend") or "secondary")
            result = str(vals.get("result") or "unknown")
            if result in {"fixed", "failed"}:
                _dual_write_catch_up_sources.labels(
                    result=result, secondary_backend=secondary
                ).inc()
            try:
                chunks = int(vals.get("copied_chunks") or 0)
            except (TypeError, ValueError):
                chunks = 0
            if chunks > 0 and result == "fixed":
                _dual_write_catch_up_chunks.labels(secondary_backend=secondary).inc(
                    chunks
                )
            try:
                remaining = float(vals.get("remaining") or 0)
            except (TypeError, ValueError):
                remaining = 0.0
            _dual_write_catch_up_remaining.labels(secondary_backend=secondary).set(
                remaining
            )
        elif kind == "vector_dual_write_shadow":
            assert _dual_write_shadow_overlap is not None
            secondary = str(vals.get("secondary_backend") or "secondary")
            try:
                _dual_write_shadow_overlap.labels(secondary_backend=secondary).set(
                    float(vals.get("mean_overlap") or 0)
                )
            except (TypeError, ValueError):
                pass
        elif kind == "dual_write_webhook":
            assert _dual_write_webhook_total is not None
            assert _dual_write_webhook_dlq_depth is not None
            assert _dual_write_webhook_dlq_quarantine_depth is not None
            assert _dual_write_webhook_circuit_open is not None
            result = str(vals.get("result") or "unknown")
            _dual_write_webhook_total.labels(result=result).inc()
            if "dlq_depth" in vals:
                try:
                    _dual_write_webhook_dlq_depth.set(float(vals.get("dlq_depth") or 0))
                except (TypeError, ValueError):
                    pass
            if "dlq_quarantine_depth" in vals:
                try:
                    _dual_write_webhook_dlq_quarantine_depth.set(
                        float(vals.get("dlq_quarantine_depth") or 0)
                    )
                except (TypeError, ValueError):
                    pass
            try:
                _dual_write_webhook_circuit_open.set(
                    1.0 if vals.get("circuit_open") else 0.0
                )
            except Exception:
                pass
        elif kind == "inhibit_equal_canary_resolve":
            assert _inhibit_equal_canary_resolve_total is not None
            result = str(vals.get("result") or "unknown")
            via = str(vals.get("via") or "none")
            _inhibit_equal_canary_resolve_total.labels(result=result, via=via).inc()


def render_prometheus() -> bytes:
    """Prometheus text exposition formatında metrik çıktısı."""
    if not prometheus_available():
        return b"# prometheus_client not installed\n"
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    _ = CONTENT_TYPE_LATEST
    with _lock:
        if ENABLE_PROMETHEUS:
            try:
                _ensure_metrics()
            except RuntimeError:
                pass
    return generate_latest()


def ensure_prometheus_server(
    *,
    port: Optional[int] = None,
    addr: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Bir kez /metrics HTTP sunucusunu başlatır. True = çalışıyor."""
    global _server_started
    use = ENABLE_PROMETHEUS if enabled is None else enabled
    if not use:
        return False
    if not prometheus_available():
        return False

    with _lock:
        if _server_started:
            return True
        from prometheus_client import start_http_server

        _ensure_metrics()
        start_http_server(port or PROMETHEUS_PORT, addr=addr or PROMETHEUS_ADDR)
        _server_started = True
        return True
