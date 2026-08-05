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
