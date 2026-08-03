"""OpenTelemetry stub: endpoint yoksa no-op."""

from rag.otel import otel_enabled, otel_endpoint, set_span_attrs, start_span


def test_otel_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("RAG_OTEL_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    import rag.otel as otel

    otel._enabled = None
    otel._initialized = False
    otel._tracer = None
    assert otel_endpoint() == ""
    assert otel_enabled() is False
    with start_span("rag.test", attributes={"k": "v"}) as span:
        assert span is None
        set_span_attrs(span, {"x": 1})


def test_otel_endpoint_prefers_rag_env(monkeypatch):
    monkeypatch.setenv("RAG_OTEL_ENDPOINT", "http://otel:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://other:4318")
    import rag.otel as otel

    otel._enabled = None
    assert otel_endpoint() == "http://otel:4318"
