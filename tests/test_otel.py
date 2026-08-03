"""OpenTelemetry stub / OTLP wiring tests."""

from rag.otel import (
    build_resource_attrs,
    force_flush,
    otel_enabled,
    otel_endpoint,
    reset_for_tests,
    set_span_attrs,
    shutdown,
    start_span,
)


def test_otel_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("RAG_OTEL_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    reset_for_tests()
    assert otel_endpoint() == ""
    assert otel_enabled() is False
    with start_span("rag.test", attributes={"k": "v"}) as span:
        assert span is None
        set_span_attrs(span, {"x": 1})
    assert force_flush() is False
    shutdown()


def test_otel_endpoint_prefers_rag_env(monkeypatch):
    monkeypatch.setenv("RAG_OTEL_ENDPOINT", "http://otel:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://other:4318")
    reset_for_tests()
    assert otel_endpoint() == "http://otel:4318"


def test_build_resource_attrs(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "rag-test")
    monkeypatch.setenv("OTEL_SERVICE_VERSION", "1.2.3")
    monkeypatch.setenv("RAG_ENVIRONMENT", "staging")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "team=platform,region=eu")
    attrs = build_resource_attrs()
    assert attrs["service.name"] == "rag-test"
    assert attrs["service.version"] == "1.2.3"
    assert attrs["deployment.environment"] == "staging"
    assert attrs["team"] == "platform"
    assert attrs["region"] == "eu"


def test_otel_enabled_path_creates_span(monkeypatch):
    pytest = __import__("pytest")
    otel = pytest.importorskip("opentelemetry")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    monkeypatch.setenv("RAG_OTEL_ENDPOINT", "http://127.0.0.1:4318")
    reset_for_tests()

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Force enabled + use existing provider path
    import rag.otel as mod

    mod._enabled = True
    mod._initialized = False
    mod._tracer = None
    mod._provider = None

    with start_span("rag.test.enabled", attributes={"rag.k": 1}) as span:
        assert span is not None
        set_span_attrs(span, {"rag.k2": "v"})

    spans = exporter.get_finished_spans()
    assert any(s.name == "rag.test.enabled" for s in spans)
    shutdown()
    # restore a clean provider for other tests
    trace.set_tracer_provider(TracerProvider())
    reset_for_tests()
    _ = otel
