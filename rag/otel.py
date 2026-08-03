"""Opsiyonel OpenTelemetry OTLP export.

`OTEL_EXPORTER_OTLP_ENDPOINT` veya `RAG_OTEL_ENDPOINT` set değilse no-op.
`opentelemetry-api` / `opentelemetry-sdk` yoksa sessizce kapanır.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_initialized = False
_tracer = None
_provider = None
_enabled: Optional[bool] = None


def otel_endpoint() -> str:
    return (
        os.environ.get("RAG_OTEL_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )


def otel_enabled() -> bool:
    global _enabled
    if _enabled is not None:
        return _enabled
    if not otel_endpoint():
        _enabled = False
        return False
    try:
        import opentelemetry  # noqa: F401

        _enabled = True
    except Exception:
        _enabled = False
    return _enabled


def _parse_resource_attributes(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v:
            out[k] = v
    return out


def build_resource_attrs() -> Dict[str, str]:
    attrs: Dict[str, str] = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "rag-assistant").strip()
        or "rag-assistant",
    }
    version = os.environ.get("OTEL_SERVICE_VERSION", "").strip()
    if version:
        attrs["service.version"] = version
    env = (
        os.environ.get("RAG_ENVIRONMENT", "").strip()
        or os.environ.get("OTEL_ENVIRONMENT", "").strip()
        or os.environ.get("ENVIRONMENT", "").strip()
    )
    if env:
        attrs["deployment.environment"] = env
    attrs.update(_parse_resource_attributes(os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")))
    return attrs


def _ensure_tracer():
    global _initialized, _tracer, _provider
    if _initialized:
        return _tracer
    _initialized = True
    if not otel_enabled():
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except Exception:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

        existing = trace.get_tracer_provider()
        # ProxyTracerProvider (default) has no force_flush; real providers do.
        if existing is not None and hasattr(existing, "add_span_processor"):
            _provider = existing
            _tracer = trace.get_tracer("rag")
            return _tracer

        resource = Resource.create(build_resource_attrs())
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otel_endpoint())
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider = provider
        _tracer = trace.get_tracer("rag")
        return _tracer
    except Exception:
        return None


def force_flush(timeout_millis: int = 5000) -> bool:
    provider = _provider
    if provider is None:
        return False
    try:
        return bool(provider.force_flush(timeout_millis))
    except Exception:
        return False


def shutdown() -> None:
    global _initialized, _tracer, _provider, _enabled
    provider = _provider
    _tracer = None
    _provider = None
    _initialized = False
    _enabled = None
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception:
        pass


def reset_for_tests() -> None:
    """Test izolasyonu: tracer state sıfırla (export kapatmadan)."""
    global _initialized, _tracer, _provider, _enabled
    _initialized = False
    _tracer = None
    _provider = None
    _enabled = None


@contextmanager
def start_span(
    name: str,
    *,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Span context; OTEL yoksa null context."""
    tracer = _ensure_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes and span is not None:
            for k, v in attributes.items():
                if v is None:
                    continue
                try:
                    span.set_attribute(str(k), v)
                except Exception:
                    pass
        yield span


def set_span_attrs(span: Any, attributes: Dict[str, Any]) -> None:
    if span is None:
        return
    for k, v in attributes.items():
        if v is None:
            continue
        try:
            span.set_attribute(str(k), v)
        except Exception:
            pass
