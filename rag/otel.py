"""Opsiyonel OpenTelemetry stub (OTLP endpoint varsa).

`OTEL_EXPORTER_OTLP_ENDPOINT` veya `RAG_OTEL_ENDPOINT` set değilse no-op.
`opentelemetry-api` / `opentelemetry-sdk` yoksa sessizce kapanır (hard dep yok).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_initialized = False
_tracer = None
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


def _ensure_tracer():
    global _initialized, _tracer
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

        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "rag-assistant"),
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otel_endpoint())
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("rag")
        return _tracer
    except Exception:
        return None


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
