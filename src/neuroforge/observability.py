"""OpenTelemetry setup (section 49). Exists to operate NeuroForge itself, not as a general-purpose
observability product — traces cover experiment execution (one span per generation) and API
request handling, nothing else. Exports to OTLP if `OTEL_EXPORTER_OTLP_ENDPOINT` is set, otherwise
to the console (visible locally without standing up a collector)."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

_configured = False


def configure_tracing(service_name: str) -> None:
    """Idempotent: safe to call from every entrypoint (API, CLI, worker) without double-exporting."""
    global _configured
    if _configured:
        return
    _configured = True

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    elif os.environ.get("NEUROFORGE_TRACE_TO_CONSOLE", "").lower() in ("1", "true"):
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    # else: no exporter configured — spans are created but dropped, which is the right default
    # for tests/CI (no noisy stdout, no dependency on a collector being reachable).

    trace.set_tracer_provider(provider)


def get_tracer(name: str):
    return trace.get_tracer(name)
