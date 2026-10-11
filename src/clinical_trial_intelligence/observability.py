from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from clinical_trial_intelligence.config import settings

run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
batch_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("batch_id", default="")
nct_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("nct_id", default="")
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

skill_confidence_histogram = None
mcp_call_duration = None
batch_success_rate = None
constitution_flags_counter = None


def setup_observability() -> None:
    global skill_confidence_histogram, mcp_call_duration, batch_success_rate, constitution_flags_counter

    resource = Resource.create({"service.name": settings.otel_service_name})

    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint)))
    trace.set_tracer_provider(trace_provider)

    if settings.enable_metrics:
        metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=settings.otel_endpoint))
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        meter = metrics.get_meter(__name__)
        skill_confidence_histogram = meter.create_histogram("skill_confidence", unit="1", description="Skill confidence scores")
        mcp_call_duration = meter.create_histogram("mcp_call_duration_ms", unit="ms", description="MCP call duration")
        batch_success_rate = meter.create_gauge("batch_success_rate", unit="1", description="Batch success rate")
        constitution_flags_counter = meter.create_counter("constitution_flags_total", unit="1", description="Constitutional flags count")

    LoggingInstrumentor().instrument(set_logging_format=True)
    HTTPXClientInstrumentor().instrument()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.CallsiteParameterAdder([
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]),
            structlog.processors.JSONRenderer() if settings.log_json else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger = structlog.get_logger(__name__)
    logger.info("observability_initialized", service=settings.otel_service_name)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name).bind(
        run_id=run_id_var.get(),
        batch_id=batch_id_var.get(),
        nct_id=nct_id_var.get(),
        trace_id=trace_id_var.get(),
    )


def set_run_id(run_id: str) -> None:
    run_id_var.set(run_id)


def set_batch_id(batch_id: str) -> None:
    batch_id_var.set(batch_id)


def set_nct_id(nct_id: str) -> None:
    nct_id_var.set(nct_id)


def record_skill_confidence(confidence: float, skill_name: str, nct_id: str) -> None:
    if skill_confidence_histogram:
        skill_confidence_histogram.record(confidence, {"skill": skill_name, "nct_id": nct_id})


def record_constitution_flag(principle: str, severity: str) -> None:
    if constitution_flags_counter:
        constitution_flags_counter.add(1, {"principle": principle, "severity": severity})


def traced_skill(skill_name: str | None = None):
    def decorator(func):
        import functools
        import time

        from opentelemetry.trace import SpanKind

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            name = skill_name or getattr(self, "name", func.__name__)
            tracer = trace.get_tracer(__name__)

            with tracer.start_as_current_span(f"skill.{name}", kind=SpanKind.INTERNAL) as span:
                span.set_attribute("skill.name", name)
                span.set_attribute("skill.version", getattr(self, "version", "1.0.0"))

                start_time = time.monotonic()
                log = get_logger(func.__module__)

                log.info("skill_start", input_type=type(args[0]).__name__ if args else "none")

                try:
                    result = await func(self, *args, **kwargs)
                    duration_ms = (time.monotonic() - start_time) * 1000

                    span.set_attribute("skill.duration_ms", duration_ms)
                    span.set_attribute("skill.confidence", result.confidence)
                    span.set_status(trace.Status(trace.StatusCode.OK))

                    log.info("skill_complete", confidence=result.confidence, duration_ms=duration_ms, provenance=result.provenance)

                    if hasattr(result, "confidence") and isinstance(result.confidence, (int, float)):
                        record_skill_confidence(result.confidence, name, getattr(args[0], "nct_id", "unknown") if args else "unknown")

                    return result

                except Exception as e:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    span.set_attribute("skill.duration_ms", duration_ms)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    log.exception("skill_failed", duration_ms=duration_ms, error=str(e))
                    raise

        return wrapper

    return decorator