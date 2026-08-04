from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from opentelemetry.instrumentation.kedi import KediTelemetryConfig, OpenTelemetryBackend
from opentelemetry.instrumentation.kedi.backend import (
    OpenTelemetrySpan,
    _attributes,
    _attribute_value,
)


@contextmanager
def providers() -> Iterator[
    tuple[TracerProvider, MeterProvider, InMemorySpanExporter, InMemoryMetricReader]
]:
    resource = Resource.create({"service.name": "kedi"})
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    try:
        yield tracer_provider, meter_provider, span_exporter, metric_reader
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()


def make_backend(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    *,
    config: KediTelemetryConfig | None = None,
) -> OpenTelemetryBackend:
    return OpenTelemetryBackend(
        version="test",
        config=config or KediTelemetryConfig(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )


def test_backend_emits_scoped_span_with_kedi_resource() -> None:
    with providers() as (tracer_provider, meter_provider, exporter, _reader):
        backend = make_backend(tracer_provider, meter_provider)
        with backend.start_span(
            scope="runtime",
            name="kedi run demo.kedi",
            operation="run_program",
            kind="internal",
            level="lifecycle",
            attributes={"kedi.entrypoint": "cli"},
        ) as active:
            active.set_attribute("kedi.result.type", "builtins.str")
            active.add_event("kedi.run.completed", {"kedi.output.present": True})

        [recorded] = exporter.get_finished_spans()
        assert recorded.instrumentation_scope is not None
        assert recorded.attributes is not None
        assert recorded.name == "kedi run demo.kedi"
        assert recorded.instrumentation_scope.name == "kedi.runtime"
        assert recorded.resource.attributes["service.name"] == "kedi"
        assert recorded.attributes["kedi.operation.name"] == "run_program"
        assert recorded.attributes["kedi.entrypoint"] == "cli"
        assert recorded.attributes["kedi.result.type"] == "builtins.str"
        assert recorded.events[0].name == "kedi.run.completed"


def test_backend_respects_runtime_detail_and_scope_switches() -> None:
    config = KediTelemetryConfig(
        runtime_detail="lifecycle",
        agent_enabled=False,
        agentic_enabled=False,
        artifacts_enabled=False,
    )
    with providers() as (tracer_provider, meter_provider, exporter, _reader):
        backend = make_backend(tracer_provider, meter_provider, config=config)
        with backend.start_span(
            scope="runtime",
            name="run python",
            operation="run_python",
            kind="internal",
            level="detailed",
            attributes=None,
        ):
            pass
        with backend.start_span(
            scope="agent",
            name="agent kedi",
            operation="run_agent",
            kind="internal",
            level="lifecycle",
            attributes=None,
        ):
            pass

        assert exporter.get_finished_spans() == ()
        assert backend.enabled("runtime", "lifecycle") is True
        assert backend.enabled("runtime", "detailed") is False
        assert backend.enabled("agent", "lifecycle") is False
        assert backend.enabled("agentic", "lifecycle") is False
        assert backend.enabled("artifacts", "lifecycle") is False


def test_backend_emits_metrics_and_filters_disabled_metrics() -> None:
    config = KediTelemetryConfig(runtime_detail="detailed")
    with providers() as (tracer_provider, meter_provider, _exporter, reader):
        backend = make_backend(tracer_provider, meter_provider, config=config)
        backend.add_counter(
            scope="runtime",
            name="kedi.python.execution.count",
            value=1,
            unit="1",
            attributes={"outcome": "success"},
        )
        backend.add_counter(
            scope="runtime",
            name="kedi.python.execution.count",
            value=1,
            unit="1",
            attributes={"outcome": "success"},
        )
        backend.record_histogram(
            scope="runtime",
            name="kedi.python.execution.duration",
            value=0.25,
            unit="s",
            attributes={"outcome": "success"},
        )
        backend.add_up_down_counter(
            scope="runtime",
            name="kedi.run.active",
            value=1,
            unit="1",
            attributes={"entrypoint": "cli"},
        )

        data = reader.get_metrics_data()
        assert data is not None
        names = {
            metric.name
            for resource_metrics in data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }
        assert names == {
            "kedi.python.execution.count",
            "kedi.python.execution.duration",
            "kedi.run.active",
        }

    with providers() as (tracer_provider, meter_provider, _exporter, reader):
        backend = make_backend(
            tracer_provider,
            meter_provider,
            config=KediTelemetryConfig(runtime_detail="off"),
        )
        backend.add_counter(
            scope="runtime",
            name="kedi.run.count",
            value=1,
            unit="1",
            attributes=None,
        )
        backend.add_up_down_counter(
            scope="runtime",
            name="kedi.run.active",
            value=1,
            unit="1",
            attributes=None,
        )
        backend.record_histogram(
            scope="runtime",
            name="kedi.run.duration",
            value=0.1,
            unit="s",
            attributes=None,
        )
        assert reader.get_metrics_data() is None


def test_backend_marks_gen_ai_operations_and_sanitizes_attributes() -> None:
    with providers() as (tracer_provider, meter_provider, exporter, _reader):
        backend = make_backend(tracer_provider, meter_provider)
        with backend.start_span(
            scope="agent",
            name="agent reviewer",
            operation="run_agent",
            kind="client",
            level="lifecycle",
            attributes={"safe.sequence": ("one", "two")},
        ):
            pass

        [recorded] = exporter.get_finished_spans()
        assert recorded.attributes is not None
        assert recorded.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert recorded.attributes["safe.sequence"] == ("one", "two")


class _RawSpan:
    def __init__(self) -> None:
        self.attributes: object = object()
        self.recorded: list[BaseException] = []
        self.name = "before"
        self.events: list[tuple[str, object]] = []

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, name: str, value: object) -> None:
        if not isinstance(self.attributes, dict):
            self.attributes = {}
        self.attributes[name] = value

    def add_event(self, name: str, attributes: object = None) -> None:
        self.events.append((name, attributes))

    def record_exception(self, exc: BaseException) -> None:
        self.recorded.append(exc)

    def update_name(self, name: str) -> None:
        self.name = name


def test_otel_span_adapter_and_attribute_sanitization() -> None:
    raw = _RawSpan()
    span = OpenTelemetrySpan(raw, KediTelemetryConfig())  # type: ignore[arg-type]
    assert span.is_recording() is True
    assert span.get_attribute("missing") is None

    span.set_attribute("safe", "value")
    span.set_attribute("unsafe", (1, "two"))
    assert span.get_attribute("safe") == "value"
    assert span.get_attribute("unsafe") is None
    span.add_event("filtered", {"items": (1, "two")})
    span.add_event("event", {"items": (1, 2)})
    error = RuntimeError("failed")
    span.record_exception(error)
    span.update_name("after")

    assert raw.events == [
        ("filtered", {}),
        ("event", {"items": (1, 2)}),
        ("exception", {"exception.type": "builtins.RuntimeError"}),
    ]
    assert raw.recorded == []
    assert raw.name == "after"
    assert span.get_attribute("logfire.msg") == "after"
    assert _attribute_value((1, 2)) == (1, 2)
    assert _attribute_value((1, "two")) is None
    assert _attribute_value((1, object())) is None
    assert _attribute_value(object()) is None
    assert _attributes(None) is None
    assert _attributes({"safe": "value", "unsafe": Any}) == {"safe": "value"}


def test_backend_reports_capture_and_native_ownership() -> None:
    backend = OpenTelemetryBackend(
        version="test",
        config=KediTelemetryConfig(
            capture_content=True,
            capture_source_paths=True,
            capture_source_snippets=False,
            native_span_owners={"pydantic": frozenset({"run_agent", "chat"})},
        ),
    )
    assert backend.capture_enabled("content") is True
    assert backend.capture_enabled("source_path") is True
    assert backend.capture_enabled("source_snippet") is False
    assert backend.owns_native_spans("pydantic", "run_agent") is True
    assert backend.owns_native_spans("pydantic", "call_tool") is False


def test_backend_preserves_context_across_asyncio_to_thread() -> None:
    with providers() as (tracer_provider, meter_provider, exporter, _reader):
        backend = make_backend(tracer_provider, meter_provider)

        def child() -> None:
            with backend.start_span(
                scope="runtime",
                name="compile module child",
                operation="compile_module",
                kind="internal",
                level="lifecycle",
                attributes=None,
            ):
                pass

        async def run() -> None:
            with backend.start_span(
                scope="runtime",
                name="kedi run demo.kedi",
                operation="run_program",
                kind="internal",
                level="lifecycle",
                attributes=None,
            ):
                await asyncio.to_thread(child)

        asyncio.run(run())
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert spans["compile module child"].parent is not None
        assert spans["kedi run demo.kedi"].context is not None
        assert (
            spans["compile module child"].parent.span_id
            == spans["kedi run demo.kedi"].context.span_id
        )


def test_exception_details_are_private_by_default_and_cancellation_is_not_error() -> None:
    secret = "SECRET_PROMPT_TOKEN_7f4b"
    with providers() as (tracer_provider, meter_provider, exporter, _reader):
        backend = make_backend(tracer_provider, meter_provider)
        with pytest.raises(RuntimeError, match=secret):
            with backend.start_span(
                scope="runtime",
                name="secret failure",
                operation="run_program",
                kind="internal",
                level="lifecycle",
                attributes=None,
            ):
                raise RuntimeError(secret)
        with pytest.raises(asyncio.CancelledError):
            with backend.start_span(
                scope="runtime",
                name="cancelled",
                operation="run_program",
                kind="internal",
                level="lifecycle",
                attributes=None,
            ):
                raise asyncio.CancelledError

        spans = {span.name: span for span in exporter.get_finished_spans()}
        failure = spans["secret failure"]
        [event] = failure.events
        assert event.name == "exception"
        assert event.attributes == {"exception.type": "builtins.RuntimeError"}
        assert secret not in repr(failure)
        assert failure.status.status_code is StatusCode.ERROR
        cancelled = spans["cancelled"]
        assert cancelled.events == ()
        assert cancelled.status.status_code is StatusCode.UNSET


def test_exception_message_and_stacktrace_have_separate_bounded_opt_ins() -> None:
    secret = "s" * 2_000
    config = KediTelemetryConfig(
        capture_exception_messages=True,
        capture_exception_stacktraces=True,
    )
    with providers() as (tracer_provider, meter_provider, exporter, _reader):
        backend = make_backend(tracer_provider, meter_provider, config=config)
        with pytest.raises(RuntimeError):
            with backend.start_span(
                scope="runtime",
                name="detailed failure",
                operation="run_program",
                kind="internal",
                level="lifecycle",
                attributes=None,
            ):
                raise RuntimeError(secret)

        [recorded] = exporter.get_finished_spans()
        [event] = recorded.events
        assert event.attributes is not None
        assert len(event.attributes["exception.message"]) == 1_000
        assert len(event.attributes["exception.stacktrace"]) <= 8_000


def test_metric_instrument_cache_respects_configured_budget() -> None:
    config = KediTelemetryConfig(max_metric_instruments=1)
    with providers() as (tracer_provider, meter_provider, _exporter, reader):
        backend = make_backend(tracer_provider, meter_provider, config=config)
        backend.add_counter(
            scope="runtime",
            name="kedi.first",
            value=1,
            unit="1",
            attributes=None,
        )
        backend.add_counter(
            scope="runtime",
            name="kedi.second",
            value=1,
            unit="1",
            attributes=None,
        )

        data = reader.get_metrics_data()
        assert data is not None
        names = {
            metric.name
            for resource_metrics in data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }
        assert names == {"kedi.first"}
