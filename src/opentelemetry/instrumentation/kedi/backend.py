from __future__ import annotations

import asyncio
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, cast

from opentelemetry import metrics, trace
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import Status, StatusCode, TracerProvider

from kedi.telemetry import (
    Attributes,
    AttributeValue,
    CaptureKind,
    NoOpTelemetryBackend,
    SpanKind,
    SpanLevel,
    TelemetryScope,
    TelemetrySpan,
)

RuntimeDetail = Literal["off", "lifecycle", "detailed"]

_SCOPE_NAMES: dict[TelemetryScope, str] = {
    "runtime": "kedi.runtime",
    "agent": "kedi.agent",
    "agentic": "kedi.agentic",
    "artifacts": "kedi.artifacts",
}
_GEN_AI_OPERATIONS = {
    "run_agent": "invoke_agent",
    "chat": "chat",
    "call_tool": "execute_tool",
}
_MAX_EXCEPTION_MESSAGE_CHARS = 1_000
_MAX_EXCEPTION_STACKTRACE_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class KediTelemetryConfig:
    runtime_detail: RuntimeDetail = "lifecycle"
    agent_enabled: bool = True
    agentic_enabled: bool = True
    artifacts_enabled: bool = True
    capture_content: bool = False
    capture_binary_content: bool = False
    capture_source_paths: bool = False
    capture_source_snippets: bool = False
    capture_model_request_parameters: bool = False
    capture_tool_definitions: bool = False
    capture_exception_messages: bool = False
    capture_exception_stacktraces: bool = False
    max_metric_instruments: int = 128
    native_span_owners: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def scope_enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool:
        if scope == "runtime":
            if self.runtime_detail == "off":
                return False
            return level == "lifecycle" or self.runtime_detail == "detailed"
        if scope == "agent":
            return self.agent_enabled
        if scope == "agentic":
            return self.agentic_enabled
        return self.artifacts_enabled


class OpenTelemetrySpan:
    __slots__ = ("_attributes", "_config", "_exception_recorded", "_span")

    def __init__(
        self,
        span: trace.Span,
        config: KediTelemetryConfig,
        initial_attributes: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        self._span = span
        self._config = config
        self._attributes = dict(initial_attributes or {})
        self._exception_recorded = False

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def get_attribute(self, name: str) -> AttributeValue | None:
        return self._attributes.get(name)

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        safe = _attribute_value(value)
        if safe is not None:
            self._attributes[name] = safe
            self._span.set_attribute(name, cast(Any, safe))

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        self._span.add_event(name, attributes=cast(Any, _attributes(attributes)))

    def record_exception(self, exc: BaseException) -> None:
        if self._exception_recorded:
            return
        self._exception_recorded = True
        attributes: dict[str, AttributeValue] = {
            "exception.type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        }
        if self._config.capture_exception_messages:
            attributes["exception.message"] = str(exc)[:_MAX_EXCEPTION_MESSAGE_CHARS]
        if self._config.capture_exception_stacktraces:
            attributes["exception.stacktrace"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[:_MAX_EXCEPTION_STACKTRACE_CHARS]
        self.add_event("exception", attributes)

    def update_name(self, name: str) -> None:
        self._span.update_name(name)
        self.set_attribute("logfire.msg", name)


def _attribute_value(value: object) -> AttributeValue | None:
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        safe = tuple(item for item in value if isinstance(item, (bool, str, int, float)))
        if len(safe) != len(value):
            return None
        value_types = {type(item) for item in safe}
        return safe if len(value_types) <= 1 else None
    return None


def _attributes(values: Attributes | None) -> dict[str, AttributeValue] | None:
    if values is None:
        return None
    return {
        key: safe for key, value in values.items() if (safe := _attribute_value(value)) is not None
    }


class OpenTelemetryBackend:
    def __init__(
        self,
        *,
        version: str,
        config: KediTelemetryConfig,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self.config = config
        self._tracers = {
            scope: trace.get_tracer(
                name,
                version,
                tracer_provider=tracer_provider,
            )
            for scope, name in _SCOPE_NAMES.items()
        }
        self._meters = {
            scope: metrics.get_meter(
                name,
                version,
                meter_provider=meter_provider,
            )
            for scope, name in _SCOPE_NAMES.items()
        }
        self._instrument_lock = RLock()
        self._counters: dict[tuple[TelemetryScope, str, str], Any] = {}
        self._up_down_counters: dict[tuple[TelemetryScope, str, str], Any] = {}
        self._histograms: dict[tuple[TelemetryScope, str, str], Any] = {}
        self._metric_instrument_count = 0
        self._noop = NoOpTelemetryBackend()
        self._noop_meter = metrics.NoOpMeterProvider().get_meter("kedi.noop")

    @contextmanager
    def start_span(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        operation: str,
        kind: SpanKind,
        level: SpanLevel,
        attributes: Attributes | None,
    ) -> Iterator[TelemetrySpan]:
        if not self.config.scope_enabled(scope, level):
            with self._noop.start_span(
                scope=scope,
                name=name,
                operation=operation,
                kind=kind,
                level=level,
                attributes=attributes,
            ) as span:
                yield span
            return

        span_attributes = dict(_attributes(attributes) or {})
        span_attributes["kedi.operation.name"] = operation
        span_attributes["logfire.msg"] = name
        if semantic_operation := _GEN_AI_OPERATIONS.get(operation):
            span_attributes["gen_ai.operation.name"] = semantic_operation
        otel_kind = OtelSpanKind.CLIENT if kind == "client" else OtelSpanKind.INTERNAL
        with self._tracers[scope].start_as_current_span(
            name,
            kind=otel_kind,
            attributes=cast(Any, span_attributes),
            record_exception=False,
            set_status_on_exception=False,
        ) as raw_span:
            active = OpenTelemetrySpan(raw_span, self.config, span_attributes)
            try:
                yield active
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                active.record_exception(exc)
                raw_span.set_status(Status(StatusCode.ERROR))
                raise

    def current_span(self) -> TelemetrySpan:
        return OpenTelemetrySpan(trace.get_current_span(), self.config)

    def enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool:
        return self.config.scope_enabled(scope, level)

    def add_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        if not self.config.scope_enabled(scope, "lifecycle"):
            return
        instrument = self._instrument(self._counters, scope, name, unit, "counter")
        instrument.add(value, attributes=_attributes(attributes))

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        if not self.config.scope_enabled(scope, "lifecycle"):
            return
        instrument = self._instrument(
            self._up_down_counters,
            scope,
            name,
            unit,
            "up_down_counter",
        )
        instrument.add(value, attributes=_attributes(attributes))

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        if not self.config.scope_enabled(scope, "lifecycle"):
            return
        instrument = self._instrument(self._histograms, scope, name, unit, "histogram")
        instrument.record(value, attributes=_attributes(attributes))

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool:
        return operation in self.config.native_span_owners.get(adapter_shortname, ())

    def capture_enabled(self, kind: CaptureKind) -> bool:
        if kind == "content":
            return self.config.capture_content
        if kind == "source_path":
            return self.config.capture_source_paths
        return self.config.capture_source_snippets

    def _instrument(
        self,
        cache: dict[tuple[TelemetryScope, str, str], Any],
        scope: TelemetryScope,
        name: str,
        unit: str,
        kind: Literal["counter", "up_down_counter", "histogram"],
    ) -> Any:
        key = (scope, name, unit)
        with self._instrument_lock:
            instrument = cache.get(key)
            if instrument is not None:
                return instrument
            if self._metric_instrument_count >= self.config.max_metric_instruments:
                return self._noop_instrument(kind, name, unit)
            meter = self._meters[scope]
            if kind == "counter":
                instrument = meter.create_counter(name, unit=unit)
            elif kind == "up_down_counter":
                instrument = meter.create_up_down_counter(name, unit=unit)
            else:
                instrument = meter.create_histogram(name, unit=unit)
            cache[key] = instrument
            self._metric_instrument_count += 1
            return instrument

    def _noop_instrument(
        self,
        kind: Literal["counter", "up_down_counter", "histogram"],
        name: str,
        unit: str,
    ) -> Any:
        if kind == "counter":
            return self._noop_meter.create_counter(name, unit=unit)
        if kind == "up_down_counter":
            return self._noop_meter.create_up_down_counter(name, unit=unit)
        return self._noop_meter.create_histogram(name, unit=unit)
