from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, cast

from opentelemetry import metrics, trace
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import TracerProvider

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
_GEN_AI_OPERATIONS = frozenset({"run_agent", "chat", "call_tool"})


@dataclass(frozen=True, slots=True)
class KediTelemetryConfig:
    runtime_detail: RuntimeDetail = "lifecycle"
    agent_enabled: bool = True
    agentic_enabled: bool = True
    artifacts_enabled: bool = True
    capture_content: bool = False
    capture_source: bool = False
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
    __slots__ = ("_span",)

    def __init__(self, span: trace.Span) -> None:
        self._span = span

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def get_attribute(self, name: str) -> AttributeValue | None:
        attributes = getattr(self._span, "attributes", None)
        if not isinstance(attributes, Mapping):
            return None
        value = attributes.get(name)
        return _attribute_value(value)

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        safe = _attribute_value(value)
        if safe is not None:
            self._span.set_attribute(name, cast(Any, safe))

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        self._span.add_event(name, attributes=cast(Any, _attributes(attributes)))

    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)

    def update_name(self, name: str) -> None:
        self._span.update_name(name)
        self._span.set_attribute("logfire.msg", name)


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
        self._noop = NoOpTelemetryBackend()

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
        if operation in _GEN_AI_OPERATIONS:
            span_attributes["gen_ai.operation.name"] = operation
        otel_kind = OtelSpanKind.CLIENT if kind == "client" else OtelSpanKind.INTERNAL
        with self._tracers[scope].start_as_current_span(
            name,
            kind=otel_kind,
            attributes=cast(Any, span_attributes),
            record_exception=True,
            set_status_on_exception=True,
        ) as raw_span:
            yield OpenTelemetrySpan(raw_span)

    def current_span(self) -> TelemetrySpan:
        return OpenTelemetrySpan(trace.get_current_span())

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
        return self.config.capture_content if kind == "content" else self.config.capture_source

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
            meter = self._meters[scope]
            if kind == "counter":
                instrument = meter.create_counter(name, unit=unit)
            elif kind == "up_down_counter":
                instrument = meter.create_up_down_counter(name, unit=unit)
            else:
                instrument = meter.create_histogram(name, unit=unit)
            cache[key] = instrument
            return instrument
