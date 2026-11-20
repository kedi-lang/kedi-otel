from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import signature
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.trace import (
    Link,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    TracerProvider,
)
from opentelemetry.util.types import Attributes, AttributeValue

from .backend import KediTelemetryConfig, _exception_attributes

_MODEL_REQUEST_PARAMETERS = "model_request_parameters"
_TOOL_DEFINITIONS = "gen_ai.tool.definitions"


def _filtered(attributes: Attributes, config: KediTelemetryConfig) -> Attributes:
    if attributes is None:
        return None
    filtered = dict(attributes)
    if not config.capture_model_request_parameters:
        filtered.pop(_MODEL_REQUEST_PARAMETERS, None)
    if not config.capture_tool_definitions:
        filtered.pop(_TOOL_DEFINITIONS, None)
    return filtered


def _filtered_links(
    links: Sequence[Link] | None, config: KediTelemetryConfig
) -> Sequence[Link] | None:
    if links is None:
        return None
    return [Link(link.context, _filtered(link.attributes, config)) for link in links]


class _RedactingSpan(Span):
    """Keep late span updates subject to the same capture policy as creation."""

    def __init__(self, span: Span, config: KediTelemetryConfig) -> None:
        self._span = span
        self._config = config

    def end(self, end_time: int | None = None) -> None:
        self._span.end(end_time)

    def get_span_context(self) -> SpanContext:
        return self._span.get_span_context()

    def is_recording(self) -> bool:
        return self._span.is_recording()

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        self._span.set_attributes(_filtered(attributes, self._config) or {})

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self.set_attributes({key: value})

    def add_event(
        self, name: str, attributes: Attributes = None, timestamp: int | None = None
    ) -> None:
        self._span.add_event(name, _filtered(attributes, self._config), timestamp)

    def add_link(self, context: SpanContext, attributes: Attributes = None) -> None:
        self._span.add_link(context, _filtered(attributes, self._config))

    def update_name(self, name: str) -> None:
        self._span.update_name(name)

    def set_status(self, status: Status | StatusCode, description: str | None = None) -> None:
        self._span.set_status(status, description)

    def record_exception(
        self,
        exception: BaseException,
        attributes: Attributes = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        # Follow the same capture policy as OpenTelemetrySpan: never hand the
        # SDK's record_exception() the exception, it always exports the full
        # message and stacktrace. Emit the event ourselves instead.
        event_attributes = _exception_attributes(exception, self._config)
        if attributes is not None:
            event_attributes = {**(_filtered(attributes, self._config) or {}), **event_attributes}
        if escaped:
            event_attributes["exception.escaped"] = True
        self._span.add_event("exception", event_attributes, timestamp)


class _LazyCurrentSpan(AbstractContextManager[Span]):
    """Start the span on ``__enter__``, not when the manager is built.

    The previous implementation started the span eagerly and wrapped it in
    ``trace.use_span``; a manager that was created but never entered leaked
    a started, never-ended span.
    """

    def __init__(
        self,
        tracer: _RedactingTracer,
        name: str,
        context: Any = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._context = context
        self._kind = kind
        self._attributes = attributes
        self._links = links
        self._start_time = start_time
        self._record_exception = record_exception
        self._set_status_on_exception = set_status_on_exception
        self._end_on_exit = end_on_exit
        self._use_span: AbstractContextManager[Span] | None = None

    def __enter__(self) -> Span:
        span = self._tracer.start_span(
            self._name,
            context=self._context,
            kind=self._kind,
            attributes=self._attributes,
            links=self._links,
            start_time=self._start_time,
            record_exception=self._record_exception,
            set_status_on_exception=self._set_status_on_exception,
        )
        self._use_span = trace.use_span(
            span,
            record_exception=self._record_exception,
            set_status_on_exception=self._set_status_on_exception,
            end_on_exit=self._end_on_exit,
        )
        return self._use_span.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool | None:
        if self._use_span is None:
            return None
        return self._use_span.__exit__(exc_type, exc_value, traceback)


class _RedactingTracer:
    """Filter sensitive Pydantic AI attributes before provider export."""

    def __init__(self, tracer: Tracer, config: KediTelemetryConfig) -> None:
        self._tracer = tracer
        self._config = config

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        span = self._tracer.start_span(
            name,
            context=context,
            kind=kind,
            attributes=_filtered(attributes, self._config),
            links=_filtered_links(links, self._config),
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        return _RedactingSpan(span, self._config)

    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> AbstractContextManager[Span]:
        return _LazyCurrentSpan(
            self,
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        )


class _RedactingTracerProvider:
    def __init__(self, provider: TracerProvider, config: KediTelemetryConfig) -> None:
        self._provider = provider
        self._config = config

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Tracer:
        tracer = self._provider.get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            schema_url,
            attributes,
        )
        return cast(Tracer, _RedactingTracer(tracer, self._config))


@dataclass(frozen=True, slots=True)
class PydanticInstrumentationSession:
    previous: object
    installed: object


def install_pydantic_instrumentation(
    *,
    config: KediTelemetryConfig,
    tracer_provider: TracerProvider | None,
    meter_provider: object | None,
) -> PydanticInstrumentationSession:
    from pydantic_ai import Agent
    from pydantic_ai.models.instrumented import InstrumentationSettings

    previous = Agent._instrument_default
    provider = tracer_provider or cast(TracerProvider, trace.get_tracer_provider())
    settings_kwargs: dict[str, object] = {
        "tracer_provider": _RedactingTracerProvider(provider, config),
        "meter_provider": meter_provider,
        "include_content": config.capture_content,
        "include_binary_content": config.capture_binary_content,
    }
    if "include_model_request_parameters" in signature(InstrumentationSettings).parameters:
        settings_kwargs["include_model_request_parameters"] = (
            config.capture_model_request_parameters
        )
    installed = InstrumentationSettings(**settings_kwargs)
    Agent.instrument_all(installed)
    return PydanticInstrumentationSession(previous=previous, installed=installed)


def restore_pydantic_instrumentation(session: PydanticInstrumentationSession) -> bool:
    from pydantic_ai import Agent

    if Agent._instrument_default is not session.installed:
        return False
    Agent.instrument_all(cast(Any, session.previous))
    return True
