from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Collection, cast

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import TracerProvider

from kedi.telemetry import TelemetryBackend, install_backend, restore_backend

from .backend import KediTelemetryConfig, OpenTelemetryBackend, RuntimeDetail

_FALLBACK_VERSION = "0.1.0"
_PYDANTIC_NATIVE_OPERATIONS = frozenset({"run_agent", "chat", "call_tool"})
_UNSET = object()


def _package_version() -> str:
    try:
        return version("kedi-otel-instrumentation")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


class KediInstrumentor(BaseInstrumentor):
    """Install Kedi's OpenTelemetry backend and optional child instrumentors."""

    def __init__(self) -> None:
        super().__init__()
        self._previous_backend: TelemetryBackend | None = None
        self._previous_pydantic_instrumentation: object = _UNSET
        self._httpx_instrumentor: BaseInstrumentor | None = None
        self._owns_httpx_instrumentation = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("kedi >= 0.4.0", "pydantic-ai >= 1.0.17")

    def _instrument(self, **kwargs: object) -> None:
        runtime_detail = kwargs.get("runtime_detail", "lifecycle")
        if runtime_detail not in {"off", "lifecycle", "detailed"}:
            raise ValueError(f"Unknown Kedi runtime detail: {runtime_detail!r}")
        runtime_detail = cast(RuntimeDetail, runtime_detail)
        tracer_provider = cast(TracerProvider | None, kwargs.get("tracer_provider"))
        meter_provider = cast(MeterProvider | None, kwargs.get("meter_provider"))
        instrument_pydantic_ai = bool(kwargs.get("instrument_pydantic_ai", True))
        instrument_httpx = bool(kwargs.get("instrument_httpx", False))
        native_owners = {"pydantic": _PYDANTIC_NATIVE_OPERATIONS} if instrument_pydantic_ai else {}
        config = KediTelemetryConfig(
            runtime_detail=runtime_detail,
            agent_enabled=bool(kwargs.get("agent_enabled", True)),
            agentic_enabled=bool(kwargs.get("agentic_enabled", True)),
            artifacts_enabled=bool(kwargs.get("artifacts_enabled", True)),
            capture_content=bool(kwargs.get("capture_content", False)),
            capture_source=bool(kwargs.get("capture_source", False)),
            native_span_owners=native_owners,
        )
        backend = OpenTelemetryBackend(
            version=_package_version(),
            config=config,
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        self._previous_backend = install_backend(backend)
        try:
            if instrument_pydantic_ai:
                from pydantic_ai import Agent
                from pydantic_ai.models.instrumented import InstrumentationSettings

                self._previous_pydantic_instrumentation = Agent._instrument_default
                Agent.instrument_all(
                    InstrumentationSettings(
                        tracer_provider=tracer_provider,
                        meter_provider=meter_provider,
                        include_content=config.capture_content,
                        include_binary_content=config.capture_content,
                    )
                )

            if instrument_httpx:
                try:
                    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
                except ImportError as exc:
                    raise ModuleNotFoundError(
                        "HTTPX instrumentation requires kedi-otel-instrumentation[httpx]"
                    ) from exc
                httpx_instrumentor = HTTPXClientInstrumentor()
                already_instrumented = httpx_instrumentor.is_instrumented_by_opentelemetry
                self._httpx_instrumentor = httpx_instrumentor
                if not already_instrumented:
                    httpx_instrumentor.instrument(
                        tracer_provider=tracer_provider,
                        meter_provider=meter_provider,
                    )
                    self._owns_httpx_instrumentation = True
        except BaseException:
            self._restore_integrations()
            raise

    def _uninstrument(self, **kwargs: object) -> None:
        self._restore_integrations()

    def _restore_integrations(self) -> None:
        if self._httpx_instrumentor is not None and self._owns_httpx_instrumentation:
            self._httpx_instrumentor.uninstrument()
        if self._httpx_instrumentor is not None:
            self._httpx_instrumentor = None
            self._owns_httpx_instrumentation = False
        if self._previous_pydantic_instrumentation is not _UNSET:
            from pydantic_ai import Agent

            Agent.instrument_all(cast(Any, self._previous_pydantic_instrumentation))
            self._previous_pydantic_instrumentation = _UNSET
        if self._previous_backend is not None:
            restore_backend(self._previous_backend)
            self._previous_backend = None


__all__ = [
    "KediInstrumentor",
    "KediTelemetryConfig",
    "OpenTelemetryBackend",
    "RuntimeDetail",
]
