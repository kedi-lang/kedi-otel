from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from threading import RLock
from typing import Any, Collection, cast

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import TracerProvider

from kedi.telemetry import (
    TelemetryBackend,
    install_backend,
    restore_backend_if_current,
)

from ._pydantic import (
    PydanticInstrumentationSession,
    install_pydantic_instrumentation,
    restore_pydantic_instrumentation,
)
from .backend import KediTelemetryConfig, OpenTelemetryBackend, RuntimeDetail

_FALLBACK_VERSION = "0.1.0"
_PYDANTIC_NATIVE_OPERATIONS = frozenset({"run_agent", "chat", "call_tool"})
_CONFIGURATION_KEYS = frozenset(
    {
        "agent_enabled",
        "artifacts_enabled",
        "capture_binary_content",
        "capture_content",
        "capture_exception_messages",
        "capture_exception_stacktraces",
        "capture_model_request_parameters",
        "capture_source_paths",
        "capture_source_snippets",
        "capture_tool_definitions",
        "instrument_httpx",
        "instrument_pydantic_ai",
        "max_metric_instruments",
        "meter_provider",
        "runtime_detail",
        "tracer_provider",
    }
)


def _package_version() -> str:
    try:
        return version("opentelemetry-instrumentation-kedi")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


@dataclass(frozen=True, slots=True)
class _InstrumentationSession:
    backend: OpenTelemetryBackend
    previous_backend: TelemetryBackend
    pydantic: PydanticInstrumentationSession | None = None


class KediInstrumentationCleanupError(RuntimeError):
    """One or more independently attempted instrumentation cleanup steps failed."""

    def __init__(self, causes: tuple[BaseException, ...]) -> None:
        self.causes = causes
        summary = "; ".join(f"{type(exc).__name__}: {exc}" for exc in causes)
        super().__init__(f"Kedi instrumentation cleanup failed: {summary}")


class KediInstrumentor(BaseInstrumentor):
    """Install Kedi's OpenTelemetry backend and optional child instrumentors."""

    _lifecycle_lock = RLock()
    _active_session: _InstrumentationSession | None = None
    _pending_cleanup_error: KediInstrumentationCleanupError | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("kedi >= 0.4.0", "pydantic-ai >= 1.95.1, < 3")

    def instrument(self, **kwargs: Any) -> Any:
        with self._lifecycle_lock:
            return super().instrument(**kwargs)

    def uninstrument(self, **kwargs: Any) -> Any:
        with self._lifecycle_lock:
            self._pending_cleanup_error = None
            result = super().uninstrument(**kwargs)
            cleanup_error = self._pending_cleanup_error
            self._pending_cleanup_error = None
            if cleanup_error is not None:
                raise cleanup_error
            return result

    def _instrument(self, **kwargs: object) -> None:
        config, tracer_provider, meter_provider, instrument_pydantic_ai, instrument_httpx = (
            _validated_configuration(kwargs)
        )
        backend = OpenTelemetryBackend(
            version=_package_version(),
            config=config,
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )
        session = _InstrumentationSession(
            backend=backend,
            previous_backend=install_backend(backend),
        )
        try:
            if instrument_pydantic_ai:
                pydantic_session = install_pydantic_instrumentation(
                    config=config,
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                )
                session = replace(session, pydantic=pydantic_session)
            if instrument_httpx:
                _ensure_httpx_instrumented(
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                )
        except BaseException as exc:
            cleanup_error = _cleanup_session(session)
            if cleanup_error is not None:
                raise exc from cleanup_error
            raise
        self._active_session = session

    def _uninstrument(self, **kwargs: object) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown Kedi uninstrumentation options: {unknown}")
        session = self._active_session
        self._active_session = None
        if session is not None:
            self._pending_cleanup_error = _cleanup_session(session)

    def _restore_integrations(self) -> None:
        """Best-effort teardown retained for explicit lifecycle recovery."""

        with self._lifecycle_lock:
            session = self._active_session
            self._active_session = None
            if session is None:
                return
            cleanup_error = _cleanup_session(session)
            if cleanup_error is not None:
                raise cleanup_error


def _cleanup_session(
    session: _InstrumentationSession,
) -> KediInstrumentationCleanupError | None:
    failures: list[BaseException] = []
    if session.pydantic is not None:
        try:
            restore_pydantic_instrumentation(session.pydantic)
        except BaseException as exc:
            failures.append(exc)
    try:
        restore_backend_if_current(
            expected=session.backend,
            replacement=session.previous_backend,
        )
    except BaseException as exc:
        failures.append(exc)
    return KediInstrumentationCleanupError(tuple(failures)) if failures else None


def _validated_configuration(
    kwargs: dict[str, object],
) -> tuple[KediTelemetryConfig, TracerProvider | None, MeterProvider | None, bool, bool]:
    unknown = set(kwargs) - _CONFIGURATION_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown Kedi instrumentation options: {names}")

    runtime_detail = kwargs.get("runtime_detail", "lifecycle")
    if runtime_detail not in {"off", "lifecycle", "detailed"}:
        raise ValueError(f"Unknown Kedi runtime detail: {runtime_detail!r}")
    instrument_pydantic_ai = _strict_bool(kwargs, "instrument_pydantic_ai", True)
    instrument_httpx = _strict_bool(kwargs, "instrument_httpx", False)
    max_metric_instruments = kwargs.get("max_metric_instruments", 128)
    if type(max_metric_instruments) is not int or max_metric_instruments < 1:
        raise TypeError("max_metric_instruments must be a positive int")
    native_owners = {"pydantic": _PYDANTIC_NATIVE_OPERATIONS} if instrument_pydantic_ai else {}
    config = KediTelemetryConfig(
        runtime_detail=cast(RuntimeDetail, runtime_detail),
        agent_enabled=_strict_bool(kwargs, "agent_enabled", True),
        artifacts_enabled=_strict_bool(kwargs, "artifacts_enabled", True),
        capture_content=_strict_bool(kwargs, "capture_content", False),
        capture_binary_content=_strict_bool(kwargs, "capture_binary_content", False),
        capture_source_paths=_strict_bool(kwargs, "capture_source_paths", False),
        capture_source_snippets=_strict_bool(kwargs, "capture_source_snippets", False),
        capture_model_request_parameters=_strict_bool(
            kwargs, "capture_model_request_parameters", False
        ),
        capture_tool_definitions=_strict_bool(kwargs, "capture_tool_definitions", False),
        capture_exception_messages=_strict_bool(kwargs, "capture_exception_messages", False),
        capture_exception_stacktraces=_strict_bool(kwargs, "capture_exception_stacktraces", False),
        max_metric_instruments=max_metric_instruments,
        native_span_owners=native_owners,
    )
    return (
        config,
        cast(TracerProvider | None, kwargs.get("tracer_provider")),
        cast(MeterProvider | None, kwargs.get("meter_provider")),
        instrument_pydantic_ai,
        instrument_httpx,
    )


def _strict_bool(kwargs: dict[str, object], name: str, default: bool) -> bool:
    value = kwargs.get(name, default)
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return cast(bool, value)


def _ensure_httpx_instrumented(
    *,
    tracer_provider: TracerProvider | None,
    meter_provider: MeterProvider | None,
) -> None:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError as exc:
        raise ModuleNotFoundError(
            "HTTPX instrumentation requires opentelemetry-instrumentation-kedi[httpx]"
        ) from exc
    instrumentor = HTTPXClientInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        return
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )


__all__ = [
    "KediInstrumentationCleanupError",
    "KediInstrumentor",
    "KediTelemetryConfig",
    "OpenTelemetryBackend",
    "RuntimeDetail",
]
