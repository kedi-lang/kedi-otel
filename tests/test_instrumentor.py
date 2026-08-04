from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Barrier, Thread
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.telemetry import agent_call_span, model_call_span
from kedi.agent_adapter.tool_override import ToolSpec, with_tool_telemetry
from kedi.telemetry import (
    NoOpTelemetryBackend,
    get_backend,
    install_backend,
    restore_backend,
)
from opentelemetry.instrumentation.kedi import (
    KediInstrumentationCleanupError,
    KediInstrumentor,
    OpenTelemetryBackend,
)


@contextmanager
def providers() -> Iterator[tuple[TracerProvider, MeterProvider, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter_provider = MeterProvider(metric_readers=[InMemoryMetricReader()])
    try:
        yield tracer_provider, meter_provider, exporter
    finally:
        tracer_provider.shutdown()
        meter_provider.shutdown()


@contextmanager
def instrumented(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    **kwargs: object,
) -> Iterator[KediInstrumentor]:
    instrumentor = KediInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        **kwargs,
    )
    try:
        yield instrumentor
    finally:
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_instrumentor_installs_and_restores_kedi_and_pydantic() -> None:
    previous_backend = get_backend()
    previous_pydantic = Agent._instrument_default
    with providers() as (tracer_provider, meter_provider, _exporter):
        with instrumented(tracer_provider, meter_provider):
            assert isinstance(get_backend(), OpenTelemetryBackend)
            assert Agent._instrument_default is not previous_pydantic
            instrumentation = Agent._instrument_default
            assert isinstance(instrumentation, InstrumentationSettings)
            assert instrumentation.include_content is False
            assert instrumentation.include_binary_content is False

        assert get_backend() is previous_backend
        assert Agent._instrument_default is previous_pydantic


def test_instrumentor_rolls_back_when_child_instrumentation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.instrumentation import httpx

    previous_backend = get_backend()
    previous_pydantic = Agent._instrument_default

    class BrokenHTTPXInstrumentor:
        is_instrumented_by_opentelemetry = False

        def instrument(self, **kwargs: object) -> None:
            raise RuntimeError("broken httpx")

    monkeypatch.setattr(httpx, "HTTPXClientInstrumentor", BrokenHTTPXInstrumentor)
    with providers() as (tracer_provider, meter_provider, _exporter):
        instrumentor = KediInstrumentor()
        with pytest.raises(RuntimeError, match="broken httpx"):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                instrument_httpx=True,
            )

    assert get_backend() is previous_backend
    assert Agent._instrument_default is previous_pydantic


def test_pydantic_native_agent_span_is_reclassified_without_duplication() -> None:
    with providers() as (tracer_provider, meter_provider, exporter):
        with instrumented(tracer_provider, meter_provider):
            adapter = PydanticAdapter(
                TestModel(call_tools=[], custom_output_text="done", model_name="test")
            )

            async def run() -> str:
                with agent_call_span(
                    adapter,
                    profile_name="reviewer",
                    call_kind="invoke",
                    prompt_chars=5,
                    output_field_count=0,
                    tool_count=0,
                    mcp_server_count=0,
                    artifacts_enabled=True,
                    configured_model=None,
                ):
                    with model_call_span(
                        adapter,
                        call_kind="invoke",
                        configured_model=None,
                    ):
                        return await adapter.invoke(prompt="hello")

            assert asyncio.run(run()) == "done"

        spans = exporter.get_finished_spans()
        run_spans = [
            span
            for span in spans
            if span.attributes is not None
            and span.attributes.get("kedi.operation.name") == "run_agent"
        ]
        assert len(run_spans) == 1
        [agent_span] = run_spans
        assert agent_span.attributes is not None
        assert agent_span.name == "agent reviewer"
        assert agent_span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert agent_span.attributes["gen_ai.agent.name"] == "reviewer"
        assert agent_span.attributes["kedi.adapter.name"] == "pydantic"
        assert "hello" not in str(agent_span.attributes)
        assert sum(span.name == "chat test" for span in spans) == 1


def test_pydantic_native_tool_span_is_reclassified_without_duplication() -> None:
    calls: list[str] = []

    def lookup(*, query: str = "kedi") -> str:
        calls.append(query)
        return "private result"

    tool = with_tool_telemetry(
        ToolSpec(fn=lookup, name="lookup", risk="read_only"),
        adapter_shortname="pydantic",
    )
    with providers() as (tracer_provider, meter_provider, exporter):
        with instrumented(tracer_provider, meter_provider):
            adapter = PydanticAdapter(TestModel(call_tools=["lookup"], model_name="test"))

            async def run() -> str:
                with agent_call_span(
                    adapter,
                    profile_name="researcher",
                    call_kind="invoke",
                    prompt_chars=11,
                    output_field_count=0,
                    tool_count=1,
                    mcp_server_count=0,
                    artifacts_enabled=True,
                    configured_model=None,
                ):
                    with adapter.tool_scope((tool,)):
                        return await adapter.invoke(prompt="Call lookup")

            assert asyncio.run(run()) == '{"lookup":"private result"}'

        spans = exporter.get_finished_spans()
        semantic_agent_spans = [
            span
            for span in spans
            if span.attributes is not None
            and span.attributes.get("kedi.operation.name") == "run_agent"
        ]
        semantic_tool_spans = [
            span
            for span in spans
            if span.attributes is not None
            and span.attributes.get("kedi.operation.name") == "call_tool"
        ]
        assert len(semantic_agent_spans) == 1
        assert len(semantic_tool_spans) == 1
        [agent_span] = semantic_agent_spans
        [tool_span] = semantic_tool_spans
        assert agent_span.context is not None
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == agent_span.context.span_id
        assert tool_span.name == "call lookup"
        assert tool_span.attributes is not None
        assert tool_span.attributes["gen_ai.tool.name"] == "lookup"
        assert tool_span.attributes["kedi.tool.risk"] == "read_only"
        assert "private result" not in str(tool_span.attributes)
        assert calls == ["kedi"]


def test_invalid_configuration_does_not_install_backend() -> None:
    previous_backend = get_backend()
    instrumentor = KediInstrumentor()
    with pytest.raises(ValueError, match="Unknown Kedi runtime detail"):
        instrumentor.instrument(runtime_detail="everything")
    assert get_backend() is previous_backend


def test_removed_agentic_switch_is_rejected() -> None:
    previous_backend = get_backend()
    instrumentor = KediInstrumentor()
    with pytest.raises(
        TypeError,
        match="Unknown Kedi instrumentation options: agentic_enabled",
    ):
        instrumentor.instrument(agentic_enabled=False)
    assert get_backend() is previous_backend


def test_instrumentor_can_disable_pydantic_native_ownership() -> None:
    with providers() as (tracer_provider, meter_provider, _exporter):
        with instrumented(
            tracer_provider,
            meter_provider,
            instrument_pydantic_ai=False,
        ):
            backend = get_backend()
            assert isinstance(backend, OpenTelemetryBackend)
            assert backend.owns_native_spans("pydantic", "run_agent") is False


def test_instrumentor_capture_content_is_explicit() -> None:
    with providers() as (tracer_provider, meter_provider, _exporter):
        with instrumented(
            tracer_provider,
            meter_provider,
            capture_content=True,
        ):
            instrumentation = Agent._instrument_default
            assert isinstance(instrumentation, InstrumentationSettings)
            assert instrumentation.include_content is True
            assert instrumentation.include_binary_content is False


def test_httpx_extra_error_restores_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_backend = get_backend()
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.httpx", None)
    with providers() as (tracer_provider, meter_provider, _exporter):
        instrumentor = KediInstrumentor()
        with pytest.raises(ModuleNotFoundError, match="requires.*httpx"):
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
                instrument_pydantic_ai=False,
                instrument_httpx=True,
            )

    assert get_backend() is previous_backend


def test_httpx_instrumentation_respects_existing_owner_and_owns_new_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.instrumentation import httpx

    calls: list[str] = []

    class FakeHTTPXInstrumentor:
        already_instrumented = False

        @property
        def is_instrumented_by_opentelemetry(self) -> bool:
            return self.already_instrumented

        def instrument(self, **_kwargs: object) -> None:
            calls.append("instrument")

        def uninstrument(self) -> None:
            calls.append("uninstrument")

    monkeypatch.setattr(httpx, "HTTPXClientInstrumentor", FakeHTTPXInstrumentor)
    with providers() as (tracer_provider, meter_provider, _exporter):
        with instrumented(
            tracer_provider,
            meter_provider,
            instrument_pydantic_ai=False,
            instrument_httpx=True,
        ):
            pass
    assert calls == ["instrument"]

    calls.clear()
    FakeHTTPXInstrumentor.already_instrumented = True
    with providers() as (tracer_provider, meter_provider, _exporter):
        with instrumented(
            tracer_provider,
            meter_provider,
            instrument_pydantic_ai=False,
            instrument_httpx=True,
        ):
            pass
    assert calls == []


def test_restore_integrations_is_idempotent_before_instrumentation() -> None:
    KediInstrumentor()._restore_integrations()


def test_repeated_singleton_construction_preserves_restore_state() -> None:
    previous = get_backend()
    with providers() as (tracer_provider, meter_provider, _exporter):
        first = KediInstrumentor()
        first.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            instrument_pydantic_ai=False,
        )
        installed = get_backend()
        second = KediInstrumentor()
        assert second is first
        second.uninstrument()

    assert get_backend() is previous
    assert installed is not previous


def test_concurrent_instrumentation_installs_one_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.instrumentation.kedi as kedi_instrumentation

    created: list[OpenTelemetryBackend] = []
    barrier = Barrier(2)
    original_backend = OpenTelemetryBackend

    class TrackingBackend(original_backend):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            created.append(self)

    monkeypatch.setattr(kedi_instrumentation, "OpenTelemetryBackend", TrackingBackend)
    previous = get_backend()
    with providers() as (tracer_provider, meter_provider, _exporter):
        instrumentor = KediInstrumentor()
        errors: list[BaseException] = []

        def install() -> None:
            try:
                barrier.wait(timeout=5)
                instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                    instrument_pydantic_ai=False,
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [Thread(target=install) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        instrumentor.uninstrument()

    assert errors == []
    assert len(created) == 1
    assert get_backend() is previous


def test_teardown_preserves_newer_backend_and_pydantic_owner() -> None:
    previous_backend = get_backend()
    previous_pydantic = Agent._instrument_default
    newer_backend = NoOpTelemetryBackend()
    newer_pydantic = InstrumentationSettings()
    try:
        with providers() as (tracer_provider, meter_provider, _exporter):
            instrumentor = KediInstrumentor()
            instrumentor.instrument(
                tracer_provider=tracer_provider,
                meter_provider=meter_provider,
            )
            install_backend(newer_backend)
            Agent.instrument_all(newer_pydantic)
            instrumentor.uninstrument()

        assert get_backend() is newer_backend
        assert Agent._instrument_default is newer_pydantic
    finally:
        restore_backend(previous_backend)
        Agent.instrument_all(previous_pydantic)


def test_cleanup_failure_does_not_skip_backend_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.instrumentation.kedi as kedi_instrumentation

    previous_backend = get_backend()
    previous_pydantic = Agent._instrument_default
    with providers() as (tracer_provider, meter_provider, _exporter):
        instrumentor = KediInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
        )

        def fail_restore(_session: object) -> bool:
            raise RuntimeError("pydantic cleanup failed")

        monkeypatch.setattr(
            kedi_instrumentation,
            "restore_pydantic_instrumentation",
            fail_restore,
        )
        with pytest.raises(KediInstrumentationCleanupError) as exc_info:
            instrumentor.uninstrument()

    assert "pydantic cleanup failed" in str(exc_info.value)
    assert get_backend() is previous_backend
    Agent.instrument_all(previous_pydantic)


def test_install_failure_remains_primary_when_rollback_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.instrumentation.kedi as kedi_instrumentation
    from opentelemetry.instrumentation import httpx

    class BrokenHTTPXInstrumentor:
        is_instrumented_by_opentelemetry = False

        def instrument(self, **_kwargs: object) -> None:
            raise RuntimeError("primary install failure")

    def fail_restore(_session: object) -> bool:
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(httpx, "HTTPXClientInstrumentor", BrokenHTTPXInstrumentor)
    monkeypatch.setattr(
        kedi_instrumentation,
        "restore_pydantic_instrumentation",
        fail_restore,
    )
    previous_backend = get_backend()
    previous_pydantic = Agent._instrument_default
    try:
        with providers() as (tracer_provider, meter_provider, _exporter):
            instrumentor = KediInstrumentor()
            with pytest.raises(RuntimeError, match="primary install failure") as exc_info:
                instrumentor.instrument(
                    tracer_provider=tracer_provider,
                    meter_provider=meter_provider,
                    instrument_httpx=True,
                )

        assert isinstance(exc_info.value.__cause__, KediInstrumentationCleanupError)
        assert "secondary cleanup failure" in str(exc_info.value.__cause__)
        assert get_backend() is previous_backend
    finally:
        Agent.instrument_all(previous_pydantic)


def test_configuration_rejects_bool_strings_typos_and_invalid_metric_budget() -> None:
    instrumentor = KediInstrumentor()
    with pytest.raises(TypeError, match="capture_content must be a bool"):
        instrumentor.instrument(instrument_pydantic_ai=False, capture_content="false")
    with pytest.raises(TypeError, match="capture_conent"):
        instrumentor.instrument(instrument_pydantic_ai=False, capture_conent=True)
    with pytest.raises(TypeError, match="positive int"):
        instrumentor.instrument(instrument_pydantic_ai=False, max_metric_instruments=0)


def test_per_agent_disabled_instrumentation_does_not_reclassify_runtime_parent() -> None:
    with providers() as (tracer_provider, meter_provider, exporter):
        with instrumented(tracer_provider, meter_provider):
            adapter = PydanticAdapter(
                TestModel(call_tools=[], custom_output_text="done", model_name="test")
            )
            adapter.instrument = False

            async def run() -> str:
                backend = get_backend()
                with backend.start_span(
                    scope="runtime",
                    name="kedi run parent.kedi",
                    operation="run_program",
                    kind="internal",
                    level="lifecycle",
                    attributes=None,
                ):
                    with agent_call_span(
                        adapter,
                        profile_name="reviewer",
                        call_kind="invoke",
                        prompt_chars=5,
                        output_field_count=0,
                        tool_count=0,
                        mcp_server_count=0,
                        artifacts_enabled=True,
                        configured_model=None,
                    ):
                        return await adapter.invoke(prompt="hello")

            assert asyncio.run(run()) == "done"

        spans = exporter.get_finished_spans()
        assert any(span.name == "kedi run parent.kedi" for span in spans)
        assert any(span.name == "agent reviewer" for span in spans)


def test_pydantic_sensitive_request_metadata_is_redacted_by_default() -> None:
    secret = "SECRET_TOOL_DESCRIPTION_7f4b"

    def lookup(*, query: str = "kedi") -> str:
        return query

    tool = with_tool_telemetry(
        ToolSpec(
            fn=lookup,
            name="lookup",
            description=secret,
            risk="read_only",
        ),
        adapter_shortname="pydantic",
    )
    with providers() as (tracer_provider, meter_provider, exporter):
        with instrumented(tracer_provider, meter_provider):
            adapter = PydanticAdapter(
                TestModel(call_tools=[], custom_output_text="done", model_name="test")
            )

            async def run() -> str:
                with adapter.tool_scope((tool,)):
                    return await adapter.invoke(prompt="hello")

            assert asyncio.run(run()) == "done"

        spans = exporter.get_finished_spans()
        assert secret not in repr(spans)
        chat = next(span for span in spans if span.name == "chat test")
        assert chat.attributes is not None
        assert "model_request_parameters" not in chat.attributes
        assert "gen_ai.tool.definitions" not in chat.attributes
