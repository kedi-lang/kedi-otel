from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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
from kedi.agent_adapter.telemetry import agent_call_span
from kedi.agent_adapter.tool_override import ToolSpec, with_tool_telemetry
from kedi.telemetry import get_backend
from opentelemetry.instrumentation.kedi import KediInstrumentor, OpenTelemetryBackend


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
        assert agent_span.attributes["gen_ai.operation.name"] == "run_agent"
        assert agent_span.attributes["gen_ai.agent.name"] == "reviewer"
        assert agent_span.attributes["kedi.adapter.name"] == "pydantic"
        assert "hello" not in str(agent_span.attributes)
        assert any(span.name == "chat test" for span in spans)


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
            assert instrumentation.include_binary_content is True


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
    assert calls == ["instrument", "uninstrument"]

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
