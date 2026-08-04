# Kedi OpenTelemetry Instrumentation

`opentelemetry-instrumentation-kedi` connects Kedi's dependency-free telemetry seam to
OpenTelemetry. Importing Kedi remains a no-op; telemetry starts only after the
instrumentor is enabled.

```python
from opentelemetry.instrumentation.kedi import KediInstrumentor

KediInstrumentor().instrument()
```

Configure the application's tracer and meter providers with
`service.name=kedi`. Kedi does not replace the application's SDK, sampler,
resource, exporter, or Logfire configuration.

## Configuration

All boolean options require actual `bool` values. Unknown option names and invalid
types raise immediately instead of silently weakening telemetry policy.

```python
KediInstrumentor().instrument(
    runtime_detail="lifecycle",  # "off", "lifecycle", or "detailed"
    agent_enabled=True,
    artifacts_enabled=True,
    capture_content=False,
    capture_binary_content=False,
    capture_source_paths=False,
    capture_source_snippets=False,
    capture_model_request_parameters=False,
    capture_tool_definitions=False,
    capture_exception_messages=False,
    capture_exception_stacktraces=False,
    max_metric_instruments=128,
    instrument_pydantic_ai=True,
    instrument_httpx=False,
    tracer_provider=tracer_provider,
    meter_provider=meter_provider,
)
```

`agent_enabled` controls the complete agent surface, including model and tool
calls, MCP initialization, approvals, subagents, and dynamic workflows.

Content, binary content, source paths, source snippets, model request parameters,
tool definitions, exception messages, and exception stack traces are separate
privacy classes. All are disabled by default. Exception type remains observable;
message and stack data require their respective explicit opt-ins and are bounded.

Pydantic AI instrumentation is enabled by default. Kedi filters model request
parameters and tool definitions before they reach the configured tracer provider
unless their dedicated capture options are enabled. Pydantic's process-wide
instrumentation setting is restored only while Kedi still owns the exact setting;
a newer application owner is never overwritten.

HTTPX instrumentation requires the `httpx` extra and explicit
`instrument_httpx=True`. Enabling it is a one-way, application-owned setup action.
Kedi deliberately does not call global HTTPX `uninstrument()` because the upstream
instrumentor exposes no owner token and teardown could remove another library's
instrumentation.

## Lifecycle

Instrumentation transitions are serialized and idempotent. Teardown attempts every
owned cleanup step even when one fails, restores the Kedi backend with an
owner-aware compare-and-restore operation, and raises
`KediInstrumentationCleanupError` with all cleanup causes. If setup itself fails,
the setup exception remains primary and any rollback failure is chained beneath it.
