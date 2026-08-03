# Kedi OpenTelemetry Instrumentation

`kedi-otel-instrumentation` connects Kedi's dependency-free telemetry seam to
OpenTelemetry. Importing Kedi remains a no-op; telemetry starts only after the
instrumentor is enabled.

```python
from opentelemetry.instrumentation.kedi import KediInstrumentor

KediInstrumentor().instrument()
```

Configure the application's tracer and meter providers with
`service.name=kedi`. Kedi does not replace the application's SDK, sampler, or
exporter configuration. Pydantic AI instrumentation is enabled by default;
HTTPX instrumentation is opt-in with `instrument_httpx=True` and the `httpx`
extra.
