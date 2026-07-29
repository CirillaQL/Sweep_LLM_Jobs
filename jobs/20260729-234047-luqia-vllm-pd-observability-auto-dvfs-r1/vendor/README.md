# Offline OpenTelemetry bundle

`otel_bundle.zip` is a merged, importable zip of universal Python wheels:

- `opentelemetry-api==1.39.0`
- `opentelemetry-sdk==1.39.0`
- `opentelemetry-semantic-conventions==0.60b0`
- `opentelemetry-exporter-otlp-proto-http==1.39.0`
- `opentelemetry-exporter-otlp-proto-common==1.39.0`
- `opentelemetry-proto==1.39.0`
- `googleapis-common-protos==1.72.0`
- `importlib-metadata==8.7.0`
- `typing-extensions==4.15.0`
- `zipp==3.23.0`

SHA-256:

```text
82267b3062e8100a726fca36a991ce7d67d37e3950bb86db3f015ae2ff3cab4f
```

The shared vLLM environment supplies `requests` and `protobuf`; vLLM 0.15.1
requires protobuf 6.33.5 or newer, while this bundle supports protobuf 5.x/6.x.
The bundle was verified with Python 3.10 by importing the SDK, HTTP/protobuf
exporter, and trace collector protobuf classes, then starting the Job's OTLP
collector and probing its health and trace endpoints.
